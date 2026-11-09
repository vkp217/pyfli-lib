# pyfli/simulator/main_common.py

"""
Shared engine-wrapping pipelines for the continuous (ICCD-style) and discrete
(TCSPC-style) simulator pairs.

``combined/main_factory.py`` (:class:`~pyfli.simulator.combined.main_factory.MacroSimulator`,
:class:`~pyfli.simulator.combined.main_factory.TCSPCSimulator`, wrapping
:class:`~pyfli.simulator.combined.simulator_engine.FLIEngine`) and
``separate/main_factory_gen.py`` (:class:`~pyfli.simulator.separate.main_factory_gen.ContinuousSimulator`,
:class:`~pyfli.simulator.separate.main_factory_gen.PhotonCountSimulator`, wrapping
:class:`~pyfli.simulator.separate.model_simulator.FLIModelSimulator`) implement the identical
noise/scaling pipeline; they differ only in which engine class they wrap, the names of
a few methods on that engine, and whether the "mono" branch trims the returned maps
dict. Subclasses declare those differences as class attributes; this module owns the
actual pipeline math so it is defined exactly once.
"""

from typing import Any

import numpy as np

from .distributions import ParameterSampler
from .noise_models import NoiseEngine


def _build_maps(
    p: Any,
    photon_count: Any,
    mono_only_maps: bool,
    jitter_bins: float,
    dt: float,
    v_shift: float = 0.0,
) -> dict[Any, Any]:
    """
    Build the parameter-maps dict, optionally trimmed for mono pixels, in the solver's
    convention: ``photon_count_map`` is ``S``, ``v_shift_map`` the mean background per bin,
    ``h_shift_map`` the jitter delay in ns (``h_shift_jit_map`` is the same delay in bins).
    """
    shifts = {
        "h_shift_jit_map": jitter_bins,
        "h_shift_map": jitter_bins * dt,
        "v_shift_map": v_shift,
    }
    if mono_only_maps and p["mono"]:
        return {
            "tau_map": p["tau"],
            "photon_count_map": photon_count,
            "mono_map": p["mono"],
            **shifts,
        }
    return {
        "tau1_map": p["tau1"],
        "tau2_map": p["tau2"],
        "alpha1_map": p["f"],
        "A1_map": p["A1"],
        "A2_map": p["A2"],
        "fret_efficiency_map": p["E"],
        "tau_mean_map": p["tau1"] * p["f"] + p["tau2"] * (1 - p["f"]),
        "photon_count_map": photon_count,
        "mono_map": p["mono"],
        **shifts,
    }


class BaseContinuousSimulator:
    """
    Shared ``__call__`` pipeline for MacroSimulator/ContinuousSimulator: samples
    lifetime parameters, scales the expected decay (the engine's
    :meth:`~pyfli.simulator.sim_engine_common.BaseFLIEngine.expected_decay`, i.e. the
    solver's forward model, circular for ``pileup_mode="wrap"``) to a Beta-sampled peak
    intensity, then applies, in the order of a camera, QE, jitter, shot noise on the
    signal, dark counts, read noise, rounding and clipping. ``fit_map`` is the
    expectation of the result before read noise, rounding and clipping (signal + dark
    counts).

    Subclasses set the following class attributes:

    engine_cls : type
        The engine class to wrap (``FLIEngine`` / ``FLIModelSimulator``).
    sample_params_name : str
        Name of the engine method that samples lifetime parameters.
    analytical_decay_name : str
        Name of the engine method that returns the clean un-convolved decay (the
        pipeline itself uses the engine's ``expected_decay``).
    mono_only_maps : bool
        Whether ``p["mono"]`` trims the returned maps dict down to
        ``{tau_map, photon_count_map, mono_map}`` (``True``, as in
        ``ContinuousSimulator``) or the full map set is always returned
        regardless of mono status (``False``, as in ``MacroSimulator``).
    """

    engine_cls: type
    sample_params_name: str
    analytical_decay_name: str
    mono_only_maps: bool = False

    def __init__(
        self, irf_data: np.ndarray, sensor_type: str = "continuous", **cfg: Any
    ) -> None:
        # Toggles
        self.use_jitter = cfg.get("jitter", True)
        self.use_dcr = cfg.get("dcr_on", True)
        self.use_poisson = cfg.get("poisson", True)
        self.use_qe = cfg.get("qe_on", True)
        self.use_read_noise = cfg.get("read_noise_on", True)
        self.use_rounding = cfg.get("round_on", True)
        self.use_clipping = cfg.get("clip_on", True)

        self.sensor_type = sensor_type.upper()
        self.engine = self.engine_cls(irf_data, **cfg)

    def __call__(self) -> dict[Any, Any]:
        """
        Run the instance as a callable.

        Returns
        -------
        dict[Any, Any]
            Dictionary containing the data produced by call.
        """
        p = getattr(self.engine, self.sample_params_name)()

        # 1. Determine target intensity (A) based on bit-depth
        alpha_pc, beta_pc = self.engine.params_cfg["pc"]
        bit_depth = self.engine.params_cfg["bit"]
        max_adc_val = (2**bit_depth) - 1

        A = ParameterSampler.beta_sample(alpha_pc, beta_pc, scale=max_adc_val)

        unit = self.engine.expected_decay(p)
        S = A / unit.max() if unit.max() > 0 else 0.0
        if self.use_qe:
            S = S * ParameterSampler.sample_qe(self.sensor_type)

        jitter = self.engine.sample_jitter() if self.use_jitter else 0.0
        signal = self.engine.expected_decay(p, S, jitter)

        obs = NoiseEngine.apply_poisson(signal) if self.use_poisson else signal.copy()

        background = 0.0
        if self.use_dcr:
            background = self.engine.params_cfg["dcr"] * (bit_depth / 8.0)
            obs = NoiseEngine.apply_dcr(obs, background)

        if self.use_read_noise and self.sensor_type == "CONTINUOUS":
            hw = ParameterSampler.sample_noise_params(bit_depth, self.sensor_type)
            obs = NoiseEngine.apply_read_noise(obs, hw["read_sigma"])

        if self.use_rounding:
            obs = np.round(obs)

        if self.use_clipping:
            obs = np.clip(obs, 0, max_adc_val)

        fit_map = signal + background

        maps = _build_maps(
            p, S, self.mono_only_maps, jitter, self.engine.dt, background
        )

        return {
            "raw_data": {"decay": obs, "irf": self.engine.irf},
            "results": {
                "maps": maps,
                "TR_maps": {"fit_map": fit_map, "residual_map": obs - fit_map},
            },
        }


class BaseDiscreteSimulator:
    """
    Shared ``__call__`` pipeline for TCSPCSimulator/PhotonCountSimulator: samples
    lifetime parameters, runs a TCSPC photon-by-photon histogram (with the timing jitter
    applied to the photons), and adds dark counts and counter clipping. ``fit_map`` is the
    exact expectation of the histogram, the engine's
    :meth:`~pyfli.simulator.sim_engine_common.BaseFLIEngine.expected_decay` (the solver's
    forward model, circular for ``pileup_mode="wrap"``) plus the dark counts.

    Subclasses set the following class attributes:

    engine_cls : type
        The engine class to wrap (``FLIEngine`` / ``FLIModelSimulator``).
    sample_params_name : str
        Name of the engine method that samples lifetime parameters.
    analytical_decay_name : str
        Name of the engine method that returns the clean un-convolved decay (the
        pipeline itself uses the engine's ``expected_decay``).
    simulate_tcspc_name : str
        Name of the engine method that runs the TCSPC photon histogram.
    mono_only_maps : bool
        As in :class:`BaseContinuousSimulator`.
    """

    engine_cls: type
    sample_params_name: str
    analytical_decay_name: str
    simulate_tcspc_name: str
    mono_only_maps: bool = False

    def __init__(
        self, irf_data: np.ndarray, sensor_type: str = "discrete", **cfg: Any
    ) -> None:
        self.use_jitter = cfg.get("jitter", True)
        self.use_dcr = cfg.get("dcr_on", True)
        self.use_qe = cfg.get("qe_on", False)

        # TCSPC is inherently integer-based, but clipping simulates counter overflow
        self.use_clipping = cfg.get("clip_on", True)

        self.sensor_type = sensor_type.upper()
        # TCSPC counters are 16-bit by default
        cfg.setdefault("bit", 16)
        self.engine = self.engine_cls(irf_data, **cfg)

    def __call__(self) -> dict[Any, Any]:
        """
        Run the instance as a callable.

        Returns
        -------
        dict[Any, Any]
            Dictionary containing the data produced by call.
        """
        p = getattr(self.engine, self.sample_params_name)()
        low_cycles, high_cycles = self.engine.params_cfg["cycles"]
        alpha_cyc, beta_cyc = self.engine.params_cfg["pc"]
        n_cycles = round(
            ParameterSampler.sample_beta(
                alpha_cyc,
                beta_cyc,
                scale=high_cycles - low_cycles,
                offset=low_cycles,
            )
        )
        mu_per_cycle = 0.01
        bit_depth = self.engine.params_cfg["bit"]
        max_bin_count = (2**bit_depth) - 1

        effective_mu = (
            mu_per_cycle * ParameterSampler.sample_qe(self.sensor_type)
            if self.use_qe
            else mu_per_cycle
        )
        jitter = self.engine.sample_jitter() if self.use_jitter else 0.0
        obs = getattr(self.engine, self.simulate_tcspc_name)(
            p, n_cycles, effective_mu, jitter
        )

        total_photons_expected = effective_mu * n_cycles

        background = 0.0
        if self.use_dcr:
            background = self.engine.params_cfg["dcr"]
            obs = NoiseEngine.apply_dcr(obs, background)
        fit = self.engine.expected_decay(p, total_photons_expected, jitter, background)

        if self.use_clipping:
            obs = np.clip(obs, 0, max_bin_count)

        maps = _build_maps(
            p,
            total_photons_expected,
            self.mono_only_maps,
            jitter,
            self.engine.dt,
            background,
        )

        return {
            "raw_data": {"decay": obs, "irf": self.engine.irf},
            "results": {
                "maps": maps,
                "TR_maps": {"fit_map": fit, "residual_map": obs - fit},
            },
        }
