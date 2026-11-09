# pyfli/simulator/sim_engine_common.py

"""
Shared IRF/timing setup, cycles-range validation, and TCSPC photon-binning logic
for the sampling engines wrapped by the "combined" and "separate" simulator pairs.

This module belongs to :mod:`pyfli.simulator` and is part of PyFLI synthetic FLI/FLIM
data generation, hardware noise modeling, calibration, and validation tools.

:class:`~pyfli.simulator.combined.simulator_engine.FLIEngine` and
:class:`~pyfli.simulator.separate.model_simulator.FLIModelSimulator` both wrap a
single IRF into per-pixel decay/parameter sampling and TCSPC photon-histogram
simulation; they differ in how lifetime parameters are sampled and how many
exponential components the analytical decay supports. :class:`BaseFLIEngine`
owns the parts that are identical between them — IRF validation, timing setup,
``n_cycles`` range normalization, photon-to-amplitude fraction conversion, the
expected decay (the solver's :func:`~pyfli.cm_funcs.forward_model.model_numpy`), and the
IRF-convolution/pile-up/binning tail of the TCSPC simulation — so that logic is
defined exactly once.
"""

from typing import Any

import numpy as np

from pyfli.cm_funcs.forward_model import (
    decay_kernel,
    model_numpy,
    validate_excitation,
    validate_gate_width,
)

from .noise_models import NoiseEngine
from .sim_helper import irf_picker


def model_spec(p: dict[str, Any]) -> tuple[str, tuple[float, ...]]:
    """``(model_type, lifetime parameters)`` of a sampled pixel, in the solver's order."""
    if p.get("mono", False) and "tau" in p:
        return "mono-exponential", (p["tau"],)
    return "bi-exponential", (p["f"], p["tau1"], p["tau2"])


class BaseFLIEngine:
    """
    Shared setup and helpers for FLI sampling engines. The config keys ``excitation``
    (``"bin_start"`` default, or ``"uniform"`` IRF photons spread over their bin) and
    ``gate_width`` (ns, gated continuous sensors only) select the forward-model options
    of :func:`pyfli.cm_funcs.forward_model.model_numpy` for both the photons and the
    expected decay.

    Parameters
    ----------
    irf_full : np.ndarray
        Full instrument response function sampled over the decay window.
    laser_feq : int
        Laser repetition frequency used by the simulation.
    seed : int | None
        Seed for reproducible random sampling.
    """

    def __init__(
        self,
        irf_full: np.ndarray,
        laser_feq: int = 80,
        seed: int | None = None,
    ) -> None:
        irf, _ = irf_picker(irf_full)
        # Timing and Normalization
        irf_sum = irf.sum()
        if not np.isfinite(irf_sum) or irf_sum <= 0:
            raise ValueError(
                f"Invalid IRF: sum={irf_sum}. IRF must be non-negative and non-zero."
            )
        self.irf = irf / irf_sum
        self.rng = np.random.default_rng(seed)
        self.laser_period = 1000 / laser_feq
        # N bins each of width dt covering [0, laser_period); endpoint-exclusive
        self.dt = self.laser_period / len(self.irf)
        self.t = np.arange(len(self.irf)) * self.dt

    @staticmethod
    def _normalize_cycles_range(n_cycles: int | tuple[int, int]) -> tuple[int, int]:
        """
        Normalizes ``n_cycles`` into a validated ``(low, high)`` range.

        A bare int is the upper bound, with the lower bound fixed at 1000.
        A ``(low, high)`` tuple sets both bounds explicitly. Both bounds must
        be >= 1000, and ``high`` must be >= ``low``.
        """
        if isinstance(n_cycles, (int, np.integer)):
            cycles_range = (1000, int(n_cycles))
        else:
            low_cycles, high_cycles = n_cycles
            cycles_range = (int(low_cycles), int(high_cycles))
        if cycles_range[0] < 1000 or cycles_range[1] < 1000:
            raise ValueError(f"n_cycles bounds {cycles_range} must both be >= 1000.")
        if cycles_range[1] < cycles_range[0]:
            raise ValueError(
                f"n_cycles upper bound ({cycles_range[1]}) must be >= "
                f"lower bound ({cycles_range[0]})."
            )
        return cycles_range

    @staticmethod
    def _amplitude_fractions(f: float, tau1: float, tau2: float) -> tuple[float, float]:
        """Pre-exponential amplitude fractions ``(A1, A2)`` of the photon fractions ``(f, 1 - f)``."""
        w1 = f / max(tau1, 1e-12)
        w2 = (1.0 - f) / max(tau2, 1e-12)
        total = w1 + w2
        return (w1 / total, w2 / total) if total > 0 else (f, 1.0 - f)

    @property
    def shift_method(self) -> str:
        """Solver ``shift_method`` matching ``pileup_mode`` (``"wrap"`` is the periodic steady state)."""
        wrap = self.params_cfg.get("pileup_mode", "wrap") == "wrap"
        return "circular" if wrap else "zero_pad"

    @property
    def excitation(self) -> str:
        """Excitation within each IRF gate (``params_cfg["excitation"]``, default ``"bin_start"``)."""
        return validate_excitation(self.params_cfg.get("excitation", "bin_start"))

    @property
    def gate_width(self) -> float | None:
        """Detection-gate width in ns (``params_cfg["gate_width"]``, default the bin width)."""
        return validate_gate_width(self.params_cfg.get("gate_width"))

    def sample_jitter(self, max_bins: int = 2) -> float:
        """Timing jitter in bins: an integer roll for ``"circular"``, continuous for ``"zero_pad"``."""
        if not max_bins:
            return 0.0
        if self.shift_method == "circular":
            return float(np.random.randint(-max_bins, max_bins + 1))
        return float(np.random.uniform(-max_bins, max_bins))

    def expected_decay(
        self,
        p: dict[str, Any],
        photon_count: float = 1.0,
        jitter_bins: float = 0.0,
        v_shift: float = 0.0,
    ) -> np.ndarray:
        """Expected counts per gate: the solver's forward model at the pixel's true parameters,
        delayed by `jitter_bins` (``h_shift = jitter_bins * dt``)."""
        model_type, lifetimes = model_spec(p)
        options = (self.excitation, self.gate_width)
        if self.shift_method == "circular":
            params = [photon_count, *lifetimes, v_shift, 0.0]
            base = model_numpy(
                self.t, self.irf, params, model_type, "circular", *options
            )
            return np.roll(base, int(jitter_bins))
        params = [photon_count, *lifetimes, v_shift, jitter_bins * self.dt]
        return model_numpy(self.t, self.irf, params, model_type, "zero_pad", *options)

    def analytical_decay(self, p: dict[str, Any]) -> np.ndarray:
        """Noise-free photon fraction per gate before the IRF (onset 0, periodic for ``"wrap"``)."""
        model_type, lifetimes = model_spec(p)
        kernel, _ = decay_kernel(
            self.t,
            (1.0, *lifetimes, 0.0),
            model_type,
            0.0,
            self.shift_method,
            self.excitation,
            self.gate_width,
        )
        return kernel

    def _bin_tcspc_photons(self, times: np.ndarray, jitter_bins: float = 0.0) -> Any:
        """
        Delays photon emission times by an IRF draw (inverse-transform sampling of the
        IRF's CDF), applies the pile-up/repetitive-excitation window filter, and bins the
        result into a TCSPC histogram. The jitter delay is a histogram roll by
        `jitter_bins` for ``"wrap"`` and ``jitter_bins * dt`` ns per photon for ``"truncate"``.
        With ``excitation="uniform"`` each IRF photon is placed uniformly inside its bin.
        """
        if self.gate_width is not None:
            raise ValueError(
                "gate_width applies to gated detection, not TCSPC photons."
            )
        total_photons = len(times)
        irf_cdf = np.cumsum(self.irf)
        irf_cdf[-1] = 1.0
        irf_bins = np.searchsorted(irf_cdf, self.rng.random(total_photons))
        if self.excitation == "uniform":
            irf_shifts = (irf_bins + self.rng.random(total_photons)) * self.dt
        else:
            irf_shifts = irf_bins * self.dt

        circular = self.shift_method == "circular"
        delay = 0.0 if circular else jitter_bins * self.dt
        pileup_mode = self.params_cfg.get("pileup_mode", "wrap")
        arrival_times = NoiseEngine.tcspc_pileup_filter(
            times + irf_shifts + delay, self.laser_period, mode=pileup_mode
        )

        bins = (arrival_times / self.dt).astype(np.int32)
        hist = np.bincount(bins[bins < len(self.t)], minlength=len(self.t))
        if circular:
            hist = np.roll(hist, int(jitter_bins))
        return hist.astype(np.float64)
