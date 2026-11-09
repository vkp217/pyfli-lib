# simulator/separate/model_simulator
"""
Generate mono- and bi-exponential model parameters and simulated TCSPC observations.

This module belongs to :mod:`pyfli.simulator.separate` and is part of PyFLI synthetic
FLI/FLIM data generation, hardware noise modeling, calibration, and validation tools.
Public API includes classes :class:`FLIModelSimulator`.

Shared IRF/timing setup, ``n_cycles`` range validation, and TCSPC photon-binning logic
live in :mod:`pyfli.simulator.sim_engine_common`.
"""

from typing import Any

import numpy as np

from ..distributions import ParameterSampler
from ..sim_engine_common import BaseFLIEngine


class FLIModelSimulator(BaseFLIEngine):
    """
    Run the flimodel simulator routine.
    observations. The class is useful for controlled model-generation experiments
    independent of image geometry.

    Parameters
    ----------
    irf_full : np.ndarray
        Full instrument response function sampled over the decay window.
    tau2 : tuple[int, float]
        Long lifetime component.
    tau2_dist : str
        Distribution used to sample donor-only lifetimes.
    tau2_beta_range : tuple[float, ...]
        Shape-parameter range for beta-distributed tau2 values.
    efficiency : tuple[int, ...]
        FRET transfer efficiency used to derive simulated lifetime components.
    A1_fraction : tuple[int, ...]
        Beta shape of the photon (intensity) fraction ``f`` of the first component, the
        solver's ``alpha1``; ``A1_map``/``A2_map`` report the matching pre-exponential
        amplitude fractions.
    photo_count : tuple[float, int]
        Expected photon count used to scale the simulated decay.
    mono_fraction : float
        Fraction of pixels or events assigned to the mono-exponential component.
    bit : int
        Bit depth or quantization setting for simulated detector output.
    n_cycles : int | tuple[int, int]
        Number of excitation cycles used when constructing the simulated decay.
        A bare int is the upper bound (lower bound fixed at 1000); a
        ``(low, high)`` tuple sets both bounds, and both must be >= 1000.
        The per-call cycle count is drawn from a Beta distribution (shaped
        by ``photo_count``) over that range.
    dcr : float
        Detector dark-count rate used by the noise model.
    laser_feq : int
        Laser repetition frequency used by the simulation.
    pileup_mode : str
        'wrap' (default) folds photons back via modulo; 'truncate' drops them.
    seed : int | None
        Seed for reproducible random sampling.
    **kwargs : Any
        Additional keyword arguments forwarded to the underlying implementation.
    """

    def __init__(
        self,
        irf_full: np.ndarray,
        tau2: tuple[int, float] = (1, 0.5),
        tau2_dist: str = "normal",  # 'normal' -> truncated_normal | 'beta' -> sample_beta
        tau2_beta_range: tuple[float, ...] = (
            4.8,
            0.2,
        ),  # (scale, offset), only used when tau2_dist='beta'
        efficiency: tuple[int, ...] = (5, 5),
        A1_fraction: tuple[int, ...] = (5, 5),
        photo_count: tuple[float, int] = (1.0, 1.0),
        mono_fraction: float = 0.2,
        bit: int = 8,
        n_cycles: int | tuple[int, int] = 800_000,
        dcr: float = 0.05,
        laser_feq: int = 80,
        pileup_mode: str = "wrap",
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(irf_full, laser_feq=laser_feq, seed=seed)

        cycles_range = self._normalize_cycles_range(n_cycles)

        #  Parameters Storage
        self.params_cfg = {
            "tau2": tau2,
            "tau2_dist": tau2_dist,
            "tau2_beta_range": tau2_beta_range,
            "eff": efficiency,
            "A1": A1_fraction,
            "pc": photo_count,
            "mono": mono_fraction,
            "bit": bit,
            "cycles": cycles_range,
            "dcr": dcr,
            "pileup_mode": pileup_mode,
            **kwargs,
        }

    def _sample_tau2(self) -> Any:
        """Draws tau2 from either a truncated normal or a beta prior, per tau2_dist."""
        if self.params_cfg["tau2_dist"] == "beta":
            scale, offset = self.params_cfg["tau2_beta_range"]
            t2 = ParameterSampler.sample_beta(
                *self.params_cfg["tau2"], scale=scale, offset=offset, rng=self.rng
            )
        else:
            t2 = ParameterSampler.truncated_normal(*self.params_cfg["tau2"])
        # Guard against a zero (or negative) lifetime, which would blow up any 1/tau2 term downstream
        return max(t2, 1e-3)

    def sample_mono_params(self) -> dict[Any, Any]:
        """Samples the lifetime parameter for a single pixel (pure single-exponential)."""
        t2 = self._sample_tau2()
        return {"mono": True, "tau": t2}

    def sample_bi_params(self) -> dict[Any, Any]:
        """Samples lifetime and fraction parameters for a single pixel."""
        t2 = self._sample_tau2()

        E = ParameterSampler.sample_beta(
            *self.params_cfg["eff"], scale=0.998, offset=0.001, rng=self.rng
        )
        f = ParameterSampler.sample_beta(
            *self.params_cfg["A1"], scale=0.998, offset=0.001, rng=self.rng
        )
        t1 = t2 * (1 - E)
        A1, A2 = self._amplitude_fractions(f, t1, t2)
        return {
            "mono": False,
            "E": E,
            "f": f,
            "tau1": t1,
            "tau2": t2,
            "A1": A1,
            "A2": A2,
        }

    def sample_params(self) -> Any:
        """
        Sample params.

        Returns
        -------
        Any
            Object produced by sample params.
        """
        if self.rng.random() < self.params_cfg["mono"]:
            return self.sample_mono_params()
        return self.sample_bi_params()

    def get_model_analytical_decay(self, p: Any) -> Any:
        """Noise-free photon fraction per gate before the IRF (:meth:`analytical_decay`)."""
        return self.analytical_decay(p)

    def simulate_model_tcspc(
        self,
        p: Any,
        n_cycles: int,
        mu_per_cycle: np.ndarray,
        jitter_bins: float = 0.0,
    ) -> Any:
        """
        Simulate model TCSPC.

        Parameters
        ----------
        p : Any
            Detector parameter object or fitted parameter vector.
        n_cycles : int
            Number of simulated laser cycles.
        mu_per_cycle : np.ndarray
            Expected photons per laser cycle in the TCSPC simulation.

        Returns
        -------
        Any
            Object produced by simulate model TCSPC.
        """
        total_photons = self.rng.poisson(mu_per_cycle * n_cycles)
        if total_photons == 0:
            return np.zeros_like(self.t)

        # Emission (Inverse Transform Sampling)
        times = np.empty(total_photons)
        if p.get("mono", False):
            times[:] = self.rng.exponential(p["tau"], size=total_photons)
        else:
            comp1 = self.rng.random(total_photons) < p["f"]
            times[comp1] = self.rng.exponential(p["tau1"], size=comp1.sum())
            times[~comp1] = self.rng.exponential(p["tau2"], size=(~comp1).sum())

        return self._bin_tcspc_photons(times, jitter_bins)
