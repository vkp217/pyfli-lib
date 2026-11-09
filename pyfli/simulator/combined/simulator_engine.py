# simulator/combined/simulator_engine
"""
Drive the main FLI parameter sampler and TCSPC simulation engine.

This module belongs to :mod:`pyfli.simulator.combined` and is part of PyFLI synthetic
FLI/FLIM data generation, hardware noise modeling, calibration, and validation tools.
Public API includes classes :class:`FLIEngine`.

Shared IRF/timing setup, ``n_cycles`` range validation, and TCSPC photon-binning logic
live in :mod:`pyfli.simulator.sim_engine_common`.
"""

from typing import Any

import numpy as np

from ..distributions import ParameterSampler
from ..sim_engine_common import BaseFLIEngine


class FLIEngine(BaseFLIEngine):
    """
    Run the fliengine routine.
    analytical decays, and simulates TCSPC counts from a shared IRF and configuration.

    Parameters
    ----------
    irf_full : np.ndarray
        Full instrument response function sampled over the decay window.
    tau2 : tuple[int, float]
        Long lifetime component.
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

    def sample_all_params(self) -> dict[Any, Any]:
        """Samples lifetime and fraction parameters for a single pixel."""
        t2 = ParameterSampler.truncated_normal(*self.params_cfg["tau2"])
        is_mono = self.rng.random() < self.params_cfg["mono"]

        if is_mono:
            if self.rng.random() < 0.9:
                E, f = 0.0, self.rng.uniform(0.99, 0.9999)
            else:
                E, f = self.rng.uniform(0.99, 1.0), self.rng.uniform(0.0001, 0.01)
            t1 = t2 * (1 - E)
            A1, A2 = self._amplitude_fractions(f, t1, t2)
            return {
                "mono": True,
                "E": E,
                "f": f,
                "tau1": t1,
                "tau2": t2,
                "A1": A1,
                "A2": A2,
            }

        E = ParameterSampler.sample_beta(
            *self.params_cfg["eff"], scale=0.9, offset=0.1, rng=self.rng
        )
        f = ParameterSampler.sample_beta(
            *self.params_cfg["A1"], scale=0.9, offset=0.05, rng=self.rng
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

    def get_analytical_decay(self, p: Any) -> Any:
        """Noise-free photon fraction per gate before the IRF (:meth:`analytical_decay`)."""
        return self.analytical_decay(p)

    def simulate_tcspc(
        self,
        p: Any,
        n_cycles: int,
        mu_per_cycle: np.ndarray,
        jitter_bins: float = 0.0,
    ) -> Any:
        """Photon-by-photon logic for TCSPC mode."""
        total_photons = self.rng.poisson(mu_per_cycle * n_cycles)
        if total_photons == 0:
            return np.zeros_like(self.t)

        # Emission (Inverse Transform Sampling)
        comp1 = self.rng.random(total_photons) < p["f"]
        times = np.empty(total_photons)
        times[comp1] = self.rng.exponential(p["tau1"], size=comp1.sum())
        times[~comp1] = self.rng.exponential(p["tau2"], size=(~comp1).sum())

        return self._bin_tcspc_photons(times, jitter_bins)
