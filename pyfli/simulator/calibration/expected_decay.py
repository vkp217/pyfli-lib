"""Analytic noise-free expected decay of the pyfli simulator for given lifetimes and IRF."""

from typing import Any

import numpy as np

from pyfli.cm_funcs.forward_model import model_numpy, validate_excitation
from pyfli.reconstruction.common_reconstruct import (
    circular_convolve,
    periodic_bin_exponential,
)

from ..irf_sim.irf_generator import IRFGenerator


def wrapped_gate_exponential(tau: Any, n_bins: int, laser_period: float) -> np.ndarray:
    """
    Probability that a photon emitted with an exponential delay of lifetime(s) `tau`
    (ns, any shape ``S``) falls in each of `n_bins` equal gates covering one laser
    period, with delays wrapped modulo the period
    (:func:`pyfli.reconstruction.common_reconstruct.periodic_bin_exponential`). Returns
    ``S + (n_bins,)``, summing to 1 along the last axis.
    """
    return periodic_bin_exponential(tau, n_bins, laser_period)


def gated_bin_probabilities(
    tau: Any, irf: np.ndarray, laser_period: float
) -> np.ndarray:
    """
    Expected per-gate photon distribution of ``PhotonCountSimulator`` for lifetime(s)
    `tau`: the wrapped, gate-integrated exponential (:func:`wrapped_gate_exponential`)
    circularly convolved with the IRF.

    Parameters
    ----------
    tau : float | np.ndarray
        Lifetime(s) in ns; any shape ``S``.
    irf : np.ndarray
        IRF of shape ``(T,)`` (shared) or ``S + (T,)`` (one per lifetime); normalised
        to unit sum here.
    laser_period : float
        Laser repetition period in ns; the ``T`` gates cover ``[0, laser_period)``.

    Returns
    -------
    np.ndarray
        Array of shape ``S + (T,)`` whose last axis sums to 1.
    """
    irf = np.asarray(irf, dtype=float)
    irf = irf / irf.sum(axis=-1, keepdims=True)
    p_emit = wrapped_gate_exponential(tau, irf.shape[-1], laser_period)
    return np.clip(circular_convolve(p_emit, irf), 0.0, None)


class DecayForwardModel:
    def __init__(
        self,
        n_bins: int,
        period_ns: float,
        detector_type: str,
        jitter: bool = True,
        inactive_bins=(),
        excitation: str = "bin_start",
    ) -> None:
        self.excitation = validate_excitation(excitation)
        self.n_bins = n_bins
        self.period = period_ns
        self.dt = period_ns / n_bins
        self.t = np.arange(n_bins) * self.dt
        self.circular = True
        self.jitter = jitter
        self.active = np.ones(n_bins, dtype=bool)
        self.active[np.asarray(inactive_bins, dtype=int)] = False

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    # ------------------------------------------------------------------ IRF
    def simulated_irf(self, mu: float, sigma_ns: float, tail_ns: float) -> np.ndarray:
        """Peak-normalised exponentially modified Gaussian IRF (pyfli ``IRFGenerator``)."""
        return IRFGenerator.gaussianExpIRF(
            mu=mu,
            T=self.period,
            num_bins=self.n_bins,
            sigma=sigma_ns,
            tau=max(tail_ns, 1e-3),
        )

    @staticmethod
    def with_baseline(irf_peak1: np.ndarray, b: float) -> np.ndarray:
        """What ``OffsetGen`` does to the IRF (``+ b``) followed by the simulator's sum-normalisation."""
        irf = irf_peak1 + b
        return irf / irf.sum()

    # ------------------------------------------------------------- lifetimes
    def components(self, taus, fracs) -> np.ndarray:
        """Photon fraction per gate before the IRF (wrapped for circular, one pulse otherwise)."""
        taus = np.clip(np.atleast_1d(np.asarray(taus, float)), 1e-3, None)
        fracs = np.atleast_1d(np.asarray(fracs, float))
        if self.circular:
            p = wrapped_gate_exponential(taus, self.n_bins, self.period)
        else:
            edges = np.append(self.t, self.period)[None, :]
            p = -np.diff(np.exp(-edges / taus[:, None]), axis=1)
        return (fracs[:, None] * p).sum(0)

    def _jitter_average(self, y: np.ndarray) -> np.ndarray:
        """Mean over the simulator's circular jitter, a roll by U{-2..2} bins."""
        if not self.jitter:
            return y
        return np.mean([np.roll(y, s) for s in range(-2, 3)], axis=0)

    def _zero_pad_shape(self, irf_norm: np.ndarray, taus, fracs) -> np.ndarray:
        """Single-pulse model averaged over the simulator's continuous U(-2, 2)-bin jitter."""
        offsets = np.linspace(-2.0, 2.0, 41) if self.jitter else np.zeros(1)
        y = np.zeros(self.n_bins)
        for tau, frac in zip(taus, fracs):
            for s in offsets:
                y += frac * model_numpy(
                    self.t,
                    irf_norm,
                    [1.0, tau, 0.0, s * self.dt],
                    "mono-exponential",
                    "zero_pad",
                    self.excitation,
                )
        return y / len(offsets)

    def shape(self, irf_norm: np.ndarray, taus, fracs) -> np.ndarray:
        """Expected decay shape (sums to 1 over the active bins) for a sum-normalised IRF."""
        taus = np.clip(np.atleast_1d(np.asarray(taus, float)), 1e-3, None)
        fracs = np.atleast_1d(np.asarray(fracs, float))
        if self.circular and self.excitation == "bin_start":
            y = fracs @ gated_bin_probabilities(taus, irf_norm, self.period)
            y = self._jitter_average(np.clip(y, 0, None))
        elif self.circular:
            y = sum(
                frac
                * model_numpy(
                    self.t,
                    irf_norm,
                    [1.0, tau, 0.0, 0.0],
                    "mono-exponential",
                    "circular",
                    self.excitation,
                )
                for tau, frac in zip(taus, fracs)
            )
            y = self._jitter_average(np.clip(y, 0, None))
        else:
            y = np.clip(self._zero_pad_shape(irf_norm, taus, fracs), 0, None)
        y = y * self.active
        s = y.sum()
        return y / s if s > 0 else self.active / self.n_active

    def flat(self) -> np.ndarray:
        """Flat background spread over the active bins (sums to 1)."""
        return self.active / self.n_active
