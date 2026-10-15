"""Analytic noise-free expected decay of the pyfli simulator for given lifetimes and IRF."""

from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from pyfli.reconstruction.common_reconstruct import periodic_bin_exponential

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
    n = irf.shape[-1]
    p_emit = wrapped_gate_exponential(tau, n, laser_period)
    p = np.fft.irfft(
        np.fft.rfft(p_emit, axis=-1) * np.fft.rfft(irf, axis=-1), n=n, axis=-1
    )
    return np.clip(p, 0.0, None)


class DecayForwardModel:
    def __init__(
        self,
        n_bins: int,
        period_ns: float,
        detector_type: str,
        jitter: bool = True,
        inactive_bins=(),
    ) -> None:
        self.n_bins = n_bins
        self.period = period_ns
        self.dt = period_ns / n_bins
        self.t = np.arange(n_bins) * self.dt
        self.circular = detector_type == "discrete"
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
        taus = np.clip(np.atleast_1d(np.asarray(taus, float)), 1e-3, None)
        fracs = np.atleast_1d(np.asarray(fracs, float))
        if self.circular:
            p = wrapped_gate_exponential(taus, self.n_bins, self.period)
        else:
            tt = taus[:, None]
            sf = 1.0 / (1.0 - np.exp(-self.period / tt))
            p = sf / tt * np.exp(-self.t[None, :] / tt)
        return (fracs[:, None] * p).sum(0)

    def _jitter_average(self, y: np.ndarray) -> np.ndarray:
        if not self.jitter:
            return y
        out = np.zeros_like(y)
        n = len(y)
        for s in range(
            -2, 3
        ):  # BaseDiscrete/ContinuousSimulator: shift ~ U{-2..2}, zero filled
            if s > 0:
                out[s:] += y[: n - s]
            elif s < 0:
                out[:s] += y[-s:]
            else:
                out += y
        return out / 5.0

    def shape(self, irf_norm: np.ndarray, taus, fracs) -> np.ndarray:
        """Expected decay shape (sums to 1 over the active bins) for a sum-normalised IRF."""
        if self.circular:
            taus = np.clip(np.atleast_1d(np.asarray(taus, float)), 1e-3, None)
            fracs = np.atleast_1d(np.asarray(fracs, float))
            y = fracs @ gated_bin_probabilities(taus, irf_norm, self.period)
        else:
            y = fftconvolve(self.components(taus, fracs), irf_norm, mode="full")[
                : self.n_bins
            ]
        y = self._jitter_average(np.clip(y, 0, None)) * self.active
        s = y.sum()
        return y / s if s > 0 else self.active / self.n_active

    def flat(self) -> np.ndarray:
        """Flat background spread over the active bins (sums to 1)."""
        return self.active / self.n_active
