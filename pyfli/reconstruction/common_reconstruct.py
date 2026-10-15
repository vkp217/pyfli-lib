import numpy as np


def periodic_bin_exponential(tau, n_bins, period):
    """
    Fraction of the photons of a periodically excited mono-exponential decay (lifetime
    ``tau``, any shape ``S``) that falls in each of ``n_bins`` equal bins covering one
    repetition ``period`` (same time unit as ``tau``). In steady state the tails of all
    earlier pulses add up to the same exponential shape, so bin ``k`` receives
    ``exp(-k dt / tau) (1 - exp(-dt / tau)) / (1 - exp(-period / tau))`` with
    ``dt = period / n_bins``. Returns ``S + (n_bins,)``, summing to 1 along the last axis.
    """
    tau = np.asarray(tau, dtype=float)[..., None]
    dt = period / n_bins
    m = np.arange(n_bins)
    return -np.expm1(-dt / tau) * np.exp(-m * dt / tau) / -np.expm1(-n_bins * dt / tau)


def periodic_instrument_exponential(tau, n_bins, period):
    """
    Per-bin photon distribution of a periodically excited mono-exponential decay as a
    real time-resolved instrument records it: excitation (IRF) photons arrive anywhere
    inside their bin, not at its start, so this is the decay kernel that, circularly
    convolved with the binned IRF, gives the binned measurement when the IRF density is
    uniform within each bin. With ``x = dt / tau`` and ``dt = period / n_bins``:

    * bin 0: ``1 - (1 - e^-x) / x`` (decays starting and ending in the excitation bin)
      plus the wrapped tails of earlier pulses;
    * bin ``m >= 1``: ``e^(-m x) (1 - e^-x)(e^x - 1) / x``, divided by
      ``1 - exp(-period / tau)`` for the periodic steady state.

    Its discrete Fourier transform at the laser frequency equals the continuous
    phasor ``1 / (1 - i omega tau)`` up to ~1e-4, so IRF-calibrated phasors of such data
    lie on the universal semicircle. :func:`periodic_bin_exponential` is the kernel when
    every excitation is at a bin start (pyfli's ``PhotonCountSimulator``). Returns
    ``S + (n_bins,)``, summing to 1 along the last axis.
    """
    tau = np.asarray(tau, dtype=float)[..., None]
    dt = period / n_bins
    x = dt / tau
    m = np.arange(n_bins)
    inner = -np.expm1(-x) * np.expm1(x) / x
    steady = 1.0 / -np.expm1(-n_bins * x)
    out = np.exp(-m * x) * inner * steady
    first = 1.0 - (-np.expm1(-x)) / x + np.exp(-n_bins * x) * inner * steady
    out[..., 0] = first[..., 0]
    return out


def mono_reconstruction(t, tau, S=1.0):
    return (S / tau) * np.exp(-t / tau)


def bi_reconstruction(t, tau1, tau2, A1, A2):
    return (A1 / tau1) * np.exp(-t / tau1) + (A2 / tau2) * np.exp(-t / tau2)


def mono_reconstruction_torch(t, tau, S=1.0):
    return (S / tau) * (-t / tau).exp()


def bi_reconstruction_torch(t, tau1, tau2, A1, A2):
    return (A1 / tau1) * (-t / tau1).exp() + (A2 / tau2) * (-t / tau2).exp()
