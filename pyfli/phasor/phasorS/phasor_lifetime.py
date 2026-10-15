"""Apparent single-exponential lifetimes of a phasor relative to the universal semicircle."""

import numpy as np
from numpy.typing import ArrayLike

from .phasor_locus_tools import _omega_rad_per_ns, modulation_lifetime, phase_lifetime

APPARENT_LIFETIMES = ("tau_phi", "tau_m", "tau_n")
"""Keys returned by :func:`apparent_lifetimes`: phase, modulation and normal lifetime."""


def _semicircle_angle(g: ArrayLike, s: ArrayLike) -> np.ndarray:
    """
    Angle ``theta`` in ``[0, pi]`` of the semicircle point nearest to each phasor, seen
    from the centre ``(1/2, 0)``. Phasors below the ``G`` axis map to the nearest end of
    the semicircle (``theta = 0`` at ``(1, 0)``, ``theta = pi`` at ``(0, 0)``); the centre
    itself and non-finite phasors are ``NaN``.
    """
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    dg = g - 0.5
    theta = np.arctan2(s, dg)
    theta = np.where(theta < 0, np.where(dg >= 0, 0.0, np.pi), theta)
    undefined = ~(np.isfinite(g) & np.isfinite(s)) | ((dg == 0) & (s == 0))
    return np.where(undefined, np.nan, theta)


def semicircle_nearest_point(
    g: ArrayLike, s: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """
    Nearest point ``(G_N, S_N)`` of the universal semicircle to each phasor ``(g, s)``.

    Parameters
    ----------
    g, s : array_like
        Phasor coordinates (any shape).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(G_N, S_N)`` with the shape of `g`; ``NaN`` for the semicircle centre and for
        non-finite phasors.
    """
    theta = _semicircle_angle(g, s)
    return 0.5 + 0.5 * np.cos(theta), 0.5 * np.sin(theta)


def normal_lifetime(
    g: ArrayLike, s: ArrayLike, frequency_hz: float, harmonic: int = 1
) -> np.ndarray:
    """
    Normal lifetime ``tau_N``: the lifetime of the semicircle point nearest to each phasor.

    Parameters
    ----------
    g, s : array_like
        IRF-calibrated phasor coordinates (any shape).
    frequency_hz : float
        Laser repetition (modulation) frequency in hertz.
    harmonic : int
        Harmonic of the phasor.

    Returns
    -------
    np.ndarray
        ``tau_N = tan(theta / 2) / omega`` in nanoseconds, same shape as `g`: ``0`` at
        ``(1, 0)``, ``inf`` at ``(0, 0)`` (and for phasors below the axis nearest to those
        ends), ``NaN`` at the semicircle centre and for non-finite phasors.
    """
    theta = _semicircle_angle(g, s)
    omega = _omega_rad_per_ns(frequency_hz, harmonic)
    with np.errstate(invalid="ignore"):
        return np.where(theta == np.pi, np.inf, np.tan(theta / 2.0) / omega)


def apparent_lifetimes(
    g: ArrayLike, s: ArrayLike, frequency_hz: float, harmonic: int = 1
) -> dict[str, np.ndarray]:
    """
    Phase, modulation and normal lifetimes of each phasor.

    Parameters
    ----------
    g, s : array_like
        IRF-calibrated phasor coordinates (any shape, e.g. ``(H, W)`` maps).
    frequency_hz : float
        Laser repetition (modulation) frequency in hertz.
    harmonic : int
        Harmonic of the phasor.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"tau_phi", "tau_m", "tau_n"}`` (see :data:`APPARENT_LIFETIMES`), each in
        nanoseconds with the shape of `g`.
    """
    return {
        "tau_phi": phase_lifetime(g, s, frequency_hz, harmonic),
        "tau_m": modulation_lifetime(g, s, frequency_hz, harmonic),
        "tau_n": normal_lifetime(g, s, frequency_hz, harmonic),
    }
