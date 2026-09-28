"""
Acquisition configuration and lifetime helpers built on :class:`MonoLocus`.

This module belongs to :mod:`pyfli.phasor.phasorS`. It complements
:class:`~pyfli.phasor.phasorS.phasor_locus.MonoLocus` with:

- :class:`AcquisitionMode` / :class:`AcquisitionConfig` -- one validated object
  describing the acquisition geometry, which traces its own locus;
- :func:`phase_lifetime` / :func:`modulation_lifetime` -- lifetime estimators for
  any harmonic;
- :func:`lifetime_from_locus` -- nearest-point lifetime on any traced locus;
- :func:`phase_lifetime_gated` -- exact phase inversion of the single-gate locus;
- :func:`discrete_locus_circle` -- analytic centre and radius of the binned locus;
- :func:`plot_discrete_n_sweep` -- convergence of the binned locus with the number
  of bins.

Units follow :class:`MonoLocus`: frequencies in hertz, times in nanoseconds, window
lengths as fractions of the laser period.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from .phasor_locus import MonoLocus
from .phasor_simple_utils import (
    _add_frequency_label,
    _style_phasor_ax,
    _universal_circle_xy,
)


class AcquisitionMode(StrEnum):
    """Acquisition geometries with an analytical mono-exponential locus."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    GATED_SINGLE = "gated_single"
    GATED_N = "gated_n"
    TRUNCATED = "truncated"
    OFFSET = "offset"


@dataclass
class AcquisitionConfig:
    """
    Validated description of an acquisition geometry.

    Parameters
    ----------
    mode : AcquisitionMode | str
        Acquisition geometry (``"continuous"``, ``"discrete"``, ``"gated_single"``,
        ``"gated_n"``, ``"truncated"`` or ``"offset"``).
    frequency_hz : float
        Laser repetition frequency in hertz.
    harmonic : int
        Phasor harmonic.
    n_bins : int
        Number of bins over one period (``discrete``).
    gate_width_frac : float
        Gate width as a fraction of the period (``gated_single``, ``gated_n``).
    n_gates : int
        Number of equidistant gates over one period (``gated_n``).
    t_rec_frac : float
        Recorded window as a fraction of the period (``truncated``).
    t0_frac : float
        Excitation offset as a fraction of the period (``offset``).
    """

    mode: AcquisitionMode | str = AcquisitionMode.CONTINUOUS
    frequency_hz: float = 80e6
    harmonic: int = 1
    n_bins: int = 256
    gate_width_frac: float = 0.5
    n_gates: int = 4
    t_rec_frac: float = 1.0
    t0_frac: float = 0.0

    def __post_init__(self) -> None:
        self.mode = AcquisitionMode(self.mode)
        if self.frequency_hz <= 0:
            raise ValueError(f"frequency_hz must be positive, got {self.frequency_hz}")
        if self.harmonic < 1:
            raise ValueError(f"harmonic must be >= 1, got {self.harmonic}")
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {self.n_bins}")
        if not 0 < self.gate_width_frac <= 1:
            raise ValueError(
                f"gate_width_frac must be in (0, 1], got {self.gate_width_frac}"
            )
        if self.n_gates < 1:
            raise ValueError(f"n_gates must be >= 1, got {self.n_gates}")
        if not 0 < self.t_rec_frac <= 1:
            raise ValueError(f"t_rec_frac must be in (0, 1], got {self.t_rec_frac}")
        if not 0 <= self.t0_frac < 1:
            raise ValueError(f"t0_frac must be in [0, 1), got {self.t0_frac}")

    @property
    def period_ns(self) -> float:
        """Laser period in nanoseconds."""
        return 1e9 / self.frequency_hz

    @property
    def omega_rad_per_ns(self) -> float:
        """Angular frequency of the harmonic, in rad/ns."""
        return 2.0 * np.pi * self.harmonic / self.period_ns

    def locus(
        self,
        tau_ns: ArrayLike | None = None,
        *,
        tau_max_ns: float = 10.0,
        n_points: int = 500,
        draw: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        """
        Trace the locus of this geometry with :class:`MonoLocus`.

        Parameters
        ----------
        tau_ns : array_like | None
            Lifetimes in nanoseconds; default ``n_points`` values up to
            ``tau_max_ns``.
        tau_max_ns, n_points : float, int
            Default lifetime grid.
        draw : bool
            Draw the locus (``ax=``, ``title=``, ``color=`` ... are passed on).

        Returns
        -------
        tuple[Any, ...]
            ``(g, s, tau_ns, ax)`` as returned by the ``MonoLocus`` locus methods.
        """
        locus = MonoLocus(self.frequency_hz, tau_max_ns=tau_max_ns, n_points=n_points)
        common = {"harmonic": self.harmonic, "tau_ns": tau_ns, "draw": draw, **kwargs}
        mode = self.mode
        if mode is AcquisitionMode.CONTINUOUS:
            return locus.continuous_locus(**common)
        if mode is AcquisitionMode.DISCRETE:
            return locus.discrete_locus(self.n_bins, **common)
        if mode is AcquisitionMode.GATED_SINGLE:
            return locus.gated_single_locus(self.gate_width_frac, **common)
        if mode is AcquisitionMode.GATED_N:
            return locus.gated_n_locus(self.gate_width_frac, self.n_gates, **common)
        if mode is AcquisitionMode.TRUNCATED:
            return locus.truncated_locus(self.t_rec_frac, **common)
        return locus.offset_locus(self.t0_frac, **common)

    def describe(self) -> str:
        """Readable summary of the geometry."""
        lines = [
            "AcquisitionConfig",
            f"  mode       : {self.mode.value}",
            f"  frequency  : {self.frequency_hz / 1e6:.3f} MHz  (T = {self.period_ns:.3f} ns)",
            f"  harmonic   : {self.harmonic}",
        ]
        mode = self.mode
        if mode is AcquisitionMode.DISCRETE:
            lines.append(f"  n_bins     : {self.n_bins}")
        if mode in (AcquisitionMode.GATED_SINGLE, AcquisitionMode.GATED_N):
            lines.append(
                f"  gate width : {self.gate_width_frac * self.period_ns:.3f} ns  ({self.gate_width_frac:.3f} T)"
            )
        if mode is AcquisitionMode.GATED_N:
            lines.append(f"  n_gates    : {self.n_gates}")
        if mode is AcquisitionMode.TRUNCATED:
            lines.append(
                f"  window     : {self.t_rec_frac * self.period_ns:.3f} ns  ({self.t_rec_frac:.3f} T)"
            )
        if mode is AcquisitionMode.OFFSET:
            lines.append(
                f"  offset t0  : {self.t0_frac * self.period_ns:.3f} ns  ({self.t0_frac:.3f} T)"
            )
        return "\n".join(lines)


def _omega_rad_per_ns(frequency_hz: float, harmonic: int) -> float:
    return 2.0 * np.pi * harmonic * frequency_hz * 1e-9


def phase_lifetime(
    g: ArrayLike, s: ArrayLike, frequency_hz: float, harmonic: int = 1
) -> np.ndarray:
    """
    Phase lifetime ``tau = s / (g * omega)`` in nanoseconds, with ``omega`` the
    angular frequency of `harmonic`. Exact for mono-exponential decays on the
    universal semicircle.
    """
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return s / (g * _omega_rad_per_ns(frequency_hz, harmonic))


def modulation_lifetime(
    g: ArrayLike, s: ArrayLike, frequency_hz: float, harmonic: int = 1
) -> np.ndarray:
    """
    Modulation lifetime ``tau = sqrt(1/m^2 - 1) / omega`` in nanoseconds, with
    ``m^2 = g^2 + s^2``. Exact for mono-exponential decays on the universal
    semicircle.
    """
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    m2 = g**2 + s**2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(m2 > 0, (1.0 - m2) / m2, np.nan)
    return np.sqrt(np.maximum(ratio, 0.0)) / _omega_rad_per_ns(frequency_hz, harmonic)


def lifetime_from_locus(
    g: ArrayLike,
    s: ArrayLike,
    locus_g: ArrayLike,
    locus_s: ArrayLike,
    locus_tau: ArrayLike,
    mask: ArrayLike | None = None,
) -> np.ndarray:
    """
    Lifetime of the nearest point of a traced locus, for every phasor.

    Parameters
    ----------
    g, s : array_like
        Phasor coordinates (any shape).
    locus_g, locus_s, locus_tau : array_like
        A locus from :class:`MonoLocus` or :meth:`AcquisitionConfig.locus`.
    mask : array_like | None
        Phasors to evaluate; the others (and non-finite phasors) are ``NaN``.

    Returns
    -------
    np.ndarray
        Lifetimes in nanoseconds, same shape as `g`.
    """
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    use = np.isfinite(g) & np.isfinite(s)
    if mask is not None:
        use &= np.asarray(mask, dtype=bool)
    tau = np.full(g.shape, np.nan)
    tree = cKDTree(np.column_stack([np.ravel(locus_g), np.ravel(locus_s)]))
    _, nearest = tree.query(np.column_stack([g[use], s[use]]))
    tau[use] = np.ravel(locus_tau)[nearest]
    return tau


def phase_lifetime_gated(
    g: ArrayLike,
    s: ArrayLike,
    frequency_hz: float,
    gate_width_frac: float,
    harmonic: int = 1,
    tau_range_ns: tuple[float, float] = (1e-3, 100.0),
) -> np.ndarray:
    """
    Lifetime whose single-gate locus point has the measured phase.

    The standard phase lifetime is biased for a single square gate of width
    ``gate_width_frac * T``. This solves ``arg(z_gate(tau)) = arg(g + i s)`` for
    ``tau`` in `tau_range_ns` (Brent's method), phasor by phasor. Phases outside the
    range of the locus give ``NaN``.

    Returns
    -------
    np.ndarray
        Lifetimes in nanoseconds, same shape as `g`.
    """
    locus = MonoLocus(frequency_hz)
    phi = np.arctan2(np.asarray(s, dtype=float), np.asarray(g, dtype=float))

    def locus_phase(tau: float) -> float:
        gg, ss = locus._gated_single_gs(np.array([tau]), gate_width_frac, harmonic)
        return float(np.arctan2(ss[0], gg[0]))

    def solve(p: float) -> float:
        if not np.isfinite(p):
            return np.nan
        try:
            return brentq(lambda t: locus_phase(t) - p, *tau_range_ns)
        except ValueError:
            return np.nan

    return np.array([solve(p) for p in phi.ravel()]).reshape(phi.shape)


def discrete_locus_circle(n_bins: int, harmonic: int = 1) -> tuple[float, float, float]:
    """
    Centre ``(gc, sc)`` and radius ``r`` of the circle the binned locus lies on.

    For ``n_bins`` equal bins over one period, the phasor of a mono-exponential
    decay is ``z = (1 - x) / (1 - x e^{i phi})`` with ``x = exp(-T / (n_bins tau))``
    and ``phi = 2 pi harmonic / n_bins``; as ``x`` goes from 0 to 1 it traces an arc
    of the circle through (1, 0) and (0, 0) with

        gc = 1/2,  sc = -tan(phi / 2) / 2,  r = 1 / (2 cos(phi / 2)).

    The circle tends to the universal semicircle as ``n_bins`` grows. When
    ``cos(phi / 2) = 0`` (e.g. ``n_bins = 2``, ``harmonic = 1``) the locus is the
    segment [0, 1] and ``(0.5, 0.0, 0.5)`` is returned.
    """
    half_phi = np.pi * harmonic / n_bins
    if abs(np.cos(half_phi)) < 1e-12:
        return 0.5, 0.0, 0.5
    sc = -0.5 * np.tan(half_phi)
    return 0.5, float(sc), float(np.hypot(0.5, sc))


def plot_discrete_n_sweep(
    frequency_hz: float,
    n_values: tuple[int, ...] = (4, 8, 16, 64, 256),
    harmonic: int = 1,
    tau_ns: ArrayLike | None = None,
    ax: Any | None = None,
    cmap: str = "viridis",
    title: str | None = None,
    figsize: tuple[float, float] = (8, 5.5),
) -> Any:
    """
    Binned loci for several numbers of bins, converging to the universal
    semicircle as the number of bins grows.

    Returns
    -------
    Any
        The axes drawn into.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    locus = MonoLocus(frequency_hz)
    colors = plt.get_cmap(cmap)(np.linspace(0.0, 1.0, len(n_values)))
    ug, us = _universal_circle_xy(half_circle=True)
    ax.plot(ug, us, "k--", lw=1, alpha=0.5, zorder=1, label="Universal semicircle")
    for n_bins, color in zip(n_values, colors):
        g, s, _, _ = locus.discrete_locus(
            n_bins, harmonic=harmonic, tau_ns=tau_ns, draw=False
        )
        ax.plot(g, s, color=color, lw=1.6, zorder=3, label=f"N = {n_bins}")
    _style_phasor_ax(
        ax, title=title or "Binned locus vs number of bins", half_circle=True
    )
    _add_frequency_label(ax, harmonic * frequency_hz)
    ax.legend(fontsize=8, title="bins", loc="upper right")
    return ax
