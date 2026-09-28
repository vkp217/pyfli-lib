"""
Theoretical phasor locus construction and plotting.

This module belongs to :mod:`pyfli.phasor.phasorS` and is part of PyFLI's compact
phasor analyzer for CPU and optional GPU FLI workflows. Public API includes the
:class:`MonoLocus` class.

Unlike :class:`~pyfli.phasor.phasorS.phasor_simple.PhasorAnalyzer`, which estimates
phasor coordinates from measured decay data, :class:`MonoLocus` traces the
theoretical phasor curve a perfect, Dirac-excited, single-exponential decay
traces under a given acquisition geometry (binning, gating, truncation, or
excitation offset). These loci are the reference curves against which measured
phasors can be checked for acquisition-induced bias.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .phasor_simple_utils import (
    _TAU_MARKS_NS,
    _add_frequency_label,
    _draw_lifetime_ticks,
    _style_phasor_ax,
    _universal_circle_xy,
)

MODE_COLORS: dict[str, str] = {
    "continuous": "#3266ad",
    "discrete": "#1d9e75",
    "gated_single": "#d85a30",
    "gated_n": "#ba7517",
    "truncated": "#d4537e",
    "offset": "#7f77dd",
}


class MonoLocus:
    """
    Trace and draw loci for different acquisition-geometry models.

    Parameters
    ----------
    frequency_hz : float
        Excitation / laser repetition frequency in hertz.
    tau_max_ns : float
        Default maximum lifetime, in nanoseconds, spanned by a traced locus.
    n_tau : int
        Default number of lifetime samples used to trace a locus.
    """

    def __init__(
        self,
        frequency_hz: float,
        tau_max_ns: float = 10.0,
        n_points: int = 500,
    ) -> None:
        self.frequency = float(frequency_hz)
        self.omega = 2.0 * np.pi * self.frequency
        self.tau_max_ns = float(tau_max_ns)
        self.n_points = int(n_points)
        self.eps = 1e-12

    def _tau_grid_ns(self, tau_max_ns: float | None = None) -> np.ndarray:
        """
        Build the default nanosecond lifetime grid used to trace a locus.

        Parameters
        ----------
        tau_max_ns : float | None
            Maximum lifetime in nanoseconds. Falls back to ``self.tau_max_ns``.

        Returns
        -------
        np.ndarray
            1-D lifetime grid in nanoseconds.
        """
        tau_max_ns = self.tau_max_ns if tau_max_ns is None else tau_max_ns
        return np.linspace(1e-3, tau_max_ns, self.n_points)

    @staticmethod
    def _finite(g: np.ndarray, s: np.ndarray, tau_ns: np.ndarray) -> tuple[Any, ...]:
        """
        Drop non-finite (g, s) samples produced near locus edge cases.

        Parameters
        ----------
        g, s : np.ndarray
            Phasor coordinates traced along ``tau_ns``.
        tau_ns : np.ndarray
            Lifetime values corresponding to ``g``/``s``.

        Returns
        -------
        tuple[Any, ...]
            Filtered ``(g, s, tau_ns)`` with non-finite entries removed.
        """
        mask = np.isfinite(g) & np.isfinite(s)
        return g[mask], s[mask], tau_ns[mask]

    # ── analytical phasor formulas (Hz / ns convention, matches PhasorAnalyzer) ─

    def _continuous_gs(self, tau_ns: np.ndarray, harmonic: int = 1) -> tuple[Any, ...]:
        """simple locus for full period"""
        tau_s = np.asarray(tau_ns, dtype=float) * 1e-9
        wt = harmonic * self.omega * tau_s
        denom = 1.0 + wt**2
        return 1.0 / denom, wt / denom

    def _discrete_gs(
        self, tau_ns: np.ndarray, n_bins: int, harmonic: int = 1
    ) -> tuple[Any, ...]:
        """Phasor of a PSED sampled into ``n_bins`` equal bins over one period."""
        tau_s = np.asarray(tau_ns, dtype=float) * 1e-9
        T_s = 1.0 / self.frequency
        dt = T_s / n_bins
        k = np.arange(n_bins, dtype=float)
        phase = 2.0 * np.pi * harmonic * k / n_bins

        tau2 = tau_s[:, None]
        e_kdt = np.exp(-k * dt / tau2)
        e_dt = np.exp(-dt / tau2)
        I_k = e_kdt * (1.0 - e_dt)

        I_sum = I_k.sum(axis=1)
        g = (I_k * np.cos(phase)).sum(axis=1) / I_sum
        s = (I_k * np.sin(phase)).sum(axis=1) / I_sum
        return g, s

    def _gated_single_gs(
        self, tau_ns: np.ndarray, gate_width_frac: float, harmonic: int = 1
    ) -> tuple[Any, ...]:
        """Phasor of a PSED viewed through a single square gate starting at t=0."""
        tau_s = np.asarray(tau_ns, dtype=float) * 1e-9
        T_s = 1.0 / self.frequency
        W_s = gate_width_frac * T_s
        omega_k = harmonic * self.omega

        wt = omega_k * tau_s
        eW = np.exp(-W_s / tau_s)
        eT = np.exp(-T_s / tau_s)
        wW = omega_k * W_s

        denom = (1.0 + wt**2) * (1.0 - eT)
        cos_int = 1.0 - eW * (np.cos(wW) - wt * np.sin(wW))
        sin_int = wt * (1.0 - eW * np.cos(wW)) - eW * np.sin(wW)
        return cos_int / denom, sin_int / denom

    def _gated_n_gs(
        self,
        tau_ns: np.ndarray,
        gate_width_frac: float,
        n_gates: int,
        harmonic: int = 1,
    ) -> tuple[Any, ...]:
        """Phasor of a PSED viewed through ``n_gates`` equidistant square gates."""
        tau_s = np.asarray(tau_ns, dtype=float) * 1e-9
        T_s = 1.0 / self.frequency
        W_s = gate_width_frac * T_s
        theta = T_s / n_gates
        k = np.arange(n_gates, dtype=float)
        t_k = k * theta
        phase = 2.0 * np.pi * harmonic * k / n_gates

        tau2 = tau_s[:, None]
        I_k = np.exp(-t_k / tau2) - np.exp(-(t_k + W_s) / tau2)

        I_sum = I_k.sum(axis=1)
        g = (I_k * np.cos(phase)).sum(axis=1) / I_sum
        s = (I_k * np.sin(phase)).sum(axis=1) / I_sum
        return g, s

    def _truncated_gs(
        self, tau_ns: np.ndarray, t_rec_frac: float, harmonic: int = 1
    ) -> tuple[Any, ...]:
        """Phasor of a PSED whose recording window is shorter than the period."""
        tau_s = np.asarray(tau_ns, dtype=float) * 1e-9
        T_s = 1.0 / self.frequency
        Trec = t_rec_frac * T_s
        omega_k = harmonic * self.omega

        wt = omega_k * tau_s
        eTr = np.exp(-Trec / tau_s)
        wTr = omega_k * Trec

        cos_int = (1.0 - eTr * (np.cos(wTr) - wt * np.sin(wTr))) / (1.0 + wt**2)
        sin_int = (wt - eTr * (wt * np.cos(wTr) + np.sin(wTr))) / (1.0 + wt**2)
        norm = 1.0 - eTr
        return cos_int / norm, sin_int / norm

    def _offset_gs(
        self, tau_ns: np.ndarray, t0_frac: float, harmonic: int = 1
    ) -> tuple[Any, ...]:
        """Phasor of a PSED whose excitation pulse is offset within the window."""
        g0, s0 = self._continuous_gs(tau_ns, harmonic)
        T_s = 1.0 / self.frequency
        phi = harmonic * self.omega * (t0_frac * T_s)
        c, si = np.cos(phi), np.sin(phi)
        return g0 * c + s0 * si, s0 * c - g0 * si

    # ── drawing (built on phasor_simple_utils, same look as PhasorPlotsMixin) ──

    def _draw_locus(
        self,
        ax: Any | None,
        g: np.ndarray,
        s: np.ndarray,
        tau_ns: np.ndarray,
        *,
        harmonic: int,
        label: str,
        color: str,
        half_circle: bool,
        title: str,
        show_universal: bool,
        figsize: tuple[float, float],
    ) -> Any:
        """
        Render one locus curve using the shared phasor-plot styling helpers.

        Parameters
        ----------
        ax : Any | None
            Existing axes to draw into; a new figure/axes is created if ``None``.
        g, s, tau_ns : np.ndarray
            Locus coordinates and the lifetime grid they were traced over.
        harmonic : int
            Harmonic index, used for the frequency annotation.
        label : str
            Legend label for the traced locus.
        color : str
            Line color for the traced locus.
        half_circle : bool
            Whether to draw only the upper half of the universal circle.
        title : str
            Axes title.
        show_universal : bool
            Whether to overlay the ideal universal semicircle for reference.
        figsize : tuple[float, float]
            Figure size used when a new figure is created.

        Returns
        -------
        Any
            The axes the locus was drawn into.
        """
        created_fig = ax is None
        if created_fig:
            _, ax = plt.subplots(figsize=figsize)

        if show_universal:
            ug, us = _universal_circle_xy(half_circle=half_circle)
            ax.plot(
                ug, us, "k--", lw=1, alpha=0.5, zorder=1, label="Universal semicircle"
            )

        ax.plot(g, s, color=color, lw=2.2, zorder=3, label=label)

        tick_taus = _TAU_MARKS_NS[_TAU_MARKS_NS <= tau_ns.max()]
        if tick_taus.size:
            g_mark = np.interp(tick_taus, tau_ns, g)
            s_mark = np.interp(tick_taus, tau_ns, s)
            _draw_lifetime_ticks(
                ax, g_mark, s_mark, color=color, lw=2, fontsize=8, show_units=True
            )

        _style_phasor_ax(ax, title=title, half_circle=half_circle)
        _add_frequency_label(ax, harmonic * self.frequency)
        ax.legend(fontsize=8, loc="upper right")

        if created_fig:
            plt.tight_layout()
        return ax

    # ── public loci ──────────────────────────────────────────────────────────

    def continuous_locus(
        self,
        harmonic: int = 1,
        tau_ns: np.ndarray | None = None,
        *,
        draw: bool = True,
        ax: Any | None = None,
        color: str | None = None,
        half_circle: bool = True,
        title: str | None = None,
        show_universal: bool = True,
        filter_finite: bool = True,
        figsize: tuple[float, float] = (6, 4.5),
    ) -> tuple[Any, ...]:
        """
        Trace (and by default draw) the ideal universal-semicircle locus.

        Returns
        -------
        tuple[Any, ...]
            ``(g, s, tau_ns, ax)``; ``ax`` is ``None`` when ``draw=False``.
        """
        tau = (
            self._tau_grid_ns()
            if tau_ns is None
            else np.atleast_1d(np.asarray(tau_ns, dtype=float))
        )
        g, s = self._continuous_gs(tau, harmonic)
        if filter_finite:
            g, s, tau = self._finite(g, s, tau)

        if draw:
            ax = self._draw_locus(
                ax,
                g,
                s,
                tau,
                harmonic=harmonic,
                label="Continuous",
                color=color or MODE_COLORS["continuous"],
                half_circle=half_circle,
                title=title or "Continuous locus",
                show_universal=show_universal,
                figsize=figsize,
            )
        return g, s, tau, ax

    def discrete_locus(
        self,
        n_bins: int,
        harmonic: int = 1,
        tau_ns: np.ndarray | None = None,
        *,
        draw: bool = True,
        ax: Any | None = None,
        color: str | None = None,
        half_circle: bool = True,
        title: str | None = None,
        show_universal: bool = True,
        filter_finite: bool = True,
        figsize: tuple[float, float] = (6, 4.5),
    ) -> tuple[Any, ...]:
        """
        Trace (and by default draw) the discrete, ``n_bins``-binned locus arc.

        Returns
        -------
        tuple[Any, ...]
            ``(g, s, tau_ns, ax)``; ``ax`` is ``None`` when ``draw=False``.
        """
        tau = (
            self._tau_grid_ns()
            if tau_ns is None
            else np.atleast_1d(np.asarray(tau_ns, dtype=float))
        )
        g, s = self._discrete_gs(tau, n_bins, harmonic)
        if filter_finite:
            g, s, tau = self._finite(g, s, tau)

        if draw:
            ax = self._draw_locus(
                ax,
                g,
                s,
                tau,
                harmonic=harmonic,
                label=f"Discrete (N={n_bins})",
                color=color or MODE_COLORS["discrete"],
                half_circle=half_circle,
                title=title or "Discrete locus",
                show_universal=show_universal,
                figsize=figsize,
            )
        return g, s, tau, ax

    def gated_single_locus(
        self,
        gate_width_frac: float,
        harmonic: int = 1,
        tau_ns: np.ndarray | None = None,
        *,
        draw: bool = True,
        ax: Any | None = None,
        color: str | None = None,
        half_circle: bool = True,
        title: str | None = None,
        show_universal: bool = True,
        filter_finite: bool = True,
        figsize: tuple[float, float] = (6, 4.5),
    ) -> tuple[Any, ...]:
        """
        Trace (and by default draw) the single-square-gate locus.

        Returns
        -------
        tuple[Any, ...]
            ``(g, s, tau_ns, ax)``; ``ax`` is ``None`` when ``draw=False``.
        """
        tau = (
            self._tau_grid_ns()
            if tau_ns is None
            else np.atleast_1d(np.asarray(tau_ns, dtype=float))
        )
        g, s = self._gated_single_gs(tau, gate_width_frac, harmonic)
        if filter_finite:
            g, s, tau = self._finite(g, s, tau)

        if draw:
            ax = self._draw_locus(
                ax,
                g,
                s,
                tau,
                harmonic=harmonic,
                label=f"Single gate (W={gate_width_frac:.2f}T)",
                color=color or MODE_COLORS["gated_single"],
                half_circle=half_circle,
                title=title or "Single-gate locus",
                show_universal=show_universal,
                figsize=figsize,
            )
        return g, s, tau, ax

    def gated_n_locus(
        self,
        gate_width_frac: float,
        n_gates: int,
        harmonic: int = 1,
        tau_ns: np.ndarray | None = None,
        *,
        draw: bool = True,
        ax: Any | None = None,
        color: str | None = None,
        half_circle: bool = True,
        title: str | None = None,
        show_universal: bool = True,
        filter_finite: bool = True,
        figsize: tuple[float, float] = (6, 4.5),
    ) -> tuple[Any, ...]:
        """
        Trace (and by default draw) the ``n_gates``-equidistant-gate locus.

        Returns
        -------
        tuple[Any, ...]
            ``(g, s, tau_ns, ax)``; ``ax`` is ``None`` when ``draw=False``.
        """
        tau = (
            self._tau_grid_ns()
            if tau_ns is None
            else np.atleast_1d(np.asarray(tau_ns, dtype=float))
        )
        g, s = self._gated_n_gs(tau, gate_width_frac, n_gates, harmonic)
        if filter_finite:
            g, s, tau = self._finite(g, s, tau)

        if draw:
            ax = self._draw_locus(
                ax,
                g,
                s,
                tau,
                harmonic=harmonic,
                label=f"Gated ×{n_gates} (W={gate_width_frac:.2f}T)",
                color=color or MODE_COLORS["gated_n"],
                half_circle=half_circle,
                title=title or "Gated-N locus",
                show_universal=show_universal,
                figsize=figsize,
            )
        return g, s, tau, ax

    def truncated_locus(
        self,
        t_rec_frac: float,
        harmonic: int = 1,
        tau_ns: np.ndarray | None = None,
        *,
        draw: bool = True,
        ax: Any | None = None,
        color: str | None = None,
        half_circle: bool = True,
        title: str | None = None,
        show_universal: bool = True,
        filter_finite: bool = True,
        figsize: tuple[float, float] = (6, 4.5),
    ) -> tuple[Any, ...]:
        """
        Trace (and by default draw) the locus for a recording window shorter
        than the excitation period (``t_rec_frac`` of the period ``T``).

        Returns
        -------
        tuple[Any, ...]
            ``(g, s, tau_ns, ax)``; ``ax`` is ``None`` when ``draw=False``.
        """
        tau = (
            self._tau_grid_ns()
            if tau_ns is None
            else np.atleast_1d(np.asarray(tau_ns, dtype=float))
        )
        g, s = self._truncated_gs(tau, t_rec_frac, harmonic)
        if filter_finite:
            g, s, tau = self._finite(g, s, tau)

        if draw:
            ax = self._draw_locus(
                ax,
                g,
                s,
                tau,
                harmonic=harmonic,
                label=f"Truncated (T_rec={t_rec_frac:.2f}T)",
                color=color or MODE_COLORS["truncated"],
                half_circle=half_circle,
                title=title or "Truncated locus",
                show_universal=show_universal,
                figsize=figsize,
            )
        return g, s, tau, ax

    def offset_locus(
        self,
        t0_frac: float,
        harmonic: int = 1,
        tau_ns: np.ndarray | None = None,
        *,
        draw: bool = True,
        ax: Any | None = None,
        color: str | None = None,
        half_circle: bool = True,
        title: str | None = None,
        show_universal: bool = True,
        filter_finite: bool = True,
        figsize: tuple[float, float] = (6, 4.5),
    ) -> tuple[Any, ...]:
        """
        Trace (and by default draw) the locus for an excitation pulse offset
        by ``t0_frac`` of the period ``T`` within the recording window.

        Returns
        -------
        tuple[Any, ...]
            ``(g, s, tau_ns, ax)``; ``ax`` is ``None`` when ``draw=False``.
        """
        tau = (
            self._tau_grid_ns()
            if tau_ns is None
            else np.atleast_1d(np.asarray(tau_ns, dtype=float))
        )
        g, s = self._offset_gs(tau, t0_frac, harmonic)
        if filter_finite:
            g, s, tau = self._finite(g, s, tau)

        if draw:
            ax = self._draw_locus(
                ax,
                g,
                s,
                tau,
                harmonic=harmonic,
                label=f"Offset (t0={t0_frac:.2f}T)",
                color=color or MODE_COLORS["offset"],
                half_circle=half_circle,
                title=title or "Offset locus",
                show_universal=show_universal,
                figsize=figsize,
            )
        return g, s, tau, ax
