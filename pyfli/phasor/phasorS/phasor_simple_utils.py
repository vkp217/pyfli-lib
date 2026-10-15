"""Shared phasor plotting geometry and axis styling helpers."""

from typing import Any

import numpy as np

_TAU_MARKS_NS = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 5, 7, 10]
)
_UNIVERSAL_CIRCLE_CENTER = (0.5, 0.0)
_UNIVERSAL_CIRCLE_RADIUS = 0.5


def _universal_circle_xy(
    n_points: int = 500, half_circle: bool = False
) -> tuple[Any, ...]:
    """
    Run the universal circle xy routine.

    Parameters
    ----------
    n_points : int
        Number of points sampled for a curve or density.
    half_circle : bool
        Whether to draw only the upper half of the universal phasor circle.

    Returns
    -------
    tuple[Any, ...]
        Tuple containing x and y coordinates for the universal semicircle.
    """
    theta = np.linspace(0, np.pi if half_circle else 2 * np.pi, n_points)
    cx, cy = _UNIVERSAL_CIRCLE_CENTER
    r = _UNIVERSAL_CIRCLE_RADIUS
    return cx + r * np.cos(theta), cy + r * np.sin(theta)


def _draw_lifetime_ticks(
    ax: Any,
    G_mark: Any,
    S_mark: Any,
    tick_length: float = 0.02,
    text_offset: float = 0.035,
    color: str = "black",
    lw: float = 2,
    fontsize: int = 7,
    show_units: bool = False,
    taus: Any = None,
) -> None:
    """
    Draw lifetime ticks.

    Parameters
    ----------
    ax : Any
        Matplotlib axes object on which the plot is drawn.
    G_mark : Any
        Phasor real coordinate for a lifetime tick mark.
    S_mark : Any
        Phasor imaginary coordinate for a lifetime tick mark.
    tick_length : float
        Length of the lifetime tick mark in phasor coordinates.
    text_offset : float
        Offset applied to lifetime labels on the phasor plot.
    color : str
        Matplotlib color used for drawing the plot element.
    lw : float
        Line width used when drawing lifetime tick marks.
    fontsize : int
        Font size used for phasor plot labels.
    show_units : bool
        If ``True``, include lifetime units in text labels.
    taus : Any
        Lifetimes (ns) labelling the ticks, matched positionally to `G_mark`;
        defaults to ``_TAU_MARKS_NS``.

    Returns
    -------
    None
        No object is returned; the function perform draw lifetime ticks.
    """
    cx, cy = _UNIVERSAL_CIRCLE_CENTER
    for tau, Gm, Sm in zip(_TAU_MARKS_NS if taus is None else taus, G_mark, S_mark):
        normal = np.array([Gm - cx, Sm - cy])
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal /= norm

        tick_start = np.array([Gm, Sm]) - tick_length * normal / 2
        tick_end = np.array([Gm, Sm]) + tick_length * normal / 2
        ax.plot(
            [tick_start[0], tick_end[0]],
            [tick_start[1], tick_end[1]],
            color=color,
            lw=lw,
        )

        label = f"{tau:.1f} ns" if show_units else f"{tau:.1f}"
        text_pos = tick_end + text_offset * normal
        ax.text(
            text_pos[0], text_pos[1], label, color=color, fontsize=fontsize, ha="center"
        )


def _add_frequency_label(
    ax: Any,
    frequency_hz: float,
    fontsize: int = 8,
    color: str = "gray",
) -> None:
    """
    Add frequency label.

    Parameters
    ----------
    ax : Any
        Matplotlib axes object on which the plot is drawn.
    frequency_hz : float
        Excitation frequency in hertz to display on the plot.
    fontsize : int
        Font size used for the frequency label.
    color : str
        Matplotlib color used for the frequency label text.

    Returns
    -------
    None
        No object is returned; the function perform add frequency label.
    """
    ax.text(
        1.0,
        -0.14,
        f"{frequency_hz / 1e6:.2f} MHz",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
        color=color,
    )


def _style_phasor_ax(
    ax: Any,
    title: str = "Phasor Diagram",
    xlim: tuple[float, ...] = (-0.1, 1.1),
    ylim: tuple[float, ...] = (0.0, 0.6),
    half_circle: bool = True,
) -> None:
    """
    Run the style phasor ax routine.

    Parameters
    ----------
    ax : Any
        Matplotlib axes object on which the plot is drawn.
    title : str
        Title displayed on the generated plot.
    xlim : tuple[float, ...]
        X-axis limits for the phasor plot.
    ylim : tuple[float, ...]
        Y-axis limits for the phasor plot.
    half_circle : bool
        Whether to draw only the upper half of the universal phasor circle.

    Returns
    -------
    None
        No object is returned; the function perform style phasor ax.
    """
    ax.set_xlabel("G")
    ax.set_ylabel("S")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.tick_params(direction="in", length=6, width=1)
