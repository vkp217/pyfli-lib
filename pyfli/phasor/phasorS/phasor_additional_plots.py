"""Supplementary phasor plots as a :class:`PhasorAnalyzer` subclass."""

import colorsys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage
from scipy.stats import gaussian_kde

from .phasor_simple import PhasorAnalyzer
from .phasor_simple_utils import (
    _TAU_MARKS_NS,
    _add_frequency_label,
    _draw_lifetime_ticks,
    _style_phasor_ax,
    _universal_circle_xy,
)


class PhasorAdditionalPlots(PhasorAnalyzer):
    """
    Extra phasor visualizations layered on top of :class:`PhasorAnalyzer`.

    An instance is a full phasor analyzer (compute, calibrate, lifetime
    conversion, and every :class:`PhasorPlotsMixin` plot) plus:

    * :meth:`phasorlifetime_to_phasor` -- the phase-lifetime map beside the
      matching phasor scatter, sharing one colormap so a pixel's color and its
      phasor point's color are identical.
    * :meth:`phasor_kde_to_px` -- the filled kernel-density regions of the
      phasor cloud carried back onto the grayscale intensity image.

    Construction matches :class:`PhasorAnalyzer` (``frequency_hz``,
    ``time_axis_ns``, ``n_harmonics``, ``device``).
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _iso_proportion_levels(density: np.ndarray, isoprop: np.ndarray) -> np.ndarray:
        """
        Convert enclosed-probability-mass proportions to iso-density thresholds.

        This mirrors how :func:`seaborn.kdeplot` turns ``levels`` into contour
        values: a proportion ``p`` maps to the density threshold below which a
        fraction ``p`` of the mass lies, i.e. the ``density >= threshold`` region
        encloses a fraction ``1 - p`` of the mass.

        Parameters
        ----------
        density : np.ndarray
            KDE density sampled on a grid.
        isoprop : np.ndarray
            Enclosed-mass proportions in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Ascending, unique density thresholds.
        """
        values = np.ravel(density)
        total = float(values.sum())
        if total <= 0:
            return np.asarray([], dtype=float)
        sorted_values = np.sort(values)[::-1]
        cdf = np.cumsum(sorted_values) / total
        idx = np.searchsorted(cdf, 1.0 - np.asarray(isoprop, dtype=float))
        levels = np.take(sorted_values, idx, mode="clip")
        return np.unique(levels)

    @staticmethod
    def _distinct_colors(n: int, cmap_name: str) -> list[tuple[float, float, float]]:
        """
        Return ``n`` visually distinct RGB colors.

        A qualitative (listed) colormap is used verbatim when it holds enough
        entries; otherwise evenly spaced full-saturation hues are generated so
        the colors stay distinct for any ``n``.

        Parameters
        ----------
        n : int
            Number of colors requested.
        cmap_name : str
            Name of the preferred qualitative colormap.

        Returns
        -------
        list[tuple[float, float, float]]
            RGB triples in ``[0, 1]``.
        """
        if n <= 0:
            return []
        cmap = plt.get_cmap(cmap_name)
        listed = getattr(cmap, "colors", None)
        if listed is not None and len(listed) >= n:
            return [to_rgb(listed[i]) for i in range(n)]
        return [colorsys.hsv_to_rgb((i / n) % 1.0, 0.85, 0.95) for i in range(n)]

    # ── plots ───────────────────────────────────────────────────────────────

    def phasorlifetime_to_phasor(
        self,
        Gc: np.ndarray,
        Sc: np.ndarray,
        boolean_mask: np.ndarray | None = None,
        cmap: str | Colormap = "jet",
        cmap_scale: tuple[float, float] = (0.0, 5.0),
        harmonic: int = 0,
        axes: Any | None = None,
        figsize: tuple[float, float] = (14, 6),
        half_circle: bool = True,
        xlim: tuple[float, float] = (-0.1, 1.1),
        ylim: tuple[float, float] = (0.0, 0.6),
        point_size: float = 6.0,
        background: str = "black",
        title: str = "Phasor Lifetime",
    ) -> Any:
        """
        Plot the phase-lifetime map beside its matching phasor scatter.

        Panel ``(0, 0)`` is the per-pixel phase-lifetime map
        (:meth:`compute_lifetime` of the calibrated coordinates) shown with
        ``imshow`` using ``cmap`` and ``vmin, vmax = cmap_scale``; pixels outside
        ``boolean_mask`` or with an undefined lifetime render in ``background``.
        Panel ``(0, 1)`` scatters the calibrated phasor points
        ``(Gc[harmonic], Sc[harmonic])`` for the masked pixels, colored through
        the *same* ``cmap`` and ``Normalize(vmin, vmax)`` as the map, so a
        pixel's color and its phasor point's color are identical.

        Parameters
        ----------
        Gc : np.ndarray
            Calibrated phasor real coordinate. Either a ``(H, W)`` map or a
            ``(n_harmonics, H, W)`` stack (the ``harmonic`` slice is used).
        Sc : np.ndarray
            Calibrated phasor imaginary coordinate, matching ``Gc``.
        boolean_mask : np.ndarray | None
            Boolean ``(H, W)`` mask selecting the pixels to display and scatter.
            When ``None`` every pixel is used.
        cmap : str | matplotlib.colors.Colormap
            Colormap shared by the lifetime map and the phasor scatter.
        cmap_scale : tuple[float, float]
            ``(vmin, vmax)`` in nanoseconds;
            ``vmin, vmax = cmap_scale[0], cmap_scale[1]``.
        harmonic : int
            Harmonic slice used when ``Gc``/``Sc`` are 3-D stacks (``0`` is the
            first harmonic). The phase lifetime of slice ``k`` is computed at its own
            frequency ``(k + 1) * f``.
        axes : Any | None
            Length-2 sequence of Matplotlib axes ``(map_ax, scatter_ax)``. A new
            1x2 figure is created when ``None``.
        figsize : tuple[float, float]
            Figure size used when a new figure is created.
        half_circle : bool
            Whether to draw only the upper half of the universal phasor circle.
        xlim, ylim : tuple[float, float]
            Axis limits for the phasor scatter panel.
        point_size : float
            Marker size for the phasor scatter points.
        background : str
            Color used for masked / undefined pixels in the lifetime map.
        title : str
            Base title; panel titles are derived from it.

        Returns
        -------
        Any
            The Matplotlib figure containing both panels.
        """
        vmin, vmax = float(cmap_scale[0]), float(cmap_scale[1])

        Gc = np.asarray(Gc)
        Sc = np.asarray(Sc)
        G_2d = Gc[harmonic] if Gc.ndim == 3 else Gc
        S_2d = Sc[harmonic] if Sc.ndim == 3 else Sc

        tau_map_ns = np.asarray(self.compute_lifetime(G_2d, S_2d), dtype=float) / (
            harmonic + 1
        )

        if boolean_mask is None:
            mask_2d = np.ones(tau_map_ns.shape, dtype=bool)
        else:
            mask_2d = np.asarray(boolean_mask).astype(bool)

        # One color pipeline shared by both panels: value -> Normalize -> cmap.
        # imshow(norm=norm, cmap=cmap_obj) and sm.to_rgba(tau) apply exactly this
        # same mapping, so a pixel and its phasor point get an identical color.
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        cmap_obj = cmap_obj.copy()
        cmap_obj.set_bad(background)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])

        created_fig = axes is None
        if created_fig:
            fig, axes_arr = plt.subplots(1, 2, figsize=figsize, squeeze=False)
            ax_map, ax_scatter = axes_arr[0, 0], axes_arr[0, 1]
        else:
            axes_flat = np.asarray(axes, dtype=object).ravel()
            ax_map, ax_scatter = axes_flat[0], axes_flat[1]
            fig = ax_map.get_figure()

        # ── (0, 0) phase-lifetime map ──────────────────────────────────────
        tau_display = np.where(mask_2d, tau_map_ns, np.nan)
        im = ax_map.imshow(
            tau_display,
            origin="upper",
            cmap=cmap_obj,
            norm=norm,
            interpolation="nearest",
        )
        ax_map.set_title(f"{title} Map (ns)")
        ax_map.axis("off")
        fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04).set_label("Lifetime (ns)")

        # ── (0, 1) matching phasor scatter ────────────────────────────────
        mask_flat = mask_2d.ravel()
        g_flat = np.ravel(G_2d)[mask_flat]
        s_flat = np.ravel(S_2d)[mask_flat]
        tau_flat = np.ravel(tau_map_ns)[mask_flat]
        valid = np.isfinite(g_flat) & np.isfinite(s_flat) & np.isfinite(tau_flat)
        g_v, s_v, tau_v = g_flat[valid], s_flat[valid], tau_flat[valid]

        ug, us = _universal_circle_xy(half_circle=half_circle)
        ax_scatter.plot(ug, us, "k--", alpha=0.8, zorder=1)
        if g_v.size:
            ax_scatter.scatter(
                g_v,
                s_v,
                c=sm.to_rgba(tau_v),
                s=point_size,
                edgecolors="none",
                zorder=2,
            )
        else:
            ax_scatter.text(0.5, 0.25, "No valid data points", ha="center", color="red")

        try:
            g_mark, s_mark = self.lifetime_to_phasor(
                _TAU_MARKS_NS, (harmonic + 1) * self.frequency
            )
            _draw_lifetime_ticks(
                ax_scatter, g_mark, s_mark, color="black", lw=2, fontsize=9
            )
        except Exception:
            pass

        _style_phasor_ax(
            ax_scatter,
            title=f"{title} — Phasor Scatter",
            xlim=xlim,
            ylim=ylim,
            half_circle=half_circle,
        )
        _add_frequency_label(ax_scatter, (harmonic + 1) * self.frequency)

        if created_fig:
            plt.tight_layout()
        return fig

    def phasor_kde_to_px(
        self,
        Gc: np.ndarray,
        Sc: np.ndarray,
        decay: np.ndarray,
        boolean_mask: np.ndarray | None = None,
        hexbin_color: str = "autumn",
        harmonic: int = 0,
        kde_levels: int = 3,
        kde_color: str = "red",
        kde_linewidths: float = 1.0,
        kde_alpha: float = 0.5,
        kde_thresh: float = 0.05,
        fill_alpha: float = 0.4,
        gridsize: int = 200,
        bw_method: Any | None = None,
        region_cmap: str = "tab10",
        peak_footprint: int | None = None,
        min_peak_height: float = 0.1,
        max_centers: int | None = 8,
        axes: Any | None = None,
        figsize: tuple[float, float] = (14, 5),
        half_circle: bool = True,
        xlim: tuple[float, float] = (-0.1, 1.1),
        ylim: tuple[float, float] = (0.0, 0.6),
        title: str = "Phasor KDE",
    ) -> Any:
        r"""
        Fill KDE regions on a phasor plot and carry those colors back to pixels.

        Panel ``(0, 0)`` reproduces the usual phasor diagram
        (:meth:`PhasorPlotsMixin.plot_phasor_diagram`: hexbin density, universal
        semicircle, lifetime ticks), overlays the KDE iso-density contour
        *lines* (like ``kdeplot=True``, in ``kde_color``), and then *fills* the
        kernel-density regions of the masked phasor cloud. Filled bands run from
        the innermost (highest density) outward; when the KDE has several modes
        each mode's territory is filled with its own clearly distinct color, so
        every ``(mode, level)`` region is a unique color (drawn at
        ``fill_alpha``).

        Panel ``(0, 1)`` shows the grayscale intensity image
        ``np.sum(decay, axis=-1)``. Every masked pixel whose phasor point lands
        in a filled KDE region is tinted, at ``fill_alpha``, with that region's
        color, so the phasor-space grouping is projected back onto the image.

        Parameters
        ----------
        Gc : np.ndarray
            Calibrated phasor real coordinate, a ``(H, W)`` map or a
            ``(n_harmonics, H, W)`` stack.
        Sc : np.ndarray
            Calibrated phasor imaginary coordinate, matching ``Gc``.
        decay : np.ndarray
            Decay cube ``(H, W, T)``; the intensity image is
            ``np.sum(decay, -1)``.
        boolean_mask : np.ndarray | None
            Boolean ``(H, W)`` mask selecting the pixels fed to the KDE and
            tinted in the overlay. When ``None`` every finite pixel is used.
        hexbin_color : str
            Colormap for the phasor hexbin density in panel ``(0, 0)``.
        harmonic : int
            Harmonic slice used when ``Gc``/``Sc`` are 3-D stacks.
        kde_levels : int
            Number of iso-proportion KDE contour levels (as in
            ``seaborn.kdeplot``).
        kde_color : str
            Color of the KDE contour lines.
        kde_linewidths : float
            Width of the KDE contour lines.
        kde_alpha : float
            Opacity of the KDE contour lines.
        kde_thresh : float
            Lowest enclosed-mass proportion contoured (``seaborn``'s ``thresh``).
        fill_alpha : float
            Opacity of the region fills and of the pixel tint in the overlay.
        gridsize : int
            Resolution of the square grid the KDE is evaluated on.
        bw_method : Any | None
            Bandwidth selector passed to :class:`scipy.stats.gaussian_kde`.
        region_cmap : str
            Preferred qualitative colormap for the distinct region colors.
        peak_footprint : int | None
            Neighborhood size, in grid cells, for the local-maximum search that
            locates KDE modes. Defaults to ``max(3, gridsize // 20)``.
        min_peak_height : float
            A density local maximum counts as a KDE mode only when it exceeds
            this fraction of the global peak density (rejects tail noise bumps).
        max_centers : int | None
            Cap on the number of KDE modes kept (strongest first); ``None``
            keeps all of them.
        axes : Any | None
            Length-2 sequence of axes ``(phasor_ax, overlay_ax)``; a new 1x2
            figure is created when ``None``.
        figsize : tuple[float, float]
            Figure size used when a new figure is created.
        half_circle : bool
            Whether to draw only the upper half of the universal phasor circle.
        xlim, ylim : tuple[float, float]
            Axis limits for the phasor panel.
        title : str
            Base title; panel titles are derived from it.

        Returns
        -------
        Any
            The Matplotlib figure containing both panels.
        """
        Gc = np.asarray(Gc)
        Sc = np.asarray(Sc)
        G_2d = Gc[harmonic] if Gc.ndim == 3 else Gc
        S_2d = Sc[harmonic] if Sc.ndim == 3 else Sc

        if boolean_mask is None:
            mask_2d = np.ones(G_2d.shape, dtype=bool)
        else:
            mask_2d = np.asarray(boolean_mask).astype(bool)

        intensity = np.asarray(np.sum(decay, axis=-1), dtype=float)

        created_fig = axes is None
        if created_fig:
            fig, axes_arr = plt.subplots(1, 2, figsize=figsize, squeeze=False)
            ax_phasor, ax_overlay = axes_arr[0, 0], axes_arr[0, 1]
        else:
            axes_flat = np.asarray(axes, dtype=object).ravel()
            ax_phasor, ax_overlay = axes_flat[0], axes_flat[1]
            fig = ax_phasor.get_figure()

        # ── (0, 0) base phasor diagram (same look as plot_phasor_diagram) ──
        self.plot_phasor_diagram(
            G_2d,
            S_2d,
            mask=mask_2d,
            hexbin_color=hexbin_color,
            ax=ax_phasor,
            half_circle=half_circle,
            title=f"{title} — regions",
            xlim=xlim,
            ylim=ylim,
            kdeplot=False,
        )

        finite_2d = np.isfinite(G_2d) & np.isfinite(S_2d)
        sel_2d = mask_2d & finite_2d
        py_idx, px_idx = np.where(sel_2d)
        g_pts = G_2d[py_idx, px_idx]
        s_pts = S_2d[py_idx, px_idx]

        pix_center = np.zeros(G_2d.shape, dtype=int)
        pix_band = np.zeros(G_2d.shape, dtype=int)
        region_color: dict[tuple[int, int], tuple[float, float, float]] = {}

        kde_ok = g_pts.size >= 5 and np.ptp(g_pts) > 0 and np.ptp(s_pts) > 0
        kde = None
        if kde_ok:
            try:
                kde = gaussian_kde(np.vstack([g_pts, s_pts]), bw_method=bw_method)
            except Exception:
                kde_ok = False

        if kde_ok and kde is not None:
            bw_g = float(np.sqrt(kde.covariance[0, 0]))
            bw_s = float(np.sqrt(kde.covariance[1, 1]))
            x0, x1 = g_pts.min() - 3.0 * bw_g, g_pts.max() + 3.0 * bw_g
            y0, y1 = s_pts.min() - 3.0 * bw_s, s_pts.max() + 3.0 * bw_s
            xs = np.linspace(x0, x1, gridsize)
            ys = np.linspace(y0, y1, gridsize)
            grid_x, grid_y = np.meshgrid(xs, ys)
            dens = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(
                grid_x.shape
            )

            if np.iterable(kde_levels):
                isoprop = np.asarray(kde_levels, dtype=float)
            else:
                isoprop = np.linspace(kde_thresh, 1.0, int(kde_levels))
            d_levels = self._iso_proportion_levels(dens, isoprop)

            if d_levels.size:
                band = np.digitize(dens, d_levels)  # 0 = outside .. k_max = core
                k_max = int(band.max())

                if k_max >= 1:
                    # KDE "centers" are the local maxima of the density inside
                    # the outermost contour; each becomes its own territory.
                    footprint = peak_footprint or max(3, gridsize // 20)
                    floor = max(float(d_levels[0]), min_peak_height * float(dens.max()))
                    is_peak = (ndimage.maximum_filter(dens, size=footprint) == dens) & (
                        dens > floor
                    )
                    peak_labels, n_peaks = ndimage.label(is_peak)
                    if n_peaks == 0:
                        centroids = np.asarray(
                            [ndimage.center_of_mass(band >= 1)], dtype=float
                        ).reshape(1, 2)
                    else:
                        coms = np.asarray(
                            ndimage.center_of_mass(
                                is_peak, peak_labels, index=range(1, n_peaks + 1)
                            ),
                            dtype=float,
                        ).reshape(n_peaks, 2)
                        rr = np.clip(np.round(coms[:, 0]), 0, gridsize - 1).astype(int)
                        cc = np.clip(np.round(coms[:, 1]), 0, gridsize - 1).astype(int)
                        strongest = np.argsort(dens[rr, cc])[::-1]
                        if max_centers is not None:
                            strongest = strongest[:max_centers]
                        centroids = coms[strongest]
                    n_centers = int(centroids.shape[0])

                    # assign each filled grid cell to its nearest center (Voronoi)
                    fy, fx = np.where(band >= 1)
                    dist = np.linalg.norm(
                        np.column_stack([fy, fx])[:, None, :] - centroids[None, :, :],
                        axis=2,
                    )
                    center_grid = np.zeros(band.shape, dtype=int)
                    center_grid[fy, fx] = np.argmin(dist, axis=1) + 1

                    # distinct color per (center, band): innermost band of
                    # center 1 first, then outward, then on to the next center
                    palette = self._distinct_colors(n_centers * k_max, region_cmap)
                    region_color = {
                        (c, b): palette[(c - 1) * k_max + (k_max - b)]
                        for c in range(1, n_centers + 1)
                        for b in range(1, k_max + 1)
                    }

                    overlay_rgba = np.zeros((*band.shape, 4), dtype=float)
                    drawn: set[tuple[int, int]] = set()
                    for (c, b), color in region_color.items():
                        reg = (center_grid == c) & (band == b)
                        if reg.any():
                            overlay_rgba[reg, :3] = color
                            overlay_rgba[reg, 3] = fill_alpha
                            drawn.add((c, b))
                    region_color = {k: v for k, v in region_color.items() if k in drawn}

                    ax_phasor.imshow(
                        overlay_rgba,
                        extent=(x0, x1, y0, y1),
                        origin="lower",
                        interpolation="nearest",
                        aspect="auto",
                        zorder=1.5,
                    )
                    ax_phasor.contour(
                        grid_x,
                        grid_y,
                        dens,
                        levels=d_levels,
                        colors=kde_color,
                        linewidths=kde_linewidths,
                        alpha=kde_alpha,
                        zorder=3,
                    )

                    ix = np.clip(np.searchsorted(xs, g_pts) - 1, 0, gridsize - 1)
                    iy = np.clip(np.searchsorted(ys, s_pts) - 1, 0, gridsize - 1)
                    pix_band[py_idx, px_idx] = band[iy, ix]
                    pix_center[py_idx, px_idx] = center_grid[iy, ix]

        # imshow with aspect="auto" drops the phasor framing -- restore it
        ax_phasor.set_aspect("equal")
        ax_phasor.set_xlim(*xlim)
        ax_phasor.set_ylim(*ylim)

        # ── (0, 1) grayscale intensity image + matching per-region tint ────
        i_min, i_max = float(np.nanmin(intensity)), float(np.nanmax(intensity))
        gray = (intensity - i_min) / (i_max - i_min + 1e-12)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        for (c, b), color in region_color.items():
            sel = (pix_center == c) & (pix_band == b)
            if sel.any():
                rgb[sel] = (1.0 - fill_alpha) * rgb[sel] + fill_alpha * np.asarray(
                    color
                )
        ax_overlay.imshow(
            np.clip(rgb, 0.0, 1.0), origin="upper", interpolation="nearest"
        )
        ax_overlay.set_title(f"{title} → intensity overlay")
        ax_overlay.axis("off")

        if region_color:
            handles = [
                Patch(
                    facecolor=(*region_color[(c, b)], max(fill_alpha, 0.6)),
                    edgecolor="none",
                    label=f"center {c} · level {b}",
                )
                for (c, b) in region_color
            ]
            ax_overlay.legend(
                handles=handles,
                fontsize=7,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
            )

        if created_fig:
            plt.tight_layout()
        return fig

    def multidata_phasor_plot(
        self,
        gs_list: list[tuple[np.ndarray, np.ndarray]],
        labels: list[str] | None = None,
        colors: Any | None = None,
        harmonic: int = 0,
        region_cmap: str = "tab10",
        ax: Any | None = None,
        figsize: tuple[float, float] = (8, 6),
        half_circle: bool = True,
        xlim: tuple[float, float] = (-0.1, 1.1),
        ylim: tuple[float, float] = (0.0, 0.6),
        title: str = "Multi-dataset Phasor",
        legend: bool = True,
    ) -> Any:
        """
        Scatter several phasor clouds on one shared phasor diagram.

        Each ``(G, S)`` pair in ``gs_list`` is raveled to points and drawn on a
        single diagram produced by
        :meth:`PhasorPlotsMixin.plot_phasor_diagram` (universal semicircle,
        lifetime ticks, frequency label, axis styling). Every dataset is handed
        to that one call as a concatenated point cloud together with a matching
        per-point RGB array, so each dataset keeps its own distinct color; a
        legend then maps the colors back to ``labels``.

        Parameters
        ----------
        gs_list : list[tuple[np.ndarray, np.ndarray]]
            One ``(G, S)`` pair per dataset. Each array is either a ``(H, W)``
            map or a ``(n_harmonics, H, W)`` stack (the ``harmonic`` slice is
            used); the datasets need not share a shape.
        labels : list[str] | None
            Legend label for each dataset. Defaults to ``["dataset 1", ...]``.
        colors : Any | None
            Explicit color per dataset (anything :func:`matplotlib.colors.to_rgb`
            accepts). When ``None``, visually distinct colors are taken from
            ``region_cmap`` via :meth:`_distinct_colors`.
        harmonic : int
            Harmonic slice used for any 3-D ``G``/``S`` stack (``0`` is the
            first harmonic).
        region_cmap : str
            Qualitative colormap the automatic per-dataset colors come from.
        ax : Any | None
            Axes to draw into. A new figure/axes is created when ``None``.
        figsize : tuple[float, float]
            Figure size used when a new figure is created.
        half_circle : bool
            Whether to draw only the upper half of the universal phasor circle.
        xlim, ylim : tuple[float, float]
            Axis limits for the phasor panel.
        title : str
            Title for the phasor diagram.
        legend : bool
            Whether to draw the color-to-label legend.

        Returns
        -------
        Any
            The Matplotlib figure containing the phasor diagram.
        """
        if len(gs_list) == 0:
            raise ValueError("gs_list must contain at least one (G, S) pair")

        n_sets = len(gs_list)
        if labels is None:
            labels = [f"dataset {i + 1}" for i in range(n_sets)]
        elif len(labels) != n_sets:
            raise ValueError("labels must have one entry per (G, S) pair")

        if colors is None:
            rgb_colors = self._distinct_colors(n_sets, region_cmap)
        elif len(colors) != n_sets:
            raise ValueError("colors must have one entry per (G, S) pair")
        else:
            rgb_colors = [to_rgb(c) for c in colors]

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Concatenate every dataset into one point cloud plus a matching
        # per-point RGB array, then let plot_phasor_diagram render it all in a
        # single call (framing + colored scatter).
        g_parts: list[np.ndarray] = []
        s_parts: list[np.ndarray] = []
        c_parts: list[np.ndarray] = []
        for (g, s), color in zip(gs_list, rgb_colors):
            g_arr = np.asarray(g, dtype=float)
            s_arr = np.asarray(s, dtype=float)
            g_2d = g_arr[harmonic] if g_arr.ndim == 3 else g_arr
            s_2d = s_arr[harmonic] if s_arr.ndim == 3 else s_arr
            g_flat = np.ravel(g_2d)
            s_flat = np.ravel(s_2d)
            g_parts.append(g_flat)
            s_parts.append(s_flat)
            c_parts.append(
                np.broadcast_to(np.asarray(color, dtype=float), (g_flat.size, 3))
            )

        g_all = np.concatenate(g_parts)
        s_all = np.concatenate(s_parts)
        c_all = np.concatenate(c_parts, axis=0)

        self.plot_phasor_diagram(
            g_all,
            s_all,
            colors=c_all,
            ax=ax,
            half_circle=half_circle,
            title=title,
            xlim=xlim,
            ylim=ylim,
            kdeplot=False,
        )

        if legend:
            handles = [
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=color,
                    markeredgecolor="none",
                    markersize=6,
                    label=label,
                )
                for color, label in zip(rgb_colors, labels)
            ]
            ax.legend(handles=handles, loc="best", fontsize=8, frameon=False)

        if created_fig:
            plt.tight_layout()
        return fig
