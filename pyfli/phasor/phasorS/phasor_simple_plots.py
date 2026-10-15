"""Plotting mixins for phasor maps, overlays, pixel fits and harmonics."""

from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize

from pyfli.reconstruction.common_reconstruct import periodic_instrument_exponential

from .phasor_simple_utils import (
    _TAU_MARKS_NS,
    _add_frequency_label,
    _draw_lifetime_ticks,
    _style_phasor_ax,
    _universal_circle_xy,
)


class PhasorPlotsMixin:
    # Plotting and color-mapping methods for PhasorAnalyzer.

    """
    Add plotting methods to phasor analyzers. The mixin renders phasor diagrams, color
    maps, overlays, pixel fits, harmonic views, and traceable analysis panels without
    owning acquisition state.
    """

    def phasor_colormap(
        self,
        G: np.ndarray,
        S: np.ndarray,
        intensity: np.ndarray | None = None,
        colormap: str = "viridis",
    ) -> Any:
        """
        Run the phasor colormap routine.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.
        intensity : np.ndarray | None
            Intensity map used to weight or color phasor output.
        colormap : str
            Colormap used to convert phasor coordinates to colors.

        Returns
        -------
        Any
            Object produced by phasor colormap.
        """
        G_col = G[0] if G.ndim == 3 else G
        S_col = S[0] if S.ndim == 3 else S
        phasor_val = np.sqrt(G_col**2 + S_col**2)
        p_min, p_max = np.nanmin(phasor_val), np.nanmax(phasor_val)
        phasor_val = (phasor_val - p_min) / (p_max - p_min + self.eps)
        colors = plt.colormaps[colormap](phasor_val)[:, :, :3]
        if intensity is not None:
            denom = intensity.max() - intensity.min() + self.eps
            int_norm = (intensity - intensity.min()) / denom
            colors = colors * int_norm[:, :, np.newaxis]
        return colors

    def _normal_lifetime_colorbar(
        self,
        G: np.ndarray,
        S: np.ndarray,
        mask: np.ndarray,
        colormap: str,
        half_circle: bool,
    ) -> tuple[Any, ...]:
        """
        Colour bar matching :meth:`phasor_radial_color`, labelled in lifetimes.

        A pixel's hue is ``H = (theta - theta_lo) / (pi - theta_lo)`` with ``theta`` its
        angle around the semicircle centre ``(1/2, 0)`` (``theta_lo = 0`` for the half
        circle, ``-pi`` otherwise). The semicircle point at angle ``theta`` has lifetime
        ``tau_N = tan(theta / 2) / omega`` (the normal lifetime), so the bar shows the
        colormap over the data's ``H`` range, with ticks at ``theta = 2 atan(omega tau)``
        for round lifetimes.

        Returns
        -------
        tuple[Any, ...]
            ``(mappable, tick_positions, tick_lifetimes_ns)``.
        """
        theta_lo = 0.0 if half_circle else -np.pi
        span = np.pi - theta_lo
        theta = np.arctan2(np.asarray(S, dtype=float), np.asarray(G, dtype=float) - 0.5)
        h = np.clip((theta - theta_lo) / span, 0.0, 1.0)
        h = h[np.asarray(mask, dtype=bool) & np.isfinite(h)]
        h_lo, h_hi = (float(h.min()), float(h.max())) if h.size else (0.0, 1.0)
        if h_hi - h_lo < 1e-3:
            h_lo, h_hi = max(h_lo - 0.01, 0.0), min(h_hi + 0.01, 1.0)
        cmap = plt.colormaps[colormap]
        bar_cmap = ListedColormap(cmap(np.linspace(h_lo, h_hi, 256)))
        sm = ScalarMappable(cmap=bar_cmap, norm=Normalize(vmin=h_lo, vmax=h_hi))
        sm.set_array([])
        omega_ns = 2.0 * np.pi * self.frequency * 1e-9
        candidates = np.array(
            [
                0.01,
                0.02,
                0.05,
                0.1,
                0.2,
                0.3,
                0.5,
                0.7,
                1,
                1.5,
                2,
                3,
                4,
                5,
                7,
                10,
                15,
                20,
                50,
                100,
            ]
        )
        h_cand = (2.0 * np.arctan(omega_ns * candidates) - theta_lo) / span
        keep = (h_cand >= h_lo) & (h_cand <= h_hi)
        if keep.sum() < 2:
            ends = theta_lo + np.array([h_lo, h_hi]) * span
            ends = np.clip(ends, 0.0, np.pi - 1e-9)
            tau_ends = np.round(np.tan(ends / 2.0) / omega_ns, 3)
            return (
                sm,
                (2.0 * np.arctan(omega_ns * tau_ends) - theta_lo) / span,
                tau_ends,
            )
        return sm, h_cand[keep], candidates[keep]

    def phasor_radial_color(
        self,
        G: np.ndarray,
        S: np.ndarray,
        colormap: str = "viridis",
        norm_color: bool = False,
        half_circle: bool = True,
    ) -> Any:
        """
        Run the phasor radial color routine.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.
        colormap : str
            Colormap used to convert phasor coordinates to colors.
        norm_color : bool
            Whether phasor colors are normalized before display.
        half_circle : bool
            Whether to draw only the upper half of the universal phasor circle.

        Returns
        -------
        Any
            Object produced by phasor radial color.
        """
        G_col = G[0] if G.ndim == 3 else np.asarray(G)
        S_col = S[0] if S.ndim == 3 else np.asarray(S)

        # Angle and radius relative to universal circle centre (0.5, 0)
        dG = G_col - 0.5
        dS = S_col
        phi = np.arctan2(dS, dG)
        r = np.sqrt(dG**2 + dS**2)

        # Normalise angle to [0, 1] for colormap lookup
        if norm_color:
            phi_min = np.nanmin(phi)
            phi_max = np.nanmax(phi)
            if phi_max <= phi_min:
                phi_max = phi_min + 1.0
            H = np.clip((phi - phi_min) / (phi_max - phi_min + self.eps), 0, 1)
        else:
            phi_lo = 0.0 if half_circle else -np.pi
            phi_hi = np.pi
            H = np.clip((phi - phi_lo) / (phi_hi - phi_lo + self.eps), 0, 1)

        # Base colour from named colormap
        base_rgb = plt.colormaps[colormap](H)[..., :3]

        norm_r = r / 0.5
        V = np.clip(norm_r, 0, 1)

        # Inside/on circle: dim proportionally toward centre
        colors = base_rgb * V[..., np.newaxis]

        # Outside circle: restore brightness, blend toward white (desaturate)
        outside = norm_r > 1.0
        if outside.any():
            sat = np.clip(1.0 / (norm_r[outside] + self.eps), 0, 1)
            colors[outside] = base_rgb[outside] * sat[:, np.newaxis] + (
                1.0 - sat[:, np.newaxis]
            )

        nan_mask = np.isnan(phi) | np.isnan(r)
        if nan_mask.any():
            colors[nan_mask] = 0.0

        return colors

    def plot_phasor_diagram(
        self,
        G: np.ndarray,
        S: np.ndarray,
        mask: np.ndarray | None = None,
        colors: Any | None = None,
        hexbin_color: np.ndarray | None = None,
        ax: Any | None = None,
        figsize: tuple[int, ...] = (8, 3),
        half_circle: bool = True,
        title: str = "Phasor Diagram",
        xlim: tuple[float, ...] = (-0.1, 1.1),
        ylim: tuple[float, ...] = (0.0, 0.6),
        kdeplot: bool = False,
        kde_color: str = "white",
        kde_levels: int = 5,
        kde_linewidths: float = 1,
        kde_alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Plot phasor diagram.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.
        mask : np.ndarray | None
            Boolean or labeled mask selecting pixels for the operation.
        colors : Any | None
            Color sequence used for plotted sources or groups.
        hexbin_color : np.ndarray | None
            Optional values used to color phasor hexbin density.
        ax : Any | None
            Matplotlib axes object on which the plot is drawn.
        figsize : tuple[int, ...]
            Figure size passed to Matplotlib.
        half_circle : bool
            Whether to draw only the upper half of the universal phasor circle.
        title : str
            Title displayed on the generated plot.
        xlim : tuple[float, ...]
            X-axis limits for the phasor plot.
        ylim : tuple[float, ...]
            Y-axis limits for the phasor plot.
        kdeplot : bool
            Whether to overlay a KDE density contour on the phasor points.
        kde_color : str
            Color used for the KDE density contour lines.
        kde_levels : int
            Number of contour levels drawn for the KDE overlay.
        kde_linewidths : float
            Line width of the KDE contour lines.
        kde_alpha : float
            Opacity of the KDE contour lines.

        Returns
        -------
        np.ndarray
            Matplotlib figure or axes containing the phasor diagram.
        """
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ug, us = _universal_circle_xy(half_circle=half_circle)
        ax.plot(ug, us, "k--")

        G_2d = G[0] if (np.ndim(G) == 3) else np.asarray(G)
        S_2d = S[0] if (np.ndim(S) == 3) else np.asarray(S)
        g_flat = np.ravel(G_2d)
        s_flat = np.ravel(S_2d)

        if mask is not None:
            mask_flat = np.ravel(mask).astype(
                bool
            )  # uint8 masks must be bool; int array would integer-index otherwise
            g_plot = g_flat[mask_flat]
            s_plot = s_flat[mask_flat]
        else:
            g_plot = g_flat
            s_plot = s_flat

        valid = ~np.isnan(g_plot) & ~np.isnan(s_plot)
        g_plot = g_plot[valid]
        s_plot = s_plot[valid]

        if colors is None:
            cmap_to_use = hexbin_color if hexbin_color is not None else "autumn"
            hb = ax.hexbin(
                g_plot, s_plot, gridsize=[800, 400], cmap=cmap_to_use, mincnt=1
            )
            fig.colorbar(hb, ax=ax).set_label("Pixel Count")
        else:
            if isinstance(colors, str):
                c_vals = g_plot
                path = ax.scatter(
                    g_plot, s_plot, cmap=colors, c=c_vals, s=8, marker="o"
                )
                fig.colorbar(path, ax=ax).set_label("Phasor G Value")
            else:
                c_flat = np.reshape(colors, (-1, 3))
                if mask is not None:
                    c_plot = c_flat[mask_flat][valid]
                else:
                    c_plot = c_flat[valid]
                ax.scatter(g_plot, s_plot, c=c_plot, s=8, marker="o")

        if kdeplot:
            sns.kdeplot(
                x=g_plot,
                y=s_plot,
                ax=ax,
                levels=kde_levels,
                color=kde_color,
                linewidths=kde_linewidths,
                alpha=kde_alpha,
            )

        G_mark, S_mark = self.lifetime_to_phasor(_TAU_MARKS_NS, self.frequency)
        _draw_lifetime_ticks(
            ax, G_mark, S_mark, color="black", lw=4, fontsize=10, show_units=True
        )
        _style_phasor_ax(ax, title=title, xlim=xlim, ylim=ylim, half_circle=half_circle)
        _add_frequency_label(ax, self.frequency)

        if created_fig:
            plt.tight_layout()
        return fig

    def plot_map(
        self,
        image: np.ndarray,
        scales: list[Any] = [0, 2],
        title: str = "",
        ax: Any | None = None,
        figsize: tuple[int, ...] = (8, 6),
    ) -> np.ndarray:
        """
        Plot map.

        Parameters
        ----------
        image : np.ndarray
            Image array displayed or processed by the routine.
        scales : list[Any]
            Display limits used when plotting an image.
        title : str
            Title displayed on the generated plot.
        ax : Any | None
            Matplotlib axes object on which the plot is drawn.
        figsize : tuple[int, ...]
            Figure size passed to Matplotlib.

        Returns
        -------
        np.ndarray
            Matplotlib figure or axes containing the rendered map.
        """
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        im = ax.imshow(
            np.clip(image, scales[0], scales[1]), origin="upper", cmap="viridis"
        )
        fig.colorbar(im, ax=ax).set_label("Lifetime (ns)")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(False)
        if created_fig:
            plt.tight_layout()
        return fig

    def plot_phasor_overlay(
        self,
        decay: np.ndarray,
        G: np.ndarray,
        S: np.ndarray,
        colormap: str = "viridis",
        ax: Any | None = None,
        figsize: tuple[int, ...] = (8, 8),
    ) -> np.ndarray:
        """
        Plot phasor overlay.

        Parameters
        ----------
        decay : np.ndarray
            Time-resolved decay signal or decay cube.
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.
        colormap : str
            Colormap used to convert phasor coordinates to colors.
        ax : Any | None
            Matplotlib axes object on which the plot is drawn.
        figsize : tuple[int, ...]
            Figure size passed to Matplotlib.

        Returns
        -------
        np.ndarray
            Matplotlib figure or axes containing the phasor overlay.
        """
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        intensity_img = self.generate_intensity_image(decay)
        phasor_colors = self.phasor_colormap(G, S, colormap=colormap)
        int_norm = (intensity_img - intensity_img.min()) / (
            intensity_img.max() - intensity_img.min() + self.eps
        )
        overlay = np.stack([int_norm] * 3, axis=2) * phasor_colors
        ax.imshow(overlay, origin="upper")
        ax.set_title("Intensity + Phasor Color Overlay")
        ax.axis("off")
        if created_fig:
            plt.tight_layout()
        return fig

    def plot_pure_phasor_map(
        self,
        G: np.ndarray,
        S: np.ndarray,
        decay: np.ndarray,
        noise_removed: bool = True,
        colormap: str = "viridis",
        ax: Any | None = None,
        figsize: tuple[int, ...] = (4, 4),
    ) -> np.ndarray:
        """
        Plot pure phasor map.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.
        decay : np.ndarray
            Time-resolved decay signal or decay cube.
        noise_removed : bool
            Whether noise-filtered phasor points are shown.
        colormap : str
            Colormap used to convert phasor coordinates to colors.
        ax : Any | None
            Matplotlib axes object on which the plot is drawn.
        figsize : tuple[int, ...]
            Figure size passed to Matplotlib.

        Returns
        -------
        np.ndarray
            Matplotlib figure or axes containing the pure phasor map.
        """
        phasor_colors = self.phasor_colormap(G, S, colormap=colormap)
        if phasor_colors.shape[-1] == 4:
            phasor_colors = phasor_colors[..., :3]

        if noise_removed:
            intensity_img = self.generate_intensity_image(decay)
            i_min, i_max = intensity_img.min(), intensity_img.max()
            denom = (i_max - i_min) if (i_max - i_min) != 0 else 1
            int_norm = (intensity_img - i_min) / denom
            final_mask = int_norm > 0.1
        else:
            final_mask = np.ones(phasor_colors.shape[:2], dtype=bool)

        pure_overlay = np.zeros_like(phasor_colors)
        pure_overlay[final_mask] = phasor_colors[final_mask]

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.imshow(pure_overlay, origin="upper")
        ax.set_title(f"Pure Phasor Map (Noise Removed: {noise_removed})")
        ax.axis("off")
        if created_fig:
            plt.tight_layout()
        return fig

    def plot_overlay_subplots(
        self,
        decay: np.ndarray,
        G: np.ndarray,
        S: np.ndarray,
        mask: np.ndarray | None = None,
        colormaps: list[Any] = ["jet", "jet", "viridis", "jet"],
        noise_removed: bool = True,
        figsize: tuple[int, ...] = (15, 10),
        half_circle: bool = True,
        xlim: tuple[float, ...] = (-0.1, 1.1),
        ylim: tuple[float, ...] = (0.0, 0.6),
        bg_color: str = "black",
        transpose: bool = False,
        kdeplot: bool = True,
        kde_levels: int = 3,
    ) -> np.ndarray:
        """
        Plot overlay subplots.

        Parameters
        ----------
        decay : np.ndarray
            Time-resolved decay signal or decay cube.
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.
        mask : np.ndarray | None
            Boolean or labeled mask selecting pixels for the operation.
        colormaps : list[Any]
            Colormap specifications used for overlay panels.
        noise_removed : bool
            Whether noise-filtered phasor points are shown.
        figsize : tuple[int, ...]
            Figure size passed to Matplotlib.
        half_circle : bool
            Whether to draw only the upper half of the universal phasor circle.
        xlim : tuple[float, ...]
            X-axis limits for the phasor plot.
        ylim : tuple[float, ...]
            Y-axis limits for the phasor plot.
        bg_color : str
            Background color used behind the phasor overlay.
        transpose : bool
            Whether image-like arrays are transposed before display.
        kdeplot : bool
            Whether to overlay a KDE density contour on the color-scatter phasor panel.
        kde_levels : int
            Number of contour levels drawn for the KDE overlay.

        Returns
        -------
        np.ndarray
            Matplotlib figure or axes containing overlay subplot panels.
        """
        _t = (lambda a: np.swapaxes(a, 0, 1)) if transpose else (lambda a: a)

        G_2d = G[0] if G.ndim == 3 else G
        S_2d = S[0] if S.ndim == 3 else S
        intensity_img = self.generate_intensity_image(decay)

        phasor_colors_raw = self.phasor_radial_color(
            G_2d, S_2d, colormap=colormaps[2], half_circle=half_circle
        )

        if mask is None:
            if noise_removed:
                int_norm = (intensity_img - intensity_img.min()) / (
                    intensity_img.max() - intensity_img.min() + self.eps
                )
                active_mask = int_norm > 0.1
            else:
                active_mask = np.ones(G_2d.shape, dtype=bool)
        else:
            active_mask = mask.astype(bool)

        bg_val = 0.0 if bg_color == "black" else 1.0

        def _resolve_cmap(spec: Any) -> np.ndarray:
            """
            Run the resolve cmap routine.

            Parameters
            ----------
            spec : Any
                Colormap specification resolved by the plotting helper.

            Returns
            -------
            np.ndarray
                Resolved Matplotlib colormap object.
            """
            cmap = plt.colormaps[spec] if isinstance(spec, str) else spec
            cmap = cmap.copy()
            cmap.set_bad(bg_color)
            return cmap

        cmap1 = _resolve_cmap(colormaps[0])
        cmap3 = _resolve_cmap(colormaps[1])

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 3, figure=fig)

        # (0,0) Intensity — inactive pixels → NaN → bg_color
        ax1 = fig.add_subplot(gs[0, 0])
        int_masked = np.where(active_mask, intensity_img.astype(float), np.nan)
        im1 = ax1.imshow(_t(int_masked), origin="upper", cmap=cmap1)
        ax1.set_title("Intensity")
        ax1.axis("off")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # (0,1) Phasor colour projection — inactive pixels → bg_val
        ax2 = fig.add_subplot(gs[0, 1])
        pure_overlay = np.full_like(phasor_colors_raw, bg_val)
        pure_overlay[active_mask] = phasor_colors_raw[active_mask]
        ax2.imshow(_t(pure_overlay), origin="upper")
        ax2.set_title("Phasor Color Projections")
        ax2.axis("off")

        # (1,0) Lifetime map — inactive pixels → NaN → bg_color
        ax3 = fig.add_subplot(gs[1, 0])
        tau_map_ns = np.clip(self.compute_lifetime(G_2d, S_2d), 0, None)
        tau_masked = np.where(active_mask, tau_map_ns, np.nan)
        im3 = ax3.imshow(_t(tau_masked), origin="upper", cmap=cmap3)
        ax3.set_title("Lifetime Map (ns)")
        ax3.axis("off")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).set_label("ns")

        # (1,1) Intensity-weighted overlay — inactive pixels → bg_val
        ax4 = fig.add_subplot(gs[1, 1])
        int_norm_3d = (intensity_img - intensity_img.min()) / (
            intensity_img.max() - intensity_img.min() + self.eps
        )
        weighted_overlay = np.full_like(phasor_colors_raw, bg_val)
        weighted_overlay[active_mask] = (
            np.stack([int_norm_3d] * 3, axis=2) * phasor_colors_raw
        )[active_mask]
        ax4.imshow(_t(weighted_overlay), origin="upper")
        ax4.set_title("Intensity-weighted Overlay")
        ax4.axis("off")

        mask_flat = np.ravel(active_mask)
        g_plot = np.ravel(G_2d)[mask_flat]
        s_plot = np.ravel(S_2d)[mask_flat]
        c_plot = np.reshape(phasor_colors_raw, (-1, 3))[mask_flat]
        valid = ~np.isnan(g_plot) & ~np.isnan(s_plot)
        g_v, s_v, c_v = g_plot[valid], s_plot[valid], c_plot[valid]

        ug, us = _universal_circle_xy(half_circle=half_circle)
        try:
            G_mark, S_mark = self.lifetime_to_phasor(_TAU_MARKS_NS, self.frequency)
        except Exception:
            G_mark = S_mark = None

        # scatter color by phasor_radial_color
        ax5 = fig.add_subplot(gs[0, 2])
        ax5.plot(ug, us, "k--", alpha=0.8, zorder=1)
        if len(g_v):
            ax5.scatter(g_v, s_v, c=c_v, s=2, alpha=0.6, edgecolors="none", zorder=2)
            if kdeplot:
                sns.kdeplot(
                    x=g_v,
                    y=s_v,
                    ax=ax5,
                    levels=kde_levels,
                    color="w",
                    linewidths=1,
                    zorder=3,
                )
        if G_mark is not None:
            _draw_lifetime_ticks(ax5, G_mark, S_mark, color="black", lw=2, fontsize=9)
        _style_phasor_ax(
            ax5, title="Phasor (color)", xlim=xlim, ylim=ylim, half_circle=half_circle
        )
        _add_frequency_label(ax5, self.frequency)

        # ax6 (1,2): phasor hexbin density plot
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(ug, us, "k--", alpha=0.8, zorder=1)
        if len(g_v):
            hb = ax6.hexbin(
                g_v, s_v, gridsize=[800, 400], cmap=colormaps[3], mincnt=1, zorder=2
            )
            fig.colorbar(hb, ax=ax6, fraction=0.046, pad=0.04).set_label("Pixel count")
        if G_mark is not None:
            _draw_lifetime_ticks(ax6, G_mark, S_mark, color="black", lw=2, fontsize=9)
        _style_phasor_ax(
            ax6, title="Phasor (density)", xlim=xlim, ylim=ylim, half_circle=half_circle
        )
        _add_frequency_label(ax6, self.frequency)

        plt.tight_layout()
        return fig

    def plot_pixel_fit(
        self,
        irf: np.ndarray,
        decay: np.ndarray,
        reconstructed_decay: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        log_scale: bool = True,
        ax: Any | None = None,
        figsize: tuple[int, ...] = (10, 6),
    ) -> np.ndarray:
        """
        Plot pixel fit.

        Parameters
        ----------
        irf : np.ndarray
            Instrument response function aligned with the decay signal.
        decay : np.ndarray
            Time-resolved decay signal or decay cube.
        reconstructed_decay : np.ndarray
            Model decay reconstructed from fitted parameters.
        x : np.ndarray
            Input array, coordinate, or signal being transformed.
        y : np.ndarray
            Observed signal, target data, or coordinate array.
        log_scale : bool
            Whether the decay axis is drawn on a logarithmic scale.
        ax : Any | None
            Matplotlib axes object on which the plot is drawn.
        figsize : tuple[int, ...]
            Figure size passed to Matplotlib.

        Returns
        -------
        np.ndarray
            Matplotlib figure or axes containing the pixel fit.
        """
        irf_trace = irf[y, x, :] if irf.ndim == 3 else np.asarray(irf)
        raw_trace = decay[y, x, :]
        fit_trace = reconstructed_decay[y, x, :]

        traces_np = np.stack([irf_trace, raw_trace, fit_trace], axis=0).astype(
            np.float32
        )
        traces_t = torch.tensor(traces_np, device=self.device)
        maxvals = traces_t.amax(dim=1, keepdim=True).clamp(min=self.eps)
        norm_t = (traces_t / maxvals).cpu().numpy()

        irf_norm, raw_norm, fit_norm = norm_t[0], norm_t[1], norm_t[2]

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.plot(self.time_axis_ns, irf_norm, "k--", alpha=0.5, label="IRF (Normalized)")
        ax.plot(
            self.time_axis_ns,
            raw_norm,
            "ro",
            markersize=4,
            alpha=0.6,
            label=f"Raw Decay (Pixel {x},{y})",
        )
        ax.plot(self.time_axis_ns, fit_norm, "b-", lw=2, label="Reconstructed Fit")

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1.2)
            ax.set_ylabel("Normalized Intensity (Log Scale)")
        else:
            ax.set_ylabel("Normalized Intensity (Linear Scale)")

        ax.set_xlabel("Time (ns)")
        ax.set_title(
            f"Decay Analysis at Pixel (X: {x}, Y: {y})  [device: {self.device}]"
        )
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        if created_fig:
            plt.tight_layout()
        return fig

    def plot_pixel_fit_single_exp(
        self,
        irf: np.ndarray,
        decay: np.ndarray,
        tau_ns: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        log_scale: bool = True,
        ax: Any | None = None,
        figsize: tuple[int, ...] = (10, 6),
    ) -> np.ndarray:
        """
        Plot pixel fit single exp.

        Parameters
        ----------
        irf : np.ndarray
            Instrument response function aligned with the decay signal.
        decay : np.ndarray
            Time-resolved decay signal or decay cube.
        tau_ns : np.ndarray
            Lifetime value in nanoseconds.
        x : np.ndarray
            Input array, coordinate, or signal being transformed.
        y : np.ndarray
            Observed signal, target data, or coordinate array.
        log_scale : bool
            Whether the decay axis is drawn on a logarithmic scale.
        ax : Any | None
            Matplotlib axes object on which the plot is drawn.
        figsize : tuple[int, ...]
            Figure size passed to Matplotlib.

        Returns
        -------
        np.ndarray
            Matplotlib figure or axes containing the single-exponential pixel fit.
        """
        if isinstance(tau_ns, (torch.Tensor, np.ndarray)):
            if tau_ns.ndim >= 2:
                tau_val = tau_ns[y, x]
            else:
                tau_val = tau_ns
            if torch.is_tensor(tau_val):
                tau_val = tau_val.item()
        else:
            tau_val = tau_ns

        model_t = torch.tensor(
            periodic_instrument_exponential(
                float(tau_val), len(self.t_s_np), 1e9 / self.frequency
            ),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        irf_trace_np = irf[y, x, :] if irf.ndim == 3 else np.asarray(irf)
        irf_trace_t = torch.tensor(
            irf_trace_np.astype(np.float32), device=self.device
        ).unsqueeze(0)

        irf_norm_t = irf_trace_t / irf_trace_t.sum(dim=1, keepdim=True).clamp(
            min=self.eps
        )
        fit_t = self._convolve_batch(model_t, irf_norm_t)
        fit_trace_np = fit_t.squeeze(0).cpu().numpy()

        raw_trace_np = decay[y, x, :]

        traces_np = np.stack([irf_trace_np, raw_trace_np, fit_trace_np], axis=0).astype(
            np.float32
        )
        traces_t = torch.tensor(traces_np, device=self.device)
        maxvals = traces_t.amax(dim=1, keepdim=True).clamp(min=self.eps)
        norm_t = (traces_t / maxvals).cpu().numpy()

        irf_norm, raw_norm, fit_norm = norm_t[0], norm_t[1], norm_t[2]

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.plot(self.time_axis_ns, irf_norm, "k--", alpha=0.5, label="IRF (Normalized)")
        ax.plot(
            self.time_axis_ns,
            raw_norm,
            "ro",
            markersize=4,
            alpha=0.6,
            label=f"Raw Decay (Pixel {x},{y})",
        )
        ax.plot(
            self.time_axis_ns,
            fit_norm,
            "b-",
            lw=2,
            label=f"Single-Exp Fit  τ = {tau_val} ns",
        )

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1.2)
            ax.set_ylabel("Normalized Intensity (Log Scale)")
        else:
            ax.set_ylabel("Normalized Intensity (Linear Scale)")

        ax.set_xlabel("Time (ns)")
        ax.set_title(
            f"Single-Exponential Decay at Pixel (X: {x}, Y: {y})  "
            f"τ = {tau_val} ns  [device: {self.device}]"
        )
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        if created_fig:
            plt.tight_layout()
        return fig

    def plot_phasor_harmonics(
        self,
        G: np.ndarray,
        S: np.ndarray,
        harmonics: tuple[int, ...] = (1, 2, 3, 4),
        mask: np.ndarray | None = None,
        colors: Any | None = None,
        hexbin_color: np.ndarray | None = None,
        figsize: tuple[int, ...] = (22, 5),
        axes: Any | None = None,
        half_circle: bool = True,
        xlim: tuple[float, ...] = (-0.1, 1.1),
        ylim: tuple[float, ...] = (0.0, 0.6),
    ) -> np.ndarray:
        """
        Plot phasor harmonics.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.
        harmonics : tuple[int, ...]
            Harmonic indices included in the phasor plot.
        mask : np.ndarray | None
            Boolean or labeled mask selecting pixels for the operation.
        colors : Any | None
            Color sequence used for plotted sources or groups.
        hexbin_color : np.ndarray | None
            Optional values used to color phasor hexbin density.
        figsize : tuple[int, ...]
            Figure size passed to Matplotlib.
        axes : Any | None
            Matplotlib axes collection used for drawing subplots.
        half_circle : bool
            Whether to draw only the upper half of the universal phasor circle.
        xlim : tuple[float, ...]
            X-axis limits for the phasor plot.
        ylim : tuple[float, ...]
            Y-axis limits for the phasor plot.

        Returns
        -------
        np.ndarray
            Matplotlib figure or axes containing phasor harmonics.
        """
        G = np.asarray(G)
        S = np.asarray(S)
        n_panels = len(harmonics)

        created_fig = axes is None
        if created_fig:
            fig, axes = plt.subplots(1, n_panels, figsize=figsize)
            if n_panels == 1:
                axes = [axes]
        else:
            fig = axes[0].get_figure()
        mask_flat = np.ravel(mask).astype(bool) if mask is not None else None

        for ax, k in zip(axes, harmonics):
            if G.ndim == 3 and k <= G.shape[0]:
                g_panel = G[k - 1]
                s_panel = S[k - 1]
            else:
                g_panel = G[0] if G.ndim == 3 else G
                s_panel = S[0] if S.ndim == 3 else S

            ug, us = _universal_circle_xy(half_circle=half_circle)
            ax.plot(ug, us, "k--", lw=1.2)

            g_flat = np.ravel(g_panel)
            s_flat = np.ravel(s_panel)

            if mask_flat is not None:
                g_plot = g_flat[mask_flat]
                s_plot = s_flat[mask_flat]
            else:
                g_plot = g_flat
                s_plot = s_flat

            valid = ~np.isnan(g_plot) & ~np.isnan(s_plot)
            g_plot = g_plot[valid]
            s_plot = s_plot[valid]

            if colors is None:
                cmap_to_use = hexbin_color if hexbin_color is not None else "jet"
                hb = ax.hexbin(
                    g_plot, s_plot, gridsize=[800, 400], cmap=cmap_to_use, mincnt=1
                )
                fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04).set_label(
                    "Pixel Count"
                )
            else:
                if isinstance(colors, str):
                    path = ax.scatter(
                        g_plot, s_plot, cmap=colors, c=g_plot, s=8, marker="o"
                    )
                    fig.colorbar(path, ax=ax, fraction=0.046, pad=0.04).set_label(
                        "Phasor G Value"
                    )
                else:
                    # Bug fix 2: if colors has a harmonic axis (n, H, W, 3) use slice k-1;
                    # otherwise treat as a single (H, W, 3) array shared across all harmonics
                    colors_arr = np.asarray(colors)
                    if colors_arr.ndim == 4 and k - 1 < colors_arr.shape[0]:
                        c_panel = colors_arr[k - 1]
                    else:
                        c_panel = colors_arr
                    c_flat = np.reshape(c_panel, (-1, 3))
                    if mask_flat is not None:
                        c_plot = c_flat[mask_flat][valid]
                    else:
                        c_plot = c_flat[valid]
                    ax.scatter(g_plot, s_plot, c=c_plot, s=8, marker="o")

            G_mark, S_mark = self.lifetime_to_phasor(_TAU_MARKS_NS, k * self.frequency)
            _draw_lifetime_ticks(
                ax,
                G_mark,
                S_mark,
                color="black",
                lw=2,
                fontsize=7,
                show_units=(k == harmonics[0]),
                tick_length=0.03,
                text_offset=0.05,
            )
            _style_phasor_ax(
                ax,
                title=rf"Harmonic {k} ($\omega_{{{k}}}$)",
                xlim=xlim,
                ylim=ylim,
                half_circle=half_circle,
            )
            _add_frequency_label(ax, k * self.frequency)

        if created_fig:
            fig.suptitle(
                "Phasor Diagram — Multiple Harmonics",
                fontsize=12,
                fontweight="bold",
                y=1.01,
            )
            plt.tight_layout()
        return fig

    def plot_traceable_analysis(
        self,
        G: np.ndarray,
        S: np.ndarray,
        mask: np.ndarray | None = None,
        colormap: str = "viridis",
        figsize: tuple[int, ...] = (14, 6),
        axes: Any | None = None,
        half_circle: bool = True,
        xlim: tuple[float, ...] = (-0.1, 1.1),
        ylim: tuple[float, ...] = (0.0, 0.6),
    ) -> np.ndarray:
        """
        Plot traceable analysis.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.
        mask : np.ndarray | None
            Boolean or labeled mask selecting pixels for the operation.
        colormap : str
            Colormap used to convert phasor coordinates to colors.
        figsize : tuple[int, ...]
            Figure size passed to Matplotlib.
        axes : Any | None
            Matplotlib axes collection used for drawing subplots.
        half_circle : bool
            Whether to draw only the upper half of the universal phasor circle.
        xlim : tuple[float, ...]
            X-axis limits for the phasor plot.
        ylim : tuple[float, ...]
            Y-axis limits for the phasor plot.

        Returns
        -------
        np.ndarray
            Matplotlib figure or axes containing traceable phasor analysis panels.
        """
        G_2d = G[0] if G.ndim == 3 else G
        S_2d = S[0] if S.ndim == 3 else S

        phasor_colors_raw = self.phasor_radial_color(
            G_2d, S_2d, colormap=colormap, half_circle=half_circle
        )

        if mask is None:
            active_mask = np.ones(G_2d.shape, dtype=bool)
        else:
            active_mask = mask.astype(bool)

        created_fig = axes is None
        if created_fig:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        else:
            fig = axes[0].get_figure()

        pure_overlay = np.zeros_like(phasor_colors_raw)
        pure_overlay[active_mask] = phasor_colors_raw[active_mask]
        axes[0].imshow(pure_overlay, origin="upper")
        axes[0].set_title("Phasor Color Projections")
        axes[0].axis("off")

        sm, h_ticks, tau_ticks = self._normal_lifetime_colorbar(
            G_2d, S_2d, active_mask, colormap, half_circle
        )
        cbar = fig.colorbar(sm, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_ticks(h_ticks)
        cbar.set_ticklabels([f"{t:g}" for t in tau_ticks])
        cbar.set_label("Normal lifetime τ_N (ns)")

        ug, us = _universal_circle_xy(half_circle=half_circle)
        axes[1].plot(ug, us, "k--", alpha=0.8, zorder=1)

        mask_flat = np.ravel(active_mask)
        g_plot = np.ravel(G_2d)[mask_flat]
        s_plot = np.ravel(S_2d)[mask_flat]
        c_plot = np.reshape(phasor_colors_raw, (-1, 3))[mask_flat]
        valid_data = ~np.isnan(g_plot) & ~np.isnan(s_plot)
        g_plot, s_plot, c_plot = (
            g_plot[valid_data],
            s_plot[valid_data],
            c_plot[valid_data],
        )

        if len(g_plot) > 0:
            axes[1].scatter(
                g_plot, s_plot, c=c_plot, s=10, alpha=0.8, edgecolors="none", zorder=2
            )
        else:
            axes[1].text(0.5, 0.25, "No valid data points", ha="center", color="red")

        try:
            G_mark, S_mark = self.lifetime_to_phasor(_TAU_MARKS_NS, self.frequency)
            _draw_lifetime_ticks(
                axes[1], G_mark, S_mark, color="black", lw=2, fontsize=9
            )
        except Exception:
            pass

        _style_phasor_ax(
            axes[1],
            title="Phasor Distribution",
            xlim=xlim,
            ylim=ylim,
            half_circle=half_circle,
        )
        _add_frequency_label(axes[1], self.frequency)

        if created_fig:
            plt.tight_layout()
        return fig
