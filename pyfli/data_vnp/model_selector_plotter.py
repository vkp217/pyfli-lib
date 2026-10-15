"""Figures for per-pixel mono- vs bi-exponential model selection."""

import os
from typing import Any, ClassVar

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from pyfli import logging

from .model_selector import ModelSelector

FLI_CMAP = LinearSegmentedColormap.from_list(
    "fli",
    [
        "#ef2312",
        "#f78c00",
        "#e6bc07",
        "#cadd08",
        "#9ff200",
        "#5af308",
        "#00ef5e",
        "#00eba2",
        "#00dacf",
        "#00bae4",
        "#0089ef",
        "#0817e6",
    ],
    N=256,
)
FRET_CMAP = FLI_CMAP.reversed()
MM = 1 / 25.4


def _sans_font() -> str:
    names = {x.name for x in fm.fontManager.ttflist}
    return next(
        (
            f
            for f in ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
            if f in names
        ),
        "sans-serif",
    )


_BLACK_RC = {
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "axes.facecolor": "white",
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.grid": False,
    "savefig.dpi": 600,
}


class ModelSelectorPlotter:
    """
    Publication figures (Nature Methods style) driven by mono/bi model selection.

    ``model`` chooses which model every ROI pixel is shown with:

    - ``"selector"`` (default): follow :class:`ModelSelector`. Draws the ``split``
      figures when at least ``split_min_frac`` of ROI pixels are bi-justified,
      otherwise the whole-ROI mono figure. Box plots use the selector's pixel classes.
    - ``"mono"``: every ROI pixel is mono-exponential. Map figure of photon counts
      and τ; box plot of τ over the whole ROI. Only the mono fit is needed.
    - ``"bi"``: every ROI pixel is bi-exponential. Map figure of photon counts, τ1,
      τ2, τm, FRET fraction and FRET efficiency; box plots of those over the whole
      ROI. Only the bi fit is needed.
    - ``"split"``: like ``"selector"`` but always draws the split figures. Each
      figure shows only its own pixel class (mono: classes 1 and 4; bi: class 2),
      other ROI pixels are drawn in ``other_color``, and the mono figure gains an
      unmasked photon-count panel.

    Parameters
    ----------
    selector : ModelSelector | None
        Selector that has already run ``compare``/``compare_from_dir``. Needed for
        ``"selector"`` and ``"split"``; not used by ``"mono"`` or ``"bi"``.
    save_dir : str | None
        Directory holding ``{fit}_{mono|bi}-exponential.npy`` and receiving the
        saved figures. ``None`` disables saving (then ``maps`` must be given).
    b_bool_mask : np.ndarray | None
        ROI mask; defaults to ``selector.roi``.
    fit : str
        File prefix of the saved fit, e.g. ``"CPU_MLE_poisson"`` or
        ``"CPU_NLSF_least_squares"``.
    maps : dict | None
        Optional ``{"mono": maps, "bi": maps}`` result maps used instead of loading
        from ``save_dir``.
    model : str
        ``"selector"`` (default), ``"mono"``, ``"bi"`` or ``"split"``; see above.
    fallback_model : str
        Model used by ``"selector"`` when no selector result is available
        (``"mono"`` or ``"bi"``).
    split_min_frac : float
        Bi-justified fraction needed for ``"selector"`` to draw the split figures.
    other_color : str
        Colour of ROI pixels in the other class in split mode; ``"transparent"`` leaves
        them blank.
    show_other_key : bool
        Draw a small key explaining the ``other_color`` pixels.
    pixel_size : float | None
        Physical pixel size; ``None`` draws no scale bar.
    scale_bar : tuple[float, str]
        ``(length, unit)`` of the scale bar, used only when ``pixel_size`` is set.
    clip_pct : tuple[float, float]
        Percentiles for robust colour limits from the shown pixels.
    tau_bound : float
        Lifetime fit bound in ns; values at or above it are ignored for colour limits.
    v_ranges : dict | None
        Manual colour-limit overrides, e.g. ``{"tau_map": (0.5, 1.5)}``.
    crop_to_roi : bool
        Crop map panels to the ROI bounding box.
    swap_xy : bool
        Transpose every map.
    bg_color : str
        Colour of pixels outside the ROI.
    tau_box : bool
        Add a τ box-plot panel to the mono-exponential map figure.
    """

    PANELS: ClassVar[dict[str, list[tuple[str, str, str, Any, float]]]] = {
        "mono_split": [
            (
                "photon_count_map",
                "Photon counts (all pixels)",
                r"Counts ($\times10^{3}$)",
                "magma",
                1e-3,
            ),
            (
                "photon_count_map",
                "Photon counts (mono pixels)",
                r"Counts ($\times10^{3}$)",
                "magma",
                1e-3,
            ),
            ("tau_map", r"Lifetime $\tau$", r"$\tau$ (ns)", FLI_CMAP, 1),
        ],
        "mono": [
            (
                "photon_count_map",
                "Photon counts",
                r"Counts ($\times10^{3}$)",
                "magma",
                1e-3,
            ),
            ("tau_map", r"Lifetime $\tau$", r"$\tau$ (ns)", FLI_CMAP, 1),
        ],
        "bi": [
            (
                "photon_count_map",
                "Photon counts",
                r"Counts ($\times10^{3}$)",
                "magma",
                1e-3,
            ),
            ("tau1_map", r"$\tau_1$", r"$\tau_1$ (ns)", FLI_CMAP, 1),
            ("tau2_map", r"$\tau_2$", r"$\tau_2$ (ns)", FLI_CMAP, 1),
            (
                "tau_mean_map",
                r"$\tau_\mathrm{m}$",
                r"$\tau_\mathrm{m}$ (ns)",
                FLI_CMAP,
                1,
            ),
            ("alpha1_map", "FRET fraction", r"$f_\mathrm{D}$ (%)", FRET_CMAP, 100),
            ("fret_efficiency_map", "FRET efficiency", r"$E$ (%)", FRET_CMAP, 100),
        ],
    }
    LAYOUT: ClassVar[dict[str, tuple[int, int, float]]] = {
        "mono": (1, 2, 120),
        "mono_split": (1, 3, 180),
        "bi": (2, 3, 180),
    }
    BOX_COLORS: ClassVar[dict[str, str]] = {
        "tau": "#0072B2",
        "tau1": "#D55E00",
        "tau2": "#0072B2",
        "taum": "#009E73",
        "frac": "#CC79A7",
        "eff": "#E69F00",
    }
    MONO_CLASSES: ClassVar[tuple[int, ...]] = (1, 4)
    BI_CLASSES: ClassVar[tuple[int, ...]] = (2,)
    MODELS: ClassVar[tuple[str, ...]] = ("selector", "mono", "bi", "split")

    def __init__(
        self,
        selector: ModelSelector | None = None,
        save_dir: str | None = None,
        b_bool_mask: np.ndarray | None = None,
        fit: str = "CPU_MLE_poisson",
        maps: dict[str, dict[str, Any]] | None = None,
        model: str = "selector",
        fallback_model: str = "mono",
        split_min_frac: float = 0.05,
        other_color: str = "transparent",
        show_other_key: bool = True,
        pixel_size: float | None = None,
        scale_bar: tuple[float, str] = (5, "mm"),
        clip_pct: tuple[float, float] = (2, 98),
        tau_bound: float = 12.5,
        v_ranges: dict[str, tuple[float, float]] | None = None,
        crop_to_roi: bool = True,
        swap_xy: bool = True,
        bg_color: str = "grey",
        tau_box: bool = True,
    ) -> None:
        if model not in self.MODELS:
            raise ValueError(f"model must be one of {self.MODELS}, got {model!r}.")
        if fallback_model not in ("mono", "bi"):
            raise ValueError(
                f"fallback_model must be 'mono' or 'bi', got {fallback_model!r}."
            )
        if b_bool_mask is None:
            if selector is None:
                raise ValueError("Provide b_bool_mask or a ModelSelector.")
            b_bool_mask = selector.roi
        if save_dir is None and maps is None:
            raise ValueError("Provide save_dir or maps.")
        self.selector = selector
        self.save_dir = save_dir
        self.roi = np.asarray(b_bool_mask).astype(bool)
        self.fit = fit
        self.label = {v: k for k, v in ModelSelector.DEFAULT_FITS.items()}.get(fit, fit)
        self._maps = dict(maps or {})
        self.model = model
        self.fallback_model = fallback_model
        self.split_min_frac = split_min_frac
        self.other_color = other_color
        self.show_other_key = show_other_key
        self.pixel_size = pixel_size
        self.scale_bar = scale_bar
        self.clip_pct = clip_pct
        self.tau_bound = tau_bound
        self.v_ranges = dict(v_ranges or {})
        self.crop_to_roi = crop_to_roi
        self.swap_xy = swap_xy
        self.bg_color = bg_color
        self.tau_box = tau_box

    @property
    def selection(self) -> dict[str, Any] | None:
        """Model-selection result for :attr:`label`, or ``None`` if unavailable."""
        if self.selector is None:
            return None
        return self.selector.results.get(self.label)

    def _require_selection(self) -> dict[str, Any]:
        ms = self.selection
        if ms is None:
            raise RuntimeError(
                f"Needs model-selection results for '{self.label}' — run "
                "ModelSelector.compare / compare_from_dir first."
            )
        return ms

    def class_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Mono and bi pixel masks from the selector's class map.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(mono_px, bi_px)``: mono adequate (classes 1, 4) and bi justified with
            parameters inside the fit limits (class 2).
        """
        cls = self._require_selection()["class_map"]
        return np.isin(cls, self.MONO_CLASSES), np.isin(cls, self.BI_CLASSES)

    def load_maps(self, model: str) -> dict[str, Any]:
        """
        Result maps of the ``model`` (``"mono"`` or ``"bi"``) fit.

        Parameters
        ----------
        model : str
            ``"mono"`` or ``"bi"``.

        Returns
        -------
        dict
            The ``results["maps"]`` dictionary of the saved fit.
        """
        if model not in self._maps:
            if self.save_dir is None:
                raise FileNotFoundError(
                    f"No {model}-exponential maps given and no save_dir."
                )
            path = os.path.join(self.save_dir, f"{self.fit}_{model}-exponential.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"No saved {model}-exponential fit: {path}")
            self._maps[model] = np.load(path, allow_pickle=True).item()["results"][
                "maps"
            ]
        return self._maps[model]

    def _orient(self, a: Any) -> np.ndarray:
        a = np.asarray(a)
        return a.T if self.swap_xy else a

    def _save(self, fig: Any, name: str, pad_inches: float) -> None:
        if self.save_dir is None:
            return
        for ext in ("pdf", "svg", "tiff"):
            fig.savefig(
                os.path.join(self.save_dir, f"{name}.{ext}"),
                dpi=600,
                bbox_inches="tight",
                pad_inches=pad_inches,
                facecolor="white",
            )
        logging.info(f"Saved → {os.path.join(self.save_dir, name)}.[pdf|svg|tiff]")

    def map_rc(self) -> dict[str, Any]:
        """Matplotlib rcParams used for the map figures."""
        return {
            **_BLACK_RC,
            "font.family": "sans-serif",
            "font.sans-serif": [_sans_font()],
            "font.size": 6,
            "axes.titlesize": 7,
            "axes.labelsize": 6,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "figure.dpi": 200,
        }

    def box_rc(self) -> dict[str, Any]:
        """Matplotlib rcParams used for the box-plot figure."""
        return {
            **_BLACK_RC,
            "font.family": "sans-serif",
            "font.sans-serif": [_sans_font()],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.bottom": True,
            "ytick.left": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }

    def draw_fli_figure(
        self,
        model: str,
        show_mask: np.ndarray | None = None,
        other_label: str | None = None,
    ) -> Any:
        """
        Draw the ``model`` map panels.

        Parameters
        ----------
        model : str
            ``"mono"`` or ``"bi"``.
        show_mask : np.ndarray | None
            Pixels to colour; other ROI pixels get ``other_color``. ``None`` shows the
            whole ROI.
        other_label : str | None
            Description of the masked-out pixels, used in the key.

        Returns
        -------
        matplotlib.figure.Figure
            The drawn figure.
        """
        src = self.load_maps(model)
        layout = "mono_split" if (model == "mono" and show_mask is not None) else model
        panels = self.PANELS[layout]
        nrows, ncols, width_mm = self.LAYOUT[layout]
        tau_box = model == "mono" and self.tau_box
        if tau_box:
            width_mm = min(180, width_mm * (ncols + 1) / ncols)
            ncols += 1
        roi = self._orient(self.roi)
        show = (
            roi
            if show_mask is None
            else roi & self._orient(np.asarray(show_mask).astype(bool))
        )
        other = roi & ~show

        if self.crop_to_roi:
            _r, _c = np.where(roi.any(1))[0], np.where(roi.any(0))[0]
            crop = (
                slice(max(_r[0] - 2, 0), _r[-1] + 3),
                slice(max(_c[0] - 2, 0), _c[-1] + 3),
            )
        else:
            crop = (slice(None), slice(None))
        ny, nx = roi[crop].shape
        other_transparent = str(self.other_color).lower() in ("transparent", "none")
        other_rgba = np.zeros((ny, nx, 4))
        if not other_transparent:
            other_rgba[other[crop]] = mpl.colors.to_rgba(self.other_color)
        show_key = bool(self.show_other_key and other_label)

        with mpl.rc_context(self.map_rc()):
            fig_w = width_mm * MM
            panel_w = fig_w / ncols * 0.72
            key_h = 0.18 if show_key else 0
            fig_h = nrows * (panel_w * ny / nx + 0.32) + key_h
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(fig_w, fig_h),
                squeeze=False,
                gridspec_kw=dict(
                    wspace=0.45,
                    hspace=0.18,
                    left=0.02,
                    right=0.92,
                    top=0.95,
                    bottom=0.02 + key_h / fig_h,
                ),
            )

            for i, (ax, (key, title, clabel, cmap_, scale)) in enumerate(
                zip(axes.flat, panels)
            ):
                unmasked = layout == "mono_split" and i == 0
                img = self._orient(np.asarray(src[key], dtype=float)) * scale
                img = np.where(
                    (roi if unmasked else show) & np.isfinite(img), img, np.nan
                )[crop]
                lim_vals = img[img < self.tau_bound] if key.startswith("tau") else img
                vmin, vmax = self.v_ranges.get(
                    key, np.nanpercentile(lim_vals, self.clip_pct)
                )

                cmap = (
                    mpl.colormaps[cmap_] if isinstance(cmap_, str) else cmap_
                ).copy()
                cmap.set_bad(self.bg_color)
                im = ax.imshow(
                    img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                if other.any() and not other_transparent and not unmasked:
                    ax.imshow(other_rgba, interpolation="none")
                ax.set_axis_off()
                ax.set_title(title, pad=3)
                ax.text(
                    -0.04,
                    1.02,
                    chr(ord("a") + i),
                    transform=ax.transAxes,
                    fontsize=8,
                    fontweight="bold",
                    va="bottom",
                    ha="right",
                )

                cax = ax.inset_axes([1.04, 0.1, 0.05, 0.8])
                cb = fig.colorbar(im, cax=cax, orientation="vertical")
                cb.outline.set_linewidth(0.4)
                cb.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
                cb.ax.tick_params(length=1.5, width=0.4, pad=1)
                cb.set_label(clabel, labelpad=2)

                if self.pixel_size and i == 0:
                    L = self.scale_bar[0] / self.pixel_size
                    ax.plot(
                        [nx * 0.92 - L, nx * 0.92],
                        [ny * 0.96] * 2,
                        color="black",
                        lw=1.5,
                        solid_capstyle="butt",
                    )
                    ax.text(
                        nx * 0.92 - L / 2,
                        ny * 0.945,
                        f"{self.scale_bar[0]} {self.scale_bar[1]}",
                        color="black",
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                    )

            if tau_box:
                self._draw_tau_box(fig, axes, src, show, len(panels))

            for ax in axes.flat[len(panels) + int(tau_box) :]:
                ax.set_axis_off()

            if show_key:
                key_patch = (
                    Patch(
                        facecolor="none",
                        edgecolor="0.5",
                        linewidth=0.4,
                        label=f"Blank: {other_label}",
                    )
                    if other_transparent
                    else Patch(
                        facecolor=self.other_color,
                        edgecolor="none",
                        label=f"{str(self.other_color).capitalize()}: {other_label}",
                    )
                )
                fig.legend(
                    handles=[key_patch],
                    loc="lower left",
                    bbox_to_anchor=(0.02, 0.0),
                    frameon=False,
                    handlelength=1.0,
                    handleheight=1.0,
                    fontsize=6,
                )

            fig_name = f"Fig_FLI_maps_{self.fit}_{model}" + (
                "_split" if show_mask is not None else ""
            )
            self._save(fig, fig_name, pad_inches=0.02)
        return fig

    def _draw_tau_box(
        self,
        fig: Any,
        axes: np.ndarray,
        src: dict[str, Any],
        show: np.ndarray,
        idx: int,
    ) -> None:
        ax = axes.flat[idx]
        v = self._orient(np.asarray(src["tau_map"], dtype=float))[show]
        v = v[np.isfinite(v) & (v > 0) & (v < self.tau_bound)]
        ax.boxplot(
            [v],
            positions=[1],
            widths=0.5,
            whis=1.5,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="black", lw=1.0),
            whiskerprops=dict(color="black", lw=0.7),
            capprops=dict(color="black", lw=0.7),
            boxprops=dict(
                edgecolor="black", lw=0.7, facecolor=mpl.colors.to_rgba("#0072B2", 0.55)
            ),
        )
        ax.set_xlim(0.4, 1.6)
        ax.set_xticks([1], [r"$\tau$"])
        ax.tick_params(axis="x", length=0, pad=3)
        ax.tick_params(axis="y", left=True, length=2, width=0.5)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel(r"$\tau$ (ns)")
        ax.set_title("Lifetime distribution", pad=3)
        fig.canvas.draw()
        p_map, p_box = axes.flat[0].get_position(), ax.get_position()
        ax.set_position(
            [p_box.x0 + 0.2 * p_box.width, p_map.y0, 0.6 * p_box.width, p_map.height]
        )
        ax.text(
            0.5,
            -0.1,
            f"n = {v.size:,} pixels",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=5.5,
        )
        fig.canvas.draw()
        x0 = (
            ax.get_tightbbox(fig.canvas.get_renderer())
            .transformed(ax.transAxes.inverted())
            .x0
        )
        ax.text(
            x0,
            1.02,
            chr(ord("a") + idx),
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    def resolve_model(self) -> str:
        """
        Resolve :attr:`model` to ``"split"``, ``"mono"`` or ``"bi"``.

        Returns
        -------
        str
            The figure model that :meth:`plot_maps` will draw.
        """
        if self.model in ("mono", "bi"):
            return self.model
        ms = self.selection
        if ms is None:
            if self.model == "split":
                self._require_selection()
            logging.info(
                f"selector: model selection not available → whole-ROI "
                f"{self.fallback_model}-exponential figure"
            )
            return self.fallback_model
        if self.model == "selector":
            bi_frac = ms["bi_justified"].sum() / max((ms["class_map"] > 0).sum(), 1)
            resolved = "split" if bi_frac >= self.split_min_frac else "mono"
            logging.info(
                f"selector: {bi_frac:.1%} of ROI pixels bi-justified ({self.label}) → {resolved}"
            )
            return resolved
        return self.model

    def _box_mode(self) -> str:
        if self.model in ("mono", "bi"):
            return self.model
        if self.selection is None:
            if self.model == "split":
                self._require_selection()
            return self.fallback_model
        return "classes"

    def plot_maps(self) -> dict[str, Any]:
        """
        Draw the lifetime map figure(s) selected by :attr:`model`.

        Returns
        -------
        dict
            ``{"mono": fig, "bi": fig}`` in split mode, otherwise ``{model: fig}``.
        """
        fig_model = self.resolve_model()
        if fig_model != "split":
            return {fig_model: self.draw_fli_figure(fig_model)}

        cls = self._require_selection()["class_map"]
        n_roi = max((cls > 0).sum(), 1)
        mono_px, bi_px = self.class_masks()
        logging.info(
            f"mono pixels: {mono_px.sum() / n_roi:.1%} | bi pixels: {bi_px.sum() / n_roi:.1%} | "
            f"excluded (bi at fit limit, blank/other_color in both): {(cls == 3).sum() / n_roi:.1%}"
        )
        return {
            "mono": self.draw_fli_figure(
                "mono",
                mono_px,
                f"pixels not mono-exponential ({1 - mono_px.sum() / n_roi:.1%} of ROI)",
            ),
            "bi": self.draw_fli_figure(
                "bi",
                bi_px,
                f"pixels not bi-exponential ({1 - bi_px.sum() / n_roi:.1%} of ROI)",
            ),
        }

    def _box_groups(self) -> dict[str, dict[str, Any]]:
        mode = self._box_mode()
        if mode == "classes":
            mono_px, bi_px = self.class_masks()
            mono_px = self.roi & mono_px
            bi_px = self.roi & bi_px
            mono_cls, bi_cls = "Mono-exponential pixels", "Bi-exponential pixels"
        else:
            mono_px = bi_px = self.roi
            mono_cls = "All ROI pixels (mono-exponential fit)"
            bi_cls = "All ROI pixels (bi-exponential fit)"

        def vals(
            maps: dict[str, Any], key: str, mask: np.ndarray, scale: float = 1.0
        ) -> np.ndarray:
            v = np.asarray(maps[key], dtype=float)[mask] * scale
            return v[np.isfinite(v)]

        c = self.BOX_COLORS
        groups = []
        if mode in ("classes", "mono"):
            m_mono = self.load_maps("mono")
            groups.append(
                dict(
                    ylabel="Lifetime (ns)",
                    cls=mono_cls,
                    title=mono_cls,
                    show_n=True,
                    data=[
                        (r"$\tau$", "tau", vals(m_mono, "tau_map", mono_px), c["tau"])
                    ],
                )
            )
        if mode in ("classes", "bi"):
            groups.extend(self._bi_box_groups(vals, bi_px, bi_cls))
        return {chr(ord("a") + i): g for i, g in enumerate(groups)}

    def _bi_box_groups(
        self, vals: Any, bi_px: np.ndarray, bi_cls: str
    ) -> list[dict[str, Any]]:
        m_bi = self.load_maps("bi")
        c = self.BOX_COLORS
        return [
            dict(
                ylabel="Lifetime (ns)",
                cls=bi_cls,
                title=bi_cls,
                show_n=False,
                data=[
                    (r"$\tau_1$", "tau1", vals(m_bi, "tau1_map", bi_px), c["tau1"]),
                    (r"$\tau_2$", "tau2", vals(m_bi, "tau2_map", bi_px), c["tau2"]),
                    (
                        r"$\tau_\mathrm{m}$",
                        "tau_mean",
                        vals(m_bi, "tau_mean_map", bi_px),
                        c["taum"],
                    ),
                ],
            ),
            dict(
                ylabel="FRET (%)",
                cls=bi_cls,
                title=None,
                show_n=True,
                data=[
                    (
                        r"$f_\mathrm{D}$",
                        "FRET fraction",
                        vals(m_bi, "alpha1_map", bi_px, 100),
                        c["frac"],
                    ),
                    (
                        r"$E$",
                        "FRET efficiency",
                        vals(m_bi, "fret_efficiency_map", bi_px, 100),
                        c["eff"],
                    ),
                ],
            ),
        ]

    @staticmethod
    def _stats_table(groups: dict[str, dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for panel, g in groups.items():
            for _, pname, v, _ in g["data"]:
                q1, med, q3 = (
                    np.percentile(v, [25, 50, 75]) if v.size else (np.nan,) * 3
                )
                rows.append(
                    {
                        "panel": panel,
                        "class": g["cls"],
                        "parameter": pname,
                        "unit": "ns" if "ns" in g["ylabel"] else "%",
                        "n": v.size,
                        "median": med,
                        "Q1": q1,
                        "Q3": q3,
                        "mean": v.mean() if v.size else np.nan,
                        "SD": v.std(ddof=1) if v.size > 1 else np.nan,
                    }
                )
        return pd.DataFrame(rows)

    def box_stats(self) -> pd.DataFrame:
        """
        Summary statistics of the box-plot parameters per pixel class.

        Returns
        -------
        pd.DataFrame
            One row per parameter: panel, class, parameter, unit, n, median, Q1, Q3,
            mean and SD.
        """
        return self._stats_table(self._box_groups())

    def plot_boxplots(
        self,
        width_mm: float = 88,
        height_mm: float | None = None,
        show_points: bool = False,
        n_points: int = 300,
        show_n: bool = True,
        whis: float = 1.5,
        seed: int = 0,
    ) -> tuple[Any, pd.DataFrame]:
        """
        Box plots of the fitted parameters, following :attr:`model`.

        - ``"selector"`` / ``"split"``: (a) τ of mono pixels, (b) τ1, τ2, τm of bi
          pixels, (c) FRET fraction and efficiency of bi pixels.
        - ``"mono"``: (a) τ of all ROI pixels.
        - ``"bi"``: (a) τ1, τ2, τm and (b) FRET fraction and efficiency of all ROI pixels.

        The statistics are also saved as ``Fig_FLI_boxplots_{fit}[_mono|_bi]_stats.csv``
        when ``save_dir`` is set.

        Parameters
        ----------
        width_mm : float
            Figure width in mm (Nature: 88 single column, 180 double column).
        height_mm : float | None
            Figure height in mm; ``None`` scales 190 mm (three panels) to the number
            of panels.
        show_points : bool
            Overlay a random subsample of pixels on each box.
        n_points : int
            Points per box when ``show_points``.
        show_n : bool
            Print the pixel count under the panels.
        whis : float
            Whisker length in IQR (Tukey); outliers are not drawn.
        seed : int
            Random seed for the point subsample and jitter.

        Returns
        -------
        tuple[matplotlib.figure.Figure, pd.DataFrame]
            The figure and the summary statistics table.
        """
        mode = self._box_mode()
        groups = self._box_groups()
        stats_df = self._stats_table(groups)
        name = f"Fig_FLI_boxplots_{self.fit}" + (
            "" if mode == "classes" else f"_{mode}"
        )
        if self.save_dir is not None:
            stats_df.to_csv(
                os.path.join(self.save_dir, f"{name}_stats.csv"), index=False
            )

        ratios = [1.0]
        for g in list(groups.values())[1:]:
            ratios += [0.24 if g["title"] is None else 0.5, 1.0]
        if height_mm is None:
            height_mm = 190 * sum(ratios) / 3.74

        rng = np.random.default_rng(seed)
        with mpl.rc_context(self.box_rc()):
            n_max = max(len(g["data"]) for g in groups.values())
            fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
            gs = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0)
            axes = [fig.add_subplot(gs[r]) for r in range(0, len(ratios), 2)]
            for ax, g in zip(axes, groups.values()):
                labels = [d[0] for d in g["data"]]
                values = [d[2] for d in g["data"]]
                colors = [d[3] for d in g["data"]]
                pos = np.arange(1, len(values) + 1) + (n_max - len(values)) / 2
                bp = ax.boxplot(
                    values,
                    positions=pos,
                    widths=0.6,
                    whis=whis,
                    showfliers=False,
                    patch_artist=True,
                    medianprops=dict(color="black", lw=1.4),
                    whiskerprops=dict(color="black", lw=0.9),
                    capprops=dict(color="black", lw=0.9),
                    boxprops=dict(edgecolor="black", lw=0.9),
                )
                for patch, col in zip(bp["boxes"], colors):
                    patch.set_facecolor(mpl.colors.to_rgba(col, 0.55))
                if show_points:
                    for p, v, col in zip(pos, values, colors):
                        sub = rng.choice(v, min(n_points, v.size), replace=False)
                        ax.scatter(
                            p + rng.uniform(-0.18, 0.18, sub.size),
                            sub,
                            s=3,
                            color=col,
                            alpha=0.5,
                            edgecolors="none",
                            zorder=3,
                        )
                ax.set_xticks(pos, labels)
                ax.set_xlim(0.4, n_max + 0.6)
                ax.set_ylabel(g["ylabel"])
                if g["title"]:
                    ax.set_title(g["title"], pad=8)
                ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
                ax.tick_params(axis="x", length=0, pad=4)
                if show_n and g["show_n"]:
                    ns = sorted({v.size for v in values})
                    ax.text(
                        0.5,
                        -0.2,
                        "n = " + " / ".join(f"{n:,}" for n in ns) + " pixels",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=9,
                        color="black",
                    )
            fig.align_ylabels(axes)
            fig.canvas.draw()
            bbs = [
                ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
                    fig.transFigure.inverted()
                )
                for ax in axes
            ]
            x0 = min(bb.x0 for bb in bbs)
            for bb, panel in zip(bbs, groups):
                fig.text(
                    x0,
                    bb.y1,
                    panel,
                    fontsize=13,
                    fontweight="bold",
                    va="top",
                    ha="left",
                )

            self._save(fig, name, pad_inches=0.03)
        return fig, stats_df
