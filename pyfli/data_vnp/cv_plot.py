"""
Plot the coefficient of variation of a fitted parameter map against photon count.

This module belongs to :mod:`pyfli.data_vnp` and is part of PyFLI visualization,
normalization, plotting, and mono-versus-bi-exponential comparison tools. Public API
includes class :class:`CVPlot`.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from ..cm_stats.goodness_of_fit import r_squared
from ..plot_style import dark_palette


class CVPlot:
    """
    Bin a fitted parameter map (typically a lifetime map) by photon count and plot its
    per-bin coefficient of variation, ``cv = std(parameter) / mean(parameter)``.

    For a shot-noise-limited lifetime estimate, ``cv`` is expected to scale as
    ``photon_count ** -0.5`` (the Cramer-Rao bound for exponential-decay data) -- this
    class is the standard way to check that scaling empirically, pooled over all pixels
    or broken out per spatial cluster.

    The class operates directly on a flat ``{key: (H, W) array}`` maps dict, i.e. exactly
    ``results["results"]["maps"]`` as returned by any of PyFLI's fitters --
    :class:`~pyfli.solver.cpu_processor.FLICPUProcessor`,
    :class:`~pyfli.solver.gpu_processor.FLIGPUProcessor`, or an
    :class:`~pyfli.solver.mle_fitter.MLEFLIFitter`-backed CPU run -- as well as the
    ground-truth ``maps`` dict produced by the simulator, since all of them share the
    same key naming convention (``tau_map`` / ``tau1_map`` / ``tau2_map`` /
    ``photon_count_map`` / ...). It has no dependency on how those maps were produced.

    Parameters
    ----------
    save_path : str | None
        Output path used when saving generated figures.
    fig_name : str | None
        Figure name or output stem used when saving; defaults to ``"cv_plot"``.
    """

    def __init__(
        self, save_path: str | None = None, fig_name: str | None = None
    ) -> None:
        self.save_path = save_path
        self.fig_name = fig_name
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)

    @staticmethod
    def _bin_stats(
        photon_vals: np.ndarray, param_vals: np.ndarray, edges: np.ndarray
    ) -> list[dict[str, Any]]:
        """
        Bin `photon_vals` by `edges` and compute per-bin mean/std/cv/count of
        `param_vals`. Returns row dicts (bin_rank, bin_center, bin_low, bin_high, mean,
        std, cv, count); ``cv`` is NaN in bins where mean is 0 (avoids a spurious
        +/-inf).
        """
        n_bins_i = len(edges) - 1
        bin_idx = pd.cut(photon_vals, bins=edges, labels=False, include_lowest=True)
        bin_centers = pd.Series(photon_vals).groupby(bin_idx).mean()

        valid = np.isfinite(param_vals)
        tmp = pd.DataFrame({"bin_rank": bin_idx[valid], "value": param_vals[valid]})
        grouped = tmp.groupby("bin_rank")["value"].agg(["mean", "std", "count"])

        rows = []
        for b in range(n_bins_i):
            if b in grouped.index:
                r = grouped.loc[b]
                n_ = r["count"]
                mean_ = r["mean"]
                rows.append(
                    {
                        "bin_rank": b,
                        "bin_center": bin_centers.loc[b],
                        "bin_low": edges[b],
                        "bin_high": edges[b + 1],
                        "mean": mean_,
                        "std": r["std"],
                        "cv": r["std"] / mean_ if mean_ else np.nan,
                        "count": int(n_),
                    }
                )
        return rows

    @staticmethod
    def _ideal_trend_fit(
        x: np.ndarray,
        y: np.ndarray,
        space: str = "linear",
        weights: np.ndarray | None = None,
    ) -> tuple[float | None, np.ndarray | None]:
        """
        Least-squares amplitude `C` for the fixed-slope shot-noise model
        ``cv = C / sqrt(N)``.

        With ``space="linear"`` (default), `C` minimizes
        ``sum(w * (y - C * x ** -0.5) ** 2)`` -- a linear fit of `y` against
        ``x ** -0.5`` through the origin, solved in closed form as
        ``C = sum(w * b * y) / sum(w * b ** 2)`` with ``b = x ** -0.5``. Low-photon
        bins have the largest `cv`, so they dominate this fit.

        With ``space="log"``, `C` minimizes
        ``sum(w * (log(y) - log(C) + 0.5 * log(x)) ** 2)``, so ``log(C)`` is the
        weighted mean of ``log(y) + 0.5 * log(x)``. Each bin then counts by its
        relative error, matching how the trend reads on a log-log plot. Points with
        ``y <= 0`` are dropped.

        `weights` (e.g. the per-bin pixel ``count``) scales each point's squared
        residual, so sparsely populated, noisy bins pull `C` less; ``None`` weights
        all points equally. Points with non-finite or non-positive weight are dropped.

        Returns ``(C, y_pred)``, or ``(None, None)`` if there are no valid points to
        fit.
        """
        if space not in ("linear", "log"):
            raise ValueError(f"space must be 'linear' or 'log', got {space!r}")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (x > 0) & (w > 0)
        if space == "log":
            valid &= y > 0
        if not np.any(valid):
            return None, None
        xv, yv, wv = x[valid], y[valid], w[valid]
        if space == "log":
            log_c = np.sum(wv * (np.log(yv) + 0.5 * np.log(xv))) / np.sum(wv)
            c = float(np.exp(log_c))
        else:
            basis = xv**-0.5
            denom = np.sum(wv * basis**2)
            if denom == 0:
                return None, None
            c = float(np.sum(wv * basis * yv) / denom)
        return c, c * x**-0.5

    @staticmethod
    def _power_law_fit(
        x: np.ndarray, y: np.ndarray
    ) -> tuple[float | None, float | None, float | None, np.ndarray | None]:
        """
        Ordinary-least-squares power-law fit ``y = a * x ** b``, via linear regression
        of ``log(y)`` against ``log(x)`` (the standard closed-form way to maximize R^2
        for this model). Returns ``(a, b, r2, y_pred)``, or all-``None`` if fewer than 2
        valid (finite, positive) points are available.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if valid.sum() < 2:
            return None, None, None, None
        log_x = np.log(x[valid])
        log_y = np.log(y[valid])
        b, log_a = np.polyfit(log_x, log_y, 1)
        pred = log_a + b * log_x
        r2 = float(r_squared(log_y - pred, log_y, empty_value=np.nan))
        a = float(np.exp(log_a))
        return a, float(b), r2, a * x**b

    def _draw_reference_curves(
        self,
        ax: Any,
        x: np.ndarray,
        y: np.ndarray,
        color: Any,
        show_ideal_trend: bool,
        show_powerlaw_fit: bool,
        ideal_fit_space: str = "linear",
        ideal_fit_weights: np.ndarray | None = None,
    ) -> None:
        """Overlay the ideal 1/sqrt(N) trend and/or the fitted a*N**b power law for one
        data series (`x`, `y`), in the same color as that series' data line.
        `ideal_fit_space` and `ideal_fit_weights` are passed to `_ideal_trend_fit`."""
        if not (show_ideal_trend or show_powerlaw_fit):
            return
        x = np.asarray(x, dtype=float)
        order = np.argsort(x)
        x_sorted = x[order]

        if show_ideal_trend:
            c, _ = self._ideal_trend_fit(
                x, y, space=ideal_fit_space, weights=ideal_fit_weights
            )
            if c is not None:
                ax.plot(
                    x_sorted,
                    c * x_sorted**-0.5,
                    linestyle=":",
                    linewidth=1.8,
                    color=color,
                    alpha=0.85,
                    label=rf"ideal $1/\sqrt{{N}}$ (C={c:.3g})",
                )

        if show_powerlaw_fit:
            a, b, r2, _ = self._power_law_fit(x, y)
            if a is not None:
                ax.plot(
                    x_sorted,
                    a * x_sorted**b,
                    linestyle="--",
                    linewidth=1.8,
                    color=color,
                    alpha=0.85,
                    label=rf"fit: {a:.3g}$\cdot N^{{{b:.2f}}}$ ($R^2$={r2:.3f})",
                )

    @staticmethod
    def _axes_with_legend_panels(
        nrows: int, ncols: int, figsize: tuple[float, float]
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        """
        Create an ``(nrows, ncols)`` grid of plot axes, each paired with a legend panel
        to its right. Within each pair the plot axes take 80% of the width and the
        legend panel 20%, so long legend labels never shrink the plot area.
        Returns ``(fig, axes, legend_axes)``; the legend panels have their axis off.
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows, 2 * ncols, width_ratios=[4, 1] * ncols)
        axes = np.empty((nrows, ncols), dtype=object)
        legend_axes = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                axes[r, c] = fig.add_subplot(gs[r, 2 * c])
                legend_axes[r, c] = fig.add_subplot(gs[r, 2 * c + 1])
                legend_axes[r, c].axis("off")
        return fig, axes, legend_axes

    @staticmethod
    def _text_width(text: str, fontsize: float) -> float:
        """Rendered width of `text` (mathtext allowed) in points at `fontsize`."""
        try:
            path = TextPath((0, 0), text, prop=FontProperties(size=fontsize))
            return float(path.get_extents().width)
        except ValueError:
            return 0.6 * fontsize * len(text)

    @staticmethod
    def _split_label_words(label: str) -> list[str]:
        """Split `label` on spaces outside ``$...$`` math, so mathtext stays intact."""
        words, current, in_math = [], "", False
        for i, ch in enumerate(label):
            if ch == "$" and (i == 0 or label[i - 1] != "\\"):
                in_math = not in_math
            if ch == " " and not in_math:
                if current:
                    words.append(current)
                current = ""
            else:
                current += ch
        if current:
            words.append(current)
        return words

    @classmethod
    def _wrap_label(cls, label: str, max_width: float, fontsize: float) -> str:
        """
        Wrap `label` at word boundaries so each line renders at most `max_width`
        points wide at `fontsize`. A single word wider than `max_width` gets a line of
        its own; words are never split, and ``$...$`` math is never broken.
        """
        lines = []
        for part in label.split("\n"):
            line = ""
            for word in cls._split_label_words(part):
                candidate = f"{line} {word}" if line else word
                if line and cls._text_width(candidate, fontsize) > max_width:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        return "\n".join(lines)

    @classmethod
    def _draw_legend(
        cls, ax: Any, legend_ax: Any, fontsize: float, title: str | None = None
    ) -> None:
        """
        Draw `ax`'s legend inside its dedicated `legend_ax` panel. The legend is kept
        as narrow as its first entry: every other label is wrapped onto extra lines
        to fit the rendered width of the first label, instead of widening the box.
        """
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
        max_width = cls._text_width(labels[0], fontsize)
        labels = [labels[0]] + [
            cls._wrap_label(label, max_width, fontsize) for label in labels[1:]
        ]
        legend_ax.legend(
            handles,
            labels,
            loc="upper left",
            borderaxespad=0.0,
            frameon=False,
            fontsize=fontsize,
            title=title,
            title_fontsize=fontsize,
        )

    def compute(
        self,
        maps: dict[str, np.ndarray],
        tau_keys: str | list[str],
        photon_map: np.ndarray | None = None,
        photon_key: str = "photon_count_map",
        mask: np.ndarray | None = None,
        cluster_mask: np.ndarray | None = None,
        cluster_names: dict[int, str] | None = None,
        n_bins: int = 10,
        bin_mode: str = "quantile",
        bin_scope: str = "pooled",
    ) -> pd.DataFrame:
        """
        Bin pixels by photon count and compute per-bin mean/std/cv for each of
        `tau_keys`, pooled over `mask` or broken out per `cluster_mask` label.

        Parameters
        ----------
        maps : dict[str, np.ndarray]
            Flat ``{key: (H, W) array}`` dict of fitted (or ground-truth) parameter
            maps, e.g. ``results["results"]["maps"]`` from any PyFLI fitter.
        tau_keys : str | list[str]
            Key(s) in `maps` to compute the coefficient of variation for (not limited to
            literal lifetimes -- any per-pixel parameter map works, e.g. ``"tau_map"``,
            ``["tau1_map", "tau2_map", "tau_mean_map"]``, ``"alpha1_map"``).
        photon_map : np.ndarray | None
            Photon-count map to bin by, e.g. ``decay.sum(axis=-1)`` computed directly
            from the raw decay cube. Takes precedence over `photon_key` when given --
            preferred when available, since a fitted ``photon_count_map`` amplitude can
            differ in scale/definition from the true detected photon count.
        photon_key : str
            Key in `maps` to use as the photon-count map when `photon_map` is not given.
        mask : np.ndarray | None
            Boolean ``(H, W)`` mask selecting pixels to include. ``None`` keeps every
            finite pixel.
        cluster_mask : np.ndarray | None
            Integer ``(H, W)`` label map for per-region binning. ``0`` = background
            (excluded); ``1, 2, 3, ...`` = cluster labels. ``None`` pools every selected
            pixel together instead.
        cluster_names : dict[int, str] | None
            Optional ``{label: name}`` mapping for display; must cover every non-zero
            label present in `cluster_mask`. Defaults to ``f"cluster_{label}"``.
        n_bins : int
            Number of photon-count bins.
        bin_mode : str
            ``"quantile"`` (equal-frequency bins) or ``"linear"`` (equal-width bins).
        bin_scope : str
            ``"pooled"`` (default) computes ONE set of bin edges shared by every
            cluster, so clusters sit at directly comparable photon-count positions.
            ``"per_group"`` gives each cluster (or the single pooled group, if no
            `cluster_mask`) its own edges from its own photon-count distribution.

        Returns
        -------
        pd.DataFrame
            Long-form frame with columns ``cluster`` (``None`` when `cluster_mask` is
            not given), ``parameter``, ``bin_rank``, ``bin_center``, ``bin_low``,
            ``bin_high``, ``mean``, ``std``, ``cv``, ``count``.
        """
        if isinstance(tau_keys, str):
            tau_keys = [tau_keys]
        missing = [k for k in tau_keys if k not in maps]
        if missing:
            raise KeyError(
                f"tau_keys {missing} not found in maps; available keys: "
                f"{sorted(maps.keys())}"
            )

        if photon_map is not None:
            photon_arr = np.asarray(photon_map, dtype=float)
            photon_label = "photon_map"
        else:
            if photon_key not in maps:
                raise KeyError(
                    f"photon_key '{photon_key}' not found in maps; available keys: "
                    f"{sorted(maps.keys())}"
                )
            photon_arr = np.asarray(maps[photon_key], dtype=float)
            photon_label = photon_key
        pixel_shape = photon_arr.shape

        for k in tau_keys:
            tk_shape = np.asarray(maps[k]).shape
            if tk_shape != pixel_shape:
                raise ValueError(
                    f"maps['{k}'] shape {tk_shape} does not match photon map shape "
                    f"{pixel_shape}"
                )

        if mask is not None:
            base_mask = np.asarray(mask).astype(bool)
            if base_mask.shape != pixel_shape:
                raise ValueError(
                    f"mask shape {base_mask.shape} does not match photon map shape "
                    f"{pixel_shape}"
                )
        else:
            base_mask = np.ones(pixel_shape, dtype=bool)

        if cluster_mask is not None:
            cm = np.asarray(cluster_mask)
            if cm.shape != pixel_shape:
                raise ValueError(
                    f"cluster_mask shape {cm.shape} does not match photon map shape "
                    f"{pixel_shape}"
                )
            cluster_ids = sorted(int(c) for c in np.unique(cm) if c != 0)
            if not cluster_ids:
                raise ValueError(
                    "cluster_mask contains no non-zero cluster labels (0 is treated "
                    "as background)."
                )
            if cluster_names is None:
                names = {cid: f"cluster_{cid}" for cid in cluster_ids}
            else:
                missing_names = [c for c in cluster_ids if c not in cluster_names]
                if missing_names:
                    raise ValueError(
                        f"cluster_names is missing label(s) {missing_names} present "
                        f"in cluster_mask"
                    )
                names = {cid: str(cluster_names[cid]) for cid in cluster_ids}
            groups = [(names[cid], (cm == cid) & base_mask) for cid in cluster_ids]
        else:
            groups = [(None, base_mask)]

        if bin_scope not in ("pooled", "per_group"):
            raise ValueError("bin_scope must be 'pooled' or 'per_group'")

        def _edges(vals: np.ndarray) -> np.ndarray:
            if bin_mode == "quantile":
                e = np.unique(np.quantile(vals, np.linspace(0, 1, n_bins + 1)))
            elif bin_mode == "linear":
                e = np.linspace(vals.min(), vals.max(), n_bins + 1)
            else:
                raise ValueError("bin_mode must be 'quantile' or 'linear'")
            if len(e) < 2:
                raise ValueError(
                    "Not enough distinct photon values to form bins; try fewer n_bins."
                )
            return e

        photon_flat = photon_arr.ravel()
        finite_photon = np.isfinite(photon_flat)

        pooled_edges = None
        if bin_scope == "pooled":
            keep_all = finite_photon & base_mask.ravel()
            if not np.any(keep_all):
                raise ValueError("No valid (finite, unmasked) pixels found.")
            pooled_edges = _edges(photon_flat[keep_all])

        rows = []
        for label, sel in groups:
            keep = finite_photon & sel.ravel()
            p_keep = photon_flat[keep]
            if p_keep.size == 0:
                continue
            edges = pooled_edges if bin_scope == "pooled" else _edges(p_keep)
            for key in tau_keys:
                param_flat = np.asarray(maps[key], dtype=float).ravel()
                p_vals = param_flat[keep]
                for row in self._bin_stats(p_keep, p_vals, edges):
                    rows.append({"cluster": label, "parameter": key, **row})

        df = pd.DataFrame(rows)
        df.attrs["photon_label"] = photon_label
        df.attrs["has_cluster"] = cluster_mask is not None
        df.attrs["n_bins"] = n_bins
        df.attrs["bin_mode"] = bin_mode
        return df

    def plot(
        self,
        df: pd.DataFrame,
        target_keys: list[str] | None = None,
        ncols: int = 3,
        figsize: tuple[float, float] | None = None,
        logx: bool = False,
        palette: dict[str, Any] | None = None,
        show_ideal_trend: bool = False,
        show_powerlaw_fit: bool = False,
        ideal_fit_space: str = "linear",
        ideal_fit_weighted: bool = False,
    ) -> tuple[Any, Any]:
        """
        Plot `cv` (from `compute`) against the photon-count bin.

        Without a cluster grouping (``compute(cluster_mask=None)``), draws ONE subplot
        with one line per `target_keys` entry. With a cluster grouping, draws a grid
        with one subplot PER `target_keys` entry, each with one line per cluster.

        Parameters
        ----------
        df : pd.DataFrame
            Output of `compute`.
        target_keys : list[str] | None
            Which `parameter` values to plot; defaults to every parameter in `df`.
        ncols : int
            Number of subplot columns (cluster grid only).
        figsize : tuple[float, float] | None
            Figure size passed to Matplotlib.
        logx : bool
            Use a log-scaled photon-count axis.
        palette : dict[str, Any] | None
            ``{cluster_name: color}`` mapping (cluster grid only); defaults to a
            qualitative Seaborn palette.
        show_ideal_trend : bool
            Overlay the theoretical shot-noise-limited trend ``cv = C / sqrt(N)`` as a
            dotted line for each data series (same color as the series), so deviation
            from ideal Poisson-limited precision is easy to spot. ``C`` is fit by
            least squares to that series' own data; the -0.5 exponent is fixed (it's
            the Cramer-Rao-bound slope, not a free fit parameter). Off by default.
        show_powerlaw_fit : bool
            Overlay an ``a * N ** b`` power-law regression fit to each data series (log-
            log ordinary least squares, the closed-form fit that maximizes R^2 for this
            model) as a dashed line, labeled with the fitted equation and R^2. Off by
            default.
        ideal_fit_space : str
            Residual space for fitting ``C`` of the ideal trend: ``"linear"`` (default;
            low-photon bins dominate) or ``"log"`` (every bin counts by its relative
            error). Only used when `show_ideal_trend` is True.
        ideal_fit_weighted : bool
            Weight each bin by its pixel ``count`` when fitting ``C`` of the ideal
            trend, so sparsely populated bins pull it less. Only used when
            `show_ideal_trend` is True. Off by default.

        Returns
        -------
        tuple[Any, Any]
            ``(fig, axes)`` -- `axes` is a length-1 array in the pooled case, or the
            full subplot grid in the cluster case.
        """
        if df.empty:
            raise ValueError("df is empty; nothing to plot.")
        if target_keys is None:
            target_keys = sorted(df["parameter"].unique())
        photon_label = df.attrs.get("photon_label", "photon count")
        has_cluster = bool(df.attrs.get("has_cluster", df["cluster"].notna().any()))

        if not has_cluster:
            fig, axes_grid, legend_grid = self._axes_with_legend_panels(
                1, 1, figsize or (7.5, 4.5)
            )
            ax = axes_grid[0, 0]
            colors = dark_palette(len(target_keys))
            for key, color in zip(target_keys, colors):
                sub = df[df["parameter"] == key].sort_values("bin_center")
                if sub.empty:
                    continue
                ax.plot(
                    sub["bin_center"],
                    sub["cv"],
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    color=color,
                    label=key,
                )
                self._draw_reference_curves(
                    ax,
                    sub["bin_center"].to_numpy(),
                    sub["cv"].to_numpy(),
                    color,
                    show_ideal_trend,
                    show_powerlaw_fit,
                    ideal_fit_space=ideal_fit_space,
                    ideal_fit_weights=(
                        sub["count"].to_numpy() if ideal_fit_weighted else None
                    ),
                )
            ax.set_xlabel(photon_label)
            ax.set_ylabel(r"coefficient of variation  $\sigma / \mathrm{mean}$")
            ax.set_title("Precision vs. photon count", fontweight="bold")
            if logx:
                ax.set_xscale("log")
            self._draw_legend(ax, legend_grid[0, 0], fontsize=9)
            sns.despine(ax=ax)
            fig.tight_layout()
            axes_out = np.array([ax])
        else:
            clusters = [c for c in df["cluster"].unique() if c is not None]
            if palette is None:
                colors = dark_palette(len(clusters), base="husl")
                palette = dict(zip(clusters, colors))
            n = len(target_keys)
            ncols_eff = min(ncols, n)
            nrows = int(np.ceil(n / ncols_eff))
            fig, axes, legend_axes = self._axes_with_legend_panels(
                nrows, ncols_eff, figsize or (6.25 * ncols_eff, 4 * nrows)
            )
            axes_flat = axes.ravel()
            legend_flat = legend_axes.ravel()
            for ax, legend_ax, key in zip(axes_flat, legend_flat, target_keys):
                sub_p = df[df["parameter"] == key]
                for c in clusters:
                    sub = sub_p[sub_p["cluster"] == c].sort_values("bin_center")
                    if sub.empty:
                        continue
                    ax.plot(
                        sub["bin_center"],
                        sub["cv"],
                        marker="o",
                        linewidth=2,
                        markersize=5,
                        color=palette[c],
                        label=str(c),
                    )
                    self._draw_reference_curves(
                        ax,
                        sub["bin_center"].to_numpy(),
                        sub["cv"].to_numpy(),
                        palette[c],
                        show_ideal_trend,
                        show_powerlaw_fit,
                        ideal_fit_space=ideal_fit_space,
                        ideal_fit_weights=(
                            sub["count"].to_numpy() if ideal_fit_weighted else None
                        ),
                    )
                ax.set_title(key, fontweight="bold")
                ax.set_xlabel(photon_label)
                ax.set_ylabel(r"$\sigma / \mathrm{mean}$")
                if logx:
                    ax.set_xscale("log")
                self._draw_legend(ax, legend_ax, fontsize=8, title="cluster")
                sns.despine(ax=ax)
            for ax in axes_flat[n:]:
                ax.axis("off")
            fig.tight_layout()
            axes_out = axes

        if self.save_path:
            plt.savefig(
                os.path.join(self.save_path, (self.fig_name or "cv_plot") + ".png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()
        return fig, axes_out

    def compute_and_plot(
        self,
        maps: dict[str, np.ndarray],
        tau_keys: str | list[str],
        photon_map: np.ndarray | None = None,
        photon_key: str = "photon_count_map",
        mask: np.ndarray | None = None,
        cluster_mask: np.ndarray | None = None,
        cluster_names: dict[int, str] | None = None,
        n_bins: int = 10,
        bin_mode: str = "quantile",
        bin_scope: str = "pooled",
        target_keys: list[str] | None = None,
        ncols: int = 3,
        figsize: tuple[float, float] | None = None,
        logx: bool = False,
        palette: dict[str, Any] | None = None,
        show_ideal_trend: bool = False,
        show_powerlaw_fit: bool = False,
        ideal_fit_space: str = "linear",
        ideal_fit_weighted: bool = False,
    ) -> tuple[pd.DataFrame, Any, Any]:
        """Convenience wrapper: `compute` then `plot` in one call. `show_ideal_trend`,
        `show_powerlaw_fit`, `ideal_fit_space` and `ideal_fit_weighted` are passed
        straight through to `plot` -- see there."""
        df = self.compute(
            maps,
            tau_keys,
            photon_map=photon_map,
            photon_key=photon_key,
            mask=mask,
            cluster_mask=cluster_mask,
            cluster_names=cluster_names,
            n_bins=n_bins,
            bin_mode=bin_mode,
            bin_scope=bin_scope,
        )
        fig, axes = self.plot(
            df,
            target_keys=target_keys,
            ncols=ncols,
            figsize=figsize,
            logx=logx,
            palette=palette,
            show_ideal_trend=show_ideal_trend,
            show_powerlaw_fit=show_powerlaw_fit,
            ideal_fit_space=ideal_fit_space,
            ideal_fit_weighted=ideal_fit_weighted,
        )
        return df, fig, axes
