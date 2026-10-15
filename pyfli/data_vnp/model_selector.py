"""Per-pixel mono- vs bi-exponential model selection (nested-model F-test + AICc)."""

import os
from typing import Any, ClassVar

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from scipy import stats

from pyfli import logging


class ModelSelector:
    """
    Per-pixel mono- versus bi-exponential model selection using a nested-model
    F-test (BH-FDR corrected) combined with an AICc criterion.

    Pixels are assigned to one of four classes: mono sufficient, bi justified,
    bi significant but with a parameter at a fit limit, and bi fit worse than mono
    (optimiser failure). Class ``0`` marks pixels outside the ROI or invalid.

    Parameters
    ----------
    b_bool_mask : np.ndarray
        ROI mask; only pixels where it is truthy are analysed.
    fdr_q : float
        False-discovery rate for the F-test.
    daicc_min : float
        Minimum ΔAICc additionally required for a pixel to be classified as bi.
    tau_bound : float
        Upper τ fit bound in ns; τ2 near it (or τ1 ≈ 0) flags a parameter at a limit.
    k_mono : int
        Number of fitted parameters in the mono-exponential model (A, τ, v, h).
    k_bi : int
        Number of fitted parameters in the bi-exponential model (A, α, τ1, τ2, v, h).
    """

    DEFAULT_FITS: ClassVar[dict[str, str]] = {
        "MLE": "CPU_MLE_poisson",
        "NLSF": "CPU_NLSF_least_squares",
    }
    CLASS_NAMES: ClassVar[list[str]] = [
        "Mono sufficient",
        "Bi justified",
        "Bi, parameter at limit",
        "Bi fit worse than mono",
    ]
    CLASS_COLORS: ClassVar[list[str]] = ["#56B4E9", "#D55E00", "#E69F00", "#7f7f7f"]
    RC: ClassVar[dict[str, Any]] = {
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.default": "regular",
    }

    def __init__(
        self,
        b_bool_mask: np.ndarray,
        fdr_q: float = 0.05,
        daicc_min: float = 4.0,
        tau_bound: float = 12.5,
        k_mono: int = 4,
        k_bi: int = 6,
    ) -> None:
        self.roi = np.asarray(b_bool_mask).astype(bool)
        self.fdr_q = fdr_q
        self.daicc_min = daicc_min
        self.tau_bound = tau_bound
        self.k_mono = k_mono
        self.k_bi = k_bi
        self.results: dict[str, dict[str, Any]] = {}
        self._rows: list[dict[str, Any]] = []

    @staticmethod
    def bh_fdr(p: np.ndarray) -> np.ndarray:
        """
        Benjamini–Hochberg adjusted p-values (q-values) for a 1-D array.

        Parameters
        ----------
        p : np.ndarray
            Raw p-values.

        Returns
        -------
        np.ndarray
            q-values in the original order, clipped to ``[0, 1]``.
        """
        p = np.asarray(p, float)
        order = np.argsort(p)
        ranked = p[order] * p.size / np.arange(1, p.size + 1)
        q = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty_like(q)
        out[order] = np.clip(q, 0, 1)
        return out

    @staticmethod
    def _load_maps(path: str) -> dict[str, Any]:
        return np.load(path, allow_pickle=True).item()["results"]["maps"]

    def _aicc(self, D: np.ndarray, k: int, n_pts: int) -> np.ndarray:
        return D + 2 * k + 2 * k * (k + 1) / (n_pts - k - 1)

    def compare(
        self, label: str, maps_mono: dict[str, Any], maps_bi: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Run model selection for one pair of mono/bi fit result maps.

        Parameters
        ----------
        label : str
            Name of the fit (e.g. ``"MLE"``); used as the key in :attr:`results`.
        maps_mono : dict
            Mono-exponential result maps; must contain ``chi2_map``.
        maps_bi : dict
            Bi-exponential result maps; must contain ``chi2_map``,
            ``reduced_chi2_map``, ``tau1_map`` and ``tau2_map``.

        Returns
        -------
        dict
            ``class_map``, ``bi_justified``, ``F_map``, ``p_map``, ``q_map``,
            ``dAICc_map``, ``dDev_map`` and ``n_points``.
        """
        roi = self.roi
        D_mono = np.asarray(maps_mono["chi2_map"], float)
        D_bi = np.asarray(maps_bi["chi2_map"], float)
        red_bi = np.asarray(maps_bi["reduced_chi2_map"], float)
        valid = (
            roi & np.isfinite(D_mono) & np.isfinite(D_bi) & (D_bi > 0) & (red_bi > 0)
        )

        n_pts = int(np.round(np.median(D_bi[valid] / red_bi[valid]))) + self.k_bi
        dof_bi = n_pts - self.k_bi
        dk = self.k_bi - self.k_mono

        dD = D_mono - D_bi
        F = np.full(roi.shape, np.nan)
        F[valid] = (dD[valid] / dk) / (D_bi[valid] / dof_bi)
        p = np.full(roi.shape, np.nan)
        p[valid] = stats.f.sf(F[valid], dk, dof_bi)
        q = np.full(roi.shape, np.nan)
        q[valid] = self.bh_fdr(p[valid])

        dAICc = np.where(
            valid,
            self._aicc(D_mono, self.k_mono, n_pts) - self._aicc(D_bi, self.k_bi, n_pts),
            np.nan,
        )

        tau1 = np.asarray(maps_bi["tau1_map"], float)
        tau2 = np.asarray(maps_bi["tau2_map"], float)
        at_limit = (tau2 >= 0.99 * self.tau_bound) | (tau1 <= 0.01)
        with np.errstate(invalid="ignore"):
            bi_sig = valid & (dD > 0) & (q < self.fdr_q) & (dAICc > self.daicc_min)

        cls = np.zeros(roi.shape, int)
        cls[valid] = 1
        cls[bi_sig & ~at_limit] = 2
        cls[bi_sig & at_limit] = 3
        cls[valid & (dD < 0)] = 4

        result = dict(
            class_map=cls,
            bi_justified=(cls == 2),
            F_map=F,
            p_map=p,
            q_map=q,
            dAICc_map=dAICc,
            dDev_map=np.where(valid, dD, np.nan),
            n_points=n_pts,
        )
        self.results[label] = result

        n_valid = valid.sum()
        self._rows = [r for r in self._rows if r["fit"] != label]
        with np.errstate(invalid="ignore"):
            self._rows.append(
                {
                    "fit": label,
                    "pixels": int(n_valid),
                    "gates": n_pts,
                    **{
                        name: f"{np.sum(cls == i + 1) / n_valid:.1%}"
                        for i, name in enumerate(self.CLASS_NAMES)
                    },
                    "mean ΔDev (null = Δk)": round(float(np.nanmean(dD[valid])), 2),
                    "F-test only (q<FDR)": f"{np.sum(valid & (dD > 0) & (q < self.fdr_q)) / n_valid:.1%}",
                    "AICc only (ΔAICc>min)": f"{np.sum(valid & (dAICc > self.daicc_min)) / n_valid:.1%}",
                }
            )
        return result

    def compare_from_dir(
        self, save_dir: str, fits: dict[str, str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """
        Load saved mono/bi fit results from ``save_dir`` and run :meth:`compare` on each.

        Files are expected as ``{prefix}_mono-exponential.npy`` and
        ``{prefix}_bi-exponential.npy``. Fits missing either file are skipped.

        Parameters
        ----------
        save_dir : str
            Directory containing the saved fit results (e.g. ``saver.save_dir``).
        fits : dict[str, str] | None
            Mapping of label to file prefix; defaults to :attr:`DEFAULT_FITS`.

        Returns
        -------
        dict
            :attr:`results`, keyed by fit label.
        """
        for label, prefix in (fits or self.DEFAULT_FITS).items():
            f_mono = os.path.join(save_dir, f"{prefix}_mono-exponential.npy")
            f_bi = os.path.join(save_dir, f"{prefix}_bi-exponential.npy")
            if not (os.path.exists(f_mono) and os.path.exists(f_bi)):
                logging.info(
                    f"[{label}] skipped — needs both {os.path.basename(f_mono)} "
                    f"and {os.path.basename(f_bi)}"
                )
                continue
            self.compare(label, self._load_maps(f_mono), self._load_maps(f_bi))
        return self.results

    def summary(self) -> pd.DataFrame:
        """
        Per-fit summary of class fractions and test statistics.

        Returns
        -------
        pd.DataFrame
            One row per compared fit, indexed by fit label.
        """
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(self._rows).set_index("fit")

    def agreement(self) -> float | None:
        """
        Jaccard index of the 'Bi justified' masks when exactly two fits were compared.

        Returns
        -------
        float | None
            Jaccard index, or ``None`` if the number of compared fits is not two.
        """
        if len(self.results) != 2:
            return None
        a, b = (r["bi_justified"] for r in self.results.values())
        return float(np.sum(a & b) / max(np.sum(a | b), 1))

    def plot(
        self, saver: Any | None = None, name: str = "model_selection_mono_vs_bi"
    ) -> Any:
        """
        Plot one row per fit: ΔAICc map, −log10 q map, class map and class fractions.

        Parameters
        ----------
        saver : Any | None
            Optional object with a ``save_plot(name, fig=..., close=...)`` method
            (e.g. :class:`pyfli.io.data_saving.DataSaver`).
        name : str
            File name passed to ``saver.save_plot``.

        Returns
        -------
        matplotlib.figure.Figure | None
            The figure, or ``None`` if nothing has been compared.
        """
        if not self.results:
            logging.info("No model-selection results to plot; run compare first.")
            return None

        roi = self.roi
        _r, _c = np.where(roi.any(1))[0], np.where(roi.any(0))[0]
        crop = (slice(_r[0], _r[-1] + 1), slice(_c[0], _c[-1] + 1))
        cls_cmap = ListedColormap(["white", *self.CLASS_COLORS])

        with mpl.rc_context(self.RC):
            nR = len(self.results)
            fig, axes = plt.subplots(
                nR,
                4,
                figsize=(11, 3.6 * nR),
                squeeze=False,
                gridspec_kw=dict(width_ratios=[1, 1, 1, 0.8], wspace=0.75, hspace=0.3),
            )
            for r_i, (label, ms) in enumerate(self.results.items()):
                ax0, ax1, ax2, ax3 = axes[r_i]

                d = np.where(roi, ms["dAICc_map"], np.nan)[crop]
                lim = np.nanpercentile(np.abs(d), 98)
                cm = mpl.colormaps["RdBu_r"].copy()
                cm.set_bad("white")
                im0 = ax0.imshow(
                    d, cmap=cm, norm=TwoSlopeNorm(0, -lim, lim), interpolation="none"
                )
                fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03).set_label(
                    "ΔAICc (mono − bi)"
                )
                ax0.set_title(f"{label}: ΔAICc  (>0 favours bi)")

                lq = np.where(roi, -np.log10(np.clip(ms["q_map"], 1e-300, 1)), np.nan)[
                    crop
                ]
                cm = mpl.colormaps["magma"].copy()
                cm.set_bad("white")
                im1 = ax1.imshow(
                    lq,
                    cmap=cm,
                    vmin=0,
                    vmax=max(np.nanpercentile(lq, 99.5), 2 * -np.log10(self.fdr_q)),
                    interpolation="none",
                )
                cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
                cb1.set_label(r"$-\log_{10}$ q  (F-test, BH-FDR)")
                cb1.ax.axhline(-np.log10(self.fdr_q), color="cyan", lw=1)
                ax1.set_title(f"{label}: F-test significance")

                ax2.imshow(
                    ms["class_map"][crop],
                    cmap=cls_cmap,
                    vmin=-0.5,
                    vmax=4.5,
                    interpolation="none",
                )
                ax2.set_title(f"{label}: model class")

                for ax in (ax0, ax1, ax2):
                    ax.set_axis_off()

                n_valid = np.sum(ms["class_map"] > 0)
                fr = [
                    100 * np.sum(ms["class_map"] == i + 1) / n_valid
                    for i in range(len(self.CLASS_NAMES))
                ]
                ypos = np.arange(len(self.CLASS_NAMES))[::-1]
                ax3.barh(ypos, fr, color=self.CLASS_COLORS, height=0.65)
                for y, v in zip(ypos, fr):
                    ax3.text(v + 1, y, f"{v:.1f}%", va="center", fontsize=6.5)
                ax3.set_yticks(ypos, self.CLASS_NAMES)
                ax3.set_xlim(0, 100 * 1.15 if max(fr) > 85 else max(fr) * 1.25)
                ax3.set_xlabel("ROI pixels (%)")
                ax3.spines[["top", "right"]].set_visible(False)
                ax3.set_title(f"q < {self.fdr_q}, ΔAICc > {self.daicc_min:g}")

            if saver is not None:
                saver.save_plot(name, fig=fig, close=False)
        return fig
