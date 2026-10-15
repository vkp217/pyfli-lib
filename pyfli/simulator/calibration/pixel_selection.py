"""Separate sample pixels from pixels that only record detector noise / background."""

from dataclasses import asdict

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from pyfli import logging

from .config import PixelSelectionCriteria
from .summary_stats import background_bins, inactive_bins


def _otsu(values: np.ndarray, nbins: int = 256) -> tuple[float, float]:
    """Otsu threshold (``skimage.filters.threshold_otsu``) and separability eta = between-class / total variance."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.ptp(values) == 0:
        return float(values[0]) if values.size else 0.0, 0.0
    thr = float(threshold_otsu(values, nbins=nbins))
    lo, hi = values[values <= thr], values[values > thr]
    var_t = values.var()
    if var_t <= 0 or not len(lo) or not len(hi):
        return thr, 0.0
    w0 = len(lo) / len(values)
    return thr, float(w0 * (1 - w0) * (lo.mean() - hi.mean()) ** 2 / var_t)


def select_signal_pixels(
    decay: np.ndarray,
    user_mask: np.ndarray | None = None,
    criteria: PixelSelectionCriteria | None = None,
    adc_ceiling: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Returns ``(mask, report)``. ``report`` records how many pixels each test removed,
    the background bins and the per-bin background level, for the calibration log.
    """
    c = criteria or PixelSelectionCriteria()
    H, W, T = decay.shape
    total = decay.sum(-1)
    finite = np.isfinite(total) & (total > 0)

    # pooled decay from the brightest pixels defines the background bins and the peak position
    bright = total >= np.percentile(total[finite], 90) if finite.any() else finite
    pooled = decay[bright].sum(0)
    bg_bins = background_bins(pooled)
    active = np.ones(T, dtype=bool)
    active[inactive_bins(decay.sum((0, 1)))] = False
    n_act = int(active.sum())
    b_px = decay[..., bg_bins].mean(-1)
    signal = total - n_act * b_px
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(total > 0, signal / np.sqrt(total), 0)

    mean_level = np.where(total > 0, total / n_act, 1)
    resid = (decay[..., active] - mean_level[..., None]) ** 2
    chi2_flat = (resid / np.maximum(mean_level, 1e-3)[..., None]).sum(-1) / (n_act - 1)
    flat_thr = 1 + c.flatness_sigma * np.sqrt(2.0 / (n_act - 1))

    smooth = ndimage.uniform_filter1d(decay, 3, axis=-1, mode="wrap")
    peak_px = smooth.argmax(-1)
    peak_ref = int(np.argmax(ndimage.uniform_filter1d(pooled, 3, mode="wrap")))
    win = c.peak_window_bins or max(T // 8, 4)
    dpk = np.abs(peak_px - peak_ref)
    dpk = np.minimum(dpk, T - dpk)  # periodic distance

    med3 = ndimage.median_filter(total, size=3)
    hot = total > c.hot_pixel_factor * np.maximum(med3, 1)
    if adc_ceiling is not None:
        hot |= (decay >= adc_ceiling).any(-1)

    tests = {
        "finite_nonzero": finite,
        "signal_photons": signal >= c.min_signal_photons,
        "snr": snr >= c.snr_min,
        "not_flat": chi2_flat >= flat_thr,
        "peak_position": dpk <= win,
        "not_hot": ~hot,
    }
    auto = np.logical_and.reduce(list(tests.values()))

    # Otsu on *all* pixels (sample + background): a bimodal histogram means a clear background population
    log_sig = np.log10(np.clip(signal, 1, None))
    otsu_thr, eta = _otsu(log_sig[finite]) if finite.sum() > 100 else (None, 0.0)
    use_otsu = c.use_otsu == "on" or (
        c.use_otsu == "auto" and eta >= c.otsu_min_separability
    )
    if use_otsu and otsu_thr is not None:
        tests["otsu"] = log_sig >= otsu_thr
        auto &= tests["otsu"]

    if c.min_component_size > 1 and auto.any():
        lab, n = ndimage.label(auto)
        sizes = ndimage.sum(auto, lab, index=np.arange(1, n + 1))
        auto = np.isin(lab, 1 + np.flatnonzero(sizes >= c.min_component_size))

    if user_mask is not None:
        mask = user_mask & finite
        if c.refine_user_mask:
            mask &= auto
        mode = "user+auto" if c.refine_user_mask else "user"
    else:
        mask = auto
        mode = "auto"

    report = {
        "mode": mode,
        "n_selected": int(mask.sum()),
        "n_total": int(H * W),
        "rejected_by_test": {k: int((~v).sum()) for k, v in tests.items()},
        "otsu": {
            "threshold_log10_signal": otsu_thr,
            "separability": eta,
            "applied": bool(use_otsu),
        },
        "background_bins": [int(bg_bins[0]), int(bg_bins[-1])] if len(bg_bins) else [],
        "background_per_bin_dark_pixels": float(np.median(b_px[~mask & finite]))
        if (~mask & finite).any()
        else 0.0,
        "background_per_bin_selected": float(np.median(b_px[mask]))
        if mask.any()
        else 0.0,
        "peak_bin": peak_ref,
        "criteria": asdict(c),
    }
    if user_mask is not None:
        inter = (user_mask & auto).sum()
        report["agreement_user_vs_auto_iou"] = float(
            inter / max((user_mask | auto).sum(), 1)
        )
    logging.info(
        f"Pixel selection ({mode}): {report['n_selected']} / {H * W} pixels kept; "
        f"rejected per test: {report['rejected_by_test']}"
    )
    if mask.sum() < 200:
        logging.warning(
            "Fewer than 200 sample pixels selected: relax the criteria or check the data."
        )
    return mask, report
