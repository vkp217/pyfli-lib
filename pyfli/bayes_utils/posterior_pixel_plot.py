"""
Posterior-predictive plot of one pixel's sampled decay reconstructions against its
measured decay.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from pyfli.bayes_utils.param_combinations import ParamSelector
from pyfli.reconstruction import ParamToDecay

#: Registry of {model_type: output_combination keys}, matching
#: ParamSelector's own params dicts (no "_map" suffix).
_MODEL_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "bi-exponential": ("alpha1", "tau1", "tau2"),
    "mono-exponential": ("tau",),
}

#: Valid values for the `center` argument.
CENTERS: tuple[str, ...] = ("best", "median", "mean")

#: Valid values for the `decay_style` argument.
DECAY_STYLES: tuple[str, ...] = ("line", "scatter")

# the whole model (central curve + credible bands).
_DECAY_COLOR = "#0D0D0D"
_FIT_COLOR = "#d62a7a"


def _reconstruct_sample_stack(
    output_combination: dict[str, np.ndarray],
    pixel: tuple[int, int],
    irf: np.ndarray,
    decay_px: np.ndarray,
    freq_acq: float,
    model_type: str,
) -> np.ndarray:
    """
    Reconstruct every posterior sample's decay curve at one pixel, scaled to
    that pixel's measured photon count the same way
    :func:`pyfli.reconstruction.compute_detailed_results` scales its fits
    (unit-amplitude reconstruction, then rescaled so its sum matches the
    measured decay's sum).

    Treats the NUM_SAMPLES axis as the reconstructor's "W" (pixel) axis, so
    every sample is batched through one vectorized reconstruction instead of
    looping in Python.

    Returns
    -------
    np.ndarray
        ``(NUM_SAMPLES, T)`` reconstructed decay curves.
    """
    x, y = pixel
    param_keys = _MODEL_PARAM_KEYS[model_type]
    num_samples = output_combination[param_keys[0]].shape[-1]

    recon_params = {
        f"{key}_map": output_combination[key][x, y, :][None, :] for key in param_keys
    }
    recon_params["photon_count_map"] = np.ones((1, num_samples), dtype=np.float32)

    irf_px = irf[x, y, :] if np.ndim(irf) == 3 else irf
    recon = ParamToDecay(model_type, freq_acq, irf=irf_px)

    unit = recon.reconstruct_unit_amplitude(recon_params)
    convolved = unit["convolved_map"]  # (1, NUM_SAMPLES, T)

    decay_rep = np.broadcast_to(decay_px, convolved.shape)
    scaled = recon.rescale_fit_to_measured_totals(convolved, decay_rep)
    return scaled[0]  # (NUM_SAMPLES, T)


def _select_best_sample_idx(
    output_combination: dict[str, np.ndarray],
    pixel: tuple[int, int],
    irf: np.ndarray,
    decay: np.ndarray,
    freq_acq: float,
    model_type: str,
    metric: str,
) -> int:
    """
    Pick the posterior sample that best fits this one pixel, by delegating to
    :class:`ParamSelector` on a 1x1-pixel crop -- reuses its tested
    per-sample goodness-of-fit logic instead of duplicating it here.
    """
    x, y = pixel
    sub_combo = {k: v[x : x + 1, y : y + 1, :] for k, v in output_combination.items()}
    sub_decay = decay[x : x + 1, y : y + 1, :]
    sub_irf = irf[x : x + 1, y : y + 1, :] if np.ndim(irf) == 3 else irf

    selector = ParamSelector(freq_acq, sub_irf, sub_decay, model_type=model_type)
    stacks = selector.evaluate_all_samples(sub_combo, progress=False)
    selection = selector.select_best_combination(sub_combo, stacks, metric=metric)
    return int(selection["best_sample_idx"][0, 0])


def plot_pixel_posterior_fit(
    output_combination: dict[str, np.ndarray],
    decay: np.ndarray,
    irf: np.ndarray,
    freq_acq: float,
    pixel: tuple[int, int],
    model_type: str = "bi-exponential",
    center: str = "median",
    metric: str = "reduced_chi2",
    ci_levels: tuple[int, ...] = (92, 68),
    decay_color: str = _DECAY_COLOR,
    fit_color: str = _FIT_COLOR,
    band_alpha: "float | tuple[float, ...]" = 0.88,
    fit_alpha: float = 1.0,
    decay_alpha: float = 0.8,
    title: str | None = None,
    ax: "plt.Axes | None" = None,
    decay_style: str = "line",
):
    """
    Plot one pixel's posterior-sample decay reconstructions as nested
    credible-interval bands, a chosen central curve, and the measured decay.

    Parameters
    ----------
    output_combination : dict[str, np.ndarray]
        Posterior-sample parameter maps, e.g.
        ``{'tau1': (H,W,NUM_SAMPLES), 'tau2': (H,W,NUM_SAMPLES), 'alpha1': (H,W,NUM_SAMPLES)}``
        for bi-exponential, or ``{'tau': (H,W,NUM_SAMPLES)}`` for mono-exponential
        -- same shape convention as :class:`ParamSelector`.
    decay : np.ndarray
        Measured decay, ``(H, W, T)``.
    irf : np.ndarray
        IRF, ``(T,)`` (shared) or ``(H, W, T)`` (per-pixel).
    freq_acq : float
        Acquisition frequency (MHz), i.e. ``freq[1]``.
    pixel : tuple[int, int]
        ``(x, y)`` pixel to plot.
    model_type : str
        ``"bi-exponential"`` or ``"mono-exponential"``.
    center : str
        Which curve to draw as the central line: ``"median"`` or ``"mean"``
        across posterior samples, or ``"best"`` (the single sample that
        optimizes ``metric`` at this pixel, via
        :meth:`ParamSelector.select_best_combination`).
    metric : str
        Only used when ``center="best"``; one of
        :attr:`ParamSelector.METRICS` (``"chi2"``, ``"reduced_chi2"``,
        ``"RMSE"``, ``"R2"``).
    ci_levels : tuple[int, ...]
        Nested credible-interval widths to shade, e.g. ``(92, 68)`` shades a
        92% and a 68% band (percentiles ``(4, 96)`` and ``(16, 84)`` of the
        per-bin sample distribution).
    decay_color : str
        Colour of the measured-decay line (defaults to ``_DECAY_COLOR``).
    fit_color : str
        Colour of the central fit curve and the credible bands (which are
        tinted-toward-white shades of it); defaults to ``_FIT_COLOR``.
    band_alpha : float or tuple[float, ...]
        Opacity of the credible bands. A scalar applies to every band; a
        sequence sets them per band, matched positionally to ``ci_levels``.
    fit_alpha : float
        Opacity of the central fit curve.
    decay_alpha : float
        Opacity of the measured decay (line or scatter points, see `decay_style`).
    title : str | None
        Axes title; defaults to ``f"Pixel ({x}, {y})"``.
    ax : matplotlib.axes.Axes | None
        Axes to draw into. If omitted, a new figure/axes is created and shown.
    decay_style : str
        How the measured decay is drawn: ``"line"`` (default) or ``"scatter"``
        (one point per time bin).

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    if model_type not in _MODEL_PARAM_KEYS:
        raise ValueError(
            f"Unknown model_type: {model_type!r}; expected one of "
            f"{tuple(_MODEL_PARAM_KEYS)}"
        )
    if center not in CENTERS:
        raise ValueError(f"Unknown center: {center!r}; expected one of {CENTERS}")
    if decay_style not in DECAY_STYLES:
        raise ValueError(
            f"Unknown decay_style: {decay_style!r}; expected one of {DECAY_STYLES}"
        )

    if np.ndim(band_alpha) == 0:
        band_alpha_by_level = {level: float(band_alpha) for level in ci_levels}
    else:
        band_alpha = tuple(band_alpha)
        if len(band_alpha) != len(ci_levels):
            raise ValueError(
                f"band_alpha must be a scalar or a sequence matching ci_levels "
                f"(len {len(ci_levels)}); got {len(band_alpha)} values"
            )
        band_alpha_by_level = dict(zip(ci_levels, band_alpha))

    x, y = pixel
    decay_px = np.asarray(decay)[x, y, :].astype(np.float64)

    stack = _reconstruct_sample_stack(
        output_combination, pixel, irf, decay_px, freq_acq, model_type
    )

    if center == "median":
        center_curve = np.median(stack, axis=0)
        center_label = "Posterior median"
    elif center == "mean":
        center_curve = np.mean(stack, axis=0)
        center_label = "Posterior mean"
    else:
        best_idx = _select_best_sample_idx(
            output_combination, pixel, irf, decay, freq_acq, model_type, metric
        )
        center_curve = stack[best_idx]
        center_label = f"Best sample ({metric})"

    t = np.arange(stack.shape[-1])

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    else:
        fig = ax.figure

    # Layering (back to front): measured decay, then the credible bands, then
    # the central fit curve.
    _Z_DECAY, _Z_BANDS, _Z_FIT = 1, 2, 3

    band_handles = []
    levels_wide_to_narrow = sorted(ci_levels, reverse=True)
    n_bands = len(levels_wide_to_narrow)
    tints = np.linspace(0.80, 0.42, n_bands) if n_bands > 1 else np.array([0.55])
    fit_rgb = np.array(mcolors.to_rgb(fit_color))
    for level, tint in zip(levels_wide_to_narrow, tints):
        half_width = (100 - level) / 2.0
        lo = np.percentile(stack, half_width, axis=0)
        hi = np.percentile(stack, 100 - half_width, axis=0)
        band_rgb = tuple((1.0 - tint) * fit_rgb + tint * np.ones(3))
        band = ax.fill_between(
            t,
            lo,
            hi,
            facecolor=band_rgb,
            edgecolor="none",
            alpha=band_alpha_by_level[level],
            zorder=_Z_BANDS,
        )
        band_handles.append((band, f"{level}% credible interval"))

    # Central fit curve on top, semi-transparent so the band reads through it.
    (center_line,) = ax.plot(
        t,
        center_curve,
        color=fit_color,
        lw=2.2,
        alpha=fit_alpha,
        zorder=_Z_FIT,
        solid_capstyle="round",
    )
    if decay_style == "scatter":
        decay_line = ax.scatter(
            t,
            decay_px,
            color=decay_color,
            s=10,
            alpha=decay_alpha,
            edgecolors="none",
            zorder=_Z_DECAY,
        )
    else:
        (decay_line,) = ax.plot(
            t,
            decay_px,
            color=decay_color,
            lw=2.0,
            alpha=decay_alpha,
            zorder=_Z_DECAY,
            solid_capstyle="round",
        )

    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Counts")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title or f"Pixel ({x}, {y})")

    handles = [h for h, _ in band_handles] + [center_line, decay_line]
    labels = [lbl for _, lbl in band_handles] + [center_label, "Decay (measured)"]
    ax.legend(handles, labels, loc="best", fontsize=8, frameon=False)

    if own_fig:
        plt.show()
    return fig, ax
