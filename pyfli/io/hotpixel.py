"""
Hot-pixel masks for gated/SPAD cameras: generate a mask from a dark or background
acquisition, save/load it as an image, and replace flagged pixels by the median of
their non-hot neighbours.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pyfli import logging
from pyfli.cm_stats.robust import MAD_TO_SIGMA, median_absolute_deviation


def hotpixel_counts(data: Any) -> np.ndarray:
    """Per-pixel total counts ``(H, W)`` of an image, an ``(H, W, T)`` cube or an ``(N, H, W, T)`` stack."""
    array = np.asarray(data, dtype=float)
    if array.ndim == 2:
        counts = array
    elif array.ndim == 3:
        counts = array.sum(axis=-1)
    elif array.ndim == 4:
        counts = array.sum(axis=(0, -1))
    else:
        raise ValueError(
            f"data must be (H, W), (H, W, T) or (N, H, W, T), got {array.shape}."
        )
    if not np.all(np.isfinite(counts)):
        raise ValueError("data contains non-finite values.")
    return counts


def generate_hotpixel_mask(
    data: Any,
    fraction: float | None = 0.05,
    threshold_sigma: float | None = None,
    save_path: str | None = None,
) -> np.ndarray:
    """
    Boolean hot-pixel mask ``(H, W)`` from a dark/background acquisition (image, cube or
    stack, summed to per-pixel counts).

    With `threshold_sigma` a pixel is hot when its counts exceed
    ``median + threshold_sigma * 1.4826 * MAD`` (as :meth:`DataIOUtils.hot_pixels`);
    otherwise the ``round(fraction * n_pixels)`` pixels with the most counts are flagged
    (``fraction=0.05`` gives a "5 per 100" mask). Ties at the cut are broken by pixel
    order, so the count is exact. `save_path` writes the mask as an image
    (:func:`save_hotpixel_mask`).
    """
    counts = hotpixel_counts(data)
    if threshold_sigma is not None:
        if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
            raise ValueError(
                f"threshold_sigma must be positive, got {threshold_sigma}."
            )
        median, mad = median_absolute_deviation(counts)
        threshold = median + threshold_sigma * MAD_TO_SIGMA * mad
        mask = counts > threshold
        rule = f"counts > {threshold:.1f} (median={median:.1f}, MAD={mad:.1f}, sigma={threshold_sigma})"
    else:
        if fraction is None or not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}.")
        n_hot = round(fraction * counts.size)
        mask = np.zeros(counts.size, dtype=bool)
        if n_hot:
            order = np.argsort(-counts.ravel(), kind="stable")
            mask[order[:n_hot]] = True
        mask = mask.reshape(counts.shape)
        rule = f"hottest {fraction:.2%} of pixels"
    logging.info(
        f"Hot-pixel mask: {int(mask.sum())} of {mask.size} pixels "
        f"({100.0 * mask.mean():.3f}%), {rule}"
    )
    if save_path:
        save_hotpixel_mask(mask, save_path)
    return mask


def save_hotpixel_mask(mask: Any, path: str) -> None:
    """Write a boolean mask as a grayscale image (hot = white), readable by :func:`load_hotpixel_mask`."""
    plt.imsave(path, np.asarray(mask, dtype=np.uint8) * 255, cmap="gray")
    logging.info(f"Hot-pixel mask saved to: {path}")


def load_hotpixel_mask(path: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    """
    Boolean mask from an image (PNG/JPEG/TIFF; first channel of a colour image, nonzero =
    hot). With `shape` ``(H, W)`` a transposed mask is turned to match, otherwise a
    mismatch raises.
    """
    mask = plt.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if shape is not None and mask.shape != tuple(shape):
        if mask.shape == tuple(shape)[::-1]:
            mask = mask.T
        else:
            raise ValueError(
                f"mask shape {mask.shape} does not match data shape {shape}."
            )
    return mask > 0


def remove_hotpixels(data: Any, mask: Any, max_radius: int = 2) -> np.ndarray:
    """
    Replace every flagged pixel of an ``(H, W)`` image or ``(H, W, T)`` cube by the median
    of its non-hot neighbours in the 3x3 window (per time gate), widening the window up to
    `max_radius` when all neighbours are hot; a pixel with no valid neighbour keeps its
    value. Uses only unflagged pixels, so the result does not depend on processing order.
    Returns float32.
    """
    array = np.asarray(data)
    image = array.ndim == 2
    cube = array[..., None] if image else array
    if cube.ndim != 3:
        raise ValueError(f"data must be (H, W) or (H, W, T), got {array.shape}.")
    hot = np.asarray(mask, dtype=bool)
    if hot.shape != cube.shape[:2]:
        raise ValueError(
            f"mask shape {hot.shape} does not match data shape {cube.shape[:2]}."
        )
    source = cube.astype(np.float32)
    source[hot] = np.nan
    cleaned = cube.astype(np.float32, copy=True)
    H, W = hot.shape
    unresolved = 0
    for y, x in zip(*np.nonzero(hot)):
        for radius in range(1, max_radius + 1):
            window = source[
                max(0, y - radius) : min(H, y + radius + 1),
                max(0, x - radius) : min(W, x + radius + 1),
            ]
            valid = np.isfinite(window).any(axis=(0, 1))
            if valid.all():
                cleaned[y, x] = np.nanmedian(window, axis=(0, 1))
                break
        else:
            unresolved += 1
    if unresolved:
        logging.warning(
            f"{unresolved} hot pixel(s) had no valid neighbour within radius {max_radius}; kept."
        )
    return cleaned[..., 0] if image else cleaned
