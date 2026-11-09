"""
Collect numerical, masking, simulation, plotting, and export utilities shared by
analysis workflows.

This module belongs to :mod:`pyfli.analysis` and is part of PyFLI post-processing,
diagnostics, statistical comparison, and result-loading utilities for fitted FLI/FLIM
datasets. Public API includes functions :func:`circular_convolution_fft`,
:func:`single_ex_decay_summed_overtime`, :func:`gate_j`, :func:`Pj_continuous_mono`,
:func:`Pj_from_samples_mono`, :func:`multimodal_normal`, :func:`recovery_plot`,
:func:`threshold_masking`, :func:`data_masking`, and
:func:`save_3d_array_as_tiff_sequence`.
"""

import math
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.integrate import quad
from scipy.stats import pearsonr

from pyfli import logging
from pyfli.plot_style import dark_palette, legend_outside
from pyfli.reconstruction.common_reconstruct import circular_convolve

from ..data_vnp.color_processor import ColorProcessor


def circular_convolution_fft(
    x: np.ndarray, h: np.ndarray, broadcast_irf: bool = True
) -> np.ndarray:
    """
    Circular convolution of ``(H, W, T)`` cubes along time
    (:func:`pyfli.reconstruction.common_reconstruct.circular_convolve`).

    Parameters
    ----------
    x : np.ndarray
        Input array, coordinate, or signal being transformed.
    h : np.ndarray
        IRF, image height, or temporal kernel used by the routine.
    broadcast_irf : bool
        Whether a shared IRF should be broadcast to every pixel.

    Returns
    -------
    np.ndarray
        Circular convolution result with the same length as the input decay.
    """
    x = np.asarray(x)
    h = np.asarray(h)

    if x.ndim != 3 or h.ndim != 3:
        raise ValueError(
            f"x and h must be 3D arrays, got x.ndim={x.ndim}, h.ndim={h.ndim}"
        )

    if x.shape[-1] != h.shape[-1]:
        raise ValueError(
            f"Last dimension (convolution axis) must match: {x.shape[-1]} vs {h.shape[-1]}"
        )

    # Broadcast h to match x (for pixel-wise or shared IRFs)
    if broadcast_irf:
        if h.shape[0] != x.shape[0] or h.shape[1] != x.shape[1]:
            h = np.broadcast_to(h, x.shape)
    # h = np.broadcast_to(h, x.shape)

    return circular_convolve(x, h, axis=2)


def single_ex_decay_summed_overtime(
    tau: np.ndarray,
    irf_data: np.ndarray,
    alpha: float = 1.0,
    err: float = 0.0,
    laser_period: float = 12.5,
    seed: int | None = None,
) -> tuple[Any, ...]:
    """
    Run the single ex decay summed overtime routine.

    Parameters
    ----------
    tau : np.ndarray
        Lifetime value or lifetime map in nanoseconds.
    irf_data : np.ndarray
        Instrument response data used to convolve or simulate decays.
    alpha : float
        Regularization strength, fraction value, or significance threshold used by the
    routine.
    err : float
        Noise or perturbation level applied to simulated decays.
    laser_period : float
        Laser repetition period in nanoseconds.
    seed : int | None
        Random seed used for reproducible sampling.

    Returns
    -------
    tuple[Any, ...]
        Tuple containing the integrated single-exponential decay and time samples.
    """
    if seed is not None:
        np.random.seed(seed)
    M, N, T = irf_data.shape
    tau = np.asarray(tau, dtype=float)

    # Ensure tau is broadcastable to (M, N, T)
    tau = np.broadcast_to(tau[..., np.newaxis], (M, N, T))

    # Time vector
    t = np.linspace(0, laser_period, T)[np.newaxis, np.newaxis, :]  # (1, 1, T)

    # --- Avoid division by zero ---
    zero_mask = (tau <= 0) | ~np.isfinite(tau)
    safe_tau = np.where(
        zero_mask, np.inf, tau
    )  # τ=0 → inf => exp(-t/inf)=1, then we zero it later

    # --- Theoretical single exponential decay ---
    f_t = (1.0 / safe_tau) * np.exp(-t / safe_tau)

    # Zero out pixels where tau=0 or invalid
    f_t[zero_mask] = 0.0

    # --- Normalize IRF per pixel ---
    I_sum = np.sum(irf_data, axis=2, keepdims=True)
    if np.any(I_sum <= 0):
        raise ValueError("One or more IRF pixels sum to zero; cannot normalize.")
    I_t = irf_data / I_sum

    # --- Circular convolution along time axis (axis=2) ---
    s_ti = circular_convolution_fft(f_t, I_t)

    # --- Add Gaussian noise ---
    if np.isscalar(err):
        noise = np.random.normal(0, err, size=s_ti.shape)
    else:
        noise = np.asarray(err, dtype=float)
        if noise.shape != s_ti.shape:
            raise ValueError("Shape mismatch: 'err' array must match signal shape")

    # --- Final weighted signal ---
    s_t = alpha * s_ti + (1.0 - alpha) * noise
    s_t = np.clip(s_t, 0.0, None)

    # Ensure f_t is zero wherever tau=0
    s_t[zero_mask] = 0.0

    return f_t, s_t, I_t, t


def gate_j(m: int, T: float) -> np.ndarray:
    """
    Run the gate j routine.

    Parameters
    ----------
    m : int
        Gate, harmonic, or interval index.
    T : float
        Time axis or acquisition period used by the calculation.

    Returns
    -------
    np.ndarray
        Integrated gate image or trace for the requested gate index.
    """
    buckets = []
    for j in range(1, m + 1):
        a = (j - 1) * T / m
        b = j * T / m
        buckets.append((a, b))
    return buckets


def Pj_continuous_mono(
    f: np.ndarray, m: int, T: float, epsabs: float = 1e-8, epsrel: float = 1e-8
) -> Any:
    """
    Run the pj continuous mono routine.

    Parameters
    ----------
    f : np.ndarray
        Decay basis, distribution, or signal function used by the calculation.
    m : int
        Gate, harmonic, or interval index.
    T : float
        Time axis or acquisition period used by the calculation.
    epsabs : float
        Absolute integration tolerance.
    epsrel : float
        Relative integration tolerance.

    Returns
    -------
    Any
        Object produced by pj continuous mono.
    """
    gates = np.array(gate_j(m, T))  # list-of-tuples → 2D array for slicing
    a_vals, b_vals = gates[:, 0], gates[:, 1]

    # Vectorized numerical integration using np.vectorize wrapper
    def integrate_interval(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Run the integrate interval routine.

        Parameters
        ----------
        a : np.ndarray
            Lower integration or interval bound.
        b : np.ndarray
            Upper integration or interval bound.

        Returns
        -------
        np.ndarray
            Integrated signal over the requested interval.
        """
        val, _ = quad(f, a, b, epsabs=epsabs, epsrel=epsrel)
        return val

    integrate_vec = np.vectorize(integrate_interval)
    Pj = integrate_vec(a_vals, b_vals)
    return Pj


def Pj_from_samples_mono(
    t_samples: np.ndarray, y_samples: np.ndarray, m: int, T: float
) -> Any:
    """
    Run the pj from samples mono routine.

    Parameters
    ----------
    t_samples : np.ndarray
        Sample times used to integrate a mono-exponential decay.
    y_samples : np.ndarray
        Sampled mono-exponential values integrated over gates.
    m : int
        Gate, harmonic, or interval index.
    T : float
        Time axis or acquisition period used by the calculation.

    Returns
    -------
    Any
        Object produced by pj from samples mono.
    """
    H, W, Tn = y_samples.shape
    gates = gate_j(m, T)

    # Ensure time axis and sample consistency
    if t_samples.shape[0] != Tn:
        raise ValueError("Length of t_samples must match y_samples.shape[-1].")

    # Interpolate gate edges and ensure inclusion
    Pj = np.zeros((H, W, m), dtype=float)
    for j, (a, b) in enumerate(gates):
        # Create boolean mask for time bins within gate
        mask = (t_samples >= a) & (t_samples <= b)

        # If gate falls outside sampled range, skip safely
        if not np.any(mask):
            continue

        # Extract y and t segments for integration
        t_sub = t_samples[mask]
        y_sub = y_samples[..., mask]

        # Include exact gate edges via vectorised linear interpolation (H,W pixels)
        if t_sub[0] > a:
            idx = int(np.clip(np.searchsorted(t_samples, a, side="right"), 1, Tn - 1))
            w = (a - t_samples[idx - 1]) / (
                t_samples[idx] - t_samples[idx - 1] + 1e-300
            )
            y_a = (
                y_samples[..., idx - 1] * (1.0 - w) + y_samples[..., idx] * w
            )  # (H, W)
            y_sub = np.concatenate((y_a[..., np.newaxis], y_sub), axis=-1)
            t_sub = np.concatenate(([a], t_sub))
        if t_sub[-1] < b:
            idx = int(np.clip(np.searchsorted(t_samples, b, side="left"), 0, Tn - 2))
            w = (b - t_samples[idx]) / (t_samples[idx + 1] - t_samples[idx] + 1e-300)
            y_b = (
                y_samples[..., idx] * (1.0 - w) + y_samples[..., idx + 1] * w
            )  # (H, W)
            y_sub = np.concatenate((y_sub, y_b[..., np.newaxis]), axis=-1)
            t_sub = np.concatenate((t_sub, [b]))

        # Integrate over time using trapezoidal rule (vectorized along last axis)
        Pj[..., j] = np.trapz(y_sub, x=t_sub, axis=-1)

    # Normalize to obtain probability distribution per pixel
    Pj_sum = np.sum(Pj, axis=-1, keepdims=True)
    Pj /= np.maximum(Pj_sum, 1e-12)

    return Pj


def multimodal_normal(
    n_samples: int = 10000,
    mus: np.ndarray | None = None,
    sigma: float | None = None,
    weights: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[Any, ...]:
    """
    Run the multimodal normal routine.

    Parameters
    ----------
    n_samples : int
        Number of samples, components, gates, or iterations used by the routine.
    mus : np.ndarray | None
        Gaussian component means used by the multimodal sampler.
    sigma : float | None
        Standard deviation used by a sampler or noise model.
    weights : np.ndarray | None
        Sampling or model weights used by the routine.
    seed : int | None
        Random seed used for reproducible sampling.

    Returns
    -------
    tuple[Any, ...]
        Tuple containing sampled values from the configured normal mixture.
    """
    np.random.seed(seed)

    if mus is None:
        raise ValueError("You must provide a list of means (mus).")
    mus = np.array(mus)
    n_modes = len(mus)

    # Ensure sigma matches mus
    if sigma is None:
        sigma = np.ones(n_modes) * 1.0  # default sigma = 1 for all modes
    elif isinstance(sigma, (int, float)):
        sigma = np.full(n_modes, sigma)
    else:
        sigma = np.array(sigma)
        assert len(sigma) == n_modes, (
            "sigma must be a single value or same length as mus"
        )

    # Equal weights if none provided
    if weights is None:
        weights = np.ones(n_modes) / n_modes
    else:
        weights = np.array(weights)
        weights /= weights.sum()  # normalize

    # Number of samples per mode
    samples_per_mode = np.random.multinomial(n_samples, weights)

    # Generate samples for each mode
    samples = []
    samples_2d = np.zeros(
        (n_modes, n_samples), dtype=float
    )  # n_samples cols = max possible
    for i, (mu_val, s, n) in enumerate(zip(mus, sigma, samples_per_mode)):
        samp = np.random.normal(loc=mu_val, scale=s, size=n)
        samples.append(samp)
        samples_2d[i, :n] = samp

    samples = np.concatenate(samples)

    # Ensure all values are positive (reflect negatives)
    samples = np.abs(samples)

    return samples, samples_2d


def recovery_plot(
    gt_dict: np.ndarray, est_dict: np.ndarray, keys_to_plot: np.ndarray | None = None
) -> np.ndarray:
    """
    Plots Ground Truth vs Estimates for specific keys.
    Handles data shapes: (N, X, Y) or (N, Batch, X, Y).

    Args:
        gt_dict: Dictionary of Ground Truth arrays.
        est_dict: Dictionary of Estimated arrays.
        keys_to_plot: List of strings (keys). If None, plots all keys in gt_dict.
    """
    if keys_to_plot is None:
        keys_to_plot = list(gt_dict.keys())

    # 1. Automatic Grid Arrangement
    num_plots = len(keys_to_plot)
    if num_plots == 0:
        return

    cols = min(num_plots, 4)
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)
    axes = axes.flatten()

    for i, key in enumerate(keys_to_plot):
        ax = axes[i]

        # Ensure data is numpy array and flatten (X, Y) -> (X*Y,)
        x = np.array(gt_dict[key]).flatten()
        y = np.array(est_dict[key]).flatten()

        # Calculate Pearson Correlation across all pixels
        r_val, _ = pearsonr(x, y)

        # 2. Scatter Plot
        # Using the style from your reference image
        ax.scatter(x, y, color="#1c3f73", alpha=0.5, s=15, edgecolors="none")

        # 3. Identity Line (y = x) - UPDATED TO RED DASH
        all_vals = np.concatenate([x, y])
        # Calculate limits: start slightly below the absolute minimum
        data_min = np.min(all_vals)
        data_max = np.max(all_vals)
        buffer = (data_max - data_min) * 0.05

        plot_min = data_min - buffer
        plot_max = data_max + buffer

        ax.plot(
            [plot_min, plot_max],
            [plot_min, plot_max],
            color="#8c2e24",
            linestyle="--",
            linewidth=1.5,
            zorder=5,
        )

        # 4. Styling & Formatting
        ax.set_title(key, fontsize=15)
        ax.set_xlabel("Ground truth", fontsize=12)

        # FORCE AXIS TO START FROM LESSER THAN MINIMUM
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)

        if i % cols == 0:
            ax.set_ylabel("Estimate", fontsize=12)

        # Display r-value
        ax.text(
            0.05,
            0.92,
            f"$r = {r_val:.3f}$",
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
        )

        # Clean background and spines
        ax.grid(True, linestyle="-", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    return fig


def threshold_masking(
    fli: np.ndarray, irf: np.ndarray, threshold: int = 100
) -> tuple[Any, ...]:
    """
    Run the threshold masking routine.

    Parameters
    ----------
    fli : np.ndarray
        FLI lifetime map or decay-derived image to threshold.
    irf : np.ndarray
        Instrument response function aligned with the decay signal.
    threshold : int
        Threshold used to mask, classify, or validate data.

    Returns
    -------
    tuple[Any, ...]
        Tuple containing thresholded mask arrays and metadata.
    """
    if threshold is None:
        raise ValueError("no thershold value provided")
    else:
        intensity = np.sum(fli, axis=-1)
        mask = intensity > threshold

    mask = mask.astype(bool)
    if mask.ndim < fli.ndim:
        mask_expanded = mask[..., np.newaxis]
        masked_fli = fli * mask_expanded
        masked_irf = irf * mask_expanded
    else:
        masked_fli = fli * mask
        masked_irf = irf * mask

    return masked_fli, masked_irf


def data_masking(*arrays: Any, mask: np.ndarray, return_list: bool = False) -> Any:
    """
    Run the data masking routine.

    Parameters
    ----------
    *arrays : Any
        Additional positional values accepted by the routine.
    mask : np.ndarray
        Boolean or labeled mask selecting pixels for the operation.
    return_list : bool
        If ``True``, return a list of masks instead of a combined mask.

    Returns
    -------
    Any
        Object produced by data masking.
    """
    mask = mask.astype(bool)
    results = []
    for arr in arrays:
        if not isinstance(arr, np.ndarray):
            raise TypeError("All inputs must be numpy arrays")
        if mask.ndim < arr.ndim:
            expand_dims = arr.ndim - mask.ndim
            mask_expanded = mask[(...,) + (None,) * expand_dims]
        else:
            mask_expanded = mask
        try:
            masked = arr * mask_expanded
        except ValueError:
            raise ValueError("Mask is not broadcastable to array shape")
        results.append(masked)
    if len(results) == 1:
        return results[0]
    return results if return_list else tuple(results)


def save_3d_array_as_tiff_sequence(
    array_3d: np.ndarray, output_folder: str, prefix: str = "frame"
) -> None:
    """
    Saves a 3D numpy array (H, W, T) as a series of 2D TIFF files.

    Parameters:
    - array_3d: The numpy array of shape (H, W, T)
    - output_folder: Path to the folder where TIFs will be saved
    - prefix: Filename prefix (e.g., 'frame_001.tif')
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    _, _, T = array_3d.shape

    logging.info(f"Saving {T} frames to '{output_folder}'...")

    for t in range(T):
        # Extract the 2D slice (X, Y) at time t
        # Note: tifffile expects (H, W), so we take [:, :, t]
        frame = array_3d[:, :, t]

        # Format filename with leading zeros for correct sorting (e.g., frame_005.tif)
        file_name = f"{prefix}_{t:03d}.tif"
        file_path = os.path.join(output_folder, file_name)

        # Save the slice
        tifffile.imwrite(file_path, frame.astype(np.float32))

    logging.info("Saving complete.")


def save_as_uint16_sequence(
    data: np.ndarray, output_folder: str, prefix: str = "frame"
) -> None:
    """
    Saves (H, W, T) array as 16-bit integer TIFFs.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Handle Negative Values (Safety for uint16)
    # Background subtraction in your class might leave tiny negatives
    data = np.maximum(data, 0)

    # 2. Optional: Auto-Scaling (Only use if data is 0.0 - 1.0 or very small)
    # If your data is already raw photon counts, skip this step.
    if data.max() <= 1.0 and data.max() > 0:
        data = data * 65535

    # 3. Cast to uint16
    # This will truncate decimals (e.g., 1.9 becomes 1)
    data_uint16 = data.astype(np.uint16)

    _, _, T = data_uint16.shape
    for t in range(T):
        frame = data_uint16[:, :, t]
        file_path = os.path.join(output_folder, f"{prefix}_{t:03d}.tif")
        tifffile.imwrite(file_path, frame)

    logging.info(f"Saved {T} files to {output_folder} in uint16 format.")


def random_true_pixel(bool_array: np.ndarray) -> Any:
    """
    Run the random true pixel routine.

    Parameters
    ----------
    bool_array : np.ndarray
        Boolean array from which a true pixel is selected.

    Returns
    -------
    Any
        Object produced by random true pixel.
    """
    true_indices = np.flatnonzero(bool_array)
    if true_indices.size == 0:
        return None
    random_linear_idx = np.random.choice(true_indices)
    pix_x, pix_y = np.unravel_index(random_linear_idx, bool_array.shape)
    return int(pix_x), int(pix_y)


def PhasorFreqComputaion(
    laser_period: float = 12.5,
    gate_delay: np.ndarray | None = None,
    num_gates: int | None = None,
) -> np.ndarray:  # all the units in ns
    """
    Run the phasor freq computaion routine.

    Parameters
    ----------
    laser_period : float
        Laser repetition period in nanoseconds.
    gate_delay : np.ndarray | None
        Delay of each gate relative to the excitation pulse.
    num_gates : int | None
        Number of acquisition gates used for frequency computation.

    Returns
    -------
    np.ndarray
        Phasor frequency-domain representation for the input decay.
    """
    freq = 1000.0 / laser_period
    if gate_delay is None or num_gates is None:
        effective_freq = freq
    else:
        effective_freq = 1000.0 / (
            num_gates * gate_delay
        )  # frequency is computed in the MHz if the gate delays are in ns
    return effective_freq


def save_plot(
    save_dir: str,
    name: str,
    fig: Any | None = None,
    dpi: int = 300,
    close: bool = False,
) -> None:
    # Saves a plot. Handles subplots (pass fig) or direct plots (uses current)
    """
    Save plot.

    Parameters
    ----------
    save_dir : str
        Directory where outputs are saved.
    name : str
        Dataset, experiment, figure, or output name.
    fig : Any | None
        Matplotlib figure object to update or save.
    dpi : int
        Resolution used when saving a figure.
    close : bool
        Whether to close the figure after saving.

    Returns
    -------
    None
        No object is returned; the function save plot.
    """
    path = os.path.join(save_dir, f"{name}.png")
    target = fig if fig is not None else plt
    try:
        target.savefig(path, bbox_inches="tight", dpi=dpi)
    except Exception as e:
        logging.error(f"ERROR saving {name}: {e!s}")
    if close:
        plt.close(fig) if fig else plt.close()


def plot_pixel_diagnostic(
    binned_decay: np.ndarray,
    all_fitset: np.ndarray,
    names: Any,
    pixel: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    t: np.ndarray | None = None,
    yscale: str = "log",
    model_type: str = "BI-EXPONENTIAL",
    colors: Any | None = None,
    figsize: tuple[int, ...] = (12, 6),
    raw_style: str = "bar",
    map_aspect: str = "equal",
    show_colorbar: bool = True,
    show: bool = True,
) -> np.ndarray:
    """
    Plot pixel diagnostic.

    Parameters
    ----------
    binned_decay : np.ndarray
        Binned decay cube used for fitting or diagnostics.
    all_fitset : np.ndarray
        Collection of fit-result dictionaries used for comparison or plotting.
    names : Any
        Dataset names used in summaries and plots.
    pixel : np.ndarray | None
        Selected pixel coordinate.
    mask : np.ndarray | None
        Boolean or labeled mask selecting pixels for the operation.
    t : np.ndarray | None
        Time axis or acquisition period used by the calculation.
    yscale : str
        Scale used for the y-axis.
    model_type : str
        FLI/FLIM model family, such as mono- or bi-exponential.
    colors : Any | None
        Color sequence used for plotted sources or groups.
    figsize : tuple[int, ...]
        Figure size passed to Matplotlib.
    raw_style : str
        Style used to draw raw pixel decay data.
    map_aspect : str
        Aspect ratio used when rendering lifetime maps.
    show_colorbar : bool
        Whether to draw a colorbar.
    show : bool
        Whether to display the generated plot.

    Returns
    -------
    np.ndarray
        Matplotlib figure or axes containing the pixel diagnostic plot.
    """
    jet_m = ColorProcessor().lowest_zero("jet")
    if pixel is None:
        if mask is None:
            raise ValueError("Provide either pixel=(row, col) or mask.")
        x, y = random_true_pixel(mask)
    else:
        x, y = pixel

    raw = np.asarray(binned_decay[x, y, :], dtype=float)
    bins = raw.shape[-1]
    if t is not None:
        xs = np.asarray(t, dtype=float).ravel()
        if len(xs) != bins:
            raise ValueError(
                f"t length ({len(xs)}) does not match decay bins ({bins}); "
                "x-axis and data would be misaligned."
            )
        xlabel = "Time (ns)"
    else:
        xs = np.arange(bins)
        xlabel = "Gate #"
    if colors is None:
        colors = dark_palette(len(all_fitset))
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 2], height_ratios=[3, 1])
    ax_map = fig.add_subplot(gs[:, 0])  # full-height left panel
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1], sharex=ax_top)
    intensity = np.sum(binned_decay, axis=-1)  # (H, W)
    display_intensity = intensity if mask is None else intensity * mask
    im = ax_map.imshow(display_intensity, cmap=jet_m, aspect=map_aspect)
    # imshow's x-axis = columns, y-axis = rows -> mark pixel at (col, row)=(y, x)
    ax_map.scatter(y, x, marker="x", c="red", s=80, linewidths=2)
    ax_map.set_title("Intensity (Summed)")
    # ax_map.set_xlabel("Column"); ax_map.set_ylabel("Row")
    if show_colorbar:
        fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

    if raw_style == "bar":
        width = (xs[1] - xs[0]) if len(xs) > 1 else 1.0
        ax_top.bar(
            xs,
            raw,
            width=width,
            color="0.8",
            edgecolor="none",
            zorder=1,
            label="Raw Data",
        )
    elif raw_style == "step":
        ax_top.plot(
            xs,
            raw,
            color="0.7",
            lw=0.9,
            drawstyle="steps-mid",
            zorder=1,
            label="Raw Data",
        )
    else:  # "line"
        ax_top.plot(xs, raw, color="0.7", lw=0.9, zorder=1, label="Raw Data")

    for i, fs in enumerate(all_fitset):
        label = names[i] if i < len(names) else f"Fit {i + 1}"
        fit = np.asarray(fs["fit_map"][x, y, :], dtype=float)
        ax_top.plot(
            xs, fit, color=colors[i], lw=1.3, zorder=2 + i, label=f"Fit: {label}"
        )

    ax_top.set_yscale(yscale)  # log / linear switch
    ax_top.set_ylabel("Photon Counts")
    ax_top.set_title(f"Fit Diagnostics ({model_type})  [pixel {x}, {y}]")
    legend_outside(ax_top, ncol=2, fontsize=8, framealpha=0.9)
    if yscale == "log":
        pos = raw[raw > 0]
        if pos.size:
            ax_top.set_ylim(bottom=max(pos.min() * 0.1, 1e-3))

    ax_bot.axhline(0, color="black", lw=0.8, zorder=1)
    for i, fs in enumerate(all_fitset):
        label = names[i] if i < len(names) else f"Fit {i + 1}"
        res = np.asarray(fs["residual_map"][x, y, :], dtype=float)
        ax_bot.plot(xs, res, color=colors[i], lw=1.0, label=f"{label} residuals")
    ax_bot.set_ylabel("Residuals")
    ax_bot.set_xlabel(xlabel)

    fig.tight_layout()
    if show:
        plt.show()
    return fig
