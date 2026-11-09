"""
Detect and fold periodic SPAD gate sequences into one excitation period.

This module belongs to :mod:`pyfli.io` and provides signal-aware temporal alignment
and folding for gated SPAD acquisitions. Detection is performed on a spatially
integrated trace, while all shifts and sums are applied to the original data cube.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from pyfli.cm_stats.robust import MAD_TO_SIGMA, median_absolute_deviation


@dataclass(frozen=True)
class SpadFoldLayout:
    """
    Store the detected or user-specified layout of a periodic SPAD acquisition.

    Parameters
    ----------
    original_bins : int
        Number of temporal gates before folding.
    period_bins : int
        Number of gates in one excitation period.
    repeat_count : int
        Number of repeated excitation periods contained in the acquisition.
    phase_origin : int
        Gate index inside one period that is treated as the start of the decay.
    phase_shift : int
        Circular shift applied on the time axis before folding.
    onset_index : int
        Detected or implied gate index of the signal onset inside one period.
    onset_lead_bins : int
        Number of gates the folded period starts before the signal onset.
    pulse_positions : tuple[int, ...]
        Expected pulse-onset positions in the original acquisition.
    period_score : float
        Circular-autocorrelation score for the selected temporal period.
    cycle_similarity : float
        Mean pairwise similarity between aligned repeated periods.
    signal_score : float
        Signal-to-baseline confidence score used during onset detection.
    confidence : float
        Combined confidence score for automatic folding.
    manual_period : bool
        Whether period_bins was supplied explicitly by the user.
    manual_phase : bool
        Whether phase_shift was supplied explicitly by the user.
    """

    original_bins: int
    period_bins: int
    repeat_count: int
    phase_origin: int
    phase_shift: int
    onset_index: int
    onset_lead_bins: int
    pulse_positions: tuple[int, ...]
    period_score: float
    cycle_similarity: float
    signal_score: float
    confidence: float
    manual_period: bool
    manual_phase: bool

    def to_metadata(self) -> dict[str, object]:
        """
        Convert the fold layout to serializable metadata.

        Returns
        -------
        dict[str, object]
            Dictionary containing the fold-detection and alignment metadata.
        """
        return asdict(self)


def build_temporal_trace(data: np.ndarray) -> np.ndarray:
    """
    Build a high-SNR one-dimensional trace by summing over the spatial dimensions.

    Parameters
    ----------
    data : np.ndarray
        Three-dimensional SPAD data cube with shape (H, W, T).

    Returns
    -------
    np.ndarray
        Spatially integrated temporal trace with shape (T,) and float64 dtype.
    """
    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError(
            f"SPAD folding requires a 3D (H, W, T) cube, got {array.shape}."
        )
    if array.shape[-1] < 2:
        raise ValueError("SPAD folding requires at least two temporal gates.")

    trace = np.nansum(array.astype(np.float64, copy=False), axis=(0, 1))
    if not np.all(np.isfinite(trace)):
        raise ValueError(
            "Temporal trace contains non-finite values after spatial integration."
        )
    if np.allclose(trace, trace[0]):
        raise ValueError(
            "Temporal trace is constant; periodic folding cannot be detected."
        )
    return trace


def _smooth_circular_trace(
    trace: np.ndarray,
    smoothing_sigma: float,
) -> np.ndarray:
    """Return a circularly smoothed float64 temporal trace."""
    values = np.asarray(trace, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D temporal trace, got shape {values.shape}.")
    if smoothing_sigma < 0:
        raise ValueError(f"smoothing_sigma must be >= 0, got {smoothing_sigma}.")
    if smoothing_sigma == 0:
        return values.copy()
    return gaussian_filter1d(values, sigma=smoothing_sigma, mode="wrap")


def _circular_autocorrelation(trace: np.ndarray) -> np.ndarray:
    """Return normalized circular autocorrelation for a one-dimensional trace."""
    values = np.asarray(trace, dtype=np.float64)
    centered = values - np.mean(values)
    energy = float(np.dot(centered, centered))
    if energy <= np.finfo(np.float64).eps:
        raise ValueError(
            "Temporal trace has insufficient variation for period detection."
        )

    spectrum = np.fft.rfft(centered)
    correlation = np.fft.irfft(
        spectrum * np.conjugate(spectrum),
        n=values.size,
    )
    correlation /= correlation[0]
    return np.real(correlation)


def _valid_period_divisors(n_bins: int) -> list[int]:
    """Return period lengths that divide n_bins into at least two full periods."""
    return [period for period in range(2, (n_bins // 2) + 1) if n_bins % period == 0]


def _period_score(
    correlation: np.ndarray,
    period_bins: int,
) -> float:
    """Score one candidate period using all repeated circular-autocorrelation lags."""
    n_bins = correlation.size
    if period_bins <= 0 or n_bins % period_bins != 0:
        raise ValueError(
            f"period_bins={period_bins} must divide the temporal length {n_bins}."
        )

    repeat_count = n_bins // period_bins
    if repeat_count < 2:
        raise ValueError("At least two excitation periods are required for folding.")

    repeated_lags = [period_bins * index for index in range(1, repeat_count)]
    score = float(np.mean(correlation[repeated_lags]))
    return float(np.clip(score, -1.0, 1.0))


def estimate_period_bins(
    trace: np.ndarray,
    expected_repeats: int | None = None,
    search_radius: float = 0.15,
) -> tuple[int, float]:
    """
    Estimate the excitation-period length from circular temporal autocorrelation.

    Parameters
    ----------
    trace : np.ndarray
        One-dimensional temporal trace.
    expected_repeats : int | None
        Expected number of repeated excitation periods. When provided, the search is
        constrained around len(trace) / expected_repeats.
    search_radius : float
        Fractional search radius around the expected period when expected_repeats is
        provided.

    Returns
    -------
    tuple[int, float]
        Detected period length in bins and its normalized autocorrelation score.
    """
    values = np.asarray(trace, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D temporal trace, got shape {values.shape}.")
    if values.size < 4:
        raise ValueError(
            "At least four temporal gates are required for period detection."
        )
    if not (0 < search_radius <= 0.5):
        raise ValueError(f"search_radius must be in (0, 0.5], got {search_radius}.")
    if expected_repeats is not None and expected_repeats < 2:
        raise ValueError(
            f"expected_repeats must be >= 2 when provided, got {expected_repeats}."
        )

    correlation = _circular_autocorrelation(values)
    candidates = _valid_period_divisors(values.size)
    if not candidates:
        raise ValueError(
            f"Temporal length {values.size} cannot be divided into repeated "
            "full periods."
        )

    if expected_repeats is not None:
        target = values.size / expected_repeats
        lower = target * (1.0 - search_radius)
        upper = target * (1.0 + search_radius)
        candidates = [period for period in candidates if lower <= period <= upper]
        if not candidates:
            raise ValueError(
                "No integer period compatible with the temporal length falls within "
                f"{search_radius:.0%} of the expected {target:.3f} bins."
            )

    scores = {period: _period_score(correlation, period) for period in candidates}

    best_score = max(scores.values())
    tolerance = 0.03
    near_best = [
        period for period, score in scores.items() if score >= best_score - tolerance
    ]

    selected_period = min(near_best)
    return selected_period, scores[selected_period]


def make_phase_folded_trace(
    trace: np.ndarray,
    period_bins: int,
) -> np.ndarray:
    """
    Sum repeated periods of a temporal trace without changing its circular phase.

    Parameters
    ----------
    trace : np.ndarray
        One-dimensional temporal trace.
    period_bins : int
        Number of gates in one excitation period.

    Returns
    -------
    np.ndarray
        Phase-folded trace with shape (period_bins,).
    """
    values = np.asarray(trace, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D temporal trace, got shape {values.shape}.")
    if period_bins < 2:
        raise ValueError(f"period_bins must be >= 2, got {period_bins}.")
    if values.size % period_bins != 0:
        raise ValueError(
            f"Temporal length {values.size} is not divisible by "
            f"period_bins={period_bins}."
        )

    repeat_count = values.size // period_bins
    return values.reshape(repeat_count, period_bins).sum(axis=0)


def detect_signal_onset(
    folded_trace: np.ndarray,
    smoothing_sigma: float = 1.0,
    threshold_fraction: float = 0.10,
) -> tuple[int, float]:
    """
    Detect the circular onset of the dominant fluorescence response in one period.

    Parameters
    ----------
    folded_trace : np.ndarray
        One-period temporal trace. The signal may wrap across the first/last gate.
    smoothing_sigma : float
        Gaussian smoothing sigma used only for onset detection.
    threshold_fraction : float
        Fraction of peak-to-baseline amplitude used for the rising-edge crossing.

    Returns
    -------
    tuple[int, float]
        Detected onset index and a signal-to-baseline confidence score in [0, 1].
    """
    values = np.asarray(folded_trace, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D folded trace, got shape {values.shape}.")
    if values.size < 3:
        raise ValueError("At least three bins are required for onset detection.")
    if not (0 < threshold_fraction < 1):
        raise ValueError(
            f"threshold_fraction must be in (0, 1), got {threshold_fraction}."
        )

    smoothed = _smooth_circular_trace(values, smoothing_sigma)
    baseline = float(np.percentile(smoothed, 10.0))
    peak_index = int(np.argmax(smoothed))
    peak_value = float(smoothed[peak_index])
    amplitude = peak_value - baseline
    scale = max(abs(peak_value), abs(baseline), 1.0)

    if amplitude <= np.finfo(np.float64).eps * scale:
        raise ValueError(
            "Folded temporal trace has no detectable fluorescence response."
        )

    baseline_cutoff = float(np.percentile(smoothed, 40.0))
    baseline_values = smoothed[smoothed <= baseline_cutoff]
    if baseline_values.size < 3:
        baseline_values = smoothed

    _, mad = median_absolute_deviation(baseline_values)
    noise = max(
        MAD_TO_SIGMA * mad,
        np.finfo(np.float64).eps * scale,
    )
    snr = amplitude / noise
    signal_score = float(
        np.clip(
            1.0 - np.exp(-snr / 5.0),
            0.0,
            1.0,
        )
    )

    threshold = baseline + threshold_fraction * amplitude
    threshold_onset = None

    for step in range(1, values.size + 1):
        below_index = (peak_index - step) % values.size
        above_index = (below_index + 1) % values.size

        if smoothed[below_index] <= threshold < smoothed[above_index]:
            threshold_onset = above_index
            break

    circular_derivative = smoothed - np.roll(smoothed, 1)
    steepest_rise = int(np.argmax(circular_derivative))
    rise_amplitude = float(circular_derivative[steepest_rise])

    if rise_amplitude > max(
        noise,
        np.finfo(np.float64).eps * scale,
    ):
        onset = steepest_rise
    elif threshold_onset is not None:
        onset = threshold_onset
    else:
        raise ValueError("Folded temporal trace has no detectable rising edge.")

    return int(onset), signal_score


def circular_align(
    data: np.ndarray,
    phase_shift: int,
) -> np.ndarray:
    """
    Circularly shift a SPAD cube along its temporal axis.

    Parameters
    ----------
    data : np.ndarray
        Three-dimensional SPAD data cube with shape (H, W, T).
    phase_shift : int
        Integer shift applied along the temporal axis. A negative value moves
        later gates toward the beginning of the temporal sequence.

    Returns
    -------
    np.ndarray
        Shifted data cube with the same shape and dtype as the input.
    """
    array = np.asarray(data)

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D (H, W, T) cube, got shape {array.shape}.")

    if not isinstance(
        phase_shift,
        (
            int,
            np.integer,
        ),
    ):
        raise TypeError(
            f"phase_shift must be an integer, got {type(phase_shift).__name__}."
        )

    return np.roll(
        array,
        shift=int(phase_shift),
        axis=-1,
    )


def _cycle_similarity(
    trace: np.ndarray,
    period_bins: int,
    phase_shift: int,
) -> float:
    """Return mean pairwise correlation between circularly aligned periods."""
    values = np.asarray(trace, dtype=np.float64)
    aligned = np.roll(values, shift=phase_shift)

    if aligned.size % period_bins != 0:
        raise ValueError(
            f"Temporal length {aligned.size} is not divisible by "
            f"period_bins={period_bins}."
        )

    cycles = aligned.reshape(
        aligned.size // period_bins,
        period_bins,
    )

    if cycles.shape[0] < 2:
        return 1.0

    centered = cycles - np.mean(
        cycles,
        axis=1,
        keepdims=True,
    )
    norms = np.linalg.norm(
        centered,
        axis=1,
    )
    valid = norms > np.finfo(np.float64).eps

    if np.count_nonzero(valid) < 2:
        return 0.0

    normalized = centered[valid] / norms[valid, None]
    correlations = normalized @ normalized.T

    upper = correlations[
        np.triu_indices(
            correlations.shape[0],
            k=1,
        )
    ]

    if upper.size == 0:
        return 1.0

    return float(
        np.clip(
            np.mean(upper),
            0.0,
            1.0,
        )
    )


def analyze_fold_layout(
    data: np.ndarray,
    expected_repeats: int | None = None,
    period_bins: int | None = None,
    phase_shift: int | None = None,
    min_confidence: float = 0.60,
    validate: bool = True,
    search_radius: float = 0.15,
    smoothing_sigma: float = 1.0,
    threshold_fraction: float = 0.10,
    onset_lead_bins: int | None = None,
) -> SpadFoldLayout:
    """
    Detect the periodic layout and circular phase required to fold SPAD data.

    Parameters
    ----------
    data : np.ndarray
        Three-dimensional SPAD data cube with shape (H, W, T).
    expected_repeats : int | None
        Expected number of repeated excitation periods.
    period_bins : int | None
        Explicit period length. When None, the period is detected automatically.
    phase_shift : int | None
        Explicit circular shift. When None, signal onset is detected automatically.
    min_confidence : float
        Minimum accepted combined confidence for automatic folding validation.
    validate : bool
        Whether to reject a detected layout whose confidence is below min_confidence.
    search_radius : float
        Fractional period-search radius around an expected period.
    smoothing_sigma : float
        Circular Gaussian smoothing sigma used only on the detection trace.
    threshold_fraction : float
        Peak-to-baseline fraction used to locate the fluorescence onset.
    onset_lead_bins : int | None
        Number of gates the folded period starts before the detected onset so the
        complete rising edge and pre-pulse baseline are kept at the start of the
        period. None selects 5 % of the period, with a minimum of two gates.
        Ignored when phase_shift is supplied explicitly.

    Returns
    -------
    SpadFoldLayout
        Validated folding layout and detection diagnostics.
    """
    if not (0 <= min_confidence <= 1):
        raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}.")

    if onset_lead_bins is not None and int(onset_lead_bins) < 0:
        raise ValueError(f"onset_lead_bins must be >= 0, got {onset_lead_bins}.")

    trace = build_temporal_trace(data)
    smoothed_trace = _smooth_circular_trace(
        trace,
        smoothing_sigma,
    )
    correlation = _circular_autocorrelation(smoothed_trace)
    original_bins = trace.size

    manual_period = period_bins is not None

    if period_bins is None:
        detected_period, period_score = estimate_period_bins(
            smoothed_trace,
            expected_repeats=expected_repeats,
            search_radius=search_radius,
        )
    else:
        detected_period = int(period_bins)

        if detected_period < 2:
            raise ValueError(f"period_bins must be >= 2, got {detected_period}.")
        if original_bins % detected_period != 0:
            raise ValueError(
                f"period_bins={detected_period} does not divide "
                f"{original_bins} temporal gates."
            )

        period_score = _period_score(
            correlation,
            detected_period,
        )

    repeat_count = original_bins // detected_period

    if expected_repeats is not None and repeat_count != expected_repeats:
        raise ValueError(
            f"Detected {repeat_count} periods but expected_repeats={expected_repeats}."
        )

    phase_trace = make_phase_folded_trace(
        smoothed_trace,
        detected_period,
    )

    detected_origin, signal_score = detect_signal_onset(
        phase_trace,
        smoothing_sigma=smoothing_sigma,
        threshold_fraction=threshold_fraction,
    )

    manual_phase = phase_shift is not None

    if onset_lead_bins is None:
        lead_bins = max(2, round(0.05 * detected_period))
    else:
        lead_bins = int(onset_lead_bins)

    if lead_bins >= detected_period:
        raise ValueError(
            f"onset_lead_bins={lead_bins} must be smaller than the period "
            f"of {detected_period} gates."
        )

    if phase_shift is None:
        phase_origin = (detected_origin - lead_bins) % detected_period
        selected_shift = -phase_origin
    else:
        selected_shift = int(phase_shift)
        phase_origin = (-selected_shift) % detected_period
        lead_bins = (detected_origin - phase_origin) % detected_period

    cycle_similarity = _cycle_similarity(
        smoothed_trace,
        period_bins=detected_period,
        phase_shift=selected_shift,
    )

    confidence = (
        0.45
        * float(
            np.clip(
                period_score,
                0.0,
                1.0,
            )
        )
        + 0.40 * cycle_similarity
        + 0.15 * signal_score
    )

    pulse_positions = tuple(
        (detected_origin + detected_period * repeat) % original_bins
        for repeat in range(repeat_count)
    )

    layout = SpadFoldLayout(
        original_bins=original_bins,
        period_bins=detected_period,
        repeat_count=repeat_count,
        phase_origin=int(phase_origin),
        phase_shift=int(selected_shift),
        onset_index=int(detected_origin),
        onset_lead_bins=int(lead_bins),
        pulse_positions=pulse_positions,
        period_score=float(period_score),
        cycle_similarity=float(cycle_similarity),
        signal_score=float(signal_score),
        confidence=float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        ),
        manual_period=manual_period,
        manual_phase=manual_phase,
    )

    if validate and layout.confidence < min_confidence:
        if layout.manual_period or layout.manual_phase:
            guidance = (
                "The explicit layout still failed periodicity validation; set "
                "fold_validate=False only if this layout is known to be correct."
            )
        else:
            guidance = (
                "Provide period_bins and/or phase_shift explicitly, or disable fold "
                "validation only when the acquisition timing is known."
            )

        raise ValueError(
            "SPAD folding confidence is too low: "
            f"{layout.confidence:.3f} < {min_confidence:.3f}. "
            f"Detected period={layout.period_bins}, "
            f"phase_shift={layout.phase_shift}, "
            f"period_score={layout.period_score:.3f}, "
            f"cycle_similarity={layout.cycle_similarity:.3f}. " + guidance
        )

    return layout


def apply_fold_layout(
    data: np.ndarray,
    layout: SpadFoldLayout,
) -> np.ndarray:
    """
    Circularly align and sum repeated periods using a validated fold layout.

    Parameters
    ----------
    data : np.ndarray
        Three-dimensional SPAD data cube with shape (H, W, T).
    layout : SpadFoldLayout
        Folding layout returned by analyze_fold_layout.

    Returns
    -------
    np.ndarray
        Folded data cube with shape (H, W, period_bins).
    """
    array = np.asarray(data)

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D (H, W, T) cube, got shape {array.shape}.")

    if array.shape[-1] != layout.original_bins:
        raise ValueError(
            f"Fold layout expects {layout.original_bins} temporal gates, "
            f"but data has {array.shape[-1]}."
        )

    if layout.period_bins * layout.repeat_count != layout.original_bins:
        raise ValueError("Fold layout is internally inconsistent.")

    aligned = circular_align(
        array,
        layout.phase_shift,
    )

    reshaped = aligned.reshape(
        *aligned.shape[:-1],
        layout.repeat_count,
        layout.period_bins,
    )

    if np.issubdtype(aligned.dtype, np.integer):
        return np.sum(
            reshaped,
            axis=-2,
            dtype=np.uint64,
        )

    return np.sum(
        reshaped,
        axis=-2,
    )
