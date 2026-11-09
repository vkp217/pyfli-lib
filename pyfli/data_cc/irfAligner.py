"""
Align decay and IRF cubes with threshold-based rise detection and circular or Fourier
shifts.

This module belongs to :mod:`pyfli.data_cc` and is part of PyFLI array preprocessing
helpers for normalization, masking, ROI extraction, and IRF alignment. Public API
includes classes :class:`IRFAligner`.
"""

import warnings
from typing import Any

import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.ndimage import uniform_filter1d

from ..analyticalWorkflow.am_utils import AnalyticalHelpers
from ..cm_stats import goodness_of_fit as _gof
from .norm import Normalization


class IRFAligner:
    """
    Run the irfaligner routine.
    edges per pixel or globally and can shift signals with Fourier or circular methods
    before fitting.

    Parameters
    ----------
    decay : np.ndarray
        Fluorescence decay trace to process.
    irf : np.ndarray
        Instrument response function aligned with the decay trace.
    decay_noise_bins : tuple[int, int]
        (start, end) bin range used to estimate the decay noise floor.
    irf_noise_bins : tuple[int, int]
        (start, end) bin range used to estimate the IRF noise floor.
    laser_period : float
        Laser repetition period in nanoseconds.
    gate_delay : float | None
        Time between gate bins in nanoseconds. Defaults to
        ``laser_period / num_gates`` when not given.
    """

    def __init__(
        self,
        decay: np.ndarray,
        irf: np.ndarray,
        decay_noise_bins: tuple[int, int] = (0, 5),
        irf_noise_bins: tuple[int, int] = (0, 5),
        laser_period: float = 12.5,
        gate_delay: float | None = None,
    ) -> None:
        self.H, self.W, self.T = decay.shape

        self.laser_period = laser_period
        self.gate_delay = (
            gate_delay if gate_delay is not None else laser_period / self.T
        )
        self.freq, self.effective_freq = AnalyticalHelpers(
            laser_period=self.laser_period, gate_delay=self.gate_delay, num_gate=self.T
        ).freq_computation()
        self.dt = self.gate_delay

        d_start, d_end = decay_noise_bins
        i_start, i_end = irf_noise_bins
        d_bg = np.mean(decay[:, :, d_start:d_end], axis=2, keepdims=True)
        i_bg = np.mean(irf[:, :, i_start:i_end], axis=2, keepdims=True)

        self.decay = np.maximum(decay - d_bg, 0)
        self.irf = np.maximum(irf - i_bg, 0)

        _threshold = 0.05
        d_peak, i_peak = np.max(decay), np.max(irf)
        if d_peak > 0 and np.mean(d_bg) > _threshold * d_peak:
            warnings.warn(
                f"Decay noise baseline ({np.mean(d_bg):.3g}) exceeds {_threshold * 100:.0f}% "
                f"of peak ({d_peak:.3g}). decay_noise_bins window may be contaminated — "
                "consider narrowing decay_noise_bins.",
                UserWarning,
                stacklevel=2,
            )
        if i_peak > 0 and np.mean(i_bg) > _threshold * i_peak:
            warnings.warn(
                f"IRF noise baseline ({np.mean(i_bg):.3g}) exceeds {_threshold * 100:.0f}% "
                f"of peak ({i_peak:.3g}). irf_noise_bins window may be contaminated — "
                "consider narrowing irf_noise_bins.",
                UserWarning,
                stacklevel=2,
            )

    def _find_rising_point(self, data: np.ndarray, fraction: float = 0.1) -> Any:
        """
        Finds the fractional bin index where the signal first reaches
        a certain percentage of its peak (the 'toe').

        Pixels with no positive signal or no threshold crossing are marked
        NaN rather than 0, so they can be distinguished from a genuine rise
        at bin 0.
        """
        H, W, T = data.shape
        rising_indices = np.full((H, W), np.nan)

        for i in range(H):
            for j in range(W):
                trace = data[i, j, :]
                peak_val = np.max(trace)
                if peak_val <= 0:
                    continue

                threshold = peak_val * fraction
                # Find the first index that exceeds the threshold
                idx_above = np.where(trace >= threshold)[0]
                if len(idx_above) == 0:
                    continue

                first_idx = idx_above[0]

                # Sub-pixel linear interpolation for the exact 'threshold' crossing
                if first_idx > 0:
                    y2 = trace[first_idx]
                    y1 = trace[first_idx - 1]
                    # Linear interp: x = x1 + (target - y1) * (dx / dy)
                    fractional_part = (threshold - y1) / (y2 - y1 + 1e-12)
                    rising_indices[i, j] = (first_idx - 1) + fractional_part
                else:
                    rising_indices[i, j] = first_idx

        return rising_indices

    def estimate_shift(
        self, fraction: float = 0.1, manual_correction: float = 0.0
    ) -> Any:
        """
        Calculates how much the IRF must move to match the decay's start.

        Pixels where either trace has no detectable rise (see
        :meth:`_find_rising_point`) get a shift of 0 rather than a spurious
        value, since there is no meaningful feature to align.
        ``manual_correction`` is subtracted from every pixel's shift,
        including those fallback pixels.
        """
        t_decay = self._find_rising_point(self.decay, fraction=fraction)
        t_irf = self._find_rising_point(self.irf, fraction=fraction)

        # Shift = Target - Source
        shifts = t_decay - t_irf
        invalid = np.isnan(shifts)
        if np.any(invalid):
            warnings.warn(
                f"{np.count_nonzero(invalid)} pixel(s) had no detectable rise in "
                "the decay or IRF trace; their shift was set to 0.",
                UserWarning,
                stacklevel=2,
            )
        return np.nan_to_num(shifts, nan=0.0) - manual_correction

    def estimate_shift_debiased(
        self,
        low_fraction: float = 0.02,
        smooth_window: int = 3,
        manual_correction: float = 0.0,
    ) -> Any:
        """
        Estimates the per-pixel IRF shift from where each trace departs
        from background, instead of :meth:`estimate_shift`'s 10%-of-peak
        threshold.

        ``decay = irf`` convolved with the fluorescence decay kernel makes
        decay's rising flank intrinsically broader than IRF's, so a
        shared peak-relative threshold is crossed later (relative to the
        true photon arrival time) for decay than for IRF, inflating the
        shift :meth:`estimate_shift` returns. Measuring the crossing much
        closer to the true onset (``low_fraction``) removes most of that
        bias — algebraically this is equivalent to subtracting each
        trace's own rise-width (10%-of-peak bin minus background-departure
        bin) from the naive shift, since that width term cancels out.

        A low threshold on raw photon-count data fires on background
        shot noise rather than the real pulse, so each trace is smoothed
        first with a ``smooth_window``-bin moving average to suppress
        that before the threshold is applied.

        ``manual_correction`` is subtracted from every pixel's shift,
        including fallback pixels with no detectable rise.
        """
        smooth_decay = uniform_filter1d(
            self.decay, size=smooth_window, axis=2, mode="nearest"
        )
        smooth_irf = uniform_filter1d(
            self.irf, size=smooth_window, axis=2, mode="nearest"
        )

        t_decay = self._find_rising_point(smooth_decay, fraction=low_fraction)
        t_irf = self._find_rising_point(smooth_irf, fraction=low_fraction)

        shifts = t_decay - t_irf
        invalid = np.isnan(shifts)
        if np.any(invalid):
            warnings.warn(
                f"{np.count_nonzero(invalid)} pixel(s) had no detectable rise in "
                "the smoothed decay or IRF trace; their shift was set to 0.",
                UserWarning,
                stacklevel=2,
            )
        return np.nan_to_num(shifts, nan=0.0) - manual_correction

    def estimate_shift_rmse(
        self,
        bin_window: tuple[int, int] | None = None,
        left: int = 10,
        right: int = 7,
        max_shift: int = 20,
        fraction: float = 0.1,
        manual_correction: float = 0.0,
    ) -> Any:
        """
        Refines the per-pixel IRF shift by minimizing RMSE between the decay
        trace and a candidate-shifted, amplitude-matched IRF within a local
        comparison window.

        If ``bin_window`` (start, end) is given, that range is used for
        every pixel. Otherwise the window is centered per-pixel on the
        decay's own rising point (from :meth:`_find_rising_point`),
        spanning `left` bins before it and `right` bins after, shifted
        inward at the trace edges rather than wrapped. Integer shifts in
        ``[-max_shift, max_shift]`` are searched and refined to sub-bin
        precision via a parabolic fit around the best candidate. Pixels
        with no detectable decay rise fall back to a shift of 0.
        ``manual_correction`` is subtracted from every pixel's shift,
        including those fallback pixels.
        """
        scaled_irf = Normalization(self.irf).norm_scale(self.decay)

        if bin_window is not None:
            start_bin, end_bin = bin_window
            width = end_bin - start_bin
            start = np.full((self.H, self.W), start_bin)
            valid = np.ones((self.H, self.W), dtype=bool)
        else:
            t_decay = self._find_rising_point(self.decay, fraction=fraction)
            valid = ~np.isnan(t_decay)
            center = np.round(np.nan_to_num(t_decay)).astype(int)

            width = left + right + 1
            start = center - left
            # Shift the window inward at the trace edges rather than wrap.
            start = start + np.maximum(0, -start)
            start = start - np.maximum(0, (start + width - 1) - (self.T - 1))

            if np.any(~valid):
                warnings.warn(
                    f"{np.count_nonzero(~valid)} pixel(s) had no detectable decay "
                    "rise; their RMSE-based shift was set to 0.",
                    UserWarning,
                    stacklevel=2,
                )

        win_idx = start[:, :, None] + np.arange(width)[None, None, :]
        decay_win = np.take_along_axis(self.decay, win_idx, axis=2)

        shift_candidates = np.arange(-max_shift, max_shift + 1)
        rmse_stack = np.empty((self.H, self.W, len(shift_candidates)))
        for k, s in enumerate(shift_candidates):
            shifted_idx = (win_idx - s) % self.T
            irf_win = np.take_along_axis(scaled_irf, shifted_idx, axis=2)
            rmse_stack[:, :, k] = _gof.rmse(irf_win - decay_win, axis=2)

        best_k = np.argmin(rmse_stack, axis=2)
        best_shift = shift_candidates[best_k].astype(float)

        # Sub-bin refinement: parabolic fit around the best integer candidate.
        interior = (best_k > 0) & (best_k < len(shift_candidates) - 1)
        k_c = np.clip(best_k, 1, len(shift_candidates) - 2)
        y0 = np.take_along_axis(rmse_stack, (k_c - 1)[:, :, None], axis=2)[:, :, 0]
        y1 = np.take_along_axis(rmse_stack, k_c[:, :, None], axis=2)[:, :, 0]
        y2 = np.take_along_axis(rmse_stack, (k_c + 1)[:, :, None], axis=2)[:, :, 0]
        denom = y0 - 2 * y1 + y2
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.where(denom != 0, 0.5 * (y0 - y2) / denom, 0.0)
        best_shift = np.where(
            interior, best_shift + np.clip(delta, -1.0, 1.0), best_shift
        )

        return np.where(valid, best_shift, 0.0) - manual_correction

    def estimate_shift_rmse_pixel(
        self,
        x: int,
        y: int,
        bin_window: tuple[int, int] | None = None,
        left: int = 10,
        right: int = 7,
        max_shift: int = 20,
        fraction: float = 0.1,
        manual_correction: float = 0.0,
    ) -> dict[str, Any]:
        """
        Runs the :meth:`estimate_shift_rmse` search for a single pixel
        (x, y) and returns the full RMSE-vs-shift curve plus the resulting
        aligned IRF trace — useful to inspect why a particular shift was
        selected, and how well it lines up with decay, without processing
        the full cube.

        ``manual_correction`` is subtracted from the RMSE-refined shift
        before it is used to build "shifted_irf", matching
        :meth:`estimate_shift_rmse`.

        Returns
        -------
        dict
            "shift_candidates" : integer shifts that were tested.
            "rmse" : RMSE at each candidate shift, same order.
            "best_shift" : final shift for this pixel (sub-bin refined,
                minus ``manual_correction``), matching what
                :meth:`estimate_shift_rmse` returns for it.
            "window" : (start, end) bin range used for the comparison.
            "decay_trace" : the pixel's full decay trace, unmodified.
            "scaled_irf_trace" : the pixel's IRF trace, amplitude-matched
                to decay via :meth:`~pyfli.data_cc.norm.Normalization.norm_scale`,
                before any shift.
            "shifted_irf" : "scaled_irf_trace" shifted by "best_shift"
                (Fourier/fractional shift, full trace) — plot this
                against "decay_trace" to see the resulting alignment.
        """
        decay_trace = self.decay[x, y, :]
        irf_trace = self.irf[x, y, :]
        scaled_irf_trace = Normalization(irf_trace).norm_scale(decay_trace)

        if bin_window is not None:
            start, end = bin_window
            width = end - start
        else:
            t_decay = self._find_rising_point(
                self.decay[x : x + 1, y : y + 1, :], fraction=fraction
            )[0, 0]
            if np.isnan(t_decay):
                raise ValueError(
                    f"Pixel ({x}, {y}) has no detectable decay rise; "
                    "pass an explicit bin_window instead."
                )
            width = left + right + 1
            start = round(t_decay) - left
            start = start + max(0, -start)
            start = start - max(0, (start + width - 1) - (self.T - 1))

        win_idx = start + np.arange(width)
        decay_win = decay_trace[win_idx]

        shift_candidates = np.arange(-max_shift, max_shift + 1)
        rmse = np.empty(len(shift_candidates))
        for k, s in enumerate(shift_candidates):
            shifted_idx = (win_idx - s) % self.T
            rmse[k] = _gof.rmse(scaled_irf_trace[shifted_idx] - decay_win, axis=None)

        best_k = int(np.argmin(rmse))
        best_shift = float(shift_candidates[best_k])
        if 0 < best_k < len(shift_candidates) - 1:
            y0, y1_, y2_ = rmse[best_k - 1], rmse[best_k], rmse[best_k + 1]
            denom = y0 - 2 * y1_ + y2_
            if denom != 0:
                best_shift += float(np.clip(0.5 * (y0 - y2_) / denom, -1.0, 1.0))
        best_shift -= manual_correction

        freqs = fftfreq(self.T)
        phase = np.exp(-2j * np.pi * freqs * best_shift)
        shifted_irf = np.maximum(np.real(ifft(fft(scaled_irf_trace) * phase)), 0)

        return {
            "shift_candidates": shift_candidates,
            "rmse": rmse,
            "best_shift": best_shift,
            "window": (int(start), int(start + width)),
            "decay_trace": decay_trace,
            "scaled_irf_trace": scaled_irf_trace,
            "shifted_irf": shifted_irf,
        }

    def apply_fourier_shift(self, shifts: np.ndarray) -> Any:
        """
        Apply fourier shift.

        Parameters
        ----------
        shifts : np.ndarray
            Per-pixel temporal shifts applied to the IRF cube.

        Returns
        -------
        Any
            Object produced by apply fourier shift.
        """
        freqs = fftfreq(self.T)
        # Apply the fractional shift in the frequency domain
        phase = np.exp(-2j * np.pi * freqs[None, None, :] * shifts[:, :, None])

        IRF_fft = fft(self.irf, axis=2)
        aligned_irf = np.real(ifft(IRF_fft * phase, axis=2))

        return np.maximum(aligned_irf, 0)

    def apply_circular_shift(self, shifts: np.ndarray) -> np.ndarray:
        """
        Applies a linear circular shift by rounding fractional shifts
        to the nearest integer and rolling the array.
        """
        aligned_irf = np.zeros_like(self.irf)
        # Round shifts to nearest integer for np.roll
        int_shifts = np.round(shifts).astype(int)

        for i in range(self.H):
            for j in range(self.W):
                # np.roll performs circular shifting
                aligned_irf[i, j, :] = np.roll(self.irf[i, j, :], int_shifts[i, j])

        return aligned_irf

    def align(
        self,
        fraction: float = 0.1,
        method: str = "fourier",
        manual_correction: float = 0.0,
    ) -> tuple[Any, ...]:
        """
        Aligns the IRF using the specified method.
        """
        shifts = self.estimate_shift(
            fraction=fraction, manual_correction=manual_correction
        )
        if method == "circular":
            return self.apply_circular_shift(shifts), shifts
        else:
            return self.apply_fourier_shift(shifts), shifts

    def align_pixel(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fraction: float = 0.1,
        method: str = "fourier",
        manual_correction: float = 0.0,
    ) -> Any:
        """
        Aligns the IRF for a single pixel (x, y) — useful for quick inspection
        of the alignment at a specific spatial location without processing the
        full data cube.
        """
        decay_trace = self.decay[x, y, :]
        irf_trace = self.irf[x, y, :]

        def _rising_point(trace: np.ndarray) -> Any:
            """
            Run the rising point routine.

            Parameters
            ----------
            trace : np.ndarray
                One-dimensional decay trace being processed.

            Returns
            -------
            Any
                Object produced by rising point.
            """
            peak_val = np.max(trace)
            if peak_val <= 0:
                return np.nan
            threshold = peak_val * fraction
            idx_above = np.where(trace >= threshold)[0]
            if len(idx_above) == 0:
                return np.nan
            first_idx = idx_above[0]
            if first_idx > 0:
                v2, v1 = trace[first_idx], trace[first_idx - 1]
                return (first_idx - 1) + (threshold - v1) / (v2 - v1 + 1e-12)
            return float(first_idx)

        raw_shift = _rising_point(decay_trace) - _rising_point(irf_trace)
        if np.isnan(raw_shift):
            warnings.warn(
                f"No detectable rise in the decay or IRF trace at pixel ({x}, {y}); "
                "shift was set to 0.",
                UserWarning,
                stacklevel=2,
            )
            raw_shift = 0.0
        shift = raw_shift - manual_correction

        if method == "circular":
            aligned_irf = np.roll(irf_trace, round(shift))
        else:
            freqs = fftfreq(self.T)
            phase = np.exp(-2j * np.pi * freqs * shift)
            aligned_irf = np.maximum(np.real(ifft(fft(irf_trace) * phase)), 0)

        return aligned_irf, shift
