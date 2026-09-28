"""
Reconstruct fit curves and goodness-of-fit maps from pre-estimated FLI
lifetime parameter maps.

This module belongs to :mod:`pyfli.reconstruction` and sits alongside
:mod:`pyfli.reconstruction.decay_reconstruction`: it drives
:class:`ParamToDecay` to turn a dictionary of already-known
lifetime maps (e.g. F-BI output, or a posterior-sample parameter combination)
back into a decay cube and its fit-quality maps, packaged in the same
structure as :class:`pyfli.solver.FLICPUProcessor`'s output so it drops
straight into Plotter / DataViewer. Public API includes class
:class:`DetailedRecon`.
"""

from typing import Any

import numpy as np

from pyfli import logging

from ..data_vnp.mono_bi_classifier import MonoBiClassifier
from ..solver.shared_metrics import pearson_chi_square, reduced_poisson_deviance
from .decay_reconstruction import ParamToDecay


class DetailedRecon:
    """
    Reconstruct fit/residual/goodness-of-fit maps from pre-estimated FLI
    lifetime parameter maps, for a fixed acquisition setup (frequency, IRF,
    measured decay).

    Three operations, all returning the same
    ``{"name", "method", "results": {"maps", "error_maps", "TR_maps"}}``
    shape (or, for :meth:`split_mono_bi`, that shape twice):

    - :meth:`reconstruct` -- direct reconstruction for either model_type, no
      classification involved. This is the general-purpose operation; the
      other two are bi-exponential-only.
    - :meth:`split_mono_bi` -- classifies bi-exponential ``params`` per pixel
      via :class:`MonoBiClassifier` and returns *two* separate results: the
      mono-classified pixel subset reconstructed as mono-exponential (using
      each such pixel's dominant/coincidence lifetime), and the
      bi-classified subset (the rest) reconstructed with the full
      bi-exponential model. Each result is NaN'd outside its own subset.
    - :meth:`collapse_to_mono` -- collapses *every* pixel (mono- and
      bi-classified alike) to a single effective lifetime and returns one
      whole-image mono-exponential reconstruction.

    Parameters
    ----------
    freq_acq : float
        Acquisition frequency freq[1] (MHz).
    binned_irf : np.ndarray
        IRF, shape (bins,) or (H, W, bins). A 1-D IRF is broadcast across
        all pixels; normalized to sum to 1 per pixel before convolving.
    binned_decay : np.ndarray | None
        Measured decay histogram per pixel, shape (H, W, bins), shared by
        every :meth:`reconstruct` call unless overridden per-call. When
        omitted (here or per-call), decay-dependent outputs (photon count,
        residuals, chi², R²) reduce to zero.
    alpha_upper, alpha_lower, tau_tol : float
        :class:`MonoBiClassifier` thresholds used by :meth:`split_mono_bi`
        and :meth:`collapse_to_mono`.
    eps : float
        Numerical floor for clip / safe division.
    """

    def __init__(
        self,
        freq_acq: float,
        binned_irf: np.ndarray,
        binned_decay: np.ndarray | None = None,
        alpha_upper: float = 0.95,
        alpha_lower: float = 0.05,
        tau_tol: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        binned_irf = np.asarray(binned_irf, dtype=np.float32)
        if binned_irf.ndim not in (1, 3):
            raise ValueError(
                f"binned_irf must be 1-D (bins,) or 3-D (H,W,bins); got shape "
                f"{binned_irf.shape}"
            )
        self.freq_acq = freq_acq
        self.binned_irf = binned_irf
        self.binned_decay = (
            None if binned_decay is None else np.asarray(binned_decay, dtype=np.float32)
        )
        self.alpha_upper = alpha_upper
        self.alpha_lower = alpha_lower
        self.tau_tol = tau_tol
        self.eps = eps

        self._recon = {
            model_type: ParamToDecay(model_type, freq_acq, irf=binned_irf)
            for model_type in ("mono-exponential", "bi-exponential")
        }

    # ------------------------------------------------------------------
    # Shared internals
    # ------------------------------------------------------------------
    @staticmethod
    def _get_map(
        params: dict[str, np.ndarray],
        key: str,
        ref_shape: tuple[int, int],
        defaults: dict[str, float],
    ) -> np.ndarray:
        """Same defaulting as ParamToDecay._get_map, kept
        local rather than reaching into that underscore-private method.
        ``defaults`` is looked up lazily (only when ``key`` is actually
        missing from ``params``) since required keys -- e.g. "tau_map" --
        have no entry in PARAM_MAP_DEFAULTS at all."""
        if key in params:
            return np.asarray(params[key], dtype=np.float32)
        return np.full(ref_shape, defaults[key], dtype=np.float32)

    @staticmethod
    def _apply_bool_mask(result: dict[str, Any], bool_mask: np.ndarray) -> None:
        """
        NaN-out every pixel where ``bool_mask`` is False, in every (H, W) or
        (H, W, ...) array under ``result["results"]["maps"]``, ``["TR_maps"]``,
        and ``["error_maps"]`` -- mirrors
        :meth:`pyfli.bayes_utils.param_combinations.ParamSelector.
        _apply_bool_mask` so excluded pixels are unambiguous downstream
        instead of looking like an ordinary (if poor) fit.
        """
        bool_mask = np.asarray(bool_mask, dtype=bool)
        results = result["results"]

        for group_key in ("maps", "TR_maps"):
            group = results.get(group_key)
            if not group:
                continue
            for key, arr in group.items():
                if not isinstance(arr, np.ndarray) or arr.shape[:2] != bool_mask.shape:
                    continue
                arr = arr.astype(np.float32, copy=True)
                arr[~bool_mask] = np.nan
                group[key] = arr

        error_maps = results.get("error_maps")
        if (
            isinstance(error_maps, np.ndarray)
            and error_maps.shape[:2] == bool_mask.shape
        ):
            error_maps = error_maps.astype(np.float32, copy=True)
            error_maps[~bool_mask] = np.nan
            results["error_maps"] = error_maps

    def _dominant_tau_map(
        self, tau1_map: np.ndarray, tau2_map: np.ndarray, alpha1_map: np.ndarray
    ) -> np.ndarray:
        """Per pixel, the tau this pixel *would* have if treated as mono:
        the dominant component's tau (by alpha1 threshold), or tau1 in the
        tau1≈tau2 coincidence case. Meaningful only where a pixel is
        actually mono-classified -- callers apply that via mono_mask."""
        return np.where(
            alpha1_map > self.alpha_upper,
            tau1_map,
            np.where(alpha1_map < self.alpha_lower, tau2_map, tau1_map),
        )

    def _classify(
        self,
        tau1_map: np.ndarray,
        tau2_map: np.ndarray,
        alpha1_map: np.ndarray,
        bool_mask: np.ndarray,
        data_name: str,
        display: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run MonoBiClassifier; return (mono_mask, bi_mask), both (H, W)
        bool and ROI-restricted to bool_mask."""
        bool_mask = np.asarray(bool_mask, dtype=bool)
        clf = MonoBiClassifier(
            bool_mask,
            names=[data_name],
            alpha_upper=self.alpha_upper,
            alpha_lower=self.alpha_lower,
            tau_tol=self.tau_tol,
            coord=None,
        )
        dataset = {
            "alpha1_map": alpha1_map,
            "tau1_map": tau1_map,
            "tau2_map": tau2_map,
        }
        classes = clf.classify([dataset], display=display)
        mono_mask = classes[0]["mono_mask"]  # (H, W) bool, ROI-restricted
        bi_mask = (~mono_mask) & bool_mask
        return mono_mask, bi_mask

    # ------------------------------------------------------------------
    # Operation 1: direct reconstruction (either model_type)
    # ------------------------------------------------------------------
    def reconstruct(
        self,
        params: dict[str, np.ndarray],
        model_type: str,
        data_name: str = "F-BI",
        n_params: int | None = None,
        binned_decay: np.ndarray | None = None,
        log_summary: bool = True,
    ) -> dict[Any, Any]:
        """
        Reconstruct fit curves + goodness-of-fit maps from pre-estimated
        lifetime parameter maps (e.g. F-BI output), packaged in the same
        structure as FLICPUProcessor.process_image so it drops straight into
        Plotter / DataViewer.

        ``params`` takes exactly the same shape as
        :class:`ParamToDecay`'s own ``params`` argument: a
        dict keyed by :attr:`ParamToDecay.PARAM_MAP_KEYS`
        ``[model_type]`` -- ``{"tau_map"}`` (plus optional
        ``"photon_count_map"``, ``"v_shift_map"``, ``"h_shift_map"``) for
        ``"mono-exponential"``, or ``{"alpha1_map", "tau1_map", "tau2_map"}``
        (plus the same three optional keys) for ``"bi-exponential"``. Missing
        optional keys default via
        :attr:`ParamToDecay.PARAM_MAP_DEFAULTS` (1.0, 0.0,
        0.0 respectively); missing required keys raise ``KeyError``.

        ``"photon_count_map"`` is accepted for schema parity but never
        changes the result: the model is always rescaled to match the
        measured decay's total (see :meth:`ParamToDecay.
        rescale_fit_to_measured_totals`), which first normalizes the model to
        a PDF -- so any literal amplitude supplied here cancels out exactly.
        ``"h_shift_map"`` is honored directly (it shifts the kernel's time
        axis before convolution, same as every other reconstruction path).
        ``"v_shift_map"`` is honored as an additive per-bin baseline: it's
        subtracted from the measured decay before total-matching the peak
        shape (so the shape-only rescale isn't skewed by the baseline), then
        added back -- mirroring :meth:`ParamToDecay.
        _build_fit_map_vectorized`'s "add v_shift after convolution"
        convention, adapted for this method's rescale-to-total amplitude
        handling. Both default to 0.0, so omitting them reproduces the
        baseline-free result exactly.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Parameter maps for ``model_type``, each (H, W). See above for the
            required/optional keys per model_type.
        model_type : str
            ``"mono-exponential"`` or ``"bi-exponential"``.
        data_name : str
            Dataset name recorded in the returned result dict.
        n_params : int | None
            Free-parameter count for the reduced-chi2 dof. Defaults to
            model_type's full parameter count -- 6 (photon_count, alpha1,
            tau1, tau2, v_shift, h_shift) for "bi-exponential", 4
            (photon_count, tau, v_shift, h_shift) for "mono-exponential" --
            matching the dof convention BaseFLIFitter/MLEFitter/
            FLIGPUProcessor and ParamToDecay
            (PARAM_MAP_KEYS) use for the same model family.
        binned_decay : np.ndarray | None
            Overrides ``self.binned_decay`` for this call only, e.g. to score
            against a different decay cube than the one this instance was
            built with. When both are None, decay-dependent outputs reduce
            to zero.

        Returns
        -------
        dict[Any, Any]
            ``{'name', 'results': {'maps', 'error_maps', 'TR_maps'}}``
        """
        if model_type not in self._recon:
            raise ValueError(
                f"Unknown model_type: {model_type!r}; expected one of "
                f"{tuple(self._recon)}"
            )
        recon = self._recon[model_type]

        missing = [k for k in recon.required_keys if k not in params]
        if missing:
            raise KeyError(
                f"params is missing required maps for model_type={model_type!r}: "
                f"{missing}"
            )

        if n_params is None:
            n_params = 6 if model_type == "bi-exponential" else 4

        ref_shape = np.asarray(params[recon.required_keys[0]]).shape
        H, W = ref_shape
        bins = self.binned_irf.shape[-1]

        decay = self.binned_decay if binned_decay is None else binned_decay
        if decay is None:
            binned_decay_arr = np.zeros((H, W, bins), dtype=np.float32)
        else:
            binned_decay_arr = np.asarray(decay, dtype=np.float32)

        def _get(key: str) -> np.ndarray:
            return self._get_map(params, key, ref_shape, recon.PARAM_MAP_DEFAULTS)

        unit_params = {k: _get(k) for k in recon.param_keys}
        unit = recon.reconstruct_unit_amplitude(unit_params)
        sdf, convolved_fit = unit["kernel_map"], unit["convolved_map"]

        v_shift_map = _get("v_shift_map")[..., None]
        decay_minus_baseline = binned_decay_arr - v_shift_map
        scaled_peak = recon.rescale_fit_to_measured_totals(
            convolved_fit, decay_minus_baseline, eps=self.eps
        )
        scaled_fit = scaled_peak + v_shift_map

        photon_count = np.sum(binned_decay_arr, axis=-1)
        fit_sum = np.sum(convolved_fit, axis=-1)
        photon_count_adjusted = np.sum(decay_minus_baseline, axis=-1)
        s_reported = np.zeros_like(photon_count, dtype=np.float32)
        np.divide(
            photon_count_adjusted, fit_sum, out=s_reported, where=fit_sum > self.eps
        )

        dof = max(bins - n_params, 1)
        residuals = binned_decay_arr - scaled_fit
        chi_sq_raw, reduced_chi2_map = reduced_poisson_deviance(
            scaled_fit, binned_decay_arr, n_params
        )
        pearson_raw = pearson_chi_square(scaled_fit, binned_decay_arr)
        pearson_stats = {
            "pearson_chi2_map": pearson_raw.astype(np.float32),
            "pearson_reduced_chi2_map": (pearson_raw / dof).astype(np.float32),
        }

        ss_res = np.sum(residuals**2, axis=-1)
        ss_tot = np.sum(
            (binned_decay_arr - np.mean(binned_decay_arr, axis=-1, keepdims=True)) ** 2,
            axis=-1,
        )
        r2_map = np.ones((H, W), dtype=np.float32)
        np.divide(ss_res, ss_tot, out=r2_map, where=ss_tot > self.eps)
        r2_map = 1.0 - r2_map
        rmse_map = np.sqrt(np.mean(residuals**2, axis=-1))

        v_shift_out = v_shift_map[..., 0]
        h_shift_out = _get("h_shift_map")
        health = (photon_count > 0).astype(np.float32)

        if model_type == "mono-exponential":
            param_maps = {
                "photon_count_map": s_reported,
                "tau_map": np.asarray(params["tau_map"], dtype=np.float32),
                "v_shift_map": v_shift_out,
                "h_shift_map": h_shift_out,
                "R2_map": r2_map.astype(np.float32),
                "chi2_map": chi_sq_raw.astype(np.float32),
                "reduced_chi2_map": reduced_chi2_map.astype(np.float32),
                **pearson_stats,
                "rmse_map": rmse_map.astype(np.float32),
                "convergence_map": health.copy(),
                "pixel_health_map": health,
            }
            # amp, tau, v_shift, h_shift -- matches ParamToDecay's
            # PARAM_MAP_KEYS["mono-exponential"] (also this call's own n_params
            # default above), so error_maps.shape[-1] lines up with every other
            # backend for this model_type even though no uncertainties are
            # estimated here (all zeros).
            error_maps = np.zeros((H, W, 4), dtype=np.float32)
        else:
            tau1_f = np.asarray(params["tau1_map"], dtype=np.float32)
            tau2_f = np.asarray(params["tau2_map"], dtype=np.float32)
            alpha1_f = np.asarray(params["alpha1_map"], dtype=np.float32)
            ratio = np.divide(
                tau1_f,
                tau2_f,
                out=np.zeros_like(tau1_f, dtype=np.float32),
                where=(tau2_f > 0),
            )
            param_maps = {
                "photon_count_map": s_reported,
                "alpha1_map": alpha1_f,
                "tau1_map": tau1_f,
                "tau2_map": tau2_f,
                "tau_mean_map": (alpha1_f * tau1_f + (1.0 - alpha1_f) * tau2_f),
                "fret_efficiency_map": np.where(tau2_f > 0, 1.0 - ratio, 0.0).astype(
                    np.float32
                ),
                "v_shift_map": v_shift_out,
                "h_shift_map": h_shift_out,
                "R2_map": r2_map.astype(np.float32),
                "chi2_map": chi_sq_raw.astype(np.float32),
                "reduced_chi2_map": reduced_chi2_map.astype(np.float32),
                **pearson_stats,
                "rmse_map": rmse_map.astype(np.float32),
                "convergence_map": health.copy(),
                "pixel_health_map": health,
            }
            # amp, alpha1, tau1, tau2, v_shift, h_shift -- matches
            # ParamToDecay's PARAM_MAP_KEYS["bi-exponential"]
            # (also this call's own n_params default above).
            error_maps = np.zeros((H, W, 6), dtype=np.float32)

        tr_maps = {
            "fit_map": scaled_fit.astype(np.float32),
            "residual_map": residuals.astype(np.float32),
            "sdf_map": sdf.astype(np.float32),
            "convolved_map": convolved_fit.astype(np.float32),
        }

        if log_summary:
            mask = photon_count > 0
            n_active = int(np.count_nonzero(mask))
            if n_active:
                logging.info(
                    f"{data_name} ({model_type}) fit over {n_active} active pixel(s):\n"
                    f"  mean reduced chi2 = {float(np.mean(reduced_chi2_map[mask])):.4f}\n"
                    f"  mean R2           = {float(np.mean(r2_map[mask])):.4f}\n"
                    f"  mean RMSE         = {float(np.mean(rmse_map[mask])):.4f}\n"
                    f"  mean chi2         = {float(np.mean(chi_sq_raw[mask])):.2f}"
                )
            else:
                logging.info(f"{data_name} ({model_type}): no active pixels to score")

        return {
            "name": data_name,
            "method": "DirectCompute",
            "results": {
                "maps": param_maps,
                "error_maps": error_maps,
                "TR_maps": tr_maps,
            },
        }

    # ------------------------------------------------------------------
    # Operation 2: bi-exponential only -- classify, then reconstruct each
    # subset with its own model.
    # ------------------------------------------------------------------
    def split_mono_bi(
        self,
        params: dict[str, np.ndarray],
        bool_mask: np.ndarray,
        data_name: str = "F-BI",
        n_params: int | None = None,
        display: bool = True,
    ) -> dict[str, Any]:
        """
        Classify bi-exponential ``params`` (``{"tau1_map", "tau2_map",
        "alpha1_map"}``) per pixel via :class:`MonoBiClassifier`, and
        reconstruct each subset with the model that actually applies to it:
        mono-classified pixels get a mono-exponential reconstruction (using
        each pixel's dominant/coincidence lifetime), and the remaining
        (bi-classified) pixels get the full bi-exponential reconstruction.
        Each returned result is NaN'd outside its own pixel subset (see
        :meth:`_apply_bool_mask`), so the two results can be recombined or
        inspected independently without the other subset's placeholder
        values being mistaken for real fits.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            ``{"tau1_map", "tau2_map", "alpha1_map"}``, each (H, W).
        bool_mask : np.ndarray
            (H, W) boolean mask selecting which pixels to classify/reconstruct
            at all (e.g. ``photon_count > 0``, or a real ROI mask).
        data_name : str
            Base dataset name; the two results are recorded as
            ``f"{data_name}_mono"`` and ``f"{data_name}_bi"``.
        n_params : int | None
            Forwarded to :meth:`reconstruct` for both subsets.
        display : bool
            Whether :class:`MonoBiClassifier` renders its mono/bi
            classification maps via DataViewer as a side effect.

        Returns
        -------
        dict[str, Any]
            ``{"mono": <reconstruct() result>, "bi": <reconstruct() result>,
            "mono_mask": (H, W) bool, "bi_mask": (H, W) bool}``.
        """
        tau1 = np.asarray(params["tau1_map"], dtype=np.float32)
        tau2 = np.asarray(params["tau2_map"], dtype=np.float32)
        alpha1 = np.asarray(params["alpha1_map"], dtype=np.float32)

        mono_mask, bi_mask = self._classify(
            tau1, tau2, alpha1, bool_mask, data_name, display
        )

        tau_eff = self._dominant_tau_map(tau1, tau2, alpha1)
        mono_result = self.reconstruct(
            {"tau_map": tau_eff},
            "mono-exponential",
            data_name=f"{data_name}_mono",
            n_params=n_params,
        )
        self._apply_bool_mask(mono_result, mono_mask)

        bi_result = self.reconstruct(
            {"tau1_map": tau1, "tau2_map": tau2, "alpha1_map": alpha1},
            "bi-exponential",
            data_name=f"{data_name}_bi",
            n_params=n_params,
        )
        self._apply_bool_mask(bi_result, bi_mask)

        return {
            "mono": mono_result,
            "bi": bi_result,
            "mono_mask": mono_mask,
            "bi_mask": bi_mask,
        }

    # ------------------------------------------------------------------
    # Operation 3: bi-exponential only -- collapse every pixel to mono,
    # single whole-image reconstruction.
    # ------------------------------------------------------------------
    def collapse_to_mono(
        self,
        params: dict[str, np.ndarray],
        bool_mask: np.ndarray,
        data_name: str = "F-BI",
        n_params: int | None = None,
        display: bool = True,
    ) -> dict[Any, Any]:
        """
        Collapse bi-exponential ``params`` (``{"tau1_map", "tau2_map",
        "alpha1_map"}``) to a single per-pixel effective lifetime via
        :class:`MonoBiClassifier` -- mono-classified pixels get their
        dominant/coincidence lifetime, bi-classified pixels get the
        amplitude-weighted mean ``alpha1*tau1 + (1-alpha1)*tau2`` -- then run
        one whole-image mono-exponential :meth:`reconstruct` on the result.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            ``{"tau1_map", "tau2_map", "alpha1_map"}``, each (H, W).
        bool_mask : np.ndarray
            (H, W) boolean mask selecting which pixels to classify/collapse.
            Pixels outside it are NaN'd in the returned result (see
            :meth:`_apply_bool_mask`).
        data_name : str
            Dataset name recorded in the returned result dict.
        n_params : int | None
            Forwarded to :meth:`reconstruct`.
        display : bool
            Whether :class:`MonoBiClassifier` renders its mono/bi
            classification maps via DataViewer as a side effect.

        Returns
        -------
        dict[Any, Any]
            :meth:`reconstruct`'s return shape, for the whole-image collapsed
            mono-exponential reconstruction.
        """
        tau1 = np.asarray(params["tau1_map"], dtype=np.float32)
        tau2 = np.asarray(params["tau2_map"], dtype=np.float32)
        alpha1 = np.asarray(params["alpha1_map"], dtype=np.float32)
        bool_mask = np.asarray(bool_mask, dtype=bool)

        mono_mask, _ = self._classify(tau1, tau2, alpha1, bool_mask, data_name, display)

        tau_mono = self._dominant_tau_map(tau1, tau2, alpha1)
        tau_bi = alpha1 * tau1 + (1.0 - alpha1) * tau2
        tau_eff = np.where(mono_mask, tau_mono, tau_bi).astype(np.float32)

        result = self.reconstruct(
            {"tau_map": tau_eff},
            "mono-exponential",
            data_name=data_name,
            n_params=n_params,
        )
        self._apply_bool_mask(result, bool_mask)
        return result
