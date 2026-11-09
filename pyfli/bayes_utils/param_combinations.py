from collections.abc import Callable
from typing import ClassVar

import numpy as np
from tqdm.auto import tqdm

from pyfli.reconstruction import DetailedRecon, ParamToDecay


class ParamSelector:
    """
    Evaluate a stack of posterior-sample parameter combinations against
    measured decay data, and select the best-fitting combination per pixel
    by a chosen goodness-of-fit metric.

    Parameters
    ----------
    freq_acq : float
        Acquisition frequency (e.g. 80 MHz), passed straight through to
        compute_detailed_results' freq_acq argument.
    irf, decay : np.ndarray
        Passed through to compute_detailed_results unchanged for every
        sample and for the final best-combination re-fit.
    bool_mask : np.ndarray | None
        Optional (H, W) boolean mask -- e.g. the same ROI mask passed to
        BiPipeline.run_inference. Not used during sample evaluation/selection
        itself (compute_detailed_results' own pixel_health_map is derived
        from decay, not this mask, so a pixel with background counts outside
        the real ROI can otherwise look "healthy" even though its params are
        meaningless placeholders). Stored as the default for
        compute_best_model_fit_result's own bool_mask argument, which uses it
        to NaN-out excluded pixels in the final result's output maps.
    model_type : str
        "bi-exponential" or "mono-exponential".
    backend : str
        Which implementation reconstructs each sample's fit/residual/
        goodness-of-fit maps: ``"compute_detailed_results"`` (default, the
        rescale-to-measured-totals implementation in
        :mod:`pyfli.reconstruction.detailed_results`)
        or ``"reconstructor"`` (:class:`pyfli.reconstruction.ParamToDecay`'s
        vectorized path). Both return the same
        ``{"name", "method", "results": {"maps", "error_maps", "TR_maps"}}``
        shape, so switching backends doesn't change any downstream code.
    shift_method : str
        Forward model, ``"zero_pad"`` or ``"circular"``, as used for the fit (see
        :class:`pyfli.reconstruction.ParamToDecay`).
    h_shift, v_shift : float | np.ndarray | None
        Decay onset (ns) and per-bin background of the forward model, a scalar or an
        (H, W) map (e.g. a solver fit's ``h_shift_map``/``v_shift_map``). ``h_shift``/
        ``v_shift`` keys in ``output_combination`` take precedence. ``None`` means 0.
    excitation, gate_width
        Forward-model options as used for the fit (see
        :class:`pyfli.reconstruction.ParamToDecay`).
    """

    #: Optional nuisance keys of ``output_combination`` (no "_map" suffix).
    SHIFT_KEYS: tuple[str, ...] = ("h_shift", "v_shift")

    #: Registry of {metric: (stack_key, "min" | "max")} -- whether the metric
    #: should be minimized (chi2/reduced_chi2/RMSE) or maximized (R2) to find
    #: the best-fitting sample per pixel.
    METRICS: ClassVar[dict[str, tuple[str, str]]] = {
        "chi2": ("chi2_stack", "min"),
        "reduced_chi2": ("reduced_chi2_stack", "min"),
        "RMSE": ("rmse_stack", "min"),
        "R2": ("r2_stack", "max"),
    }

    #: Valid values for the ``backend`` constructor argument.
    BACKENDS: tuple[str, ...] = ("compute_detailed_results", "reconstructor")

    #: Registry of {reducer: numpy function} for
    #: compute_aggregate_model_fit_result's non-"best" methods -- collapses
    #: the NUM_SAMPLES axis of each output_combination array to a single
    #: per-pixel value. Add an entry here to support another reduction
    #: (e.g. "mode") without touching the method itself.
    REDUCERS: ClassVar[dict[str, Callable]] = {
        "mean": np.mean,
        "median": np.median,
    }

    def __init__(
        self,
        freq_acq,
        irf,
        decay,
        model_type="bi-exponential",
        backend="compute_detailed_results",
        bool_mask=None,
        shift_method="zero_pad",
        h_shift=None,
        v_shift=None,
        excitation="bin_start",
        gate_width=None,
    ):
        if model_type not in ("bi-exponential", "mono-exponential"):
            raise ValueError(f"Unknown model_type: {model_type!r}")
        if backend not in self.BACKENDS:
            raise ValueError(
                f"Unknown backend: {backend!r}; expected one of {self.BACKENDS}"
            )
        self.freq_acq = freq_acq
        self.irf = irf
        self.decay = decay
        self.model_type = model_type
        self.backend = backend
        self.bool_mask = bool_mask
        self.shift_method = shift_method
        self.excitation = excitation
        self.gate_width = gate_width
        self.shifts = {"h_shift": h_shift, "v_shift": v_shift}
        self._reconstructor = None
        self._detailed_reconstructor = None

    def _shift_maps(self, params):
        """``h_shift_map``/``v_shift_map`` for ``params``: its own keys, else the constructor's values."""
        ref_shape = np.shape(next(iter(params.values())))
        out = {}
        for key in self.SHIFT_KEYS:
            value = params.get(key, self.shifts[key])
            if value is not None:
                out[f"{key}_map"] = np.broadcast_to(
                    np.asarray(value, dtype=float), ref_shape
                ).copy()
        return out

    @staticmethod
    def _apply_bool_mask(result, bool_mask):
        """
        NaN-out every pixel where ``bool_mask`` is False, in every (H, W) or
        (H, W, ...) array under ``result["results"]["maps"]``, ``["TR_maps"]``,
        and ``["error_maps"]`` -- so excluded pixels (e.g. outside the real
        ROI) are unambiguous downstream instead of looking like ordinary
        (if poor) fits.
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

    def _run_compute_detailed_results(self, params, data_name, log_summary=True):
        """Reconstruct one parameter combination's fit/residual/goodness-of-fit
        maps via the configured ``backend``, using the shared
        freq_acq/irf/decay/model_type stored on this instance.

        ``log_summary`` is forwarded to the ``"compute_detailed_results"``
        backend to suppress its per-call goodness-of-fit summary log line;
        set False when calling in a loop (e.g. evaluate_all_samples) so the
        progress bar isn't drowned out. The ``"reconstructor"`` backend emits
        no such line and ignores this flag."""
        if self.backend == "reconstructor":
            return self._run_via_reconstructor(params, data_name)
        return self._run_via_compute_detailed_results(
            params, data_name, log_summary=log_summary
        )

    def _run_via_compute_detailed_results(self, params, data_name, log_summary=True):
        if self._detailed_reconstructor is None:
            self._detailed_reconstructor = DetailedRecon(
                self.freq_acq,
                self.irf,
                binned_decay=self.decay,
                shift_method=self.shift_method,
                excitation=self.excitation,
                gate_width=self.gate_width,
            )

        if self.model_type == "bi-exponential":
            cdr_params = {
                "tau1_map": params["tau1"],
                "tau2_map": params["tau2"],
                "alpha1_map": params["alpha1"],
            }
        else:
            cdr_params = {"tau_map": params["tau"]}
        cdr_params.update(self._shift_maps(params))
        return self._detailed_reconstructor.reconstruct(
            cdr_params, self.model_type, data_name=data_name, log_summary=log_summary
        )

    def _run_via_reconstructor(self, params, data_name):
        if self._reconstructor is None:
            self._reconstructor = ParamToDecay(
                self.model_type,
                self.freq_acq,
                irf=self.irf,
                shift_method=self.shift_method,
                excitation=self.excitation,
                gate_width=self.gate_width,
            )

        if self.model_type == "bi-exponential":
            recon_params = {
                "tau1_map": params["tau1"],
                "tau2_map": params["tau2"],
                "alpha1_map": params["alpha1"],
            }
        else:
            recon_params = {"tau_map": params["tau"]}
        recon_params.update(self._shift_maps(params))

        out = self._reconstructor.reconstruct_vectorized(
            recon_params, decay=self.decay, verbose=False
        )
        fit_stats = out["fit_stats_maps"]

        # Wrapped into compute_detailed_results' own return shape so callers
        # (evaluate_all_samples, compute_best_model_fit_result) don't need to
        # know which backend actually ran.
        return {
            "name": data_name,
            "method": "ParamToDecay",
            "results": {
                "maps": {
                    "R2_map": fit_stats["R2_map"],
                    "chi2_map": fit_stats["chi2_map"],
                    "reduced_chi2_map": fit_stats["reduced_chi2_map"],
                    "rmse_map": fit_stats["rmse_map"],
                },
                "error_maps": None,
                "TR_maps": out["TR_maps"],
            },
        }

    def evaluate_all_samples(
        self, output_combination, keep_per_sample_results=False, progress=True
    ):
        """
        Run compute_detailed_results once per posterior sample.

        Parameters
        ----------
        output_combination : dict[str, np.ndarray]
            e.g. ``{'tau1': (H,W,NUM_SAMPLES), 'tau2': (H,W,NUM_SAMPLES), 'alpha1': (H,W,NUM_SAMPLES)}``
            for bi-exponential, or ``{'tau': (H,W,NUM_SAMPLES)}`` for mono-exponential.
        keep_per_sample_results : bool
            If True, also returns the full compute_detailed_results() dict for
            every sample (memory-heavy for large images/NUM_SAMPLES). Default
            False -- only the scalar metric stacks are kept.
        progress : bool
            Show a tqdm progress bar over the NUM_SAMPLES loop. Default True;
            pass False for quiet use (e.g. a 1x1-pixel crop). The per-sample
            backend log line is suppressed regardless, so it never competes
            with the bar.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``chi2_stack``, ``reduced_chi2_stack``, ``rmse_stack``, ``r2_stack`` : (H, W, NUM_SAMPLES)
            - ``per_sample_results`` : list[dict] or None
        """
        for key, arr in output_combination.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"output_combination['{key}'] is {type(arr).__name__} ({arr!r}), "
                    "not a numpy array. Did a placeholder/example assignment "
                    "(e.g. 'output_combination = {...}') accidentally overwrite "
                    "your real output_combination in a later cell?"
                )
            if arr.ndim != 3:
                raise ValueError(
                    f"output_combination['{key}'] has shape {arr.shape} (ndim={arr.ndim}); "
                    "expected 3D (H, W, NUM_SAMPLES)."
                )

        first_key = next(iter(output_combination))
        H, W, num_samples = output_combination[first_key].shape

        chi2_stack = np.zeros((H, W, num_samples), dtype=np.float32)
        reduced_chi2_stack = np.zeros((H, W, num_samples), dtype=np.float32)
        rmse_stack = np.zeros((H, W, num_samples), dtype=np.float32)
        r2_stack = np.zeros((H, W, num_samples), dtype=np.float32)

        per_sample_results = [] if keep_per_sample_results else None

        with tqdm(
            total=num_samples,
            desc="Evaluating posterior samples",
            disable=not progress,
            leave=False,
        ) as pbar:
            for s in range(num_samples):
                if self.model_type == "bi-exponential":
                    params_s = {
                        "tau1": output_combination["tau1"][..., s],
                        "tau2": output_combination["tau2"][..., s],
                        "alpha1": output_combination["alpha1"][..., s],
                    }
                else:
                    params_s = {"tau": output_combination["tau"][..., s]}
                for key in self.SHIFT_KEYS:
                    if key in output_combination:
                        params_s[key] = output_combination[key][..., s]

                result_s = self._run_compute_detailed_results(
                    params_s,
                    data_name=f"BI_MODEL_{self.model_type}_sample{s}",
                    log_summary=False,
                )

                maps = result_s["results"]["maps"]

                chi2_stack[..., s] = maps["chi2_map"]
                reduced_chi2_stack[..., s] = maps["reduced_chi2_map"]
                r2_stack[..., s] = maps["R2_map"]
                rmse_stack[..., s] = maps["rmse_map"]

                if keep_per_sample_results:
                    per_sample_results.append(result_s)

                pbar.update(1)

        return {
            "chi2_stack": chi2_stack,
            "reduced_chi2_stack": reduced_chi2_stack,
            "rmse_stack": rmse_stack,
            "r2_stack": r2_stack,
            "per_sample_results": per_sample_results,
        }

    def select_best_combination(self, output_combination, stacks, metric="RMSE"):
        """
        For each pixel independently, pick the sample index that optimizes the
        chosen metric (minimizes chi2/reduced_chi2/RMSE, maximizes R2), and
        build the corresponding best-per-pixel parameter maps.

        Parameters
        ----------
        output_combination : dict[str, np.ndarray]
            Same dict passed to evaluate_all_samples (each array (H, W, NUM_SAMPLES)).
        stacks : dict
            Output of evaluate_all_samples.
        metric : str
            One of "chi2", "reduced_chi2", "RMSE", "R2".

        Returns
        -------
        dict
            Dictionary with keys:

            - ``best_params`` : dict[str, np.ndarray] -- best (H, W) map per key
              in output_combination (e.g. tau1/tau2/alpha1)
            - ``best_sample_idx`` : (H, W) int array -- which sample index won per pixel
            - ``best_score`` : (H, W) float array -- the winning metric value per pixel
        """
        if metric not in self.METRICS:
            raise ValueError(
                f"Unknown metric: {metric!r}; expected one of {list(self.METRICS)}"
            )
        stack_key, sense = self.METRICS[metric]
        score_stack = stacks[stack_key]  # (H, W, NUM_SAMPLES)

        best_idx = (np.argmax if sense == "max" else np.argmin)(score_stack, axis=-1)
        best_idx_expanded = best_idx[..., np.newaxis]  # (H, W, 1)

        best_params = {
            key: np.take_along_axis(arr, best_idx_expanded, axis=-1).squeeze(-1)
            for key, arr in output_combination.items()
        }
        best_score = np.take_along_axis(
            score_stack, best_idx_expanded, axis=-1
        ).squeeze(-1)

        return {
            "best_params": best_params,
            "best_sample_idx": best_idx,
            "best_score": best_score,
        }

    def compute_best_model_fit_result(
        self,
        output_combination,
        metric="RMSE",
        data_name="BI_MODEL_bi_best",
        bool_mask=None,
        stacks=None,
    ):
        """
        Full pipeline: evaluate every sample, pick the best per-pixel combination
        by ``metric``, then re-run compute_detailed_results once more on that best
        combination.

        Parameters
        ----------
        output_combination : dict[str, np.ndarray]
            Same dict passed to evaluate_all_samples.
        metric : str
            One of "chi2", "reduced_chi2", "RMSE", "R2".
        data_name : str
            Dataset name recorded in the returned result dict.
        stacks : dict | None
            Precomputed :meth:`evaluate_all_samples` output for this exact
            ``output_combination``. When given, the per-sample evaluation loop
            (NUM_SAMPLES reconstructions) is skipped and these stacks are used
            directly -- so a caller that already ran ``evaluate_all_samples``
            (e.g. to inspect the raw stacks or try several metrics) doesn't pay
            for it twice. When None (default) it is computed internally. The
            stacks must come from the same ``output_combination``; passing
            mismatched stacks yields wrong selections.
        bool_mask : np.ndarray | None
            Optional (H, W) boolean mask; defaults to ``self.bool_mask`` (set at
            construction) when not given. Pixels where the mask is False are
            NaN'd out in every array under result['results']['maps'],
            ['TR_maps'], and ['error_maps'] -- so excluded pixels (e.g.
            outside the real ROI) can't be mistaken for ordinary fits
            downstream, since compute_detailed_results' own pixel_health_map
            is derived from decay, not this mask. No masking is applied if
            both this argument and ``self.bool_mask`` are None.

        Notes
        -----
        The return value is exactly the dict compute_detailed_results itself
        returns -- ``{"name", "method", "results": {"maps", "error_maps", "TR_maps"}}``
        -- so it plugs directly into the rest of the workflow (e.g.
        ``result['results']['maps'].keys()``, ``result['results']['TR_maps']['fit_map']``,
        ``saver.save_npy(...)``, DataViewer, etc.) exactly like any other
        compute_detailed_results output.

        Two extra maps are folded into ``result['results']['maps']``:

        - ``best_sample_idx_map`` : (H, W) -- which posterior sample
          (0..NUM_SAMPLES-1) won at each pixel
        - ``<metric>_selection_map`` : (H, W) -- the winning metric value at
          each pixel (e.g. ``reduced_chi2_selection_map``)

        Per-sample diagnostics (the full metric stacks across all samples, and
        the chosen metric name) are attached as an additional top-level key,
        ``sample_selection``, without disturbing the primary
        ``"name"``/``"method"``/``"results"`` structure.

        Returns
        -------
        dict
            Same shape as compute_detailed_results()'s return value, plus a
            top-level ``sample_selection`` key holding the raw per-sample
            stacks and best_params used to produce the final maps.
        """
        if stacks is None:
            stacks = self.evaluate_all_samples(output_combination)
        selection = self.select_best_combination(
            output_combination, stacks, metric=metric
        )
        best_params = selection["best_params"]

        bi_model_best = self._run_compute_detailed_results(
            best_params, data_name=data_name
        )

        # Fold the selection diagnostics directly into the same 'maps' dict that
        # every other compute_detailed_results() call produces, so they show up
        # alongside tau1_map/alpha1_map/chi2_map/etc. and can be visualized or
        # saved the same way (e.g. via DataViewer, saver.save_npy).
        bi_model_best["results"]["maps"]["best_sample_idx_map"] = selection[
            "best_sample_idx"
        ].astype(np.float32)
        bi_model_best["results"]["maps"][f"{metric}_selection_map"] = selection[
            "best_score"
        ].astype(np.float32)

        effective_mask = self.bool_mask if bool_mask is None else bool_mask
        if effective_mask is not None:
            self._apply_bool_mask(bi_model_best, effective_mask)

        bi_model_best["sample_selection"] = {
            "metric": metric,
            "best_params": best_params,
            "best_sample_idx": selection["best_sample_idx"],
            "best_score": selection["best_score"],
            "stacks": stacks,
        }

        print(
            f"Best-combination selection complete (metric={metric!r}). "
            f"Mean winning {metric}: {np.nanmean(selection['best_score']):.4f}"
        )

        return bi_model_best

    def compute_aggregate_model_fit_result(
        self,
        output_combination,
        method="best",
        metric="RMSE",
        data_name="BI_MODEL_bi_aggregate",
        bool_mask=None,
        stacks=None,
    ):
        """
        Collapse the stack of posterior-sample parameter combinations down to
        a single per-pixel parameter map via ``method``, then run the
        configured backend once on that combination.

        Parameters
        ----------
        output_combination : dict[str, np.ndarray]
            Same dict passed to evaluate_all_samples/compute_best_model_fit_result
            (each array (H, W, NUM_SAMPLES)).
        method : str
            How to reduce the NUM_SAMPLES axis to a single per-pixel value:
              - "best"            : per-pixel sample that optimizes ``metric``
                                     (delegates to compute_best_model_fit_result;
                                     see that method for the extra
                                     'best_sample_idx_map'/'sample_selection'
                                     diagnostics only this option adds).
              - any key in :attr:`REDUCERS` (default "mean", "median") :
                                     per-pixel reduction across samples, e.g.
                                     the per-pixel mean or median tau/alpha1.
        metric : str
            Only used when method="best"; one of "chi2", "reduced_chi2",
            "RMSE", "R2" (see :attr:`METRICS`).
        data_name : str
            Dataset name recorded in the returned result dict.
        bool_mask : np.ndarray | None
            Optional (H, W) boolean mask; defaults to ``self.bool_mask`` (set at
            construction) when not given. Forwarded to
            compute_best_model_fit_result for method="best"; for the reducer
            methods, applied the same way directly here -- see
            compute_best_model_fit_result's bool_mask parameter for what it does.
        stacks : dict | None
            Only used when method="best"; forwarded to
            compute_best_model_fit_result to skip the per-sample evaluation loop
            when the caller already has its output. Ignored by the reducer
            methods, which never score individual samples.

        Returns
        -------
        dict : same shape as compute_detailed_results()'s return value (see
               compute_best_model_fit_result for the exact structure notes).
               For method="best" this is exactly compute_best_model_fit_result's
               return value, including its extra 'sample_selection' key; the
               other methods have no per-sample selection to report, so that
               key is simply absent.
        """
        if method == "best":
            return self.compute_best_model_fit_result(
                output_combination,
                metric=metric,
                data_name=data_name,
                bool_mask=bool_mask,
                stacks=stacks,
            )

        reducer = self.REDUCERS.get(method)
        if reducer is None:
            raise ValueError(
                f"Unknown method: {method!r}; expected 'best' or one of "
                f"{list(self.REDUCERS)}"
            )

        agg_params = {
            key: reducer(arr, axis=-1).astype(np.float32)
            for key, arr in output_combination.items()
        }

        result = self._run_compute_detailed_results(agg_params, data_name=data_name)

        effective_mask = self.bool_mask if bool_mask is None else bool_mask
        if effective_mask is not None:
            self._apply_bool_mask(result, effective_mask)

        print(f"{method.capitalize()}-combination result computed ({data_name!r}).")

        return result
