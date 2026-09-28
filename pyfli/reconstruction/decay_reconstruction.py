"""
Reconstruct per-pixel modeled decay curves from fitted FLI parameter maps.

This module belongs to :mod:`pyfli.reconstruction` and sits downstream of
:mod:`pyfli.solver`: it reuses the solver's mono-/bi-exponential forward-model
kernels and fit-quality metrics to turn a dictionary of fitted parameter maps
(shaped like :class:`pyfli.solver.FLICPUProcessor`'s output) back into decay
cubes. Public API includes class :class:`ParamToDecay`.
"""

import itertools
from typing import Any, ClassVar

import numpy as np
from scipy.signal import fftconvolve
from tqdm import tqdm

from pyfli.solver.forward_model import (
    decay_kernel,
    gate_integrated_kernel,
    model_numpy,
    negative_lag_gates,
)
from pyfli.solver.shared_metrics import (
    compute_fli_stats,
    compute_pearson_stats,
    pearson_chi_square,
    reduced_poisson_deviance,
)

# Matches forward_model._EPS — kept as a local literal since that name is
# module-private in forward_model.
_EPS = 1e-8


class ParamToDecay:
    """
    Reconstruct per-pixel modeled decay curves from fitted parameter maps.

    Rebuilds the forward-model decay for every pixel from a dictionary of
    parameter maps shaped like :class:`pyfli.solver.FLICPUProcessor`'s output,
    and optionally derives fit-quality maps (``TR_maps`` and per-pixel
    :func:`compute_fli_stats` summaries) when the measured decay is also
    supplied. When an IRF is available, pixels are rebuilt via
    :func:`pyfli.solver.forward_model.model_numpy` (kernel convolved with the
    IRF); when no IRF is given, pixels are rebuilt directly from
    :func:`pyfli.solver.forward_model.decay_kernel` (the un-convolved model).

    ``"photon_count_map"``, ``"v_shift_map"``, and ``"h_shift_map"`` are
    optional — missing ones default to 1.0, 0.0, and 0.0 respectively (see
    :attr:`PARAM_MAP_DEFAULTS`), except ``"photon_count_map"``: if it's
    omitted *and* ``decay`` is supplied to :meth:`reconstruct`, it is instead
    solved for as the amplitude that makes the model's own convolved-kernel
    discrete sum match ``decay``'s discrete sum at each pixel (*not* simply
    ``decay.sum(axis=-1)`` — see :meth:`_fill_photon_count_from_decay`),
    rather than falling back to 1.0. The lifetime map(s) (``"tau_map"`` for
    mono-exponential; ``"alpha1_map"``, ``"tau1_map"``, ``"tau2_map"`` for
    bi-exponential) are always required.

    A 2-D boolean ``bool_mask`` can be passed to :meth:`reconstruct` to only
    reconstruct pixels where it is ``True``; every output map is ``NaN``
    elsewhere. Single-pixel reconstruction is also supported: pass scalar
    parameter values (and a 1-D IRF/decay) and every output collapses to a
    plain 1-D trace (or scalar, for the fit-quality numbers) instead of an
    ``(H, W, ...)`` map — see :meth:`reconstruct`.

    New model families can be added by extending :attr:`PARAM_MAP_KEYS` (and the
    matching kernel in :mod:`pyfli.solver.forward_model`); new post-reconstruction
    map products can be added by overriding :meth:`_compute_tr_maps` in a subclass.

    Parameters
    ----------
    model_type : str
        FLI model family; one of the keys in :attr:`PARAM_MAP_KEYS`
        (``"mono-exponential"`` or ``"bi-exponential"``).
    freq : float
        Acquisition frequency in MHz — i.e. ``freq[1]`` in
        :class:`pyfli.solver.BaseFLIFitter`'s/:class:`pyfli.solver.FLIGPUProcessor`'s
        ``(laser_freq_mhz, acq_freq_mhz)`` convention. Only the acquisition
        frequency is ever needed here, to derive the time axis as
        ``T_acq = 1000.0 / freq``.
    irf : np.ndarray | None
        Instrument response function; either a shared 1-D trace or a per-pixel
        ``(H, W, T)`` cube. The number of time gates is inferred from its last
        axis (or its length, if 1-D). If omitted, pixels are reconstructed
        without IRF convolution (see :func:`pyfli.solver.forward_model.decay_kernel`)
        and ``num_gates`` must be supplied instead.
    num_gates : int | None
        Number of time gates/bins. Required only when ``irf`` is omitted;
        otherwise it is inferred from ``irf`` and this argument is ignored.
    """

    #: Registry of {model_type: required parameter-map keys}, in the exact order
    #: expected by ``forward_model.model_numpy``/``decay_kernel`` — the last key
    #: is always the temporal shift consumed as ``h_shift``. Extend this (plus a
    #: matching kernel branch in ``forward_model.decay_kernel``) to support new
    #: model families.
    PARAM_MAP_KEYS: ClassVar[dict[str, tuple[str, ...]]] = {
        "mono-exponential": (
            "photon_count_map",
            "tau_map",
            "v_shift_map",
            "h_shift_map",
        ),
        "bi-exponential": (
            "photon_count_map",
            "alpha1_map",
            "tau1_map",
            "tau2_map",
            "v_shift_map",
            "h_shift_map",
        ),
    }

    #: Default value used for a parameter map when it's omitted from ``params``.
    #: Keys absent from this dict are required (no sensible physical default).
    PARAM_MAP_DEFAULTS: ClassVar[dict[str, float]] = {
        "photon_count_map": 1.0,
        "v_shift_map": 0.0,
        "h_shift_map": 0.0,
    }

    def __init__(
        self,
        model_type: str,
        freq: float,
        irf: np.ndarray | None = None,
        num_gates: int | None = None,
    ) -> None:
        if model_type not in self.PARAM_MAP_KEYS:
            raise ValueError(
                f"Unknown model_type {model_type!r}; expected one of "
                f"{tuple(self.PARAM_MAP_KEYS)}."
            )
        self.model_type = model_type
        # Acquisition frequency in MHz — freq[1] in BaseFLIFitter/
        # FLIGPUProcessor's (laser_freq, acq_freq) convention.
        self.freq = float(freq)
        self.T_acq = 1000.0 / self.freq

        if irf is not None:
            self.irf = np.asarray(irf)
            self.num_gates = len(self.irf) if self.irf.ndim == 1 else self.irf.shape[2]
        else:
            if num_gates is None:
                raise ValueError(
                    "num_gates must be given explicitly when irf is not provided "
                    "(it can no longer be inferred from the IRF's shape)."
                )
            self.irf = None
            self.num_gates = int(num_gates)

        self.t = np.linspace(0, self.T_acq, self.num_gates, endpoint=False)

    @property
    def param_keys(self) -> tuple[str, ...]:
        """All parameter-map keys for :attr:`model_type`, in kernel order."""
        return self.PARAM_MAP_KEYS[self.model_type]

    @property
    def required_keys(self) -> tuple[str, ...]:
        """Parameter-map keys for :attr:`model_type` that have no default."""
        return tuple(k for k in self.param_keys if k not in self.PARAM_MAP_DEFAULTS)

    def _validate_params(self, params: dict[str, Any]) -> None:
        missing = [k for k in self.required_keys if k not in params]
        if missing:
            raise KeyError(
                f"params is missing required maps for model_type={self.model_type!r}: "
                f"{missing}"
            )

    def _normalize_single_pixel(
        self, params: dict[str, Any], decay: np.ndarray | None
    ) -> tuple[bool, dict[str, np.ndarray], np.ndarray | None]:
        """
        Wrap scalar (single-pixel) inputs into 1x1 maps so the rest of the
        pipeline only has to know about the ``(H, W, ...)`` case.
        """
        ref_val = params[self.required_keys[0]]
        if np.ndim(ref_val) != 0:
            return False, params, decay

        wrapped_params = {
            k: np.asarray(v, dtype=float).reshape(1, 1) for k, v in params.items()
        }
        wrapped_decay = (
            None if decay is None else np.asarray(decay, dtype=float).reshape(1, 1, -1)
        )
        return True, wrapped_params, wrapped_decay

    def _squeeze_single_pixel(self, result: dict[str, Any]) -> dict[str, Any]:
        """Undo :meth:`_normalize_single_pixel`'s wrapping on the output dict."""
        out: dict[str, Any] = {"fit_map": result["fit_map"][0, 0, :]}
        if "TR_maps" in result:
            tr = result["TR_maps"]
            out["TR_maps"] = {
                "fit_map": tr["fit_map"][0, 0, :],
                "residual_map": tr["residual_map"][0, 0, :],
            }
        if "fit_stats_maps" in result:
            out["fit_stats_maps"] = {
                k: float(v[0, 0]) for k, v in result["fit_stats_maps"].items()
            }
        return out

    def _fill_photon_count_from_decay(
        self, params: dict[str, np.ndarray], decay: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        """
        When ``"photon_count_map"`` is omitted but ``decay`` is supplied, solve
        for the amplitude S that makes the model's own convolved-kernel
        discrete sum match ``decay``'s discrete sum at each pixel, instead of
        falling back to the constant default in :attr:`PARAM_MAP_DEFAULTS`.

        This is *not* the same as ``decay.sum(axis=-1)``: the model kernel's
        own discrete sum at unit amplitude is generally not 1 (it depends on
        tau, the acquisition window, dt, and any convolution truncation), so
        equating S directly to the raw decay total would silently over/under-
        scale the reconstructed fit by that same factor.
        """
        if "photon_count_map" in params or decay is None:
            return params
        unit_params = dict(params)
        unit_params["photon_count_map"] = np.ones(
            np.asarray(params[self.required_keys[0]]).shape, dtype=np.float32
        )
        convolved = (
            self._convolve_with_irf_vectorized(unit_params)
            if self.irf is not None
            else self._kernel_vectorized(unit_params)
        )
        fit_sum = np.sum(convolved, axis=-1)
        decay_total = np.asarray(decay).sum(axis=-1)
        S = np.zeros_like(decay_total, dtype=np.float64)
        np.divide(decay_total, fit_sum, out=S, where=fit_sum > _EPS)

        params = dict(params)
        params["photon_count_map"] = S
        return params

    def _pixel_params(
        self, params: dict[str, np.ndarray], i: int, j: int
    ) -> np.ndarray:
        """Stack one pixel's scalar parameters in kernel order, applying defaults."""
        values = [
            params[k][i, j] if k in params else self.PARAM_MAP_DEFAULTS[k]
            for k in self.param_keys
        ]
        return np.array(values, dtype=float)

    def _pixel_irf(self, i: int, j: int) -> np.ndarray:
        """Return the IRF trace to use for pixel ``(i, j)``."""
        return self.irf[i, j, :] if self.irf.ndim == 3 else self.irf

    def _pixel_fit(self, params: dict[str, np.ndarray], i: int, j: int) -> np.ndarray:
        """Reconstruct one pixel's modeled decay, with or without an IRF."""
        full_params = self._pixel_params(params, i, j)

        if self.irf is not None:
            return model_numpy(
                self.t, self._pixel_irf(i, j), full_params, self.model_type
            )

        # No IRF: use the un-convolved kernel directly (decay_kernel already
        # separates h_shift from the rest of the kernel parameters).
        h_shift = float(full_params[-1])
        kernel_params = full_params[:-1]
        kernel, v_shift = decay_kernel(
            self.t, kernel_params, self.model_type, h_shift=h_shift
        )
        return (kernel + v_shift).astype(np.float32)

    def _reconstruct_fit_map(
        self,
        params: dict[str, np.ndarray],
        bool_mask: np.ndarray | None,
        verbose: bool,
    ) -> np.ndarray:
        any_map = np.asarray(params[self.required_keys[0]])
        h, w = any_map.shape
        fit_map = np.full((h, w, self.num_gates), np.nan, dtype=np.float32)

        pixel_iterator = itertools.product(range(h), range(w))
        total = h * w if bool_mask is None else int(np.count_nonzero(bool_mask))
        with tqdm(
            total=total,
            desc="Reconstructing decay",
            disable=not verbose,
            leave=False,
        ) as pbar:
            for i, j in pixel_iterator:
                if bool_mask is not None and not bool_mask[i, j]:
                    continue
                fit_map[i, j, :] = self._pixel_fit(params, i, j)
                pbar.update(1)
        return fit_map

    def _compute_tr_maps(
        self,
        fit_map: np.ndarray,
        decay: np.ndarray,
        bool_mask: np.ndarray | None,
    ) -> dict[str, Any]:
        """
        Derive ``TR_maps`` (fit + residual) and per-pixel fit-quality maps.

        The residual map follows :class:`pyfli.solver.FLICPUProcessor`'s /
        :class:`pyfli.solver.FLIGPUProcessor`'s convention (``decay - fit``);
        the accompanying ``fit_stats_maps`` use the same key names those two
        processors expose in their ``maps`` dict (``R2_map``, ``chi2_map`` and
        ``reduced_chi2_map`` -- the Poisson deviance and deviance / (n - p) --,
        ``pearson_chi2_map``, ``pearson_reduced_chi2_map``, ``rmse_map``), and are
        computed per pixel via :func:`compute_fli_stats` /
        :func:`compute_pearson_stats` — the same functions :class:`BaseFLIFitter`
        uses — so goodness-of-fit numbers stay in lockstep with the rest of
        the solver package. Pixels outside ``bool_mask`` (or wherever
        ``fit_map`` is ``NaN``, e.g. because they were skipped) are left
        ``NaN`` in every output map. Override this method in a subclass to
        add further per-pixel map products.
        """
        h, w, _ = fit_map.shape
        n_params = len(self.param_keys)

        # NaN in fit_map (skipped pixels) propagates through the subtraction.
        residual_map = decay - fit_map
        chi2_map = np.full((h, w), np.nan, dtype=np.float32)
        reduced_chi2_map = np.full((h, w), np.nan, dtype=np.float32)
        r2_map = np.full((h, w), np.nan, dtype=np.float32)
        rmse_map = np.full((h, w), np.nan, dtype=np.float32)
        pearson_map = np.full((h, w), np.nan, dtype=np.float32)
        pearson_reduced_map = np.full((h, w), np.nan, dtype=np.float32)

        for i, j in itertools.product(range(h), range(w)):
            if bool_mask is not None and not bool_mask[i, j]:
                continue
            _ssr, chi_sq, red_chi_sq, r_sq, rmse = compute_fli_stats(
                fit_map[i, j, :], decay[i, j, :], n_params
            )
            chi2_map[i, j] = chi_sq
            reduced_chi2_map[i, j] = red_chi_sq
            r2_map[i, j] = r_sq
            rmse_map[i, j] = rmse
            pearson_map[i, j], pearson_reduced_map[i, j] = compute_pearson_stats(
                fit_map[i, j, :], decay[i, j, :], n_params
            )

        return {
            "TR_maps": {"fit_map": fit_map, "residual_map": residual_map},
            "fit_stats_maps": {
                "R2_map": r2_map,
                "chi2_map": chi2_map,
                "reduced_chi2_map": reduced_chi2_map,
                "pearson_chi2_map": pearson_map,
                "pearson_reduced_chi2_map": pearson_reduced_map,
                "rmse_map": rmse_map,
            },
        }

    # ── vectorized path ──────────────────────────────────────────────────────
    # Same contract/output as reconstruct(), just batched over the whole image
    # with numpy broadcasting + a single fftconvolve call instead of a Python
    # per-pixel loop — orders of magnitude faster on real image sizes. Numeric
    # output is identical to reconstruct() (see tests); the only behavioral
    # difference is the IRF zero-sum fallback, applied per-pixel here via
    # np.where rather than a Python-level branch.

    def _get_map(self, params: dict[str, np.ndarray], key: str) -> np.ndarray:
        """Return the (H, W) map for ``key``, broadcasting its default if absent."""
        if key in params:
            return np.asarray(params[key], dtype=float)
        any_map = np.asarray(params[self.required_keys[0]])
        return np.full(any_map.shape, self.PARAM_MAP_DEFAULTS[key], dtype=float)

    def _kernel_vectorized(
        self, params: dict[str, np.ndarray], gate_start: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Build the un-convolved gate-integrated (H, W, T') kernel, matching
        :func:`pyfli.solver.forward_model.gate_integrated_kernel` -- on the gates
        starting at ``self.t`` (as ``decay_kernel``) unless `gate_start` is given.
        """
        dt = float(self.t[1] - self.t[0]) if self.t.size > 1 else 1.0
        starts = self.t if gate_start is None else np.asarray(gate_start, dtype=float)
        S = self._get_map(params, "photon_count_map")[..., None]
        h_shift = self._get_map(params, "h_shift_map")[..., None]

        if self.model_type == "mono-exponential":
            kernel_params = (S, self._get_map(params, "tau_map")[..., None], 0.0)
        else:
            kernel_params = (
                S,
                self._get_map(params, "alpha1_map")[..., None],
                self._get_map(params, "tau1_map")[..., None],
                self._get_map(params, "tau2_map")[..., None],
                0.0,
            )
        kernel, _ = gate_integrated_kernel(
            starts[None, None, :], dt, kernel_params, self.model_type, h_shift=h_shift
        )
        return np.asarray(kernel, dtype=float)

    def _convolve_with_irf_vectorized(
        self, params: dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Batch-convolve the kernel of `params` with the (per-pixel-normalized) IRF,
        exactly as :func:`pyfli.solver.forward_model.model_numpy` does per pixel:
        the kernel is evaluated on extra negative-lag gates when any onset
        ``h_shift`` is negative, so an earlier onset shifts the curve.
        """
        t = self.num_gates
        dt = float(self.t[1] - self.t[0]) if self.t.size > 1 else 1.0
        n_neg = negative_lag_gates(self._get_map(params, "h_shift_map"), dt)
        kernel = self._kernel_vectorized(params, np.arange(-n_neg, t) * dt)
        h, w, _ = kernel.shape
        irf_arr = (
            self.irf if self.irf.ndim == 3 else np.broadcast_to(self.irf, (h, w, t))
        )
        irf_sum = np.sum(irf_arr, axis=-1, keepdims=True)
        safe_sum = np.where(irf_sum > 0, irf_sum, 1.0)
        irf_norm = np.where(irf_sum > 0, irf_arr / safe_sum, irf_arr)
        return fftconvolve(kernel, irf_norm, mode="full", axes=-1)[
            ..., n_neg : n_neg + t
        ]

    def _build_fit_map_vectorized(self, params: dict[str, np.ndarray]) -> np.ndarray:
        convolved = (
            self._convolve_with_irf_vectorized(params)
            if self.irf is not None
            else self._kernel_vectorized(params)
        )
        v_shift = self._get_map(params, "v_shift_map")
        return (convolved + v_shift[..., None]).astype(np.float32)

    def _compute_tr_maps_vectorized(
        self,
        fit_map: np.ndarray,
        decay: np.ndarray,
        bool_mask: np.ndarray | None,
    ) -> dict[str, Any]:
        """Vectorized equivalent of :meth:`_compute_tr_maps`, same formulas as
        :func:`compute_fli_stats` / :func:`compute_pearson_stats` (Poisson deviance
        and its expectation-normalized reduced value; Pearson with variance floored
        at 1.0 and dof = T - n_params)."""
        n_params = len(self.param_keys)
        dof = max(self.num_gates - n_params, 1)

        residual_map = decay - fit_map
        ssr = np.sum(residual_map**2, axis=-1)
        chi2_map, reduced_chi2_map = reduced_poisson_deviance(fit_map, decay, n_params)
        pearson_map = pearson_chi_square(fit_map, decay)
        pearson_reduced_map = pearson_map / dof
        ss_tot = np.sum((decay - np.mean(decay, axis=-1, keepdims=True)) ** 2, axis=-1)
        r2_map = np.where(
            ss_tot > 0, 1.0 - ssr / np.where(ss_tot > 0, ss_tot, 1.0), 0.0
        )
        rmse_map = np.sqrt(np.mean(residual_map**2, axis=-1))

        if bool_mask is not None:
            chi2_map = np.where(bool_mask, chi2_map, np.nan)
            reduced_chi2_map = np.where(bool_mask, reduced_chi2_map, np.nan)
            pearson_map = np.where(bool_mask, pearson_map, np.nan)
            pearson_reduced_map = np.where(bool_mask, pearson_reduced_map, np.nan)
            r2_map = np.where(bool_mask, r2_map, np.nan)
            rmse_map = np.where(bool_mask, rmse_map, np.nan)

        return {
            "TR_maps": {
                "fit_map": fit_map,
                "residual_map": residual_map.astype(np.float32),
            },
            "fit_stats_maps": {
                "R2_map": r2_map.astype(np.float32),
                "chi2_map": chi2_map.astype(np.float32),
                "reduced_chi2_map": reduced_chi2_map.astype(np.float32),
                "pearson_chi2_map": pearson_map.astype(np.float32),
                "pearson_reduced_chi2_map": pearson_reduced_map.astype(np.float32),
                "rmse_map": rmse_map.astype(np.float32),
            },
        }

    def reconstruct_unit_amplitude(
        self, params: dict[str, Any], bool_mask: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """
        Build the un-convolved kernel and IRF-convolved model at whatever
        literal amplitude ``params`` carries (a constant
        ``"photon_count_map"`` of 1.0 gives a pure shape/PDF result), without
        adding ``v_shift`` or computing any fit-quality maps.

        Pairs with :meth:`rescale_fit_to_measured_totals` for callers whose
        parameter maps don't carry a meaningful literal amplitude (e.g.
        lifetime-only estimator output) and need the final scale pinned to a
        measured photon count instead — see
        :meth:`pyfli.reconstruction.DetailedRecon.reconstruct`.

        Returns
        -------
        dict[str, np.ndarray]
            ``{"kernel_map": <H, W, T>, "convolved_map": <H, W, T>}`` (or
            ``(T,)`` each, for single-pixel input). ``convolved_map`` equals
            ``kernel_map`` when the instance has no IRF.
        """
        self._validate_params(params)
        single_pixel, params, _ = self._normalize_single_pixel(params, None)

        kernel = self._kernel_vectorized(params)
        convolved = (
            self._convolve_with_irf_vectorized(params)
            if self.irf is not None
            else kernel
        )
        kernel = kernel.astype(np.float32)
        convolved = convolved.astype(np.float32)

        if bool_mask is not None and not single_pixel:
            kernel = np.where(bool_mask[..., None], kernel, np.nan).astype(np.float32)
            convolved = np.where(bool_mask[..., None], convolved, np.nan).astype(
                np.float32
            )

        result = {"kernel_map": kernel, "convolved_map": convolved}
        if single_pixel:
            result = {k: v[0, 0, :] for k, v in result.items()}
        return result

    def rescale_fit_to_measured_totals(
        self, fit_map: np.ndarray, decay: np.ndarray, eps: float = _EPS
    ) -> np.ndarray:
        """
        Rescale ``fit_map`` per pixel so its sum along the time axis matches
        ``decay``'s exactly, instead of using whatever literal amplitude
        (``"photon_count_map"``) went into building it.

        Useful when the parameter maps don't carry a meaningful absolute
        amplitude (e.g. lifetime-only estimator output) and the fit's total
        should instead be pinned to the measured photon count — this also
        compensates for any convolution-truncation loss the way
        :meth:`pyfli.reconstruction.DetailedRecon.reconstruct` requires.
        """
        fit_map = np.asarray(fit_map, dtype=float)
        decay = np.asarray(decay, dtype=float)
        fit_sum = np.sum(fit_map, axis=-1, keepdims=True)
        decay_total = np.sum(decay, axis=-1, keepdims=True)
        fit_pdf = np.zeros_like(fit_map)
        np.divide(fit_map, fit_sum, out=fit_pdf, where=fit_sum > eps)
        return (decay_total * fit_pdf).astype(np.float32)

    def reconstruct_vectorized(
        self,
        params: dict[str, Any],
        decay: np.ndarray | None = None,
        bool_mask: np.ndarray | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Vectorized equivalent of :meth:`reconstruct` — same contract and
        (up to floating-point noise) identical numeric output, but built
        with batched numpy/``fftconvolve`` instead of a Python per-pixel
        loop. Prefer this for whole-image reconstruction; ``reconstruct``
        remains available for its progress bar and for subclasses that
        override the per-pixel hooks.
        """
        self._validate_params(params)
        single_pixel, params, decay = self._normalize_single_pixel(params, decay)
        if single_pixel:
            bool_mask = None
        params = self._fill_photon_count_from_decay(params, decay)

        if verbose:
            print("Reconstructing decay (vectorized)...")
        fit_map = self._build_fit_map_vectorized(params)
        if bool_mask is not None:
            fit_map = np.where(bool_mask[..., None], fit_map, np.nan).astype(np.float32)

        result: dict[str, Any] = {"fit_map": fit_map}
        if decay is not None:
            result.update(self._compute_tr_maps_vectorized(fit_map, decay, bool_mask))

        if single_pixel:
            result = self._squeeze_single_pixel(result)
        return result

    def reconstruct(
        self,
        params: dict[str, Any],
        decay: np.ndarray | None = None,
        bool_mask: np.ndarray | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Rebuild the per-pixel modeled decay from fitted parameter maps.

        Parameters
        ----------
        params : dict[str, Any]
            Parameter maps keyed as in :class:`pyfli.solver.FLICPUProcessor`'s
            output for :attr:`model_type` (see :attr:`param_keys`). Each value
            is normally an ``(H, W)`` array; ``"photon_count_map"``,
            ``"v_shift_map"``, and ``"h_shift_map"`` may be omitted (see
            :attr:`PARAM_MAP_DEFAULTS` and, for ``"photon_count_map"``, the
            ``decay``-derived fallback described in the class docstring). For
            single-pixel reconstruction, pass plain scalars instead.
        decay : np.ndarray | None
            Measured decay, ``(H, W, T)`` (or ``(T,)`` for a single pixel).
            When supplied, ``TR_maps`` and ``fit_stats_maps`` are also
            computed and returned.
        bool_mask : np.ndarray | None
            Optional 2-D boolean mask; only ``True`` pixels are reconstructed,
            everything else is ``NaN`` in every output map. Ignored for
            single-pixel reconstruction.
        verbose : bool
            Show a progress bar while reconstructing.

        Returns
        -------
        dict[str, Any]
            ``{"fit_map": <H, W, T>}``, plus ``"TR_maps"`` and
            ``"fit_stats_maps"`` when ``decay`` is supplied. For single-pixel
            input, every value collapses to a ``(T,)`` trace or, for
            ``fit_stats_maps``, a plain ``float``.
        """
        self._validate_params(params)
        single_pixel, params, decay = self._normalize_single_pixel(params, decay)
        if single_pixel:
            bool_mask = None
        params = self._fill_photon_count_from_decay(params, decay)

        fit_map = self._reconstruct_fit_map(params, bool_mask, verbose)

        result: dict[str, Any] = {"fit_map": fit_map}
        if decay is not None:
            result.update(self._compute_tr_maps(fit_map, decay, bool_mask))

        if single_pixel:
            result = self._squeeze_single_pixel(result)
        return result
