# solver/base_fitter.py
"""
Implement the shared least-squares FLI fitter used by CPU, GPU, and model-comparison
workflows.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLI fitting routines. Public API includes
classes :class:`BaseFLIFitter`.
"""

import warnings
from typing import Any

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit, least_squares
from scipy.stats import f

from .base_static import moment_based_guess, resolve_params_and_bounds
from .forward_model import model_numpy
from .shared_metrics import (
    compute_average_lifetime,
    compute_fli_stats,
    compute_fret_efficiency,
    enforce_tau_ordering,
)

NLSF_WEIGHTINGS = ("irls", "none", "neyman")
"""Residual weightings of :meth:`BaseFLIFitter.least_squares_fit`."""


class BaseFLIFitter:
    """
    Run the base flifitter routine.
    base class handles model construction, fit ranges, parameter guesses, bounds, post-
    processing, and model comparison support.

    Parameters
    ----------
    freq : float
        Acquisition frequency information used to derive timing constants.
    decay_px : np.ndarray
        Per-pixel fluorescence decay trace supplied to the fitter.
    irf_px : np.ndarray
        Per-pixel instrument response function supplied to the fitter.
    white_noise : float
        White-noise estimate used to weight residuals.
    guess_plugin : np.ndarray
        Optional callable that supplies initial parameter guesses.
    custom_funcs : np.ndarray | None
        Optional custom model functions used by the fitter.
    shift_method : str
        Method used to align the IRF and decay traces.
    fit_indices : tuple[int, int] | None
        Optional (gate_num_start, gate_num_end) gate range to fit over, e.g. to focus on
        the tail of the decay. ``None`` fits the full trace.
    """

    def __init__(
        self,
        freq: float,
        decay_px: np.ndarray,
        irf_px: np.ndarray,
        white_noise: float = 0.1,
        guess_plugin: np.ndarray = moment_based_guess,
        custom_funcs: np.ndarray | None = None,
        shift_method: str = "zero_pad",
        fit_indices: tuple[int, int] | None = None,
    ) -> None:
        self.decay = np.asarray(decay_px)
        self.irf = np.asarray(irf_px)
        self.white_noise = white_noise
        self.guess_plugin = guess_plugin
        self.shift_method = shift_method

        # Timing constants
        self.T_laser = 1000.0 / freq[0]
        self.T_acq = 1000.0 / freq[1]
        self.N = len(self.irf) if self.irf.ndim == 1 else self.irf.shape[2]
        self.t = np.linspace(0, self.T_acq, self.N, endpoint=False)
        if fit_indices is not None:
            gate_start, gate_end = fit_indices
            self.fit_indices = np.arange(max(gate_start, 0), min(gate_end, self.N))
        else:
            self.fit_indices = np.arange(self.N)

        # Central Solver Registry
        self.funcs = {
            "least_squares": self.least_squares_fit,
            "trust_region": self.trust_region,
            "unconstrained": self.unconstrained,
        }
        if custom_funcs:
            self.funcs.update(custom_funcs)

    def fit_with_estimator(
        self,
        estimator_type: str = "least_squares",
        model_type: str = "bi-exponential",
        p0: Any | None = None,
        bounds: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        """Unified entry point for all NLSF estimators."""
        # Now calls the external static logic from base_static.py
        p0_safe, bounds_safe = resolve_params_and_bounds(
            p0,
            bounds,
            model_type,
            self.t,
            self.decay,
            self.T_laser,
            self.guess_plugin,
            self.T_acq,
        )

        if estimator_type in self.funcs:
            return self.funcs[estimator_type](
                p0_safe, bounds_safe, model_type, **kwargs
            )
        else:
            raise ValueError(f"Estimator '{estimator_type}' not found in registry.")

    def least_squares_fit(
        self,
        p0: Any,
        bounds: np.ndarray,
        model_type: str,
        weighting: str = "irls",
        use_weights: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Weighted non-linear least-squares fit, ``min sum_k w_k (mu_k - d_k)^2``.

        Parameters
        ----------
        p0 : Any
            Initial parameter vector supplied to the optimizer.
        bounds : np.ndarray
            Lower and upper parameter bounds supplied to the optimizer.
        model_type : str
            FLI model family, such as mono- or bi-exponential.
        weighting : str
            How the residual weights ``w_k`` are chosen (see :data:`NLSF_WEIGHTINGS`):

            - ``"irls"`` (default): iteratively reweighted least squares. Each round
              solves with the weights frozen at ``1 / max(mu_k, variance_floor)``
              from the previous round's model, until the parameters stop changing.
              At convergence ``sum_k (d_k - mu_k) / mu_k * dmu_k/dtheta = 0``, the
              Poisson likelihood equations, so photon-count data gets
              MLE-equivalent (unbiased) estimates.
            - ``"none"``: unweighted; nearly unbiased but less precise for
              Poisson data, since every gate counts equally.
            - ``"neyman"``: weights ``1 / max(d_k, 1)`` from the measured data
              (Neyman chi-square). Biased towards low counts -- for decays, towards
              short lifetimes, increasingly so at low photon counts. Kept only to
              reproduce results of earlier PyFLI versions.
        use_weights : bool | None
            Deprecated: ``True`` means ``weighting="neyman"`` (the former default)
            and ``False`` means ``weighting="none"``.
        **kwargs : Any
            ``max_iter`` / ``maxiter`` (function evaluations per solve), ``ftol``,
            ``xtol``, and for IRLS ``irls_max_rounds`` (default 20), ``irls_tol``
            (relative parameter change for convergence, default 1e-6) and
            ``variance_floor`` (default 1.0 counts).

        Returns
        -------
        Any
            Object produced by least squares fit.
        """
        if use_weights is not None:
            warnings.warn(
                "use_weights is deprecated; use weighting='neyman' (former default) "
                "or weighting='none'.",
                DeprecationWarning,
                stacklevel=2,
            )
            weighting = "neyman" if use_weights else "none"
        if weighting not in NLSF_WEIGHTINGS:
            raise ValueError(
                f"weighting must be one of {NLSF_WEIGHTINGS}, got {weighting!r}"
            )

        d_fit = np.asarray(self.decay[self.fit_indices], dtype=float)
        solver_opts = {
            "bounds": bounds,
            "ftol": kwargs.get("ftol", 1e-7),
            "xtol": kwargs.get("xtol", 1e-7),
            "max_nfev": kwargs.get("max_iter", kwargs.get("maxiter", 500)),
        }

        def solve(x0: Any, weights: np.ndarray) -> Any:
            def residuals(params: Any) -> Any:
                full_model = self.model_fit(self.t, params, model_type=model_type)
                return (full_model[self.fit_indices] - d_fit) * weights

            return least_squares(residuals, x0=x0, **solver_opts)

        robust_residuals = None
        if weighting == "neyman":
            res = solve(p0, 1.0 / np.sqrt(np.clip(d_fit, 1.0, None)))
        elif weighting == "none":
            res = solve(p0, np.ones_like(d_fit))
            robust_residuals = res.fun
        else:
            floor = kwargs.get("variance_floor", 1.0)
            tol = kwargs.get("irls_tol", 1e-6)
            params = np.asarray(p0, dtype=float)
            for _ in range(kwargs.get("irls_max_rounds", 20)):
                mu = self.model_fit(self.t, params, model_type=model_type)
                weights = 1.0 / np.sqrt(np.clip(mu[self.fit_indices], floor, None))
                res = solve(params, weights)
                change = np.max(
                    np.abs(res.x - params) / np.maximum(np.abs(params), 1e-12)
                )
                params = res.x
                if change < tol:
                    break
        return self._post_process(
            res.x,
            res.jac,
            res.status,
            model_type,
            bounds=bounds,
            objective_chi_sq=2.0 * float(res.cost),
            robust_residuals=robust_residuals,
        )

    def trust_region(
        self, p0: Any, bounds: np.ndarray, model_type: str, **kwargs: Any
    ) -> Any:
        """
        Run the trust region routine.

        Parameters
        ----------
        p0 : Any
            Initial parameter vector supplied to the optimizer.
        bounds : np.ndarray
            Lower and upper parameter bounds supplied to the optimizer.
        model_type : str
            FLI model family, such as mono- or bi-exponential.
        **kwargs : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        Any
            Object produced by trust region.
        """
        max_nfev = kwargs.get("max_iter", kwargs.get("maxiter", 2000))

        def wrapper(t_sub: np.ndarray, *p: Any) -> Any:
            """
            Run the wrapper routine.

            Parameters
            ----------
            t_sub : np.ndarray
                Subset of the time axis used during fitting.
            *p : Any
                Detector parameter object or fitted parameter vector.

            Returns
            -------
            Any
                Object produced by wrapper.
            """
            return self.model_fit(self.t, p, model_type=model_type)[self.fit_indices]

        try:
            popt, pcov = curve_fit(
                wrapper,
                self.t[self.fit_indices],
                self.decay[self.fit_indices],
                p0=p0,
                method="trf",
                bounds=bounds,
                max_nfev=max_nfev,
            )
            status = 1
        except Exception:
            popt, pcov, status = p0, None, 0
        return self._post_process(
            popt, None, status, model_type, pcov=pcov, bounds=bounds
        )

    def unconstrained(
        self, p0: Any, bounds: np.ndarray, model_type: str, **kwargs: Any
    ) -> Any:
        """
        Run the unconstrained routine.

        Parameters
        ----------
        p0 : Any
            Initial parameter vector supplied to the optimizer.
        bounds : np.ndarray
            Lower and upper parameter bounds supplied to the optimizer.
        model_type : str
            FLI model family, such as mono- or bi-exponential.
        **kwargs : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        Any
            Object produced by unconstrained.
        """
        max_nfev = kwargs.get("max_iter", kwargs.get("maxiter", 2000))

        def wrapper(t_sub: np.ndarray, *p: Any) -> Any:
            """
            Run the wrapper routine.

            Parameters
            ----------
            t_sub : np.ndarray
                Subset of the time axis used during fitting.
            *p : Any
                Detector parameter object or fitted parameter vector.

            Returns
            -------
            Any
                Object produced by wrapper.
            """
            return self.model_fit(self.t, p, model_type=model_type)[self.fit_indices]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, pcov = curve_fit(
                    wrapper,
                    self.t[self.fit_indices],
                    self.decay[self.fit_indices],
                    p0=p0,
                    method="lm",
                    maxfev=max_nfev,
                )
            status = 1
        except Exception:
            return self.fit_with_estimator(
                estimator_type="trust_region",
                model_type=model_type,
                p0=p0,
                bounds=bounds,
            )
        return self._post_process(
            popt, None, status, model_type, pcov=pcov, bounds=bounds
        )

    def model_fit(
        self, t: np.ndarray, params: Any, model_type: str = "mono-exponential"
    ) -> Any:
        """
        Run the model fit routine.

        Parameters
        ----------
        t : np.ndarray
            Time axis or acquisition period used by the calculation.
        params : Any
            Model, detector, or plotting parameters used by the routine.
        model_type : str
            FLI model family, such as mono- or bi-exponential.

        Returns
        -------
        Any
            Object produced by model fit.
        """
        return model_numpy(t, self.irf, params, model_type)

    def _post_process(
        self,
        popt: np.ndarray,
        jac: Any,
        status: np.ndarray,
        model_type: str,
        pcov: np.ndarray | None = None,
        bounds: np.ndarray | None = None,
        objective_chi_sq: float | None = None,
        robust_residuals: np.ndarray | None = None,
    ) -> tuple[Any, ...]:
        """
        Run the post process routine.

        Parameters
        ----------
        popt : np.ndarray
            Optimized model parameter vector.
        jac : Any
            Jacobian matrix returned by the optimizer.
        status : np.ndarray
            Optimizer status flag used during post-processing.
        model_type : str
            FLI model family, such as mono- or bi-exponential.
        pcov : np.ndarray | None
            Parameter covariance matrix.
        bounds : np.ndarray | None
            Lower and upper parameter bounds supplied to the optimizer.
        objective_chi_sq : float | None
            Minimized weighted sum of squares, used to scale the Jacobian-based
            uncertainties; defaults to the Poisson deviance of the fit.
        robust_residuals : np.ndarray | None
            Residuals of an unweighted fit; when given, the uncertainties use the
            heteroscedasticity-robust sandwich estimator instead (see
            :meth:`calculate_uncertainties`).

        Returns
        -------
        tuple[Any, ...]
            ``(popt, perr, r2, chi2, reduced_chi2, ssr, converged, rmse)``; ``chi2``
            is the Poisson deviance and ``reduced_chi2`` is deviance / (n - p).
        """
        d_fit = self.decay[self.fit_indices]
        perr = None
        if pcov is None and jac is not None:
            if objective_chi_sq is None:
                unordered = self.model_fit(self.t, popt, model_type=model_type)
                objective_chi_sq = compute_fli_stats(
                    unordered[self.fit_indices], d_fit, len(popt)
                )[1]
            perr = self.calculate_uncertainties(
                jac,
                objective_chi_sq,
                len(d_fit),
                len(popt),
                residuals=robust_residuals,
            )

        if model_type == "bi-exponential":
            popt, perr, pcov = enforce_tau_ordering(
                popt, perr=perr, pcov=pcov, bounds=bounds
            )

        final_model = self.model_fit(self.t, popt, model_type=model_type)[
            self.fit_indices
        ]
        ssr, chi_sq, red_chi_sq, r_sq, rmse = compute_fli_stats(
            final_model, d_fit, len(popt)
        )

        if pcov is not None:
            perr = np.sqrt(np.maximum(np.diag(pcov), 0))
        elif perr is None:
            perr = np.full(len(popt), np.nan)

        return popt, perr, r_sq, chi_sq, red_chi_sq, ssr, (1 if status > 0 else 0), rmse

    def calculate_uncertainties(
        self,
        jacobian: Any,
        chi_sq: np.ndarray,
        n_data: int,
        n_params: int,
        residuals: np.ndarray | None = None,
    ) -> Any:
        """
        Calculate uncertainties.

        Parameters
        ----------
        jacobian : Any
            Jacobian matrix used to estimate parameter uncertainty.
        chi_sq : np.ndarray
            Chi-square statistic used to scale uncertainty estimates.
        n_data : int
            Number of samples, components, gates, or iterations used by the routine.
        n_params : int
            Number of fitted model parameters.
        residuals : np.ndarray | None
            Residuals of an unweighted fit. When given, the covariance is the
            heteroscedasticity-robust sandwich ``(J^T J)^-1 J^T diag(r^2) J
            (J^T J)^-1 * n / (n - p)``, since unweighted residuals of photon counts
            do not share one variance; otherwise it is ``(J^T J)^-1 * chi_sq /
            (n - p)`` for residuals already weighted by their inverse standard
            deviation.

        Returns
        -------
        Any
            One-standard-deviation uncertainty of each parameter.
        """
        try:
            dof = n_data - n_params
            if dof <= 0 or chi_sq <= 0:
                return np.zeros(n_params)
            red_chi_sq = chi_sq / dof
            jacobian = np.asarray(jacobian, dtype=float)
            col_norm = np.linalg.norm(jacobian, axis=0)
            col_norm = np.where(col_norm > 0, col_norm, 1.0)
            scaled = jacobian / col_norm
            bread = np.linalg.pinv(scaled.T @ scaled)
            if residuals is not None:
                meat = scaled.T @ (scaled * (np.asarray(residuals) ** 2)[:, None])
                cov = bread @ meat @ bread * n_data / dof
            else:
                cov = bread * red_chi_sq
            cov = cov / np.outer(col_norm, col_norm)
            return np.sqrt(np.maximum(np.diag(cov), 0))
        except Exception:
            return np.full(n_params, np.nan)

    def compare_models(self, alpha: float = 0.05) -> tuple[Any, ...]:
        """
        Compare models.

        Parameters
        ----------
        alpha : float
            Regularization strength, fraction value, or significance threshold used by the routine.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing model-comparison statistics and selected fit results.
        """
        res_m = self.fit_with_estimator(model_type="mono-exponential")
        res_b = self.fit_with_estimator(model_type="bi-exponential")
        n, p_m, p_b = len(self.fit_indices), 4, 6
        chi_m, chi_b = res_m[3], res_b[3]
        f_stat = ((chi_m - chi_b) / (p_b - p_m)) / (chi_b / (n - p_b))
        p_val = 1 - f.cdf(f_stat, p_b - p_m, n - p_b)
        winner = res_b if p_val < alpha else res_m
        return (
            ("bi-exponential" if p_val < alpha else "mono-exponential"),
            winner[0],
            winner[1],
            winner[2],
            winner[4],
            p_val,
        )

    def get_average_lifetime(self, popt: np.ndarray) -> Any:
        """
        Return average lifetime.

        Parameters
        ----------
        popt : np.ndarray
            Optimized model parameter vector.

        Returns
        -------
        Any
            Object produced by get average lifetime.
        """
        return compute_average_lifetime(popt)

    def get_fret_efficiency(self, popt: np.ndarray) -> Any:
        """
        Return fret efficiency.

        Parameters
        ----------
        popt : np.ndarray
            Optimized model parameter vector.

        Returns
        -------
        Any
            Object produced by get FRET efficiency.
        """
        return compute_fret_efficiency(popt)

    def set_fit_range(self, start_pct: int = 0, end_pct: int = 100) -> None:
        """
        Set fit range.

        Parameters
        ----------
        start_pct : int
            Start percentage of the decay range used for fitting.
        end_pct : int
            End percentage of the decay range used for fitting.

        Returns
        -------
        None
            No object is returned; the function set fit range.
        """
        start_idx = int((start_pct / 100.0) * self.N)
        end_idx = int((end_pct / 100.0) * self.N)
        self.fit_indices = np.arange(start_idx, min(end_idx, self.N))
