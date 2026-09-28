#  solver/mle_fitter.py
"""
Extend the base fitter with Poisson and chi-square maximum-likelihood estimators.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLI fitting routines. Public API includes
classes :class:`MLEFLIFitter`.
"""

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, f

from .base_fitter import BaseFLIFitter
from .base_static import resolve_params_and_bounds
from .shared_metrics import compute_fli_stats, enforce_tau_ordering

MLE_WEIGHTINGS = ("poisson", "pearson", "neyman", "none")
"""Objectives of :meth:`MLEFLIFitter.fit_with_estimator` (``"poisson"`` is the default)."""


class MLEFLIFitter(BaseFLIFitter):
    """
    Extend the base FLI fitter with Poisson, Pearson, and Neyman objective functions.
    It supports MLE-style fitting, uncertainty extraction from optimizer curvature, and
    likelihood-based model comparison.
    """

    def poisson_log_likelihood(self, params: Any, model_type: str) -> Any:
        """Standard Poisson MLE (Deviance/C-Statistic)."""
        model = self.model_fit(self.t, params, model_type=model_type)[self.fit_indices]
        if not np.all(np.isfinite(model)):
            return np.inf
        data = self.decay[self.fit_indices]
        model = np.clip(model, 1e-9, None)
        val = 2.0 * np.sum(
            model - data + data * np.log(np.clip(data, 1e-9, None) / model)
        )
        return val if np.isfinite(val) else np.inf

    def pearson_chi_square(self, params: Any, model_type: str) -> np.ndarray:
        """Pearson's Chi-square: Weighted by the MODEL [1/y_model]."""
        model = self.model_fit(self.t, params, model_type=model_type)[self.fit_indices]
        data = self.decay[self.fit_indices]
        model = np.clip(model, 1e-9, None)
        return np.sum((data - model) ** 2 / model)

    def neyman_chi_square(self, params: Any, model_type: str) -> np.ndarray:
        """Neyman's Chi-square: Weighted by the DATA [1/y_data]."""
        model = self.model_fit(self.t, params, model_type=model_type)[self.fit_indices]
        data = self.decay[self.fit_indices]
        weights = np.clip(data, 1.0, None)
        return np.sum((data - model) ** 2 / weights)

    def unweighted_sum_of_squares(self, params: Any, model_type: str) -> np.ndarray:
        """Unweighted sum of squared residuals (every gate counts equally)."""
        model = self.model_fit(self.t, params, model_type=model_type)[self.fit_indices]
        data = self.decay[self.fit_indices]
        return np.sum((data - model) ** 2)

    def fit_with_estimator(
        self,
        estimator_type: str = "poisson",
        p0: Any | None = None,
        bounds: np.ndarray | None = None,
        model_type: str = "bi-exponential",
        weighting: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Main interface for MLE/Chi-square fitting.
        Fully compatible with BaseFLIFitter registry and offset-based parameter resolving.

        Parameters
        ----------
        estimator_type : str
            Objective name (``"poisson"``, ``"pearson"`` or ``"neyman"``); used when
            `weighting` is not given. Unknown names fall back to ``"poisson"``.
        p0, bounds : Any
            Initial parameters and bounds (see :func:`resolve_params_and_bounds`).
        model_type : str
            ``"mono-exponential"`` or ``"bi-exponential"``.
        weighting : str | None
            Objective to minimize (see :data:`MLE_WEIGHTINGS`); overrides
            `estimator_type`:

            - ``"poisson"`` (default): Poisson deviance -- the maximum-likelihood
              estimator for photon counts; the consistent, least-biased choice.
            - ``"pearson"``: ``sum (d - mu)^2 / mu`` minimized directly. Because the
              weights move with the model, it is biased towards larger model values
              (for decays, longer lifetimes) at low counts.
            - ``"neyman"``: ``sum (d - mu)^2 / max(d, 1)``; biased towards low
              counts (shorter lifetimes) at low counts.
            - ``"none"``: unweighted sum of squares.
        **kwargs : Any
            ``max_iter`` (or ``maxiter``, default 2000) optimizer iterations,
            ``maxfun`` (default 50000), ``ftol`` and ``gtol``.
        """
        if weighting is None:
            weighting = (
                estimator_type if estimator_type in MLE_WEIGHTINGS else "poisson"
            )
        elif weighting not in MLE_WEIGHTINGS:
            raise ValueError(
                f"weighting must be one of {MLE_WEIGHTINGS}, got {weighting!r}"
            )
        # Call the external static logic to merge guesses and bounds
        p0_safe, (l_vec, h_vec) = resolve_params_and_bounds(
            p0,
            bounds,
            model_type,
            self.t,
            self.decay,
            self.T_laser,
            self.guess_plugin,
            self.T_acq,
        )
        bnds = list(zip(l_vec, h_vec))

        funcs = {
            "poisson": self.poisson_log_likelihood,
            "pearson": self.pearson_chi_square,
            "neyman": self.neyman_chi_square,
            "none": self.unweighted_sum_of_squares,
        }
        obj_func = funcs[weighting]
        scale = np.maximum(np.abs(p0_safe), 1.0)

        res = minimize(
            lambda z, mt: obj_func(z * scale, mt),
            x0=p0_safe / scale,
            args=(model_type,),
            bounds=[(lo / s, hi / s) for (lo, hi), s in zip(bnds, scale)],
            method="L-BFGS-B",
            options={
                "ftol": kwargs.get("ftol", 1e-12),
                "gtol": kwargs.get("gtol", 1e-9),
                "maxfun": kwargs.get("maxfun", 50000),
                "maxiter": kwargs.get("max_iter", kwargs.get("maxiter", 2000)),
            },
        )

        popt = res.x * scale
        converged = 1 if res.success else 0

        try:
            cov_scaled = 2.0 * (res.hess_inv @ np.eye(len(popt)))
            if weighting == "none":
                dof = max(len(self.fit_indices) - len(popt), 1)
                cov_scaled = cov_scaled * res.fun / dof
            perr = np.sqrt(np.maximum(np.diag(cov_scaled), 0.0)) * scale
        except Exception:
            perr = np.full(len(popt), np.nan)

        return self._post_process(
            popt, None, converged, model_type, manual_perr=perr, bounds=(l_vec, h_vec)
        )

    def _post_process(
        self,
        popt: np.ndarray,
        jac: Any,
        status: np.ndarray,
        model_type: str,
        pcov: np.ndarray | None = None,
        manual_stat: np.ndarray | None = None,
        manual_perr: np.ndarray | None = None,
        bounds: np.ndarray | None = None,
    ) -> tuple[Any, ...]:
        """
        Compatible post-processor that enforces tau1 <= tau2 and handles MLE-specific statistics.
        """
        if model_type == "bi-exponential":
            popt, manual_perr, _ = enforce_tau_ordering(
                popt, perr=manual_perr, bounds=bounds
            )

        data = self.decay[self.fit_indices]
        final_model = self.model_fit(self.t, popt, model_type=model_type)[
            self.fit_indices
        ]

        ssr, chi_sq, red_chi_sq, r_sq, rmse = compute_fli_stats(
            final_model, data, len(popt)
        )

        perr = manual_perr if manual_perr is not None else np.full(len(popt), np.nan)

        return popt, perr, r_sq, chi_sq, red_chi_sq, ssr, (1 if status > 0 else 0), rmse

    def compare_models(
        self, alpha: float = 0.05, estimator: str = "poisson"
    ) -> tuple[Any, ...]:
        """
        Statistical model selection.
        - Poisson: Uses Likelihood Ratio Test (LRT) on Deviance.
        - Chi-square/LS: Uses F-test on the reported chi-square (res[3], the Poisson
          deviance from compute_fli_stats).
        """
        res_m = self.fit_with_estimator(estimator, model_type="mono-exponential")
        res_b = self.fit_with_estimator(estimator, model_type="bi-exponential")

        if estimator == "poisson":
            # LRT: Difference in Deviance follows a Chi-square distribution
            dev_m = self.poisson_log_likelihood(res_m[0], "mono-exponential")
            dev_b = self.poisson_log_likelihood(res_b[0], "bi-exponential")
            LRT_stat = max(dev_m - dev_b, 0.0)
            p_val = 1.0 - chi2.cdf(LRT_stat, df=2)
        else:
            # F-test on the reported chi-square (res[3]: Poisson deviance)
            n, p_m, p_b = len(self.fit_indices), 4, 6
            chi_m, chi_b = res_m[3], res_b[3]
            f_stat = ((chi_m - chi_b) / (p_b - p_m)) / (chi_b / (n - p_b))
            p_val = 1.0 - f.cdf(f_stat, p_b - p_m, n - p_b)

        winner = res_b if p_val < alpha else res_m
        return (
            "bi-exponential" if p_val < alpha else "mono-exponential",
            winner[0],
            winner[1],
            winner[2],
            winner[4],
            p_val,
        )
