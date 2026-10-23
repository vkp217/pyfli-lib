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
from .shared_metrics import _FIXED_BOUND_TOL, compute_fli_stats, enforce_tau_ordering

MLE_WEIGHTINGS = ("poisson", "pearson", "neyman", "none")
"""Objectives of :meth:`MLEFLIFitter.fit_with_estimator` (``"poisson"`` is the default)."""


class MLEFLIFitter(BaseFLIFitter):
    """
    Extend the base FLI fitter with Poisson, Pearson, and Neyman objective functions.
    It supports MLE-style fitting, Fisher-information (Poisson) and robust sandwich
    (chi-square) parameter uncertainties, and likelihood-based model comparison.
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
            perr = self.parameter_errors(
                popt, model_type, weighting=weighting, bounds=(l_vec, h_vec)
            )
        except Exception:
            perr = np.full(len(popt), np.nan)

        return self._post_process(
            popt, None, converged, model_type, manual_perr=perr, bounds=(l_vec, h_vec)
        )

    def parameter_errors(
        self,
        popt: np.ndarray,
        model_type: str,
        weighting: str = "poisson",
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        One-standard-deviation uncertainties of parameters fitted with `weighting`.

        ``"poisson"`` uses the inverse Fisher information ``(J^T diag(1/mu) J)^-1`` of
        the Poisson likelihood at `popt`, i.e. the asymptotic covariance (and
        Cramer-Rao bound) of the MLE. ``"pearson"``, ``"neyman"`` and ``"none"`` use
        the heteroscedasticity-robust sandwich of :meth:`calculate_uncertainties` on
        the weighted Jacobian and residuals.

        Parameters
        ----------
        popt : np.ndarray
            Fitted parameter vector (before tau ordering).
        model_type : str
            ``"mono-exponential"`` or ``"bi-exponential"``.
        weighting : str
            Objective the parameters were fitted with (see :data:`MLE_WEIGHTINGS`).
        bounds : tuple[np.ndarray, np.ndarray] | None
            ``(low, high)`` bounds of the fit. Parameters pinned by them
            (low == high) get 0; parameters sitting on an active bound get NaN,
            because the asymptotic normal approximation does not hold there. The
            remaining uncertainties are computed with those parameters held fixed.

        Returns
        -------
        np.ndarray
            Standard error of each parameter.
        """
        popt = np.asarray(popt, dtype=float)
        n_params = len(popt)
        if bounds is None:
            low = np.full(n_params, -np.inf)
            high = np.full(n_params, np.inf)
        else:
            low = np.asarray(bounds[0], dtype=float)
            high = np.asarray(bounds[1], dtype=float)

        width = np.maximum(np.abs(popt), 1.0)
        pinned = np.abs(high - low) < _FIXED_BOUND_TOL
        on_bound = ~pinned & (
            (popt - low <= 1e-8 * width) | (high - popt <= 1e-8 * width)
        )
        perr = np.full(n_params, np.nan)
        perr[pinned] = 0.0
        free = np.flatnonzero(~pinned & ~on_bound)
        if free.size == 0:
            return perr

        data = self.decay[self.fit_indices]
        mu = self.model_fit(self.t, popt, model_type=model_type)[self.fit_indices]
        jac = np.empty((len(data), free.size))
        for j, k in enumerate(free):
            step = 1e-6 * width[k]
            up, down = popt.copy(), popt.copy()
            up[k] = min(popt[k] + step, high[k])
            down[k] = max(popt[k] - step, low[k])
            jac[:, j] = (
                self.model_fit(self.t, up, model_type=model_type)[self.fit_indices]
                - self.model_fit(self.t, down, model_type=model_type)[self.fit_indices]
            ) / (up[k] - down[k])

        if weighting == "poisson":
            info = jac.T @ (jac / np.clip(mu, 1e-9, None)[:, None])
            diag = np.diag(info)
            norm = np.sqrt(np.where(diag > 0, diag, 1.0))
            cov = np.linalg.pinv(info / np.outer(norm, norm)) / np.outer(norm, norm)
            perr[free] = np.where(
                diag > 0, np.sqrt(np.maximum(np.diag(cov), 0.0)), np.nan
            )
            return perr

        if weighting == "pearson":
            weights = 1.0 / np.clip(mu, 1e-9, None)
        elif weighting == "neyman":
            weights = 1.0 / np.clip(data, 1.0, None)
        else:
            weights = np.ones_like(data, dtype=float)
        root_w = np.sqrt(weights)
        residuals = (data - mu) * root_w
        perr[free] = self.calculate_uncertainties(
            jac * root_w[:, None],
            float(np.sum(residuals**2)),
            len(data),
            free.size,
            residuals=residuals,
        )
        return perr

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
