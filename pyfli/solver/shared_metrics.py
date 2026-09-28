"""
Centralize tau ordering, lifetime summaries, FRET efficiency, and fit-quality metrics.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLI fitting routines. Public API includes
functions :func:`enforce_tau_ordering`, :func:`poisson_deviance`,
:func:`expected_poisson_deviance`, :func:`reduced_poisson_deviance`,
:func:`pearson_chi_square`, :func:`compute_fli_stats`, :func:`compute_pearson_stats`,
:func:`compute_average_lifetime`, and :func:`compute_fret_efficiency`.

Goodness of fit is reported as the Poisson deviance ``D`` (the likelihood-ratio
statistic for photon counts). At low counts the expected deviance of a gate is not 1
(about 0.47 at 0.1 expected counts, 1.15 at 1), so the reduced value divides ``D`` by
its expectation under the fitted model, ``sum_k E[D_k](mu_k) - p`` (Kaastra 2017,
A&A 605, A51), instead of ``n - p``; it averages 1 for a correct model at any count
level. The former Pearson statistic, whose variance is floored at 1 count and which
therefore reads below 1 whenever many gates hold less than one expected count, is
still available as :func:`pearson_chi_square` / :func:`compute_pearson_stats`.
"""

from typing import Any

import numpy as np
from scipy.stats import poisson

_FIXED_BOUND_TOL = 1e-5
_ED_SWITCH = 10.0


def _exact_expected_deviance(mu: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    k = np.arange(int(np.ceil(mu.max() + 12 * np.sqrt(mu.max()) + 15)) + 1)[:, None]
    safe_k = np.where(k > 0, k, 1)
    term = 2.0 * (mu - k + np.where(k > 0, k * np.log(safe_k / mu), 0.0))
    return np.sum(poisson.pmf(k, mu) * term, axis=0)


_ED_GRID = np.geomspace(1e-8, _ED_SWITCH, 800)
_ED_TABLE = _exact_expected_deviance(_ED_GRID)


def enforce_tau_ordering(
    popt: np.ndarray,
    perr: Any | None = None,
    pcov: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Any, ...]:
    """
    Enforce tau ordering.

    Parameters
    ----------
    popt : np.ndarray
        Optimized model parameter vector.
    perr : Any | None
        One-standard-deviation parameter uncertainty estimates.
    pcov : np.ndarray | None
        Parameter covariance matrix.
    bounds : tuple[np.ndarray, np.ndarray] | None
        Optional (low, high) bound vectors used for the fit. When either tau1 or tau2
        was pinned by the caller (low == high), that parameter's slot is left alone.

    Returns
    -------
    tuple[Any, ...]
        Tuple containing the reordered parameter vector and any reordered uncertainty or covariance data.
    """
    popt = np.asarray(popt, dtype=float)

    if bounds is not None:
        low, high = bounds
        tau1_fixed = abs(float(high[2]) - float(low[2])) < _FIXED_BOUND_TOL
        tau2_fixed = abs(float(high[3]) - float(low[3])) < _FIXED_BOUND_TOL
        if tau1_fixed or tau2_fixed:
            return popt, perr, pcov

    if popt[1] > 0.999:
        popt[1], popt[3] = 1.0, popt[2]
    elif popt[1] < 0.001:
        popt[1], popt[2] = 0.0, popt[3]

    if popt[2] > popt[3]:
        popt[2], popt[3] = popt[3], popt[2]
        popt[1] = 1.0 - popt[1]
        if perr is not None:
            perr = np.asarray(perr, dtype=float)
            perr[2], perr[3] = perr[3], perr[2]
        if pcov is not None:
            pcov[[2, 3], :] = pcov[[3, 2], :]
            pcov[:, [2, 3]] = pcov[:, [3, 2]]

    return popt, perr, pcov


def poisson_deviance(model: Any, data: Any, axis: int = -1) -> Any:
    """
    Poisson deviance ``2 * sum(mu - d + d * ln(d / mu))`` along `axis` (gates with
    ``d <= 0`` contribute ``2 * (mu - d)``). For counts drawn from the model,
    ``deviance / (sum E[D_k] - p)`` averages 1 (see :func:`reduced_poisson_deviance`).
    """
    mu = np.clip(np.asarray(model, dtype=float), 1e-12, None)
    d = np.asarray(data, dtype=float)
    positive = d > 0
    log_term = np.where(positive, d * np.log(np.where(positive, d, 1.0) / mu), 0.0)
    return 2.0 * np.sum(mu - d + log_term, axis=axis)


def expected_poisson_deviance(model: Any) -> Any:
    """
    Expected Poisson deviance ``E[2 * (mu - d + d * ln(d / mu))]`` of a gate with
    expected count `mu` (element-wise), for ``d ~ Poisson(mu)``: about ``2 mu
    ln(1/mu)`` for ``mu -> 0``, peaks near 1.15 at ``mu ~ 1`` and tends to
    ``1 + 1/(6 mu)`` for large ``mu``. Tabulated exactly below ``mu = 10``.
    """
    mu = np.clip(np.asarray(model, dtype=float), 0.0, None)
    safe = np.where(mu > 0, mu, _ED_GRID[0])
    small = np.interp(np.log(safe), np.log(_ED_GRID), _ED_TABLE, left=0.0)
    large = 1.0 + 1.0 / (6.0 * safe) + 1.0 / (6.0 * safe**2)
    return np.where(mu < _ED_SWITCH, np.where(mu > 0, small, 0.0), large)


def reduced_poisson_deviance(
    model: Any, data: Any, n_params: int, axis: int = -1
) -> tuple[Any, Any]:
    """
    ``(D, D_reduced)``: the Poisson deviance along `axis` and the deviance divided
    by its expectation under the model minus the number of fitted parameters,
    ``D / max(sum E[D_k] - n_params, 1)``, which averages 1 for a correct model.
    """
    deviance = poisson_deviance(model, data, axis=axis)
    expected = np.sum(expected_poisson_deviance(model), axis=axis)
    return deviance, deviance / np.maximum(expected - n_params, 1.0)


def pearson_chi_square(model: Any, data: Any, axis: int = -1) -> Any:
    """
    Pearson chi-square ``sum((d - mu)^2 / max(mu, 1))`` along `axis` -- the fit
    statistic reported before the switch to the Poisson deviance. The variance floor
    of 1 count makes it read below ``n - p`` when many gates hold less than one
    expected count.
    """
    mu = np.asarray(model, dtype=float)
    d = np.asarray(data, dtype=float)
    return np.sum((d - mu) ** 2 / np.clip(mu, 1.0, None), axis=axis)


def compute_fli_stats(
    final_model: np.ndarray, d_fit: np.ndarray, n_params: int
) -> tuple[Any, ...]:
    """
    Compute FLI fit statistics.

    Parameters
    ----------
    final_model : np.ndarray
        Model decay evaluated at the fitted parameters.
    d_fit : np.ndarray
        Measured decay samples over the fitted range.
    n_params : int
        Number of fitted model parameters.

    Returns
    -------
    tuple[Any, ...]
        ``(ssr, chi_sq, red_chi_sq, r_sq, rmse)``: sum of squared residuals, the
        Poisson deviance (:func:`poisson_deviance`), the reduced deviance
        (:func:`reduced_poisson_deviance`), R-squared and RMSE.
    """
    residuals = final_model - d_fit
    ssr = float(np.sum(residuals**2))
    chi_sq, red_chi_sq = reduced_poisson_deviance(final_model, d_fit, n_params)
    chi_sq, red_chi_sq = float(chi_sq), float(red_chi_sq)
    ss_tot = float(np.sum((d_fit - np.mean(d_fit)) ** 2))
    r_sq = 1.0 - ssr / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return ssr, chi_sq, red_chi_sq, r_sq, rmse


def compute_pearson_stats(
    final_model: np.ndarray, d_fit: np.ndarray, n_params: int
) -> tuple[float, float]:
    """
    ``(pearson_chi2, pearson_reduced_chi2)`` with the former definition (variance
    floored at 1 count, dof = n - p), for comparison with earlier results.
    """
    chi = float(pearson_chi_square(final_model, d_fit))
    return chi, chi / max(len(d_fit) - n_params, 1)


def compute_average_lifetime(popt: np.ndarray) -> float:
    """
    Compute average lifetime.

    Parameters
    ----------
    popt : np.ndarray
        Optimized model parameter vector.

    Returns
    -------
    float
        Amplitude-weighted average lifetime for bi-exponential fits or the mono-exponential lifetime.
    """
    if len(popt) == 6:
        return float(popt[1] * popt[2] + (1.0 - popt[1]) * popt[3])
    return float(popt[1])


def compute_fret_efficiency(popt: np.ndarray) -> float:
    """
    Compute FRET efficiency.

    Parameters
    ----------
    popt : np.ndarray
        Optimized model parameter vector.

    Returns
    -------
    float
        FRET efficiency estimated from the fitted short and long lifetimes.
    """
    if len(popt) == 6:
        tau1, tau2 = popt[2], popt[3]
        if tau2 > 0:
            return float(1.0 - tau1 / tau2)
    return 0.0
