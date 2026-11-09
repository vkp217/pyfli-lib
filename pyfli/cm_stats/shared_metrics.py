"""
Centralize tau ordering, lifetime summaries, FRET efficiency, and fit-quality metrics.

This module belongs to :mod:`pyfli.cm_stats` and is shared by the PyFLI solver,
reconstruction, Laguerre, and simulator-calibration packages. Public API includes
functions :func:`enforce_tau_ordering`, :func:`poisson_deviance`,
:func:`expected_poisson_deviance`, :func:`reduced_poisson_deviance`,
:func:`pearson_chi_square`, :func:`compute_fli_stats`, :func:`compute_pearson_stats`,
:func:`compute_average_lifetime`, and :func:`compute_fret_efficiency`. The deviance,
Pearson and residual statistics themselves are defined in :mod:`.deviance`,
:mod:`.chi_square` and :mod:`.goodness_of_fit` and re-exported here.

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

from .chi_square import degrees_of_freedom, pearson_chi_square
from .deviance import (
    expected_poisson_deviance,
    poisson_deviance,
    reduced_poisson_deviance,
)
from .goodness_of_fit import r_squared, rmse, sum_squared_residuals

__all__ = [
    "compute_average_lifetime",
    "compute_fli_stats",
    "compute_fret_efficiency",
    "compute_pearson_stats",
    "enforce_tau_ordering",
    "expected_poisson_deviance",
    "pearson_chi_square",
    "poisson_deviance",
    "reduced_poisson_deviance",
]

_FIXED_BOUND_TOL = 1e-5


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
    ssr = float(sum_squared_residuals(residuals, axis=None))
    chi_sq, red_chi_sq = reduced_poisson_deviance(final_model, d_fit, n_params)
    chi_sq, red_chi_sq = float(chi_sq), float(red_chi_sq)
    r_sq = float(r_squared(residuals, d_fit, axis=None))
    return ssr, chi_sq, red_chi_sq, r_sq, float(rmse(residuals, axis=None))


def compute_pearson_stats(
    final_model: np.ndarray, d_fit: np.ndarray, n_params: int
) -> tuple[float, float]:
    """
    ``(pearson_chi2, pearson_reduced_chi2)`` with the former definition (variance
    floored at 1 count, dof = n - p), for comparison with earlier results.
    """
    chi = float(pearson_chi_square(final_model, d_fit))
    return chi, chi / degrees_of_freedom(len(d_fit), n_params)


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
