"""
Tests for ``MLEFLIFitter`` parameter uncertainties.

The reported standard errors must match the actual Monte Carlo spread of the fitted
parameters: the Poisson MLE uses the inverse Fisher information, the chi-square
objectives a robust sandwich. Pinned parameters get 0 and parameters on an active
bound get NaN.
"""

import numpy as np
import pytest

from pyfli.solver.forward_model import model_numpy
from pyfli.solver.mle_fitter import MLE_WEIGHTINGS, MLEFLIFitter

_N = 256
_T = np.arange(_N) * 12.5 / _N
_FREQ = [80.0, 80.0]
_TRUE = np.array([5000.0, 1.5, 1.0, 0.0])
_MODEL = "mono-exponential"


def _irf():
    irf = np.exp(-0.5 * ((_T - 0.8) / 0.08) ** 2)
    return irf / irf.sum()


@pytest.fixture(scope="module")
def monte_carlo():
    mu = model_numpy(_T, _irf(), _TRUE, _MODEL)
    decays = np.random.default_rng(0).poisson(mu, size=(80, _N)).astype(float)
    out = {}
    for weighting in MLE_WEIGHTINGS:
        fits = [
            MLEFLIFitter(_FREQ, d, _irf()).fit_with_estimator(
                model_type=_MODEL, weighting=weighting
            )
            for d in decays
        ]
        out[weighting] = (
            np.array([f[0] for f in fits]),
            np.array([f[1] for f in fits]),
        )
    return out


def test_poisson_se_matches_monte_carlo_spread(monte_carlo):
    popt, perr = monte_carlo["poisson"]
    spread = popt.std(axis=0, ddof=1)
    np.testing.assert_allclose(np.median(perr, axis=0), spread, rtol=0.2)


def test_poisson_se_equals_inverse_fisher_information():
    mu = model_numpy(_T, _irf(), _TRUE, _MODEL)
    decay = np.random.default_rng(1).poisson(mu).astype(float)
    fitter = MLEFLIFitter(_FREQ, decay, _irf())
    popt, perr = fitter.fit_with_estimator(model_type=_MODEL)[:2]

    jac = np.empty((_N, len(popt)))
    for k in range(len(popt)):
        step = np.zeros_like(popt)
        step[k] = 1e-5 * max(abs(popt[k]), 1.0)
        jac[:, k] = (
            model_numpy(_T, _irf(), popt + step, _MODEL)
            - model_numpy(_T, _irf(), popt - step, _MODEL)
        ) / (2 * step[k])
    mu_hat = model_numpy(_T, _irf(), popt, _MODEL)
    fisher = jac.T @ (jac / mu_hat[:, None])
    np.testing.assert_allclose(perr, np.sqrt(np.diag(np.linalg.inv(fisher))), rtol=1e-3)


@pytest.mark.parametrize("weighting", ["pearson", "neyman", "none"])
def test_chi_square_se_matches_monte_carlo_spread(monte_carlo, weighting):
    popt, perr = monte_carlo[weighting]
    spread = popt.std(axis=0, ddof=1)
    np.testing.assert_allclose(np.median(perr, axis=0), spread, rtol=0.25)


def test_pinned_and_active_bound_parameters():
    mu = model_numpy(_T, _irf(), _TRUE, _MODEL)
    decay = np.random.default_rng(2).poisson(mu).astype(float)
    fitter = MLEFLIFitter(_FREQ, decay, _irf())
    popt = fitter.fit_with_estimator(model_type=_MODEL)[0]

    low, high = popt - 10.0, popt + 10.0
    low[2] = high[2] = popt[2]
    low[3] = popt[3]
    perr = fitter.parameter_errors(popt, _MODEL, bounds=(low, high))

    assert perr[2] == 0.0
    assert np.isnan(perr[3])
    assert np.all(np.isfinite(perr[:2])) and np.all(perr[:2] > 0)
