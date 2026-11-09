"""
Poisson deviance statistics: per-gate terms, the summed and reduced deviance, and its
expectation (and variance) under the model for photon counts.
"""

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.special import gammaln, xlogy

NONPOSITIVE_MODES = ("zero", "nan", "floor")

MU_MIN = 1e-8
MU_SWITCH = 1e3
N_GRID = 4401


def poisson_deviance_terms(
    model: Any,
    data: Any,
    floor: float = 1e-12,
    nonpositive: str = "zero",
    data_floor: float = 1e-9,
) -> Any:
    """
    Per-gate Poisson deviance ``2 (mu - d + d ln(d / mu))`` with ``mu`` floored at `floor`.

    `nonpositive` sets the ``d ln(d / mu)`` term of gates with ``d <= 0``: ``"zero"`` (0),
    ``"nan"`` (scipy ``xlogy``: 0 for ``d = 0``, NaN for ``d < 0``) or ``"floor"``
    (``d ln(max(d, data_floor) / mu)`` on every gate).
    """
    if nonpositive not in NONPOSITIVE_MODES:
        raise ValueError(
            f"nonpositive must be one of {NONPOSITIVE_MODES}, got {nonpositive!r}."
        )
    mu = np.clip(np.asarray(model, dtype=float), floor, None)
    d = np.asarray(data, dtype=float)
    if nonpositive == "nan":
        return 2.0 * (xlogy(d, d / mu) - (d - mu))
    if nonpositive == "floor":
        return 2.0 * (mu - d + d * np.log(np.clip(d, data_floor, None) / mu))
    positive = d > 0
    log_term = np.where(positive, d * np.log(np.where(positive, d, 1.0) / mu), 0.0)
    return 2.0 * (mu - d + log_term)


def poisson_deviance(model: Any, data: Any, axis: int = -1, **kwargs: Any) -> Any:
    """
    Poisson deviance: :func:`poisson_deviance_terms` summed along `axis` (gates with
    ``d <= 0`` contribute ``2 * (mu - d)`` by default). For counts drawn from the model,
    ``deviance / (sum E[D_k] - p)`` averages 1 (see :func:`reduced_poisson_deviance`).
    """
    return np.sum(poisson_deviance_terms(model, data, **kwargs), axis=axis)


@lru_cache(maxsize=4)
def poisson_deviance_moment_table(
    mu_min: float = MU_MIN, mu_max: float = MU_SWITCH, n_grid: int = N_GRID
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ln mu, E[d], Var[d])`` of the per-gate deviance on a log grid of ``mu``, exact by summation."""
    mu_grid = np.logspace(np.log10(mu_min), np.log10(mu_max), n_grid)
    E = np.empty(n_grid)
    V = np.empty(n_grid)
    for i, m in enumerate(mu_grid):
        half_width = 14.0 * np.sqrt(m) + 30.0
        y = np.arange(
            max(0, int(m - half_width)), int(np.ceil(m + half_width)) + 1, dtype=float
        )
        p = np.exp(xlogy(y, m) - m - gammaln(y + 1.0))
        d = poisson_deviance_terms(m, y, nonpositive="nan")
        E[i] = np.sum(p * d)
        V[i] = np.sum(p * (d - E[i]) ** 2)
    return np.log(mu_grid), E, V


def poisson_deviance_moments(model: Any) -> tuple[Any, Any]:
    """
    Exact ``(E[d], Var[d])`` of the per-gate deviance for ``d ~ Poisson(mu)``, element-wise:
    tabulated below ``mu = 1e3`` (0 at ``mu = 0``), then ``E = 1 + 1/(6 mu) + 1/(6 mu^2)``
    and ``Var = 2 + 2/(3 mu) + 4/(3 mu^2)``.
    """
    mu = np.clip(np.asarray(model, dtype=float), 0.0, None)
    log_grid, E_tab, V_tab = poisson_deviance_moment_table()
    safe = np.where(mu > 0, mu, MU_MIN)
    log_mu = np.log(safe)
    small = mu < MU_SWITCH
    positive = mu > 0
    E = np.where(
        small,
        np.where(positive, np.interp(log_mu, log_grid, E_tab, left=0.0), 0.0),
        1.0 + 1.0 / (6.0 * safe) + 1.0 / (6.0 * safe**2),
    )
    V = np.where(
        small,
        np.where(positive, np.interp(log_mu, log_grid, V_tab, left=0.0), 0.0),
        2.0 + 2.0 / (3.0 * safe) + 4.0 / (3.0 * safe**2),
    )
    return E, V


def expected_poisson_deviance(model: Any) -> Any:
    """
    Expected Poisson deviance ``E[2 * (mu - d + d * ln(d / mu))]`` of a gate with
    expected count `mu` (element-wise), for ``d ~ Poisson(mu)``: about ``2 mu
    ln(1/mu)`` for ``mu -> 0``, peaks near 1.15 at ``mu ~ 1`` and tends to
    ``1 + 1/(6 mu)`` for large ``mu`` (:func:`poisson_deviance_moments`).
    """
    return poisson_deviance_moments(model)[0]


def deviance_dof(model: Any, n_params: int, weights: Any = None, axis: int = -1) -> Any:
    """Expected deviance minus fitted parameters, ``sum_k w_k E[d_k] - n_params``: the reduced-deviance denominator."""
    expected = expected_poisson_deviance(model)
    if weights is not None:
        expected = weights * expected
    return np.sum(expected, axis=axis) - n_params


def reduced_poisson_deviance(
    model: Any, data: Any, n_params: int, axis: int = -1
) -> tuple[Any, Any]:
    """
    ``(D, D_reduced)``: the Poisson deviance along `axis` and the deviance divided
    by its expectation under the model minus the number of fitted parameters,
    ``D / max(sum E[D_k] - n_params, 1)``, which averages 1 for a correct model.
    """
    deviance = poisson_deviance(model, data, axis=axis)
    return deviance, deviance / np.maximum(
        deviance_dof(model, n_params, axis=axis), 1.0
    )
