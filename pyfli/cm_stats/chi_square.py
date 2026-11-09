"""
Pearson and Neyman chi-square statistics and the degrees of freedom of a fit.
"""

from typing import Any

import numpy as np


def degrees_of_freedom(n: Any, n_params: Any, minimum: Any = 1) -> Any:
    """``n - n_params``, raised to at least `minimum` (``None`` leaves it unbounded)."""
    nu = n - n_params
    if minimum is None:
        return nu
    if np.ndim(nu) == 0:
        return max(nu, minimum)
    return np.maximum(nu, minimum)


def pearson_terms(model: Any, data: Any, floor: float = 1.0) -> Any:
    """Per-gate Pearson terms ``(d - mu)^2 / max(mu, floor)``."""
    mu = np.asarray(model, dtype=float)
    d = np.asarray(data, dtype=float)
    return (d - mu) ** 2 / np.clip(mu, floor, None)


def pearson_chi_square(
    model: Any, data: Any, axis: int = -1, floor: float = 1.0
) -> Any:
    """
    Pearson chi-square ``sum((d - mu)^2 / max(mu, floor))`` along `axis` -- the fit
    statistic reported before the switch to the Poisson deviance. The default variance
    floor of 1 count makes it read below ``n - p`` when many gates hold less than one
    expected count.
    """
    return np.sum(pearson_terms(model, data, floor), axis=axis)


def neyman_chi_square(model: Any, data: Any, axis: int = -1, floor: float = 1.0) -> Any:
    """Neyman chi-square ``sum((d - mu)^2 / max(d, floor))`` along `axis` (variance from the data)."""
    mu = np.asarray(model, dtype=float)
    d = np.asarray(data, dtype=float)
    return np.sum((d - mu) ** 2 / np.clip(d, floor, None), axis=axis)
