"""Generic Fisher information and Cramér-Rao lower bound (CRLB) tools."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def numerical_jacobian(
    model: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    rel_step: float = 1e-4,
    abs_step: float = 1e-8,
) -> np.ndarray:
    """
    Central-difference Jacobian ``dmu/dtheta`` of a batched model.

    Parameters
    ----------
    model : Callable
        ``model(params)`` with `params` ``(P, K)`` returning channel means ``(P, T)``.
    params : np.ndarray
        Parameter vectors ``(P, K)`` (or ``(K,)`` for a single one).
    rel_step, abs_step : float
        Step for parameter ``k`` is ``max(rel_step * |theta_k|, abs_step)``.

    Returns
    -------
    np.ndarray
        Jacobian of shape ``(P, K, T)``.
    """
    params = np.atleast_2d(np.asarray(params, dtype=float))
    columns = []
    for k in range(params.shape[1]):
        h = np.maximum(rel_step * np.abs(params[:, k]), abs_step)
        up, down = params.copy(), params.copy()
        up[:, k] += h
        down[:, k] -= h
        columns.append((model(up) - model(down)) / (2.0 * h)[:, None])
    return np.stack(columns, axis=1)


def poisson_fisher_information(mu: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    """
    Fisher matrices ``(P, K, K)`` for independent Poisson channels with means `mu`
    ``(P, T)`` and Jacobian ``(P, K, T)``. Channels with ``mu <= 0`` carry no
    information and are skipped.
    """
    mu = np.asarray(mu, dtype=float)
    weight = np.divide(1.0, mu, out=np.zeros_like(mu), where=mu > 0)
    return np.einsum("pit,pjt,pt->pij", jacobian, jacobian, weight)


def gaussian_fisher_information(sigma: Any, jacobian: np.ndarray) -> np.ndarray:
    """
    Fisher matrices ``(P, K, K)`` for independent Gaussian channels with known noise
    standard deviation `sigma` (scalar, ``(T,)`` or ``(P, T)``).
    """
    shape = (jacobian.shape[0], jacobian.shape[2])
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), shape)
    weight = np.divide(1.0, sigma**2, out=np.zeros(shape), where=sigma > 0)
    return np.einsum("pit,pjt,pt->pij", jacobian, jacobian, weight)


def fisher_information(
    model: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    noise: str = "poisson",
    sigma: Any = None,
    jacobian: np.ndarray | None = None,
    rel_step: float = 1e-4,
    abs_step: float = 1e-8,
) -> np.ndarray:
    """
    Fisher matrices ``(P, K, K)`` of a batched model at `params` ``(P, K)``.

    Parameters
    ----------
    model : Callable
        ``model(params)`` -> channel means ``(P, T)``.
    params : np.ndarray
        Parameter vectors ``(P, K)``.
    noise : str
        ``"poisson"`` (counts; the means come from the model) or ``"gaussian"``
        (needs `sigma`).
    sigma : Any
        Noise standard deviation for ``noise="gaussian"``.
    jacobian : np.ndarray | None
        Analytic Jacobian ``(P, K, T)``; computed numerically when None.
    rel_step, abs_step : float
        Finite-difference steps (see :func:`numerical_jacobian`).
    """
    params = np.atleast_2d(np.asarray(params, dtype=float))
    if jacobian is None:
        jacobian = numerical_jacobian(model, params, rel_step, abs_step)
    if noise == "poisson":
        return poisson_fisher_information(model(params), jacobian)
    if noise == "gaussian":
        if sigma is None:
            raise ValueError("noise='gaussian' needs sigma.")
        return gaussian_fisher_information(sigma, jacobian)
    raise ValueError(f"noise must be 'poisson' or 'gaussian', got {noise!r}")


def select_parameters(info: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """
    Fisher sub-matrices for the parameters in `indices`, i.e. the information when
    every other parameter is known exactly and not estimated.
    """
    idx = np.asarray(indices)
    return info[..., idx[:, None], idx[None, :]]


def crlb_covariance(info: np.ndarray) -> np.ndarray:
    """
    Cramér-Rao bound on the covariance, ``I^-1`` per batch entry. A (near-)singular
    Fisher matrix -- an unidentifiable parameter -- falls back to the
    pseudo-inverse.
    """
    info = np.asarray(info, dtype=float)
    try:
        return np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(info, hermitian=True)


def crlb_std(info: np.ndarray, index: int) -> np.ndarray:
    """Smallest achievable standard deviation of parameter `index`, per batch entry."""
    var = crlb_covariance(info)[..., index, index]
    return np.sqrt(np.clip(var, 0.0, None))


def crlb_cv(info: np.ndarray, value: Any, index: int) -> np.ndarray:
    """Smallest achievable coefficient of variation ``std / value`` of parameter `index`."""
    return crlb_std(info, index) / np.asarray(value, dtype=float)


def f_value(cv: Any, n_photons: Any) -> np.ndarray:
    """
    Photon-economy F-value ``CV * sqrt(N)``: 1 for an ideal photon-counting
    measurement, larger for any real instrument or estimator. Applies equally to a
    measured CV and to a CRLB CV.
    """
    return np.asarray(cv, dtype=float) * np.sqrt(np.asarray(n_photons, dtype=float))


def estimator_efficiency(cv_measured: Any, cv_bound: Any) -> np.ndarray:
    """
    Statistical efficiency ``(cv_bound / cv_measured) ** 2``: 1 when an estimator
    reaches the CRLB, smaller otherwise (a biased estimator can exceed 1).
    """
    cv_measured = np.asarray(cv_measured, dtype=float)
    cv_bound = np.asarray(cv_bound, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cv_measured > 0, (cv_bound / cv_measured) ** 2, np.nan)
