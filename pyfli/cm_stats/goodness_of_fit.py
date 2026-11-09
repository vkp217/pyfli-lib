"""
Residual-based goodness of fit: sum of squared residuals, RMSE and the coefficient of
determination R2, for NumPy arrays or torch tensors.
"""

from typing import Any

import numpy as np


def _is_torch(x: Any) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def sum_squared_residuals(residuals: Any, axis: Any = -1) -> Any:
    """``sum(residuals^2)`` along `axis` (``None``: all elements)."""
    if _is_torch(residuals):
        return residuals.pow(2).sum() if axis is None else (residuals**2).sum(dim=axis)
    return np.sum(residuals**2, axis=axis)


def rmse(residuals: Any, axis: Any = -1) -> Any:
    """Root-mean-square residual ``sqrt(mean(residuals^2))`` along `axis` (``None``: all elements)."""
    if _is_torch(residuals):
        import torch

        return torch.sqrt(torch.mean(residuals**2, dim=axis))
    return np.sqrt(np.mean(residuals**2, axis=axis))


def r_squared(
    residuals: Any,
    data: Any,
    axis: int = -1,
    empty_value: float = 0.0,
    threshold: float = 0.0,
    dtype: Any = None,
) -> Any:
    """
    Coefficient of determination ``1 - SS_res / SS_tot`` along `axis`, with
    ``SS_tot = sum((data - mean(data))^2)``. Where ``SS_tot <= threshold`` (a flat decay)
    the result is `empty_value`. `dtype` sets the precision of the ratio (default: that
    of ``SS_res``).
    """
    if _is_torch(residuals):
        import torch

        ss_res = torch.sum(residuals**2, dim=axis)
        ss_tot = torch.sum((data - data.mean(dim=axis, keepdim=True)) ** 2, dim=axis)
        return torch.where(
            ss_tot > threshold,
            1.0 - ss_res / ss_tot,
            torch.full_like(ss_tot, empty_value),
        )
    ss_res = np.sum(residuals**2, axis=axis)
    ss_tot = np.sum((data - np.mean(data, axis=axis, keepdims=True)) ** 2, axis=axis)
    ratio = np.full(
        np.shape(ss_res),
        1.0 - empty_value,
        dtype=np.asarray(ss_res).dtype if dtype is None else dtype,
    )
    np.divide(ss_res, ss_tot, out=ratio, where=ss_tot > threshold)
    return 1.0 - ratio
