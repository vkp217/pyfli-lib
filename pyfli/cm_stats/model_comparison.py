"""
Nested-model comparison of fit statistics (mono- vs bi-exponential).
"""

from typing import Any

from scipy.stats import f

P_VALUE_METHODS = ("sf", "cdf")


def nested_f_test(
    chi_simple: Any,
    chi_complex: Any,
    df_extra: Any,
    df_complex: Any,
    p_value: str = "sf",
) -> tuple[Any, Any]:
    """
    ``(F, p)`` of the extra-sum-of-squares F-test
    ``F = ((chi_simple - chi_complex) / df_extra) / (chi_complex / df_complex)``;
    ``p`` is ``f.sf(F)`` (``p_value="sf"``) or ``1 - f.cdf(F)`` (``"cdf"``).
    """
    if p_value not in P_VALUE_METHODS:
        raise ValueError(f"p_value must be one of {P_VALUE_METHODS}, got {p_value!r}.")
    stat = ((chi_simple - chi_complex) / df_extra) / (chi_complex / df_complex)
    if p_value == "sf":
        return stat, f.sf(stat, df_extra, df_complex)
    return stat, 1.0 - f.cdf(stat, df_extra, df_complex)
