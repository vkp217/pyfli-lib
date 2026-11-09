# ruff: noqa: F401

"""
Shared statistics for PyFLI, defined once and used across the package:
:mod:`.deviance` (Poisson deviance), :mod:`.chi_square` (Pearson/Neyman chi-square,
degrees of freedom), :mod:`.goodness_of_fit` (SSR, RMSE, R2), :mod:`.model_comparison`
(nested F-test), :mod:`.robust` (median/MAD) and :mod:`.shared_metrics` (fit summaries,
tau ordering, lifetime/FRET summaries).
"""

from .chi_square import (
    degrees_of_freedom,
    neyman_chi_square,
    pearson_chi_square,
    pearson_terms,
)
from .deviance import (
    deviance_dof,
    expected_poisson_deviance,
    poisson_deviance,
    poisson_deviance_moment_table,
    poisson_deviance_moments,
    poisson_deviance_terms,
    reduced_poisson_deviance,
)
from .goodness_of_fit import r_squared, rmse, sum_squared_residuals
from .model_comparison import nested_f_test
from .robust import MAD_TO_SIGMA, median_absolute_deviation
from .shared_metrics import (
    compute_average_lifetime,
    compute_fli_stats,
    compute_fret_efficiency,
    compute_pearson_stats,
    enforce_tau_ordering,
)
