##### inside solver.__init__.py
# ruff: noqa: F401

"""
Provide solver tools for PyFLI least-squares, maximum-likelihood, CPU, GPU, binned, and
global FLI fitting routines.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLI fitting routines. The module primarily re-
exports package symbols or constants for downstream imports.
"""

from .base_fitter import BaseFLIFitter
from .binned_fitter import BinnedFLIFitter, FLIBinner
from .comparison import FittingComparator
from .cpu_processor import FLICPUProcessor
from .forward_model import decay_kernel, model_numpy
from .global_fitter import GlobalFLIFitter
from .gpu_processor import FLIGPUProcessor
from .mle_fitter import MLEFLIFitter
from .shared_metrics import (
    compute_average_lifetime,
    compute_fli_stats,
    compute_fret_efficiency,
    compute_pearson_stats,
    enforce_tau_ordering,
    expected_poisson_deviance,
    pearson_chi_square,
    poisson_deviance,
    reduced_poisson_deviance,
)

# [BaseFLIFitter, FLICPUProcessor, FLIGPUProcessor, MLEFLIFitter, GlobalFLIFitter, FittingComparator,
# BinnedFLIFitter, FLIBinner, decay_kernel, model_numpy, enforce_tau_ordering,
# compute_fli_stats, compute_average_lifetime, compute_fret_efficiency]
