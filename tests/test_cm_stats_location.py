"""
pyfli.cm_stats owns the fit statistics and pyfli.cm_funcs the forward model; old import
paths still resolve to them.
"""

import importlib

import pyfli.cm_stats as cm_stats
import pyfli.solver as solver
from pyfli.cm_stats import shared_metrics

PUBLIC = (
    "compute_average_lifetime",
    "compute_fli_stats",
    "compute_fret_efficiency",
    "compute_pearson_stats",
    "enforce_tau_ordering",
    "expected_poisson_deviance",
    "pearson_chi_square",
    "poisson_deviance",
    "reduced_poisson_deviance",
)


def test_old_module_path_is_same_module():
    """pyfli.solver.shared_metrics is the pyfli.cm_stats.shared_metrics module."""
    old = importlib.import_module("pyfli.solver.shared_metrics")
    assert old is shared_metrics
    from pyfli.solver.shared_metrics import _FIXED_BOUND_TOL, compute_fli_stats

    assert compute_fli_stats is shared_metrics.compute_fli_stats
    assert _FIXED_BOUND_TOL == shared_metrics._FIXED_BOUND_TOL


def test_public_names_reexported():
    """Package-level names in pyfli.cm_stats and pyfli.solver are the shared functions."""
    for name in PUBLIC:
        assert getattr(cm_stats, name) is getattr(shared_metrics, name)
        assert getattr(solver, name) is getattr(shared_metrics, name)


def test_forward_model_location():
    """pyfli.cm_funcs owns the forward model; pyfli.solver keeps its old names."""
    import pyfli.cm_funcs as cm_funcs
    from pyfli.cm_funcs import forward_model

    assert importlib.import_module("pyfli.solver.forward_model") is forward_model
    for name in ("model_numpy", "decay_kernel", "gate_integrated_kernel"):
        assert getattr(cm_funcs, name) is getattr(forward_model, name)
    assert solver.model_numpy is forward_model.model_numpy
    assert solver.decay_kernel is forward_model.decay_kernel
