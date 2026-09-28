"""
Provide a compact phasor analyzer for CPU and optional GPU FLI workflows.

This module belongs to :mod:`pyfli.phasor.phasorS` and is part of PyFLI's compact
phasor analyzer for CPU and optional GPU FLI workflows. Public API includes classes
:class:`PhasorAnalyzer`, :class:`PhasorPlotsMixin`, :class:`MonoLocus`,
:class:`PhasorAdditionalPlots`, :class:`AcquisitionConfig` and
:class:`AcquisitionMode`, and the locus helpers of
:mod:`~pyfli.phasor.phasorS.phasor_locus_tools`.
"""
# ruff: noqa: F401

from .phasor_additional_plots import PhasorAdditionalPlots
from .phasor_locus import MonoLocus
from .phasor_locus_tools import (
    AcquisitionConfig,
    AcquisitionMode,
    discrete_locus_circle,
    lifetime_from_locus,
    modulation_lifetime,
    phase_lifetime,
    phase_lifetime_gated,
    plot_discrete_n_sweep,
)
from .phasor_simple import PhasorAnalyzer
from .phasor_simple_plots import PhasorPlotsMixin
