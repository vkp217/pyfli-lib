"""Compact phasor analyzer for CPU and optional GPU FLI workflows."""
# ruff: noqa: F401

from .phasor_additional_plots import PhasorAdditionalPlots
from .phasor_clustering import (
    add_label_legend,
    align_labels,
    cluster_summary,
    clustering_accuracy,
    image_to_labels,
    labels_to_image,
    labels_to_rgb,
    phasor_feature_matrix,
    plot_explained_variance,
    plot_scores,
)
from .phasor_lifetime import (
    APPARENT_LIFETIMES,
    apparent_lifetimes,
    normal_lifetime,
    semicircle_nearest_point,
)
from .phasor_locus import MonoLocus
from .phasor_locus_tools import (
    AcquisitionConfig,
    AcquisitionMode,
    discrete_locus_circle,
    discrete_phasor_lifetime,
    lifetime_from_locus,
    modulation_lifetime,
    phase_lifetime,
    phase_lifetime_gated,
    plot_discrete_n_sweep,
)
from .phasor_simple import PhasorAnalyzer
from .phasor_simple_plots import PhasorPlotsMixin
