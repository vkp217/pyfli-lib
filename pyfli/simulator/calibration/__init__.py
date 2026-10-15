"""Calibrate the pyfli simulator (priors, IRF, detector noise) to experimental data."""

from .calibrator import CalibrationResult, DetectorSimCalibrator
from .config import CalibrationInputs, PixelSelectionCriteria
from .coverage import coverage_fractions, covered_pixels, ensure_coverage
from .data_io import MaskShapeError, load_experimental_data, load_reference_maps
from .folder_calibration import calibrate_folder, find_data_files, match_masks
from .multicluster import detect_clusters, simulate_multicluster_tile
from .pixel_selection import select_signal_pixels
from .pooled_calibration import (
    PooledCalibrator,
    find_session_configs,
    load_session_calibration,
)
from .sim_bridge import load_calibrated_generator, simulate_image
from .sweep_combination import (
    combination_generator,
    combine_calibrations,
    load_combined_setup,
    make_combined_generator,
    verify_combined,
)

__all__ = [
    "CalibrationInputs",
    "CalibrationResult",
    "DetectorSimCalibrator",
    "MaskShapeError",
    "PixelSelectionCriteria",
    "PooledCalibrator",
    "calibrate_folder",
    "combination_generator",
    "combine_calibrations",
    "coverage_fractions",
    "covered_pixels",
    "detect_clusters",
    "ensure_coverage",
    "find_data_files",
    "find_session_configs",
    "load_calibrated_generator",
    "load_combined_setup",
    "load_experimental_data",
    "load_reference_maps",
    "load_session_calibration",
    "make_combined_generator",
    "match_masks",
    "select_signal_pixels",
    "simulate_image",
    "simulate_multicluster_tile",
    "verify_combined",
]
