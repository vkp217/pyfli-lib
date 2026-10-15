"""Fixtures for the pyfli.simulator.calibration tests."""

import shutil

import numpy as np
import pytest
from calib_helpers import SAMPLE, SHAPE, TINY_BUDGET, simulate_cube, true_irf

from pyfli.simulator.calibration import CalibrationInputs, DetectorSimCalibrator


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("calib_data")
    np.save(d / "exp_a.npy", simulate_cube(seed=0))
    np.save(d / "irf.npy", true_irf())
    mask = np.zeros(SHAPE, bool)
    mask[SAMPLE] = True
    np.save(d / "mask_a.npy", mask)
    return d


@pytest.fixture(scope="session")
def synthetic_cube(synthetic_dir):
    return np.load(synthetic_dir / "exp_a.npy")


@pytest.fixture(scope="session")
def calibration(synthetic_dir, tmp_path_factory):
    """One real calibration of the synthetic experiment (simulated IRF, automatic mask)."""
    out = tmp_path_factory.mktemp("calib_out") / "exp_a"
    result = DetectorSimCalibrator(
        CalibrationInputs(
            data_path=str(synthetic_dir / "exp_a.npy"),
            detector_type="discrete",
            model_type="bi_exponential",
            seed=0,
            output_dir=str(out),
            target_coverage=0.95,
            **TINY_BUDGET,
        )
    ).run()
    return result


@pytest.fixture
def calibration_copy(calibration, tmp_path):
    """A writable copy of the session calibration folder; returns the path of its JSON."""
    dst = tmp_path / "copy"
    shutil.copytree(calibration.output_dir, dst)
    return str(dst / "calibrated_sim_config.json")
