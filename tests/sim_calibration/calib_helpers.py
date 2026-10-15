"""Shared constants and helpers for the pyfli.simulator.calibration tests."""

import json
import os
import shutil

import numpy as np

from pyfli.simulator.calibration.sim_bridge import (
    draw_pixels,
    make_generator,
    stack_outputs,
)
from pyfli.simulator.irf_sim.irf_generator import IRFGenerator

N_BINS = 64
PERIOD_NS = 12.5
SHAPE = (20, 20)
SAMPLE = (slice(3, 17), slice(3, 17))
INACTIVE = np.r_[0:3, N_BINS - 3 : N_BINS]
DCR = 0.02
TRUE_CONFIG = {
    "sensor_type": "discrete",
    "tau2": (40.0, 40.0),
    "tau2_dist": "beta",
    "tau2_beta_range": (1.0, 1.5),
    "efficiency": (30.0, 20.0),
    "A1_fraction": (20.0, 20.0),
    "photo_count": (4.0, 4.0),
    "n_cycles": (30_000, 90_000),
    "mono_fraction": 0.0,
    "bit": 16,
    "dcr": DCR,
    "laser_feq": 80.0,
    "jitter": False,
    "pileup_mode": "wrap",
    "seed": None,
}
TINY_BUDGET = dict(
    n_sim_pixels=150,
    n_exp_pixels=400,
    n_fit_pixels=40,
    n_validation_pixels=200,
    n_bank_pixels=400,
    bank_rounds=1,
    max_evals_stage1=6,
    max_evals_stage2=60,
    n_coverage_pixels=600,
    max_coverage_iter=2,
)


def true_irf() -> np.ndarray:
    return IRFGenerator.gaussianExpIRF(
        mu=10, T=PERIOD_NS, num_bins=N_BINS, sigma=0.08, tau=0.05
    )


def simulate_cube(
    seed: int = 0, config: dict | None = None, shape=SHAPE, sample=SAMPLE
) -> np.ndarray:
    """Sample pixels from ``config`` inside ``sample``, dark-count-only pixels outside, inactive bins zero."""
    cfg = {**TRUE_CONFIG, **(config or {})}
    H, W = shape
    cube = (
        np.random.default_rng(seed + 1).poisson(DCR, (H, W, N_BINS)).astype(np.float32)
    )
    rows = range(H)[sample[0]]
    cols = range(W)[sample[1]]
    gen = make_generator(true_irf()[None, None, :], cfg, (0, 0), (0, 0))
    decays = stack_outputs(draw_pixels(gen, len(rows) * len(cols), seed))["decay"]
    cube[sample] = decays.reshape(len(rows), len(cols), N_BINS)
    cube[..., INACTIVE] = 0
    return cube


def copy_calibration(src_dir: str, dst_dir: str, **input_overrides) -> str:
    """Copy a calibration folder and patch its saved inputs; returns the JSON path."""
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    path = os.path.join(dst_dir, "calibrated_sim_config.json")
    with open(path) as f:
        payload = json.load(f)
    payload["inputs"].update(input_overrides)
    with open(path, "w") as f:
        json.dump(payload, f)
    return path
