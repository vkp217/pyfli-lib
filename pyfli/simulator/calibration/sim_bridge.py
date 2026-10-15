"""Build the pyfli simulator config from calibrated quantities and run SimGenerator."""

import json
import os

import numpy as np

from ..irf_sim.irf_offset_gen import OffsetGen
from ..sim_workflow import SimGenerator
from .config import SIM_FAMILY
from .prior_estimation import PriorSet


def build_sim_config(
    priors: PriorSet,
    detector_type: str,
    dcr: float,
    laser_freq_mhz: float,
    jitter: bool = True,
) -> dict:
    """Config dict for ``SimGenerator`` / ``PhotonCountSimulator`` / ``ContinuousSimulator``."""
    cfg = {
        "sensor_type": detector_type,
        "tau2": priors.tau2.as_tuple(),
        "tau2_dist": "beta",
        "tau2_beta_range": (round(priors.tau2.scale, 4), round(priors.tau2.offset, 4)),
        "efficiency": priors.efficiency.as_tuple() if priors.efficiency else (5.0, 5.0),
        "A1_fraction": priors.alpha1.as_tuple() if priors.alpha1 else (5.0, 5.0),
        "photo_count": priors.photon.as_tuple(),
        "mono_fraction": float(priors.mono_fraction),
        "bit": int(priors.bit),
        "dcr": float(dcr),
        "laser_feq": float(laser_freq_mhz),
        "jitter": bool(jitter),
        "pileup_mode": "wrap",
        "seed": None,
    }
    if detector_type == "discrete":
        cfg["n_cycles"] = tuple(int(v) for v in priors.photon_range)
    return cfg


class MultiIRFSimGenerator(SimGenerator):
    """
    ``SimGenerator`` whose IRF is drawn per pixel from a library: ``irf_cube[0, k]`` for
    k = 0..K-1, each with the usual ``OffsetGen`` shift ``a`` and baseline ``b``. Used for one
    config covering experiments recorded with different IRFs (e.g. two detector channels).
    The drawn index is returned as ``irf_index``. ``b_ranges`` optionally gives each IRF its
    own baseline range (background belongs to the detector channel, like the IRF).
    """

    def __init__(
        self, irf_cube, config, a_range, b_range, weights=None, b_ranges=None, **kwargs
    ):
        super().__init__(
            irf_cube, config, a_range=a_range, b_range=b_range, pixel=(0, 0), **kwargs
        )
        K = irf_cube.shape[1]
        b_ranges = b_ranges or [b_range] * K
        self._offset_gens = [
            OffsetGen(
                irf_cube, a_range=a_range, b_range=tuple(b_ranges[k]), pixel=(0, k)
            )
            for k in range(K)
        ]
        w = np.ones(K) if weights is None else np.asarray(weights, float)
        self.irf_weights = w / w.sum()

    def simulate_once(self):
        k = int(np.random.choice(len(self._offset_gens), p=self.irf_weights))
        self.offset_gen = self._offset_gens[k]
        out = super().simulate_once()
        out["irf_index"] = k
        return out


class TimeWindowedGenerator:
    """
    Wraps a ``SimGenerator`` and zeroes ``inactive_bins`` of every simulated decay: the time
    bins outside the detector's acquisition window (e.g. outside the TCSPC TAC range), which
    never record a count in the experiment. The pyfli simulator fills all bins, so without
    this step the simulated decays could not contain the experimental ones. Ground-truth
    ``photon_count`` stays the simulator's expected photon number over the full period.
    """

    def __init__(self, generator: SimGenerator, inactive_bins) -> None:
        self.generator = generator
        self.inactive_bins = np.asarray(inactive_bins, dtype=int)

    @property
    def config(self) -> dict:
        return self.generator.config

    def simulate_once(self) -> dict:
        out = self.generator.simulate_once()
        decay = np.array(out["decay"], dtype=float)
        decay[self.inactive_bins] = 0.0
        out["decay"] = decay
        return out


def make_generator(
    irf_cube: np.ndarray,
    config: dict,
    a_range,
    b_range,
    irf_weights=None,
    irf_b_ranges=None,
    inactive_bins=None,
):
    kw = dict(family=SIM_FAMILY, sensor_type=config["sensor_type"])
    if irf_cube.ndim == 3 and irf_cube.shape[1] > 1:
        gen = MultiIRFSimGenerator(
            irf_cube,
            dict(config),
            tuple(a_range),
            tuple(b_range),
            weights=irf_weights,
            b_ranges=irf_b_ranges,
            **kw,
        )
    else:
        gen = SimGenerator(
            irf_cube,
            dict(config),
            a_range=tuple(a_range),
            b_range=tuple(b_range),
            pixel=(0, 0),
            **kw,
        )
    if inactive_bins is not None and len(inactive_bins):
        return TimeWindowedGenerator(gen, inactive_bins)
    return gen


TRUTH_KEYS = (
    "tau1",
    "tau2",
    "tau",
    "alpha1",
    "Efficiency",
    "photon_count",
    "h_shift",
    "v_shift_bgp",
)


def draw_pixels(gen, n: int, seed: int | None = 0) -> list[dict]:
    """
    ``n`` calls of ``gen.simulate_once()``, reproducible for a given ``seed``: pixel ``i``
    gets the engine seed ``seed * 1_000_003 + i`` and the global NumPy RNG is seeded once
    (and restored afterwards). ``seed=None`` draws fresh random numbers.
    """
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed % (2**32))
    try:
        outs = []
        for i in range(n):
            if seed is not None:
                gen.config["seed"] = (seed * 1_000_003 + i) % (2**32)
            outs.append(gen.simulate_once())
    finally:
        np.random.set_state(state)
        if seed is not None:
            gen.config["seed"] = None
    return outs


def stack_outputs(outs: list[dict]) -> dict[str, np.ndarray]:
    """Stack ``simulate_once`` outputs: ``decay`` / ``irf`` as ``(n, T)``, ground truth as ``(n,)``."""
    res = {
        "decay": np.stack([o["decay"] for o in outs]).astype(np.float64),
        "irf": np.stack([o["irf"] for o in outs]).astype(np.float64),
    }
    for k in TRUTH_KEYS + (("irf_index",) if "irf_index" in outs[0] else ()):
        res[k] = np.array([float(np.asarray(o[k]).ravel()[0]) for o in outs])
    return res


def simulate_batch(
    irf_cube: np.ndarray,
    config: dict,
    a_range,
    b_range,
    n: int,
    seed: int | None = 0,
    irf_b_ranges=None,
    inactive_bins=None,
) -> dict[str, np.ndarray]:
    """Simulate ``n`` pixels; returns stacked decays and ground-truth parameters (:func:`stack_outputs`)."""
    gen = make_generator(
        irf_cube,
        config,
        a_range,
        b_range,
        irf_b_ranges=irf_b_ranges,
        inactive_bins=inactive_bins,
    )
    out = stack_outputs(draw_pixels(gen, n, seed))
    out.pop("irf")
    return out


def to_jsonable(obj):
    """Nested dicts / lists / tuples / NumPy values converted to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


TUPLE_KEYS = (
    "tau2",
    "tau2_beta_range",
    "efficiency",
    "A1_fraction",
    "photo_count",
    "n_cycles",
)


def save_calibration(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def load_calibrated_generator(json_path: str) -> SimGenerator:
    """
    Rebuild the calibrated ``SimGenerator`` from ``calibrated_sim_config.json`` (and the
    ``calibrated_irf.npy`` next to it). Use it as::

        gen = load_calibrated_generator(".../calibrated_sim_config.json")
        sample = gen.simulate_once()          # dict with decay, irf, tau1, tau2, alpha1, ...
    """
    with open(json_path) as f:
        payload = json.load(f)
    cfg = dict(payload["config"])
    for k in TUPLE_KEYS:
        if k in cfg and isinstance(cfg[k], list):
            cfg[k] = tuple(cfg[k])
    irf_file = os.path.join(os.path.dirname(json_path), payload["irf"]["file"])
    irf_cube = np.load(irf_file)
    kw = payload["sim_generator_kwargs"]
    return make_generator(
        irf_cube,
        cfg,
        kw["a_range"],
        kw["b_range"],
        kw.get("irf_weights"),
        kw.get("irf_b_ranges"),
        kw.get("inactive_bins"),
    )


def simulate_image(
    json_path: str, shape: tuple[int, int] = (8, 8), seed: int | None = 0
) -> dict[str, np.ndarray]:
    """
    Simulate an ``(H, W)`` image with the calibrated simulator. Returns the decay cube
    ``(H, W, T)``, the per-pixel IRF cube and ``(H, W)`` ground-truth maps
    (tau1, tau2, tau, alpha1, Efficiency, photon_count, h_shift, v_shift_bgp).
    """
    H, W = shape
    flat = stack_outputs(draw_pixels(load_calibrated_generator(json_path), H * W, seed))
    return {
        k: v.reshape(H, W, -1) if v.ndim == 2 else v.reshape(H, W)
        for k, v in flat.items()
    }
