"""Combine per-experiment calibrations into one sweep-style simulator setup."""

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from pyfli import logging
from pyfli.data_cc.config_combinations import ConfigCombinationGenerator

from ..sim_workflow import SimGenerator
from .config import SIM_FAMILY
from .coverage import coverage_fractions, resupport_beta
from .data_io import load_experimental_data
from .prior_estimation import BetaPrior
from .sim_bridge import (
    TUPLE_KEYS,
    TimeWindowedGenerator,
    draw_pixels,
    make_generator,
    stack_outputs,
    to_jsonable,
)
from .summary_stats import background_bins, compute_stats

SENSOR_FIXED = {
    "discrete": {"bit": 16, "qe_on": False},
    "continuous": {
        "qe_on": True,
        "read_noise_on": True,
        "round_on": True,
        "poisson": True,
    },
}
COMMON_FIXED = {
    "dcr_on": True,
    "clip_on": True,
    "tau2_dist": "beta",
    "pileup_mode": "wrap",
}
COMMON_SWEEP_KEYS = (
    "jitter",
    "efficiency",
    "A1_fraction",
    "tau2",
    "photo_count",
    "dcr",
)
RUN_SETTINGS = {
    "SAVING": False,
    "SANITY_PLOT": False,
    "COMPARE_PLOT": True,
    "BASE_OUT": "<select output folder path>",
    "GEN_MODE_DEFAULT": "weighted",
    "CARTESIAN_NUM_SAMPLE": 20000,
    "GEN_MODE": "weighted",
    "NUM_SAMPLE": 10,
    "TOTAL_SAMPLES": 20_000,
    "SEED": 0,
}
MAX_CARTESIAN = 100_000


@dataclass
class ExperimentCalibration:
    name: str
    json_path: str
    payload: dict
    weight: float = 1.0

    @property
    def config(self) -> dict:
        return self.payload["config"]

    @property
    def kwargs(self) -> dict:
        return self.payload["sim_generator_kwargs"]

    @property
    def detector(self) -> str:
        return self.payload["inputs"]["detector_type"]

    def irf_template(self) -> np.ndarray:
        cube = np.load(
            os.path.join(os.path.dirname(self.json_path), self.payload["irf"]["file"])
        )
        return cube[0, 0].astype(float)


def load_experiment(json_path: str, name: str | None = None) -> ExperimentCalibration:
    with open(json_path) as f:
        payload = json.load(f)
    return ExperimentCalibration(
        name or os.path.basename(os.path.dirname(json_path)), json_path, payload
    )


def _prior(d: dict) -> BetaPrior:
    return BetaPrior(d["a"], d["b"], d["offset"], d["scale"])


def cdf_error(old: BetaPrior, new: BetaPrior, n: int = 2001) -> float:
    """Maximum difference between the two priors' CDFs, in physical units."""
    x = np.linspace(
        min(old.offset, new.offset),
        max(old.offset + old.scale, new.offset + new.scale),
        n,
    )
    f_old = beta_dist.cdf(np.clip((x - old.offset) / old.scale, 0, 1), old.a, old.b)
    f_new = beta_dist.cdf(np.clip((x - new.offset) / new.scale, 0, 1), new.a, new.b)
    return float(np.max(np.abs(f_old - f_new)))


def _sweep_entry(values: list, weights: list[float]) -> tuple[list, list[float]]:
    """Deduplicate values (keeping first-seen order) and add up their weights."""
    out_v, out_w = [], []
    for v, w in zip(values, weights):
        if v in out_v:
            out_w[out_v.index(v)] += w
        else:
            out_v.append(v)
            out_w.append(w)
    total = sum(out_w)
    return out_v, [round(w / total, 6) for w in out_w]


def _split(per_key: dict[str, list], weights: list[float]) -> tuple[dict, dict, dict]:
    """Keys with one distinct value go to the fixed config, the rest to sweep + weights."""
    fixed, sweep, sweep_w = {}, {}, {}
    for k, vals in per_key.items():
        v, w = _sweep_entry(vals, weights)
        if len(v) == 1:
            fixed[k] = v[0]
        else:
            sweep[k], sweep_w[k] = v, w
    return fixed, sweep, sweep_w


def combine_calibrations(
    json_paths: dict[str, str], output_dir: str, weighting: str = "equal"
) -> dict:
    """
    Build the combined sweep setup from per-experiment ``calibrated_sim_config.json`` files
    (``name -> path``) and write it to ``output_dir`` as ``combined_sim_setup.json``,
    ``combined_sim_setup.py`` (same constants, importable) and ``irfs/<name>.txt``.

    ``weighting``: "equal" (every experiment counts the same) or "pixels" (by number of
    sample pixels). Returns the setup dict.
    """
    exps = [load_experiment(p, n) for n, p in json_paths.items()]
    if not exps:
        raise ValueError("No calibrations to combine")
    for key in ("detector_type", "model_type", "laser_freq_mhz"):
        vals = {str(e.payload["inputs"][key]) for e in exps}
        if len(vals) > 1:
            raise ValueError(f"Experiments disagree on {key}: {vals}")
    n_bins = {len(e.irf_template()) for e in exps}
    if len(n_bins) > 1:
        raise ValueError(f"Experiments have different numbers of time bins: {n_bins}")
    detector = exps[0].detector
    laser = float(exps[0].payload["inputs"]["laser_freq_mhz"])
    T = n_bins.pop()
    if weighting == "pixels":
        w = np.array([e.payload["pixel_selection"]["n_selected"] for e in exps], float)
    else:
        w = np.ones(len(exps))
    w = (w / w.sum()).tolist()
    for e, wi in zip(exps, w):
        e.weight = wi

    report = {}
    tau2 = {e.name: _prior(e.payload["priors"]["tau2"]) for e in exps}
    lo = min(p.offset for p in tau2.values())
    hi = max(p.offset + p.scale for p in tau2.values())
    tau2_new = {n: resupport_beta(p, lo, hi) for n, p in tau2.items()}
    report["tau2"] = {n: cdf_error(tau2[n], tau2_new[n]) for n in tau2}

    photon = {e.name: _prior(e.payload["priors"]["photon"]) for e in exps}
    sensor_fixed = dict(SENSOR_FIXED[detector])
    dcr = {e.name: float(e.config["dcr"]) for e in exps}
    if detector == "discrete":
        c_lo = min(int(e.config["n_cycles"][0]) for e in exps)
        c_hi = max(int(e.config["n_cycles"][1]) for e in exps)
        photon_new = {n: resupport_beta(p, c_lo, c_hi) for n, p in photon.items()}
        report["photo_count"] = {n: cdf_error(photon[n], photon_new[n]) for n in photon}
        sensor_fixed["n_cycles"] = (c_lo, c_hi)
    else:
        bits = {e.name: int(e.config["bit"]) for e in exps}
        b_max = max(bits.values())
        phys = {
            n: BetaPrior(p.a, p.b, 0.0, 2 ** bits[n] - 1) for n, p in photon.items()
        }
        photon_new = {n: resupport_beta(p, 0.0, 2**b_max - 1) for n, p in phys.items()}
        report["photo_count"] = {n: cdf_error(phys[n], photon_new[n]) for n in phys}
        dcr = {n: v * bits[n] / b_max for n, v in dcr.items()}
        sensor_fixed["bit"] = b_max

    per_key = {
        "jitter": [bool(e.config["jitter"]) for e in exps],
        "efficiency": [tuple(e.config["efficiency"]) for e in exps],
        "A1_fraction": [tuple(e.config["A1_fraction"]) for e in exps],
        "tau2": [tau2_new[e.name].as_tuple() for e in exps],
        "photo_count": [photon_new[e.name].as_tuple() for e in exps],
        "dcr": [round(dcr[e.name], 6) for e in exps],
    }
    fixed, sweep, sweep_w = _split(per_key, w)
    common_config = {
        **COMMON_FIXED,
        "laser_feq": laser,
        "tau2_beta_range": (round(hi - lo, 4), round(lo, 4)),
        **fixed,
    }

    irf_dir = os.path.join(output_dir, "irfs")
    os.makedirs(irf_dir, exist_ok=True)
    gate_delay = 1000.0 / laser / T
    irf_runs, irf_info = [], {}
    for e in exps:
        path = os.path.join(irf_dir, f"{e.name}.txt")
        np.savetxt(path, e.irf_template(), fmt="%.8g")
        irf_runs.append((path, detector, gate_delay))
        irf_info[path] = {
            "experiment": e.name,
            "a_range": list(e.kwargs["a_range"]),
            "b_range": list(e.kwargs["b_range"]),
            "inactive_bins": e.kwargs.get("inactive_bins") or [],
        }
    a_range = (
        min(i["a_range"][0] for i in irf_info.values()),
        max(i["a_range"][1] for i in irf_info.values()),
    )
    b_range = (0.0, max(i["b_range"][1] for i in irf_info.values()))

    n_combos = int(np.prod([len(v) for v in sweep.values()])) if sweep else 1
    worst = max(max(v.values()) for v in report.values())
    if worst > 0.05:
        logging.warning(
            f"Re-expressing priors on the common support changed a CDF by up to {worst:.3f}: "
            "the experiments' ranges differ a lot; see resupport_cdf_error"
        )
    if n_combos > MAX_CARTESIAN:
        logging.warning(
            f"{n_combos} sweep combinations: too many to enumerate. Use make_combined_generator "
            "(samples the keys independently, same distribution as GEN_MODE='weighted')."
        )

    sensors = {"discrete": ({}, {}, {}), "continuous": ({}, {}, {})}
    sensors[detector] = (sensor_fixed, {}, {})
    setup = {
        "IRF_RUNS": irf_runs,
        "IRF_RUN_INFO": irf_info,
        "MONO_FRACTIONS": sorted({float(e.config["mono_fraction"]) for e in exps}),
        "LASER_PERIOD_NS": 1000.0 / laser,
        "A_RANGE": a_range,
        "B_RANGE": b_range,
        **RUN_SETTINGS,
        "COMMON_CONFIG": common_config,
        "DISCRETE_CONFIG": sensors["discrete"][0],
        "CONTINUOUS_CONFIG": sensors["continuous"][0],
        "COMMON_SWEEP": sweep,
        "DISCRETE_SWEEP": sensors["discrete"][1],
        "CONTINUOUS_SWEEP": sensors["continuous"][1],
        "COMMON_WEIGHTS": sweep_w,
        "DISCRETE_WEIGHTS": sensors["discrete"][2],
        "CONTINUOUS_WEIGHTS": sensors["continuous"][2],
        "EXPERIMENT_CONFIGS": {e.name: os.path.abspath(e.json_path) for e in exps},
        "EXPERIMENT_WEIGHTS": {e.name: round(e.weight, 6) for e in exps},
        "EXPERIMENT_SWEEP_VALUES": {
            e.name: {k: per_key[k][i] for k in sweep} for i, e in enumerate(exps)
        },
        "N_COMBINATIONS": n_combos,
        "RESUPPORT_CDF_ERROR": {
            k: {n: round(v, 4) for n, v in d.items()} for k, d in report.items()
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "combined_sim_setup.json"), "w") as f:
        json.dump(to_jsonable(setup), f, indent=2)
    write_setup_module(setup, os.path.join(output_dir, "combined_sim_setup.py"))
    logging.info(
        f"Combined {len(exps)} calibrations into {output_dir}: sweep keys {list(sweep)}, "
        f"{n_combos} combinations, max re-expression CDF error {worst:.3f}"
    )
    return setup


def _py(obj, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        rows = [
            f"{pad}    {k!r}: {_py(v, indent + 4).lstrip()}," for k, v in obj.items()
        ]
        return "{\n" + "\n".join(rows) + f"\n{pad}}}"
    if (
        isinstance(obj, list)
        and obj
        and any(isinstance(v, (tuple, list, dict)) for v in obj)
        and len(repr(obj)) > 90
    ):
        rows = [f"{pad}    {_py(v, indent + 4).lstrip()}," for v in obj]
        return "[\n" + "\n".join(rows) + f"\n{pad}]"
    return repr(obj)


def write_setup_module(setup: dict, path: str) -> None:
    """Write ``setup`` as an importable Python module of constants, plus ``SENSOR_SETUP``."""
    lines = [
        '"""',
        "Combined pyfli simulator setup generated by pyfli.simulator.calibration.combine_calibrations.",
        "",
        "Calibrated values: IRF_RUNS, MONO_FRACTIONS, *_CONFIG, *_SWEEP, *_WEIGHTS (one sweep value per",
        "experiment; EXPERIMENT_CONFIGS lists each experiment's own calibrated config). IRF_RUN_INFO holds the",
        "exact OffsetGen ranges and inactive time bins of each IRF. SAVING ... SEED are run settings with",
        "default values, not calibration results.",
        '"""',
        "",
    ]
    for k, v in setup.items():
        lines.append(f"{k} = {_py(v)}")
        lines.append("")
    lines.append("SENSOR_SETUP = {")
    lines.append('    "discrete": (DISCRETE_CONFIG, DISCRETE_SWEEP, DISCRETE_WEIGHTS),')
    lines.append(
        '    "continuous": (CONTINUOUS_CONFIG, CONTINUOUS_SWEEP, CONTINUOUS_WEIGHTS),'
    )
    lines.append("}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def load_combined_setup(path: str) -> dict:
    """Read ``combined_sim_setup.json`` back, restoring the tuples the simulator expects."""
    with open(path) as f:
        setup = json.load(f)

    def fix(cfg: dict) -> dict:
        return {
            k: tuple(v)
            if (k in TUPLE_KEYS or k == "tau2_beta_range") and isinstance(v, list)
            else v
            for k, v in cfg.items()
        }

    def fix_sweep(sw: dict) -> dict:
        return {
            k: [tuple(x) if isinstance(x, list) else x for x in vals]
            for k, vals in sw.items()
        }

    for k in ("COMMON_CONFIG", "DISCRETE_CONFIG", "CONTINUOUS_CONFIG"):
        setup[k] = fix(setup[k])
    for k in ("COMMON_SWEEP", "DISCRETE_SWEEP", "CONTINUOUS_SWEEP"):
        setup[k] = fix_sweep(setup[k])
    setup["IRF_RUNS"] = [tuple(r) for r in setup["IRF_RUNS"]]
    setup["A_RANGE"], setup["B_RANGE"] = (
        tuple(setup["A_RANGE"]),
        tuple(setup["B_RANGE"]),
    )
    return setup


def _parts(setup: dict, sensor_type: str, mono_fraction: float | None):
    base = {
        **setup["COMMON_CONFIG"],
        **setup[f"{sensor_type.upper()}_CONFIG"],
        "sensor_type": sensor_type,
    }
    if mono_fraction is not None:
        base["mono_fraction"] = float(mono_fraction)
    sweep = {**setup["COMMON_SWEEP"], **setup[f"{sensor_type.upper()}_SWEEP"]}
    weights = {**setup["COMMON_WEIGHTS"], **setup[f"{sensor_type.upper()}_WEIGHTS"]}
    return base, sweep, weights


def combination_generator(
    setup: dict, sensor_type: str, mono_fraction: float | None = None
) -> ConfigCombinationGenerator:
    """``pyfli.data_cc.ConfigCombinationGenerator`` for the combined setup (enumerates every combination)."""
    base, sweep, weights = _parts(setup, sensor_type, mono_fraction)
    n = int(np.prod([len(v) for v in sweep.values()])) if sweep else 1
    if n > MAX_CARTESIAN:
        raise ValueError(
            f"{n} combinations are too many to enumerate; use make_combined_generator instead"
        )
    return ConfigCombinationGenerator(
        base, sweep or {"jitter": [base.get("jitter", True)]}, weights
    )


class IndependentSweepGenerator:
    """
    Draws every swept key independently with its weights, then simulates one pixel with
    pyfli's ``SimGenerator``. The resulting distribution is that of
    ``ConfigCombinationGenerator.sample`` / ``WeightedConfigSimGenerator`` (combination
    weight = product of per-key weights), but nothing is enumerated, so any number of
    experiments works. ``config_index`` in the output gives the chosen value index per key.
    """

    def __init__(
        self,
        irf_cube,
        base: dict,
        sweep: dict,
        weights: dict,
        a_range,
        b_range,
        seed: int | None = None,
    ):
        self.irf_cube, self.base, self.a_range, self.b_range = (
            irf_cube,
            base,
            tuple(a_range),
            tuple(b_range),
        )
        self.keys = list(sweep)
        self.values = [list(sweep[k]) for k in self.keys]
        self.probs = [
            np.asarray(weights.get(k, np.ones(len(v))), float)
            for k, v in zip(self.keys, self.values)
        ]
        self.probs = [p / p.sum() for p in self.probs]
        self.rng = np.random.default_rng(seed)
        self.config = dict(base)

    def simulate_once(self) -> dict:
        idx = [
            int(self.rng.choice(len(v), p=p)) for v, p in zip(self.values, self.probs)
        ]
        cfg = {**self.base, **{k: v[i] for k, v, i in zip(self.keys, self.values, idx)}}
        if self.config.get("seed") is not None:
            cfg["seed"] = self.config["seed"]
        gen = SimGenerator(
            self.irf_cube,
            cfg,
            a_range=self.a_range,
            b_range=self.b_range,
            pixel=(0, 0),
            family=SIM_FAMILY,
            sensor_type=cfg["sensor_type"],
        )
        out = gen.simulate_once()
        out["config_index"] = dict(zip(self.keys, idx))
        return out


def make_combined_generator(
    setup_path: str,
    run: int = 0,
    mono_fraction: float | None = None,
    seed: int | None = None,
    exact_ranges: bool = True,
):
    """
    Simulator for one IRF run of the combined setup: ``gen.simulate_once()`` draws a
    combination of the calibrated values and simulates one pixel with ``IRF_RUNS[run]``.
    ``exact_ranges`` uses that IRF's own calibrated ``a_range`` / ``b_range`` and inactive
    time bins (``IRF_RUN_INFO``); otherwise the global ``A_RANGE`` / ``B_RANGE``.
    """
    setup = load_combined_setup(setup_path)
    irf_path, sensor_type, _ = setup["IRF_RUNS"][run]
    info = setup["IRF_RUN_INFO"][irf_path]
    irf_cube = np.loadtxt(irf_path)[None, None, :]
    if mono_fraction is None:
        mono_fraction = setup["MONO_FRACTIONS"][0]
    base, sweep, weights = _parts(setup, sensor_type, mono_fraction)
    a_range = info["a_range"] if exact_ranges else setup["A_RANGE"]
    b_range = info["b_range"] if exact_ranges else setup["B_RANGE"]
    gen = IndependentSweepGenerator(
        irf_cube, base, sweep, weights, a_range, b_range, seed
    )
    if exact_ranges and info["inactive_bins"]:
        return TimeWindowedGenerator(gen, info["inactive_bins"])
    return gen


def verify_combined(
    setup_path: str, n_sim: int = 10000, n_exp: int = 20000, seed: int = 0
) -> pd.DataFrame:
    """
    For every experiment of the combined setup, the coverage (:func:`coverage_fractions`) of
    its sample pixels by

    * ``own values``: the combination made of that experiment's sweep values, i.e. its own
      calibration after re-expression on the common supports;
    * ``combined``: the whole combined distribution (all combinations, with weights).

    Both use the experiment's own IRF run. The experimental pixels are re-loaded from the
    experiment's data file with its calibration's bin radius and sample mask. The table is
    also written next to the setup as ``coverage_per_experiment.csv``.
    """
    setup = load_combined_setup(setup_path)
    rows = []
    for run, (irf_path, sensor_type, _) in enumerate(setup["IRF_RUNS"]):
        info = setup["IRF_RUN_INFO"][irf_path]
        name = info["experiment"]
        cfg_path = setup["EXPERIMENT_CONFIGS"][name]
        with open(cfg_path) as f:
            inp = json.load(f)["inputs"]
        data = load_experimental_data(
            inp["data_path"],
            bg_path=inp.get("bg_path"),
            bin_radius=int(inp["bin_radius"]),
        )
        mask = np.load(os.path.join(os.path.dirname(cfg_path), "signal_mask.npy"))
        coords = np.argwhere(mask)
        pick = coords[
            np.random.default_rng(seed).choice(
                len(coords), min(n_exp, len(coords)), replace=False
            )
        ]
        exp = data.decay[pick[:, 0], pick[:, 1]].astype(np.float64)
        bg = background_bins(exp.sum(0))
        exp_stats = compute_stats(exp, bg)

        base, _, _ = _parts(setup, sensor_type, setup["MONO_FRACTIONS"][0])
        own = {
            **base,
            **{
                k: tuple(v) if isinstance(v, list) else v
                for k, v in setup["EXPERIMENT_SWEEP_VALUES"][name].items()
            },
        }
        irf_cube = np.loadtxt(irf_path)[None, None, :]
        own_gen = make_generator(
            irf_cube,
            own,
            info["a_range"],
            info["b_range"],
            inactive_bins=info["inactive_bins"],
        )
        comb_gen = make_combined_generator(setup_path, run=run, seed=seed)
        for label, gen in (("own values", own_gen), ("combined", comb_gen)):
            sim_stats = compute_stats(
                stack_outputs(draw_pixels(gen, n_sim, seed + 17 * run))["decay"], bg
            )
            rows.append(
                {
                    "experiment": name,
                    "simulation": label,
                    **coverage_fractions(exp_stats, sim_stats),
                }
            )
        logging.info(f"Verified {name}: {rows[-2]} / {rows[-1]}")
    table = pd.DataFrame(rows).set_index(["experiment", "simulation"])
    table.to_csv(
        os.path.join(os.path.dirname(setup_path), "coverage_per_experiment.csv")
    )
    return table
