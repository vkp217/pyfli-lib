"""One simulator config that covers several experiments."""

import glob
import json
import os
from dataclasses import dataclass, field, replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from pyfli import logging

from .config import (
    SIM_FAMILY,
    SIM_FRACTION_OFFSET,
    SIM_FRACTION_SCALE,
    SIM_MU_PER_CYCLE,
)
from .coverage import coverage_fractions, flatten_lifetime_priors
from .data_io import load_experimental_data
from .expected_decay import DecayForwardModel
from .irf_estimation import IRFEstimate
from .optimizer import CalibrationState, ObjectiveContext, evaluate
from .prior_estimation import BetaPrior, PriorSet
from .sim_bridge import build_sim_config, save_calibration
from .summary_stats import StatsComparator, background_bins, compute_stats

LIFETIME_PRIORS = ("tau2", "efficiency", "alpha1")


# ------------------------------------------------------------ per-experiment states
@dataclass
class SessionCalibration:
    name: str
    folder: str
    json_path: str
    payload: dict
    state: CalibrationState

    @property
    def n_bins(self) -> int:
        return len(self.state.irf.template)


def load_session_calibration(
    json_path: str, name: str = "", folder: str = ""
) -> SessionCalibration:
    """Rebuild the calibrated :class:`CalibrationState` from a ``calibrated_sim_config.json``."""
    with open(json_path) as f:
        p = json.load(f)
    pr = p["priors"]

    def bp(key):
        d = pr.get(key)
        return None if d is None else BetaPrior(d["a"], d["b"], d["offset"], d["scale"])

    priors = PriorSet(
        tau2=bp("tau2"),
        efficiency=bp("efficiency"),
        alpha1=bp("alpha1"),
        photon=bp("photon"),
        photon_range=tuple(pr["photon_range"]),
        bit=int(pr["bit"]),
        mono_fraction=float(pr["mono_fraction"]),
        source=pr["source"],
    )
    template = np.load(os.path.join(os.path.dirname(json_path), p["irf"]["file"]))[0, 0]
    a_lo, a_hi = p["irf"]["a_range"]
    irf = IRFEstimate(
        p["irf"]["source"],
        template,
        (a_lo + a_hi) / 2,
        (a_hi - a_lo) / 2,
        float(p["irf"]["b_range"][1]),
        p["irf"]["params"],
        p["irf"]["pooled_fit"],
    )
    state = CalibrationState(
        irf, priors, float(p["dcr_config"]), float(p.get("photon_scale", 1.0))
    )
    return SessionCalibration(
        name or os.path.basename(os.path.dirname(json_path)),
        folder,
        json_path,
        p,
        state,
    )


# --------------------------------------------------------------------- combining
def covering_beta(
    samples: list[np.ndarray],
    weights: np.ndarray,
    offset: float,
    scale: float,
    q_exp: tuple[float, float] = (0.01, 0.99),
    q_cov: tuple[float, float] = (1e-3, 1 - 1e-3),
    min_shape: float = 1.0,
) -> tuple[BetaPrior, dict]:
    """
    Beta on ``[offset, offset + scale]`` matched to the weighted mixture of ``samples``,
    then widened until each sample set's ``q_exp`` interval lies inside the Beta's
    ``q_cov`` interval.
    """
    norm = [np.clip((np.asarray(s) - offset) / scale, 1e-6, 1 - 1e-6) for s in samples]
    w = np.asarray(weights, float) / np.sum(weights)
    m = float(sum(wi * x.mean() for wi, x in zip(w, norm)))
    m2 = float(sum(wi * (x**2).mean() for wi, x in zip(w, norm)))
    v = max(m2 - m**2, 1e-6)
    k = max(m * (1 - m) / v - 1, 2 * min_shape)
    need = [np.quantile(x, q_exp) for x in norm]

    def make(k):
        return BetaPrior(
            max(m * k, min_shape), max((1 - m) * k, min_shape), offset, scale
        )

    prior = make(k)
    for _ in range(60):
        lo, hi = beta_dist.ppf(q_cov, prior.a, prior.b)
        if all(lo <= a and b <= hi for a, b in need) or (
            prior.a <= min_shape and prior.b <= min_shape
        ):
            break
        k *= 0.85
        prior = make(k)
    lo, hi = beta_dist.ppf(q_cov, prior.a, prior.b)
    covered = [float(np.mean((x >= lo) & (x <= hi))) for x in norm]
    return prior, {
        "covered_fraction_per_experiment": covered,
        "interval": [offset + scale * lo, offset + scale * hi],
    }


def combine_states(
    sessions: list[SessionCalibration],
    weights: np.ndarray,
    fm: DecayForwardModel,
    n_samples: int = 20000,
    seed: int = 0,
) -> tuple[CalibrationState, dict]:
    rng = np.random.default_rng(seed)
    report = {}
    states = [s.state for s in sessions]

    # lifetime priors
    combined = {}
    for name in LIFETIME_PRIORS:
        priors = [getattr(st.priors, name) for st in states]
        if any(p is None for p in priors):
            combined[name] = None
            continue
        samples = [p.sample(n_samples, rng) for p in priors]
        if name == "tau2":
            lo = max(0.05, 0.9 * min(np.quantile(s, 1e-3) for s in samples))
            hi = 1.1 * max(np.quantile(s, 1 - 1e-3) for s in samples)
            offset, scale = lo, hi - lo
        else:
            offset, scale = SIM_FRACTION_OFFSET, SIM_FRACTION_SCALE
        combined[name], report[name] = covering_beta(samples, weights, offset, scale)

    # photon prior: n_cycles (discrete) or peak A / ADC max (continuous)
    detector = sessions[0].payload["inputs"]["detector_type"]
    if detector == "discrete":
        samples = [
            st.priors.photon.sample(n_samples, rng) for st in states
        ]  # offset / scale are the n_cycles range
        lo, hi = (
            min(st.priors.photon_range[0] for st in states),
            max(st.priors.photon_range[1] for st in states),
        )
        photon, report["photon"] = covering_beta(samples, weights, lo, hi - lo)
        photon_range, bit = (int(lo), int(hi)), 16
    else:
        bit = max(st.priors.bit for st in states)
        samples = [
            st.priors.photon.sample(n_samples, rng) * (2**st.priors.bit - 1)
            for st in states
        ]
        photon, report["photon"] = covering_beta(samples, weights, 0.0, 2**bit - 1)
        photon = BetaPrior(photon.a, photon.b)  # simulator scales by 2**bit - 1 itself
        photon_range = (0, 2**bit - 1)

    # IRF library: each experiment's calibrated IRF at its own position, peak-normalised
    library = np.stack(
        [st.irf.effective_irf() / st.irf.effective_irf().max() for st in states]
    )
    irf = IRFEstimate(
        "library",
        library[0],
        0.0,
        float(max(st.irf.a_half for st in states)),
        0.0,
        {
            "experiments": [s.name for s in sessions],
            "per_experiment": {
                s.name: {
                    "source": s.state.irf.source,
                    "peak_bin": int(np.argmax(lib)),
                    **s.state.irf.params,
                }
                for s, lib in zip(sessions, library)
            },
        },
        {},
        library=library,
    )

    priors = PriorSet(
        tau2=combined["tau2"],
        efficiency=combined["efficiency"],
        alpha1=combined["alpha1"],
        photon=photon,
        photon_range=photon_range,
        bit=bit,
        mono_fraction=float(
            np.average([st.priors.mono_fraction for st in states], weights=weights)
        ),
        source="pooled_covering",
    )
    dcr, b_max_k, report["background"] = covering_background(
        sessions, library, n_samples, rng
    )
    irf = replace(irf, b_max=float(b_max_k.max()), library_b_max=b_max_k)
    return CalibrationState(irf, priors, dcr, 1.0), report


def _background_fraction(
    dcr: float, b: np.ndarray, photons: np.ndarray, template_sum: float, T: int
) -> np.ndarray:
    """Fraction of a pixel's counts that is flat background: dark counts plus the IRF baseline b."""
    f_dcr = dcr * T / (photons + dcr * T)
    f_b = b * T / (template_sum + b * T)
    return f_dcr + (1 - f_dcr) * f_b


def covering_background(
    sessions: list[SessionCalibration],
    library: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    q: float = 0.995,
) -> tuple[float, np.ndarray, dict]:
    """
    One dark-count rate cannot reproduce experiments whose backgrounds differ 100-fold, and
    too much background pulls the raw phasors of clean experiments out of the simulated
    cloud. The combined config therefore uses the *lowest* dark-count rate and gets the rest
    from the per-pixel IRF baseline ``b ~ U(0, b_max_k)`` (background proportional to signal).
    Background belongs to the detector channel, so each library IRF k gets its own ``b_max_k``,
    chosen so that experiment k's background fractions are reached. A single shared range
    would make the low-background channel's clean pixels rare.
    """
    T = library.shape[1]
    dcr_c = float(min(s.state.dcr for s in sessions))
    b_max, per_exp = [], {}
    for s, template in zip(sessions, library):
        s_c = float(template.sum())
        st = s.state
        cycles = st.priors.photon.sample(n_samples, rng)
        photons = (
            cycles * SIM_MU_PER_CYCLE
            if s.payload["inputs"]["detector_type"] == "discrete"
            else cycles * (2**st.priors.bit - 1)
        )
        b = rng.uniform(0, st.irf.b_max, n_samples)
        f = _background_fraction(st.dcr, b, photons, float(st.irf.template.sum()), T)
        f_dcr_c = dcr_c * T / (photons + dcr_c * T)
        f_b = np.clip((f - f_dcr_c) / (1 - f_dcr_c), 0, 0.99)
        need = f_b * s_c / (T * (1 - f_b))
        b_k = float(max(np.quantile(need, q), 1e-6))
        b_max.append(b_k)
        per_exp[s.name] = {
            "dcr": st.dcr,
            "b_max": b_k,
            "background_fraction_q50": float(np.median(f)),
            "background_fraction_q99": float(np.quantile(f, 0.99)),
        }
    return dcr_c, np.array(b_max), {"dcr": dcr_c, "per_experiment": per_exp}


# ------------------------------------------------------------------ pooled data
def pooled_pixels(
    sessions: list[SessionCalibration],
    n_per_exp: int,
    seed: int = 0,
    decay_name: str = "clean_decay.npy",
    mask_name: str = "final_mask.npy",
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-size random subsets of each experiment's sample pixels; returns (decays (N, T), experiment index (N,))."""
    rng = np.random.default_rng(seed)
    decays, labels = [], []
    for i, s in enumerate(sessions):
        inp = s.payload["inputs"]
        if os.path.exists(inp["data_path"]):
            cube = load_experimental_data(
                inp["data_path"],
                bg_path=inp.get("bg_path"),
                bin_radius=int(inp.get("bin_radius", 0)),
            ).decay
        else:
            cube = np.load(os.path.join(s.folder, decay_name), mmap_mode="r")
        mask = np.load(os.path.join(os.path.dirname(s.json_path), "signal_mask.npy"))
        coords = np.argwhere(mask)
        pick = coords[
            np.sort(rng.choice(len(coords), min(n_per_exp, len(coords)), replace=False))
        ]
        decays.append(np.asarray(cube[pick[:, 0], pick[:, 1]], dtype=np.float64))
        labels.append(np.full(len(pick), i))
    return np.concatenate(decays), np.concatenate(labels)


# --------------------------------------------------------------------- pipeline
@dataclass
class PooledResult:
    config: dict
    sim_generator_kwargs: dict
    state: CalibrationState
    coverage: pd.DataFrame
    prior_report: dict
    validation: dict
    output_dir: str
    files: dict = field(default_factory=dict)


class PooledCalibrator:
    """
    Build one covering simulator config from per-experiment calibrations.

    Parameters
    ----------
    session_jsons : dict[str, str]
        experiment name -> ``calibrated_sim_config.json`` of that experiment.
    weighting : str
        "equal" (each experiment counts the same) or "pixels" (by number of sample pixels).
    target_coverage : float
        Required fraction of every experiment's pixels inside the simulated phasor clouds
        (harmonics 1 and 2). Priors are flattened up to ``max_flatten`` times to reach it.
    """

    def __init__(
        self,
        session_jsons: dict[str, str],
        output_dir: str,
        weighting: str = "equal",
        n_per_exp: int = 4000,
        n_validation: int = 20000,
        target_coverage: float = 0.90,
        max_flatten: int = 8,
        flatten_factor: float = 0.8,
        seed: int = 0,
        folders: dict[str, str] | None = None,
    ) -> None:
        folders = folders or {}
        self.sessions = [
            load_session_calibration(p, n, folders.get(n, ""))
            for n, p in session_jsons.items()
        ]
        inputs = [s.payload["inputs"] for s in self.sessions]
        for key in ("detector_type", "model_type", "laser_freq_mhz"):
            vals = {str(i[key]) for i in inputs}
            if len(vals) > 1:
                raise ValueError(f"Experiments disagree on {key}: {vals}")
        if len({s.n_bins for s in self.sessions}) > 1:
            raise ValueError("Experiments have different numbers of time bins")
        self.detector = inputs[0]["detector_type"]
        self.model = inputs[0]["model_type"]
        self.laser = float(inputs[0]["laser_freq_mhz"])
        self.jitter = all(bool(s.payload["config"]["jitter"]) for s in self.sessions)
        n_px = np.array(
            [s.payload["pixel_selection"]["n_selected"] for s in self.sessions], float
        )
        self.weights = (
            np.ones(len(self.sessions)) if weighting == "equal" else n_px / n_px.sum()
        )
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.n_per_exp, self.n_validation, self.seed = n_per_exp, n_validation, seed
        self.target_coverage, self.max_flatten, self.flatten_factor = (
            target_coverage,
            max_flatten,
            flatten_factor,
        )
        self.weighting = weighting

    def run(self) -> PooledResult:
        T = self.sessions[0].n_bins
        inactive = set(range(T))
        for s in self.sessions:
            inactive &= set(
                s.payload["sim_generator_kwargs"].get("inactive_bins") or []
            )
        self.inactive_bins = np.array(sorted(inactive), dtype=int)
        fm = DecayForwardModel(
            T, 1000.0 / self.laser, self.detector, self.jitter, self.inactive_bins
        )
        state, prior_report = combine_states(
            self.sessions, self.weights, fm, seed=self.seed
        )

        # pooled experimental pixels (same number per experiment, or proportional to pixel counts)
        n_each = self.n_per_exp if self.weighting == "equal" else None
        if n_each is None:
            total = self.n_per_exp * len(self.sessions)
            decays, labels = [], []
            for i, s in enumerate(self.sessions):
                d, _ = pooled_pixels(
                    [s], max(int(total * self.weights[i]), 50), self.seed + i
                )
                decays.append(d)
                labels.append(np.full(len(d), i))
            exp_decays, labels = np.concatenate(decays), np.concatenate(labels)
        else:
            exp_decays, labels = pooled_pixels(self.sessions, n_each, self.seed)
        bg_bins = background_bins(exp_decays.sum(0))
        exp_stats = compute_stats(exp_decays, bg_bins)

        ctx = ObjectiveContext(
            fm=fm,
            detector_type=self.detector,
            laser_freq_mhz=self.laser,
            jitter=self.jitter,
            comparator=StatsComparator(exp_stats),
            bg_bins=bg_bins,
            exp_counts=exp_stats["counts"],
            exp_peaks=exp_decays.max(-1),
            data_max=float(exp_decays.max()),
            bit_depth=state.priors.bit,
            prior_inflation=1.0,
            n_sim=self.n_validation,
            seed=self.seed,
            fixed_photon_prior=True,
            inactive_bins=self.inactive_bins,
        )
        # Coverage loop. Every marginal prior already covers every experiment, but joint
        # combinations (e.g. short tau2 + low background + NIR IRF) can still be drawn too
        # rarely to populate an experiment's corner of the phasor plot. While any experiment's
        # phasor coverage is below target, flatten all lifetime Betas toward uniform and re-simulate.
        trace = []
        for it in range(self.max_flatten + 1):
            dist, parts, sim, sim_stats = evaluate(
                state, ctx, n=self.n_validation, seed=self.seed + 7919
            )
            coverage = self._coverage_table(
                state, exp_stats, labels, sim_stats, prior_report
            )
            worst = float(coverage[["cover phasor1", "cover phasor2"]].min().min())
            trace.append(
                {
                    "iteration": it,
                    "min_phasor_coverage": worst,
                    "pooled_distance": dist,
                    **{
                        n: list(getattr(state.priors, n).as_tuple())
                        for n in LIFETIME_PRIORS
                        if getattr(state.priors, n) is not None
                    },
                }
            )
            logging.info(
                f"Coverage iteration {it}: worst phasor coverage {worst:.1%}, pooled distance {dist:.3f}"
            )
            if worst >= self.target_coverage or it == self.max_flatten:
                break
            state = replace(
                state, priors=flatten_lifetime_priors(state.priors, self.flatten_factor)
            )
        validation = {
            "pooled_distance": dist,
            "pooled_parts": parts,
            "n_validation_pixels": self.n_validation,
            "coverage_iterations": trace,
            "target_phasor_coverage": self.target_coverage,
        }
        logging.info(f"Minimum coverage per statistic:\n{coverage.min()}")

        config = build_sim_config(
            state.priors, self.detector, state.dcr, self.laser, self.jitter
        )
        kw = {
            "a_range": [round(v, 3) for v in state.a_range()],
            "b_range": [0.0, float(state.irf.b_max)],
            "irf_b_ranges": state.irf.b_ranges(),
            "irf_names": [s.name for s in self.sessions],
            "inactive_bins": self.inactive_bins.tolist(),
            "pixel": [0, 0],
            "family": SIM_FAMILY,
            "sensor_type": self.detector,
        }
        res = PooledResult(
            config, kw, state, coverage, prior_report, validation, self.output_dir
        )
        self._save(res, exp_stats, labels, sim, sim_stats)
        return res

    def _coverage_table(
        self, state, exp_stats, labels, sim_stats, prior_report
    ) -> pd.DataFrame:
        rows = []
        for i, s in enumerate(self.sessions):
            sel = labels == i
            e_stats = {
                k: v[sel]
                for k, v in exp_stats.items()
                if k not in ("mean_decay", "mean_decay_rel_var")
            }
            row = {
                "experiment": s.name,
                **{
                    f"cover {k}": v
                    for k, v in coverage_fractions(e_stats, sim_stats).items()
                },
            }
            for name in LIFETIME_PRIORS:
                mine, comb = getattr(s.state.priors, name), getattr(state.priors, name)
                if (
                    mine is not None and comb is not None
                ):  # recomputed: flattening changes the combined prior
                    x = mine.sample(20000, np.random.default_rng(i))
                    lo, hi = comb.offset + comb.scale * beta_dist.ppf(
                        [1e-3, 1 - 1e-3], comb.a, comb.b
                    )
                    row[f"prior {name} covered"] = float(np.mean((x >= lo) & (x <= hi)))
            row["prior photon covered"] = prior_report["photon"][
                "covered_fraction_per_experiment"
            ][i]
            rows.append(row)
        return pd.DataFrame(rows).set_index("experiment")

    # ------------------------------------------------------------------ output
    def _save(self, res: PooledResult, exp_stats, labels, sim, sim_stats) -> None:
        out, st = self.output_dir, res.state
        files = {
            "irf": "calibrated_irf.npy",
            "coverage": "coverage.csv",
            "validation_sample": "validation_simulated_pixels.npz",
            "config": "calibrated_sim_config.json",
        }
        np.save(os.path.join(out, files["irf"]), st.irf.irf_cube())
        res.coverage.to_csv(os.path.join(out, files["coverage"]))
        np.savez_compressed(os.path.join(out, files["validation_sample"]), **sim)
        payload = {
            "inputs": {
                "detector_type": self.detector,
                "model_type": self.model,
                "laser_freq_mhz": self.laser,
                "weighting": self.weighting,
                "experiments": {s.name: s.json_path for s in self.sessions},
                "weights": self.weights.tolist(),
            },
            "config": res.config,
            "sim_generator_kwargs": res.sim_generator_kwargs,
            "irf": {
                "source": st.irf.source,
                "file": files["irf"],
                "params": st.irf.params,
                "a_range": list(st.a_range()),
                "b_range": list(st.b_range()),
                "pooled_fit": st.irf.pooled_fit,
            },
            "priors": st.priors.summary(),
            "priors_per_experiment": {
                s.name: s.state.priors.summary() for s in self.sessions
            },
            "prior_coverage": res.prior_report,
            "dcr_config": st.dcr,
            "photon_scale": 1.0,
            "coverage": res.coverage.reset_index().to_dict(orient="records"),
            "validation": res.validation,
            "usage": "from pyfli.simulator.calibration import load_calibrated_generator; "
            "gen = load_calibrated_generator('<this file>'); gen.simulate_once()",
        }
        save_calibration(os.path.join(out, files["config"]), payload)
        files.update(self._plots(res, exp_stats, labels, sim, sim_stats))
        res.files = {k: os.path.join(out, v) for k, v in files.items()}
        logging.info(f"Pooled calibration saved to {out}")

    def _plots(self, res, exp_stats, labels, sim, sim_stats) -> dict:
        out, st = self.output_dir, res.state
        files = {}
        colors = plt.cm.tab10(np.arange(len(self.sessions)) % 10)

        # priors: each experiment's Beta vs the combined covering Beta
        names = [n for n in LIFETIME_PRIORS if getattr(st.priors, n) is not None]
        titles = {
            "tau2": "tau2 (ns)",
            "efficiency": "efficiency E",
            "alpha1": "A1 (photon fraction of tau1)",
        }
        fig, axes = plt.subplots(1, len(names) + 1, figsize=(5 * (len(names) + 1), 4))
        for ax, name in zip(axes, names):
            p = getattr(st.priors, name)
            x = np.linspace(p.offset, p.offset + p.scale, 400)
            for s, c in zip(self.sessions, colors):
                q = getattr(s.state.priors, name)
                xq = np.clip((x - q.offset) / q.scale, 0, 1)
                y = np.where(
                    (x >= q.offset) & (x <= q.offset + q.scale),
                    beta_dist.pdf(xq, q.a, q.b) / q.scale,
                    0,
                )
                ax.plot(x, y / max(y.max(), 1e-12), color=c, lw=1, label=s.name[:28])
            yc = beta_dist.pdf((x - p.offset) / p.scale, p.a, p.b)
            ax.fill_between(
                x, 0, yc / yc.max(), color="k", alpha=0.15, label="combined (covering)"
            )
            ax.set(title=titles[name], yticks=[])
        ax = axes[-1]
        for i, (s, c) in enumerate(zip(self.sessions, colors)):
            ax.hist(
                exp_stats["log_counts"][labels == i],
                50,
                histtype="step",
                color=c,
                density=True,
            )
        ax.hist(
            sim_stats["log_counts"],
            50,
            color="k",
            alpha=0.2,
            density=True,
            label="combined simulation",
        )
        ax.set(title="photons / pixel (log10)", yticks=[])
        axes[0].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        files["plot_priors"] = "pooled_priors.png"
        fig.savefig(os.path.join(out, files["plot_priors"]), dpi=130)
        plt.close(fig)

        # phasor coverage per experiment
        n = len(self.sessions)
        fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4), squeeze=False)
        for j, (s, c) in enumerate(zip(self.sessions, colors)):
            sel = labels == j
            for i, h in enumerate(("phasor1", "phasor2")):
                ax = axes[i, j]
                ax.hexbin(
                    sim_stats[h][:, 0],
                    sim_stats[h][:, 1],
                    gridsize=60,
                    cmap="Greys",
                    mincnt=1,
                    bins="log",
                )
                e = exp_stats[h][sel]
                ax.scatter(e[::4, 0], e[::4, 1], s=1, color=c, alpha=0.4)
                cov = res.coverage.loc[s.name, f"cover {h}"]
                ax.set_title(
                    f"{s.name[:22]}\nh={i + 1}: {cov:.1%} covered"
                    if i == 0
                    else f"h={i + 1}: {cov:.1%} covered",
                    fontsize=8,
                )
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(
            "raw phasors: combined simulation (grey) vs each experiment (colour)"
        )
        fig.tight_layout()
        files["plot_phasor_coverage"] = "pooled_phasor_coverage.png"
        fig.savefig(os.path.join(out, files["plot_phasor_coverage"]), dpi=130)
        plt.close(fig)

        # pooled decay shape and rising edges
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        for i, (s, c) in enumerate(zip(self.sessions, colors)):
            ax[1].hist(
                exp_stats["edge"][labels == i],
                60,
                histtype="step",
                color=c,
                density=True,
                label=s.name[:28],
            )
        ax[1].hist(
            sim_stats["edge"],
            60,
            color="k",
            alpha=0.2,
            density=True,
            label="combined simulation",
        )
        ax[1].set(title="rising-edge position (bins)")
        ax[1].legend(fontsize=7)
        ax[0].semilogy(exp_stats["mean_decay"], label="pooled experiments")
        ax[0].semilogy(sim_stats["mean_decay"], "--", label="combined simulation")
        ax[0].set(
            title="pooled decay shape",
            ylim=(
                exp_stats["mean_decay"].max() * 1e-4,
                exp_stats["mean_decay"].max() * 3,
            ),
        )
        ax[0].legend()
        fig.tight_layout()
        files["plot_decay"] = "pooled_decay_edges.png"
        fig.savefig(os.path.join(out, files["plot_decay"]), dpi=130)
        plt.close(fig)
        return files


def find_session_configs(root: str, pattern: str = "*") -> dict[str, str]:
    """name -> calibrated_sim_config.json for every calibrated experiment under ``root``."""
    return {
        os.path.basename(os.path.dirname(p)): p
        for p in sorted(
            glob.glob(os.path.join(root, pattern, "calibrated_sim_config.json"))
        )
    }


# photon helper for reporting
def photon_range_photons(state: CalibrationState) -> tuple[float, float]:
    lo, hi = state.priors.photon_range
    return lo * SIM_MU_PER_CYCLE, hi * SIM_MU_PER_CYCLE
