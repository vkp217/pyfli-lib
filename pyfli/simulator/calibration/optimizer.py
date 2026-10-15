"""Simulation-based calibration of the pyfli simulator config."""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist

from pyfli import logging

from .expected_decay import DecayForwardModel, gated_bin_probabilities
from .irf_estimation import (
    IRFEstimate,
    apply_irf_theta,
    irf_theta_from_estimate,
    irf_theta_names,
)
from .prior_estimation import (
    MAX_CONC,
    BetaPrior,
    PriorSet,
    photon_prior_continuous,
    photon_prior_discrete,
)
from .sim_bridge import build_sim_config, simulate_batch
from .summary_stats import PreparedSim, StatsComparator, compute_stats

STEP = {  # initial Nelder-Mead simplex step per parameter
    "mu_bin": 1.0,
    "log_sigma_ns": 0.3,
    "log_tail_ns": 0.4,
    "a_center": 1.0,
    "log_a_half": 0.5,
    "log_b_max": 1.0,
    "log_dcr": 1.0,
    "log_photon_scale": 0.15,
    "tau2_logit_m": 0.4,
    "tau2_log_k": 0.5,
    "efficiency_logit_m": 0.4,
    "efficiency_log_k": 0.5,
    "alpha1_logit_m": 0.4,
    "alpha1_log_k": 0.5,
}


@dataclass
class CalibrationState:
    irf: IRFEstimate
    priors: PriorSet
    dcr: float  # discrete: dark counts / bin / pixel; continuous: config value (scaled by bit / 8 in pyfli)
    photon_scale: float = 1.0  # continuous only

    def a_range(self):
        return self.irf.a_range

    def b_range(self):
        return self.irf.b_range


@dataclass
class ObjectiveContext:
    fm: DecayForwardModel
    detector_type: str
    laser_freq_mhz: float
    jitter: bool
    comparator: StatsComparator
    bg_bins: np.ndarray
    exp_counts: np.ndarray  # all selected pixels
    exp_peaks: np.ndarray  # bg-subtracted smoothed peaks (continuous)
    data_max: float
    bit_depth: int | None
    prior_inflation: float
    n_sim: int
    seed: int
    reference_moments: dict = field(default_factory=dict)
    reference_weight: float = 0.0
    fixed_photon_prior: bool = (
        False  # True: keep the state's photon prior (e.g. a pooled covering prior)
    )
    inactive_bins: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def active_photon_fraction(state: CalibrationState, ctx: ObjectiveContext) -> float:
    """
    Expected share of signal photons that land in active time bins, for the prior-mean
    lifetimes of `state` (photons in inactive bins are zeroed in data and simulation).
    1 for continuous detectors or when every bin is active.
    """
    fm = ctx.fm
    if ctx.detector_type != "discrete" or fm.n_active == fm.n_bins or state.irf is None:
        return 1.0
    pr = state.priors
    tau2 = pr.tau2.mean
    if pr.efficiency is None or pr.alpha1 is None or pr.mono_fraction >= 1:
        taus, fracs = [tau2], [1.0]
    else:
        e, a, m = pr.efficiency.mean, pr.alpha1.mean, pr.mono_fraction
        taus, fracs = [tau2 * (1 - e), tau2], [(1 - m) * a, (1 - m) * (1 - a) + m]
    irf = fm.with_baseline(state.irf.effective_irf(), state.irf.b_max / 2)
    p = np.asarray(fracs) @ gated_bin_probabilities(np.asarray(taus), irf, fm.period)
    return float(np.clip(p[fm.active].sum(), 1e-3, 1.0))


def refresh_photon_prior(
    state: CalibrationState, ctx: ObjectiveContext
) -> CalibrationState:
    if ctx.fixed_photon_prior:
        return state
    if ctx.detector_type == "discrete":
        prior, cyc = photon_prior_discrete(
            ctx.exp_counts,
            ctx.fm.n_active,
            state.dcr,
            ctx.prior_inflation,
            active_photon_fraction(state, ctx),
        )
        pri = replace(state.priors, photon=prior, photon_range=cyc, bit=16)
    else:
        prior, bit = photon_prior_continuous(
            ctx.exp_peaks,
            ctx.data_max,
            ctx.bit_depth,
            inflation=ctx.prior_inflation,
            scale_corr=state.photon_scale,
        )
        pri = replace(state.priors, photon=prior, photon_range=(0, 2**bit - 1), bit=bit)
    return replace(state, priors=pri)


class ParameterVector:
    """Maps a flat unconstrained vector to a :class:`CalibrationState` and back."""

    def __init__(
        self, base: CalibrationState, ctx: ObjectiveContext, names: list[str]
    ) -> None:
        self.base, self.ctx, self.names = base, ctx, names

    def initial(self) -> np.ndarray:
        s = self.base
        vals = dict(zip(irf_theta_names(s.irf.source), irf_theta_from_estimate(s.irf)))
        vals.update(
            {
                "log_a_half": np.log(max(s.irf.a_half, 0.05)),
                "log_b_max": np.log(max(s.irf.b_max, 1e-6)),
                "log_dcr": np.log(max(s.dcr, 1e-6)),
                "log_photon_scale": np.log(s.photon_scale),
            }
        )
        for name in ("tau2", "efficiency", "alpha1"):
            p = getattr(s.priors, name)
            if p is not None:
                vals[f"{name}_logit_m"] = logit(np.clip(p.mean_norm, 1e-3, 1 - 1e-3))
                vals[f"{name}_log_k"] = np.log(p.conc)
        return np.array([vals[n] for n in self.names], float)

    def state(self, x: np.ndarray) -> CalibrationState:
        v = dict(zip(self.names, x))
        s, fm = self.base, self.ctx.fm
        irf = s.irf
        theta_names = irf_theta_names(irf.source)
        if any(n in v for n in theta_names):
            theta = irf_theta_from_estimate(irf).copy()
            for i, n in enumerate(theta_names):
                if n in v:
                    theta[i] = v[n]
            if irf.source == "simulated":
                theta[0] = np.clip(theta[0], 0, fm.n_bins - 1)
                theta[1] = np.clip(theta[1], np.log(0.3 * fm.dt), np.log(1.5))
                theta[2] = np.clip(theta[2], np.log(3e-3), np.log(1.0))
            else:
                theta[0] = np.clip(theta[0], -fm.n_bins / 3, fm.n_bins / 3)
            irf = apply_irf_theta(irf, fm, theta)
        if "log_a_half" in v:
            irf = replace(
                irf, a_half=float(np.clip(np.exp(v["log_a_half"]), 0.0, fm.n_bins / 8))
            )
        if "log_b_max" in v:
            irf = replace(irf, b_max=float(np.clip(np.exp(v["log_b_max"]), 1e-7, 0.5)))
        dcr = (
            float(np.clip(np.exp(v["log_dcr"]), 1e-7, 1e4)) if "log_dcr" in v else s.dcr
        )
        scale = (
            float(np.clip(np.exp(v["log_photon_scale"]), 0.2, 5.0))
            if "log_photon_scale" in v
            else s.photon_scale
        )

        priors = s.priors
        for name in ("tau2", "efficiency", "alpha1"):
            p = getattr(priors, name)
            if p is None or f"{name}_logit_m" not in v:
                continue
            m = float(expit(v[f"{name}_logit_m"]))
            k = float(np.clip(np.exp(v[f"{name}_log_k"]), 1.0, MAX_CONC))
            priors = replace(
                priors, **{name: BetaPrior.from_mean_conc(m, k, p.offset, p.scale)}
            )
        return refresh_photon_prior(CalibrationState(irf, priors, dcr, scale), self.ctx)


def evaluate(
    state: CalibrationState,
    ctx: ObjectiveContext,
    n: int | None = None,
    seed: int | None = None,
):
    """Simulate with ``state`` and return ``(distance, parts, sim_batch, sim_stats)``."""
    cfg = build_sim_config(
        state.priors, ctx.detector_type, state.dcr, ctx.laser_freq_mhz, ctx.jitter
    )
    sim = simulate_batch(
        state.irf.irf_cube(),
        cfg,
        state.a_range(),
        state.b_range(),
        n or ctx.n_sim,
        ctx.seed if seed is None else seed,
        irf_b_ranges=state.irf.b_ranges(),
        inactive_bins=ctx.inactive_bins,
    )
    sim_stats = compute_stats(sim["decay"], ctx.bg_bins)
    dist, parts = ctx.comparator.distance(sim_stats)
    return dist, parts, sim, sim_stats


def _reference_penalty(state: CalibrationState, ctx: ObjectiveContext) -> float:
    """
    ``reference_weight * sum(((prior mean - reference mean) / reference sd)^2)`` in
    physical units, so it does not depend on the prior's support (``tau2_beta_range`` is
    re-centred between rounds). The sd is floored at 5 % of the prior's support width.
    """
    if ctx.reference_weight <= 0:
        return 0.0
    pen = 0.0
    for name, (m_ref, sd_ref) in ctx.reference_moments.items():
        p = getattr(state.priors, name, None)
        if p is not None:
            pen += ((p.mean - m_ref) / max(sd_ref, 0.05 * p.scale)) ** 2
    return ctx.reference_weight * pen


def optimize_stage(
    base: CalibrationState,
    ctx: ObjectiveContext,
    names: list[str],
    max_evals: int,
    label: str,
) -> tuple[CalibrationState, list[dict]]:
    if not names or max_evals <= 0:
        return base, []
    pv = ParameterVector(base, ctx, names)
    x0 = pv.initial()
    history = []
    best = {"f": np.inf, "x": x0}

    def objective(x):
        state = pv.state(x)
        try:
            dist, parts, *_ = evaluate(state, ctx)
        except (
            Exception
        ) as exc:  # invalid configs (e.g. degenerate IRF) count as bad, not fatal
            logging.warning(f"[{label}] simulation failed: {exc}")
            return 1e6
        f = dist + _reference_penalty(state, ctx)
        history.append(
            {
                "stage": label,
                "eval": len(history),
                "objective": f,
                **parts,
                **{n: float(val) for n, val in zip(names, x)},
            }
        )
        if f < best["f"]:
            best.update(f=f, x=np.array(x))
            logging.info(
                f"[{label}] eval {len(history)}: objective {f:.4f} "
                + " ".join(f"{k}={v:.3f}" for k, v in parts.items())
            )
        return f

    simplex = np.vstack(
        [x0] + [x0 + np.eye(len(x0))[i] * STEP.get(n, 0.3) for i, n in enumerate(names)]
    )
    minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "maxfev": max_evals,
            "initial_simplex": simplex,
            "xatol": 1e-3,
            "fatol": 1e-3,
            "adaptive": True,
        },
    )
    logging.info(
        f"[{label}] best objective {best['f']:.4f} after {len(history)} evaluations"
    )
    return pv.state(best["x"]), history


def stage_parameter_names(
    state: CalibrationState,
    ctx: ObjectiveContext,
    optimize_irf: bool,
    optimize_noise: bool,
    optimize_priors: bool,
    is_bi: bool,
) -> tuple[list[str], list[str]]:
    """(nuisance names searched with real simulations, prior names searched on the re-weighted bank)."""
    nuisance = []
    if optimize_irf:
        nuisance += [*irf_theta_names(state.irf.source), "log_a_half", "log_b_max"]
    if optimize_noise:
        nuisance += ["log_dcr"]
        if ctx.detector_type == "continuous":
            nuisance += ["log_photon_scale"]
    prior = []
    if optimize_priors:
        prior += ["tau2_logit_m", "tau2_log_k"]
        if is_bi:
            prior += [
                "efficiency_logit_m",
                "efficiency_log_k",
                "alpha1_logit_m",
                "alpha1_log_k",
            ]
    return nuisance, prior


# ----------------------------------------------------------- importance-weighted bank
LIFETIME_PRIORS = ("tau2", "efficiency", "alpha1")
TRUTH_KEY = {"tau2": "tau2", "efficiency": "Efficiency", "alpha1": "alpha1"}


class PriorBank:
    """
    Simulated pixels from a broad proposal, re-weighted to any candidate set of Beta priors.

    The bank is a defensive mixture: ``uniform_share`` of the pixels come from flat
    Beta(1, 1) priors on the current supports, the rest from the current priors widened
    about 4x in variance. For target priors ``pi`` each pixel gets the weight
    ``pi(theta) / q(theta)``, so the statistics of any candidate prior are obtained
    without re-simulating. IRF, noise and photon priors stay fixed inside one bank.
    """

    def __init__(
        self,
        state: CalibrationState,
        ctx: ObjectiveContext,
        n: int,
        seed: int,
        uniform_share: float = 0.3,
    ):
        self.ctx = ctx
        names = [p for p in LIFETIME_PRIORS if getattr(state.priors, p) is not None]
        self.names = names
        cur = {p: getattr(state.priors, p) for p in names}
        flat = {p: BetaPrior(1.0, 1.0, cur[p].offset, cur[p].scale) for p in names}
        wide = {
            p: BetaPrior.from_mean_conc(
                cur[p].mean_norm, max(cur[p].conc / 4, 2.0), cur[p].offset, cur[p].scale
            )
            for p in names
        }
        n_u = int(np.rint(n * uniform_share))
        batches = []
        for sub_n, pri, s in ((n_u, flat, seed), (n - n_u, wide, seed + 1)):
            st = replace(state, priors=replace(state.priors, **pri))
            cfg = build_sim_config(
                st.priors, ctx.detector_type, st.dcr, ctx.laser_freq_mhz, ctx.jitter
            )
            batches.append(
                simulate_batch(
                    st.irf.irf_cube(),
                    cfg,
                    st.a_range(),
                    st.b_range(),
                    sub_n,
                    s,
                    inactive_bins=ctx.inactive_bins,
                )
            )
        self.sim = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
        self.prepared = PreparedSim(compute_stats(self.sim["decay"], ctx.bg_bins))
        self.mono = (self.sim["Efficiency"] == 1.0) & (self.sim["alpha1"] == 1.0)
        self.x = {
            p: np.clip(
                (self.sim[TRUTH_KEY[p]] - cur[p].offset) / cur[p].scale, 1e-6, 1 - 1e-6
            )
            for p in names
        }
        self.log_q = np.logaddexp(
            np.log(uniform_share) + self._log_density(flat),
            np.log(1 - uniform_share) + self._log_density(wide),
        )

    def _log_density(self, priors: dict) -> np.ndarray:
        out = np.zeros(len(self.mono))
        for p in self.names:
            lp = beta_dist.logpdf(self.x[p], priors[p].a, priors[p].b)
            out += lp if p == "tau2" else np.where(self.mono, 0.0, lp)
        return out

    def weights(self, priors: PriorSet) -> np.ndarray:
        lw = self._log_density({p: getattr(priors, p) for p in self.names}) - self.log_q
        w = np.exp(lw - lw.max())
        return w / w.sum()

    def _pooled_rel_var(self, w: np.ndarray) -> np.ndarray:
        """
        Relative variance per bin of the pooled decay of an i.i.d. sample of the bank's
        size drawn from the candidate prior: ``(sum_i w_i (d_ik - m_k)^2) / n / m_k^2``,
        i.e. the candidate's per-pixel variance (importance-weighted) divided by ``n``.
        It deliberately does not use the importance-sampling variance
        ``sum_i w_i^2 (d_ik - m_k)^2``: that grows as the weights degenerate (narrow
        candidate priors), and subtracting it would reward those priors.
        """
        if not hasattr(self, "_decay_sq"):
            self._decay_sq = self.sim["decay"] ** 2
        m = w @ self.sim["decay"]
        var = (w @ self._decay_sq - m**2) / len(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(m > 0, np.clip(var, 0.0, None) / m**2, 0.0)

    def distance(self, priors: PriorSet) -> tuple[float, dict, float]:
        w = self.weights(priors)
        ess = 1.0 / np.sum(w**2)
        dist, parts = self.ctx.comparator.distance(
            self.prepared,
            mean_decay=w @ self.sim["decay"],
            pixel_weights=w,
            mean_decay_rel_var=self._pooled_rel_var(w),
        )
        return dist, parts, ess


def recenter_tau2_support(
    state: CalibrationState, margin: float = 0.3
) -> CalibrationState:
    """
    Move ``tau2_beta_range`` to the fitted prior's 0.1-99.9 % range plus ``margin`` on each
    side, keeping the prior's mean and SD. Initial supports from noisy per-pixel fits are
    usually far too wide, which wastes the bank and makes the Beta very skewed.
    """
    p = state.priors.tau2
    lo_q, hi_q = p.offset + p.scale * beta_dist.ppf([1e-3, 1 - 1e-3], p.a, p.b)
    width = hi_q - lo_q
    lo, hi = max(0.05, lo_q - margin * width), hi_q + margin * width
    if hi_q > p.offset + 0.98 * p.scale:  # mass piling at the upper bound: let it grow
        hi = p.offset + 1.5 * p.scale
    m, sd = (p.mean - lo) / (hi - lo), p.sd / (hi - lo)
    k = m * (1 - m) / max(sd**2, 1e-8) - 1
    new = BetaPrior.from_mean_conc(m, k, lo, hi - lo)
    return replace(state, priors=replace(state.priors, tau2=new))


def _prior_bounds(names: list[str]) -> list[tuple[float, float]]:
    return [
        (-6.0, 6.0) if n.endswith("_logit_m") else (0.0, float(np.log(MAX_CONC)))
        for n in names
    ]


def fit_priors_on_bank(
    base: CalibrationState,
    ctx: ObjectiveContext,
    names: list[str],
    n_bank: int,
    max_evals: int,
    round_idx: int,
    min_ess: float = 300.0,
) -> tuple[CalibrationState, list[dict]]:
    """One bank round: simulate, global search (differential evolution), Nelder-Mead polish."""
    if not names:
        return base, []
    label = f"priors_round{round_idx + 1}"
    bank = PriorBank(base, ctx, n_bank, seed=ctx.seed + 101 * (round_idx + 1))
    pv = ParameterVector(base, ctx, names)
    bounds = _prior_bounds(names)
    x0 = np.clip(pv.initial(), [b[0] for b in bounds], [b[1] for b in bounds])
    history = []
    best = {"f": np.inf, "x": x0}

    def objective(x):
        st = pv.state(x)
        dist, parts, ess = bank.distance(st.priors)
        f = dist + _reference_penalty(st, ctx) + max(0.0, min_ess / max(ess, 1.0) - 1.0)
        history.append(
            {
                "stage": label,
                "eval": len(history),
                "objective": f,
                "ess": ess,
                **parts,
                **{n: float(v) for n, v in zip(names, x)},
            }
        )
        if f < best["f"]:
            best.update(f=f, x=np.array(x))
        return f

    popsize = 8
    de_iters = max(1, int(0.7 * max_evals / (popsize * len(names))) - 1)
    differential_evolution(
        objective,
        bounds,
        x0=x0,
        popsize=popsize,
        maxiter=de_iters,
        tol=1e-3,
        seed=ctx.seed + round_idx,
        polish=False,
        init="sobol",
    )
    remaining = max(max_evals - len(history), 20)
    simplex = np.vstack(
        [best["x"]]
        + [
            best["x"] + np.eye(len(names))[i] * 0.5 * STEP.get(n, 0.3)
            for i, n in enumerate(names)
        ]
    )
    minimize(
        objective,
        best["x"],
        method="Nelder-Mead",
        options={
            "maxfev": remaining,
            "initial_simplex": simplex,
            "xatol": 1e-4,
            "fatol": 1e-4,
            "adaptive": True,
        },
    )
    state = pv.state(best["x"])
    _, parts, ess = bank.distance(state.priors)
    logging.info(
        f"[{label}] objective {best['f']:.4f}, ESS {ess:.0f} / {n_bank}: "
        + " ".join(f"{k}={v:.3f}" for k, v in parts.items())
    )
    if "tau2_logit_m" in names:
        state = recenter_tau2_support(state)
    return state, history


def run_calibration_rounds(
    state: CalibrationState,
    ctx: ObjectiveContext,
    nuisance: list[str],
    prior_names: list[str],
    n_bank: int,
    rounds: int,
    max_evals_nuisance: int,
    max_evals_priors: int,
) -> tuple[CalibrationState, list[dict]]:
    """Alternate prior fits (bank) and IRF / detector fits (real simulations)."""
    history = []
    rounds = rounds if prior_names else 1
    for r in range(rounds):
        state, h = fit_priors_on_bank(
            state, ctx, prior_names, n_bank, max_evals_priors, r
        )
        history += h
        evals = max_evals_nuisance if r == 0 else max(max_evals_nuisance // 2, 20)
        state, h = optimize_stage(
            state, ctx, nuisance, evals, f"irf_noise_round{r + 1}"
        )
        history += h
    return state, history
