"""Check that the experimental data lies within what the calibrated simulator produces."""

from dataclasses import replace

import numpy as np
from scipy import ndimage
from scipy.stats import beta as beta_dist

from pyfli import logging

from .config import SIM_MIN_CYCLES, SIM_MU_PER_CYCLE
from .optimizer import (
    CalibrationState,
    ObjectiveContext,
    active_photon_fraction,
    evaluate,
)
from .prior_estimation import BetaPrior, photon_prior_continuous, photon_prior_discrete

COVERAGE_STATS = ("phasor1", "phasor2", "log_counts", "edge", "bg_frac")
LIFETIME_PRIORS = ("tau2", "efficiency", "alpha1")


def hdr_covered(
    e: np.ndarray, s: np.ndarray, q: float = 1e-3, bins: int = 120
) -> np.ndarray:
    """
    2D points ``e`` inside the highest-density region that holds ``1 - q`` of the simulated
    points ``s``. The simulated density is a binned Gaussian kernel estimate with Scott's
    bandwidth (``sd * n ** (-1/6)`` per axis), so the region converges as the number of
    simulated pixels grows instead of depending on it, unlike a raw cell-count rule, which
    marks sparse but supported regions as uncovered.
    """
    rng = [
        [min(e[:, 0].min(), s[:, 0].min()), max(e[:, 0].max(), s[:, 0].max())],
        [min(e[:, 1].min(), s[:, 1].min()), max(e[:, 1].max(), s[:, 1].max())],
    ]
    H, xe, ye = np.histogram2d(s[:, 0], s[:, 1], bins=bins, range=rng)
    bandwidth = s.std(0) * len(s) ** (-1 / 6)
    cell = np.array([xe[1] - xe[0], ye[1] - ye[0]])
    sigma = np.where(cell > 0, bandwidth / np.where(cell > 0, cell, 1.0), 0.0)
    D = ndimage.gaussian_filter(H, sigma, mode="constant")

    def lookup(p):
        ix = np.clip(np.digitize(p[:, 0], xe) - 1, 0, bins - 1)
        iy = np.clip(np.digitize(p[:, 1], ye) - 1, 0, bins - 1)
        return D[ix, iy]

    level = np.quantile(lookup(s), q)
    return lookup(e) >= max(level, 1e-12)


def covered_pixels(
    exp_stats: dict, sim_stats: dict, q: float = 1e-3
) -> dict[str, np.ndarray]:
    """
    Per experimental pixel and statistic: is it inside the region holding ``1 - q`` of the
    simulated pixels? Raw phasors use the 2D highest-density region (:func:`hdr_covered`);
    photons, edge and background use the central ``[q, 1 - q]`` quantile range.
    """
    out = {
        h: hdr_covered(exp_stats[h], sim_stats[h], q) for h in ("phasor1", "phasor2")
    }
    for k in ("log_counts", "edge", "bg_frac"):
        s = sim_stats[k][np.isfinite(sim_stats[k])]
        lo, hi = np.quantile(s, [q, 1 - q])
        e = exp_stats[k]
        with np.errstate(invalid="ignore"):
            out[k] = (e >= lo) & (e <= hi)
    return out


def coverage_fractions(
    exp_stats: dict, sim_stats: dict, q: float = 1e-3
) -> dict[str, float]:
    """Fraction of experimental pixels inside the region populated by the simulation, per statistic."""
    flags = covered_pixels(exp_stats, sim_stats, q)
    out = {k: float(np.mean(v)) for k, v in flags.items()}
    for k in ("log_counts", "edge", "bg_frac"):
        out[k] = float(np.mean(flags[k][np.isfinite(exp_stats[k])]))
    return out


def _tails(
    exp_stats: dict, sim_stats: dict, key: str, q: float = 1e-3
) -> tuple[float, float]:
    """Fractions of experimental values below / above the simulated [q, 1 - q] range."""
    s = sim_stats[key][np.isfinite(sim_stats[key])]
    lo, hi = np.quantile(s, [q, 1 - q])
    e = exp_stats[key][np.isfinite(exp_stats[key])]
    return float(np.mean(e < lo)), float(np.mean(e > hi))


def flatten_beta(p: BetaPrior, factor: float) -> BetaPrior:
    """Move a Beta toward Beta(1, 1) on the same support: shape -> 1 + factor * (shape - 1)."""
    return BetaPrior(1 + factor * (p.a - 1), 1 + factor * (p.b - 1), p.offset, p.scale)


def flatten_lifetime_priors(priors, factor: float):
    out = priors
    for name in LIFETIME_PRIORS:
        p = getattr(priors, name)
        if p is not None:
            out = replace(out, **{name: flatten_beta(p, factor)})
    return out


def resupport_beta(p: BetaPrior, lo: float, hi: float) -> BetaPrior:
    """The same mean and SD expressed as a Beta on the new support ``[lo, hi]``."""
    m = (p.mean - lo) / (hi - lo)
    sd = p.sd / (hi - lo)
    k = m * (1 - m) / max(sd**2, 1e-10) - 1
    return BetaPrior.from_mean_conc(m, k, lo, hi - lo)


def cover_photon_range(
    state: CalibrationState,
    exp_counts: np.ndarray,
    n_active_bins: int,
    margin: float = 0.2,
    active_fraction: float = 1.0,
) -> CalibrationState:
    """
    Discrete: stretch ``n_cycles`` so that the signal photons of every experimental
    pixel are reachable, given that only ``active_fraction`` of them land in active bins.
    """
    sig = (
        np.clip(np.asarray(exp_counts, float) - state.dcr * n_active_bins, 0, None)
        / active_fraction
    )
    lo_c = max(
        int(np.floor(sig.min() * (1 - margin) / SIM_MU_PER_CYCLE)), SIM_MIN_CYCLES
    )
    hi_c = int(np.ceil(sig.max() * (1 + margin) / SIM_MU_PER_CYCLE))
    cur_lo, cur_hi = state.priors.photon_range
    lo_c, hi_c = (
        min(lo_c, int(cur_lo)),
        max(hi_c, int(cur_hi), lo_c + 10 * SIM_MIN_CYCLES),
    )
    if (lo_c, hi_c) == (int(cur_lo), int(cur_hi)):
        return state
    photon = resupport_beta(state.priors.photon, lo_c, hi_c)
    return replace(
        state, priors=replace(state.priors, photon=photon, photon_range=(lo_c, hi_c))
    )


def stretch_tau2_support(
    state: CalibrationState, below: float, above: float, step: float = 0.15
) -> CalibrationState:
    """Extend ``tau2_beta_range`` on the side(s) where experimental phasors are uncovered."""
    p = state.priors.tau2
    lo, hi = p.offset, p.offset + p.scale
    width = hi - lo
    new_lo = max(0.05, lo - step * width) if below > 0 else lo
    new_hi = hi + step * width if above > 0 else hi
    if (new_lo, new_hi) == (lo, hi):
        return state
    return replace(
        state, priors=replace(state.priors, tau2=resupport_beta(p, new_lo, new_hi))
    )


def _lifetime_side(exp_stats: dict, sim_stats: dict) -> tuple[float, float]:
    """
    Which side of the lifetime range is uncovered, as fractions ``(short, long)`` of
    experimental pixels beyond the simulated range. Raw phasors are rotated by the IRF
    delay, so the rotation-invariant first-harmonic modulus is used: it decreases with
    the lifetime.
    """
    s = np.hypot(*sim_stats["phasor1"].T)
    e = np.hypot(*exp_stats["phasor1"].T)
    lo, hi = np.quantile(s, [1e-3, 1 - 1e-3])
    return float(np.mean(e > hi)), float(np.mean(e < lo))


def step_factor(
    coverage: float, target: float, slowest: float = 0.9, fastest: float = 0.4
) -> float:
    """Flattening factor for one step: large steps far below the target, small steps near it."""
    return float(np.clip((coverage / target) ** 4, fastest, slowest))


def refit_photon_prior(
    state: CalibrationState, ctx: ObjectiveContext, exp_counts: np.ndarray
) -> CalibrationState:
    """Photon prior re-derived from the observed photon counts (or peaks) of the experiment."""
    if ctx.detector_type == "discrete":
        f_active = active_photon_fraction(state, ctx)
        prior, cyc = photon_prior_discrete(
            exp_counts, ctx.fm.n_active, state.dcr, active_fraction=f_active
        )
        state = replace(
            state, priors=replace(state.priors, photon=prior, photon_range=cyc)
        )
        return cover_photon_range(
            state, exp_counts, ctx.fm.n_active, active_fraction=f_active
        )
    prior, _ = photon_prior_continuous(
        ctx.exp_peaks, ctx.data_max, state.priors.bit, scale_corr=state.photon_scale
    )
    return replace(state, priors=replace(state.priors, photon=prior))


def widen(
    state: CalibrationState,
    failing: list[str],
    exp_stats: dict,
    sim_stats: dict,
    ctx: ObjectiveContext,
    exp_counts: np.ndarray,
    cov: dict,
    target: float,
    photon_refitted: bool,
) -> tuple[CalibrationState, list[str]]:
    """One widening step for every failing statistic; returns the new state and what was changed."""
    actions = []
    if "log_counts" in failing:
        if not photon_refitted:
            state = refit_photon_prior(state, ctx, exp_counts)
            actions.append("photon prior refitted to the observed counts")
        else:
            f = step_factor(cov["log_counts"], target)
            state = replace(
                state,
                priors=replace(
                    state.priors, photon=flatten_beta(state.priors.photon, f)
                ),
            )
            actions.append(f"photon prior flattened ({f:.2f})")
    if "phasor1" in failing or "phasor2" in failing:
        factor = step_factor(min(cov["phasor1"], cov["phasor2"]), target)
        short_side, long_side = _lifetime_side(exp_stats, sim_stats)
        p = state.priors.tau2
        near_edge = beta_dist.ppf([1e-3, 1 - 1e-3], p.a, p.b)
        below = short_side if near_edge[0] < 0.05 else 0.0
        above = long_side if near_edge[1] > 0.95 else 0.0
        state = stretch_tau2_support(state, below, above)
        state = replace(state, priors=flatten_lifetime_priors(state.priors, factor))
        actions.append(f"lifetime priors flattened ({factor:.2f})")
    if "edge" in failing:
        irf = state.irf
        a_half = float(min(irf.a_half * 1.3 + 0.5, ctx.fm.n_bins / 8))
        state = replace(state, irf=replace(irf, a_half=a_half))
        actions.append(f"a_half -> {a_half:.2f} bins")
    if "bg_frac" in failing:
        below, above = _tails(exp_stats, sim_stats, "bg_frac")
        if above >= below:
            irf = state.irf
            state = replace(
                state, irf=replace(irf, b_max=float(min(irf.b_max * 1.5 + 1e-5, 0.5)))
            )
            actions.append(f"b_max -> {state.irf.b_max:.2e}")
        else:
            state = replace(state, dcr=state.dcr * 0.7)
            actions.append(f"dcr -> {state.dcr:.2e}")
    return state, actions


def ensure_coverage(
    state: CalibrationState,
    ctx: ObjectiveContext,
    exp_stats: dict,
    exp_counts: np.ndarray,
    target: float = 0.99,
    n_sim: int = 20000,
    max_iter: int = 10,
    seed: int = 0,
) -> tuple[CalibrationState, list[dict], dict, dict, dict]:
    """
    Widen ``state`` until every statistic in :data:`COVERAGE_STATS` covers at least
    ``target`` of the experimental pixels (or ``max_iter`` widening steps were made).

    Returns ``(state, trace, coverage, sim, sim_stats)``; ``trace`` has one row per
    iteration with the coverage of each statistic, the distance to the experiment and the
    widening applied.
    """
    if ctx.detector_type == "discrete":
        state = cover_photon_range(
            state,
            exp_counts,
            ctx.fm.n_active,
            active_fraction=active_photon_fraction(state, ctx),
        )
    trace = []
    photon_refitted = False
    for it in range(max_iter + 1):
        dist, parts, sim, sim_stats = evaluate(state, ctx, n=n_sim, seed=seed)
        cov = coverage_fractions(exp_stats, sim_stats)
        failing = [k for k in COVERAGE_STATS if cov[k] < target]
        row = {
            "iteration": it,
            "distance": dist,
            **{f"cover {k}": v for k, v in cov.items()},
        }
        logging.info(
            f"Coverage iteration {it}: "
            + " ".join(f"{k}={v:.1%}" for k, v in cov.items())
            + f" distance={dist:.3f}"
        )
        if not failing or it == max_iter:
            row["widened"] = ""
            trace.append(row)
            break
        state, actions = widen(
            state,
            failing,
            exp_stats,
            sim_stats,
            ctx,
            exp_counts,
            cov,
            target,
            photon_refitted,
        )
        photon_refitted = photon_refitted or "log_counts" in failing
        row["widened"] = "; ".join(actions)
        trace.append(row)
    if failing:
        logging.warning(
            f"Coverage target {target:.0%} not reached for {failing} after {max_iter} widening steps"
        )
    return state, trace, cov, sim, sim_stats
