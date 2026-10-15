"""Beta priors for the pyfli simulator config."""

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from pyfli import logging
from pyfli.solver.shared_metrics import poisson_deviance

from .config import (
    SIM_FRACTION_OFFSET,
    SIM_FRACTION_SCALE,
    SIM_MIN_CYCLES,
    SIM_MU_PER_CYCLE,
    SIM_QE_RANGE,
)
from .expected_decay import DecayForwardModel

MIN_SHAPE = 0.3  # keep Beta shapes away from degenerate U-shapes
MAX_CONC = 1000.0
BI_FIT_PRIOR_SD = np.array(
    [0.7, 2.0, 2.0]
)  # log tau2, logit E, logit A1 (quick fits only)


@dataclass
class BetaPrior:
    a: float
    b: float
    offset: float = 0.0
    scale: float = 1.0

    @property
    def mean_norm(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def conc(self) -> float:
        return self.a + self.b

    @property
    def mean(self) -> float:
        return self.offset + self.scale * self.mean_norm

    @property
    def sd(self) -> float:
        m, k = self.mean_norm, self.conc
        return self.scale * np.sqrt(m * (1 - m) / (k + 1))

    @classmethod
    def from_mean_conc(
        cls, m: float, k: float, offset: float = 0.0, scale: float = 1.0
    ) -> "BetaPrior":
        m = float(np.clip(m, 1e-3, 1 - 1e-3))
        k = float(np.clip(k, 2 * MIN_SHAPE / min(m, 1 - m), MAX_CONC))
        return cls(max(m * k, MIN_SHAPE), max((1 - m) * k, MIN_SHAPE), offset, scale)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.offset + self.scale * rng.beta(self.a, self.b, n)

    def as_tuple(self) -> tuple[float, float]:
        return (round(self.a, 4), round(self.b, 4))


def fit_beta(
    values: np.ndarray, offset: float = 0.0, scale: float = 1.0, inflation: float = 1.0
) -> BetaPrior:
    """
    Method-of-moments Beta fit on (values - offset) / scale, with the variance multiplied
    by ``inflation``. The values are winsorized at their 0.5 / 99.5 percentiles (capped,
    not dropped), which limits single outliers without the ~3 % variance loss that
    trimming the tails causes.
    """
    x = (np.asarray(values, float) - offset) / scale
    x = x[np.isfinite(x)]
    if len(x) < 10:
        logging.warning(f"Only {len(x)} values for a Beta fit: using a broad prior")
        return BetaPrior(1.5, 1.5, offset, scale)
    lo, hi = np.percentile(x, [0.5, 99.5])
    x = np.clip(np.clip(x, lo, hi), 1e-3, 1 - 1e-3)
    m, v = x.mean(), x.var() * inflation
    v = min(max(v, 1e-6), m * (1 - m) * 0.95)
    return BetaPrior.from_mean_conc(m, m * (1 - m) / v - 1, offset, scale)


@dataclass
class PriorSet:
    tau2: BetaPrior  # physical units: tau2 (ns), offset/scale = simulator's tau2_beta_range
    efficiency: BetaPrior | None  # simulator's 0.998 / 0.001 scaling
    alpha1: BetaPrior | None
    photon: BetaPrior  # normalised photon variable (n_cycles in [lo, hi] or A / (2**bit - 1))
    photon_range: tuple[
        float, float
    ]  # discrete: n_cycles (lo, hi); continuous: (0, 2**bit - 1)
    bit: int
    mono_fraction: float
    source: str
    reference_moments: dict = field(
        default_factory=dict
    )  # physical mean / sd per parameter (tau2 in ns, efficiency / alpha1 as fractions)

    def summary(self) -> dict:
        out = {
            "source": self.source,
            "mono_fraction": self.mono_fraction,
            "bit": self.bit,
            "photon_range": list(self.photon_range),
        }
        for name in ("tau2", "efficiency", "alpha1", "photon"):
            p = getattr(self, name)
            if p is not None:
                out[name] = {**asdict(p), "mean": p.mean, "sd": p.sd}
        return out


# ------------------------------------------------------------------ per-pixel quick fits
def quick_pixel_fits(
    fm: DecayForwardModel,
    decays: np.ndarray,
    irf_norm: np.ndarray,
    bg_per_bin: np.ndarray,
    is_bi: bool,
    init: dict,
) -> dict[str, np.ndarray]:
    """
    Per-pixel Poisson fits with the IRF and the background fixed, parameterised like the
    simulator: bi -> (tau2, E, A1, N), mono -> (tau, N). Bi-exponential fits are MAP with a
    weak Gaussian prior around the pooled fit (``BI_FIT_PRIOR_SD``). They only initialise
    the priors; the simulation-based search refines them.
    """
    n = len(decays)
    out = {
        k: np.full(n, np.nan) for k in ("tau1", "tau2", "alpha1", "efficiency", "tau")
    }
    tau_hi = fm.period * 1.5
    for i, (d, bg) in enumerate(zip(decays.astype(float), bg_per_bin)):
        n0 = max(d.sum() - bg * fm.n_active, 1.0)
        if is_bi:
            x_ref = np.array(
                [
                    np.log(init["tau2"]),
                    logit(np.clip(init["efficiency"], 0.05, 0.95)),
                    logit(np.clip(init["alpha1"], 0.05, 0.95)),
                ]
            )

            def loss(x, d=d, bg=bg, x_ref=x_ref):
                t2 = min(np.exp(x[0]), tau_hi)
                E, f = expit(x[1]), expit(x[2])
                dev = poisson_deviance(
                    np.exp(x[3]) * fm.shape(irf_norm, [t2 * (1 - E), t2], [f, 1 - f])
                    + bg * fm.active,
                    d,
                )
                # weak Gaussian pull toward the pooled fit: stops poorly determined pixels running off
                return dev + np.sum(((x[:3] - x_ref) / BI_FIT_PRIOR_SD) ** 2)

            x0 = [*x_ref, np.log(n0)]
        else:

            def loss(x, d=d, bg=bg):
                return poisson_deviance(
                    np.exp(x[1])
                    * fm.shape(irf_norm, [min(np.exp(x[0]), tau_hi)], [1.0])
                    + bg * fm.active,
                    d,
                )

            x0 = [np.log(init["tau"]), np.log(n0)]
        res = minimize(loss, x0, method="L-BFGS-B", options={"maxiter": 200})
        x = res.x
        if is_bi:
            t2, E, f = min(np.exp(x[0]), tau_hi), expit(x[1]), expit(x[2])
            out["tau2"][i], out["efficiency"][i], out["alpha1"][i] = t2, E, f
            out["tau1"][i] = t2 * (1 - E)
            out["tau"][i] = f * t2 * (1 - E) + (1 - f) * t2
        else:
            out["tau"][i] = out["tau2"][i] = min(np.exp(x[0]), tau_hi)
    # fits that ran into the lifetime cap are failures, not long lifetimes
    failed = out["tau2"] >= 0.95 * tau_hi
    for k in out:
        out[k][failed] = np.nan
    logging.info(
        f"Quick per-pixel MLE fits done on {n} pixels ({int(failed.sum())} hit the lifetime cap, dropped)"
    )
    return out


# -------------------------------------------------------------------- photon priors
def photon_prior_discrete(
    counts: np.ndarray,
    n_active_bins: int,
    dcr: float,
    inflation: float = 1.0,
    active_fraction: float = 1.0,
):
    """
    Beta over n_cycles in [lo, hi] such that ``0.01 * n_cycles * active_fraction``
    matches the observed signal photons. ``active_fraction`` is the expected share of
    signal photons that land in active time bins (photons in bins outside the
    acquisition window are lost, in the data and in the windowed simulation).
    """
    sig = np.clip(counts - dcr * n_active_bins, 0, None) / active_fraction
    lo_ph = max(np.percentile(sig, 0.5) * 0.9, SIM_MIN_CYCLES * SIM_MU_PER_CYCLE)
    hi_ph = max(
        np.percentile(sig, 99.5) * 1.1, lo_ph + 10 * SIM_MIN_CYCLES * SIM_MU_PER_CYCLE
    )
    cyc = (
        int(np.rint(lo_ph / SIM_MU_PER_CYCLE)),
        int(np.rint(hi_ph / SIM_MU_PER_CYCLE)),
    )
    prior = fit_beta(
        sig / SIM_MU_PER_CYCLE,
        offset=cyc[0],
        scale=cyc[1] - cyc[0],
        inflation=inflation,
    )
    return prior, cyc


def infer_bit_depth(peaks_a: np.ndarray, data_max: float) -> int:
    need = max(data_max, np.percentile(peaks_a, 99.9) / 0.9)
    return int(np.clip(np.ceil(np.log2(need + 1)), 8, 16))


def photon_prior_continuous(
    peaks: np.ndarray,
    data_max: float,
    bit: int | None,
    qe_on: bool = True,
    inflation: float = 1.0,
    scale_corr: float = 1.0,
):
    """
    Beta over A / (2**bit - 1). The observed peak is ``A * QE`` with QE ~ U(0.15, 0.35)
    independent of A, so E[A] = E[peak] / E[QE] and E[A^2] = E[peak^2] / E[QE^2].
    """
    peaks = np.clip(np.asarray(peaks, float), 0, None) * scale_corr
    if qe_on:
        lo, hi = SIM_QE_RANGE["continuous"]
        e_qe, e_qe2 = (lo + hi) / 2, (hi**3 - lo**3) / (3 * (hi - lo))
    else:
        e_qe, e_qe2 = 1.0, 1.0
    m_a = peaks.mean() / e_qe
    m2_a = (peaks**2).mean() / e_qe2
    var_a = max(m2_a - m_a**2, (0.05 * m_a) ** 2) * inflation
    bit = bit or infer_bit_depth(peaks / e_qe, data_max)
    max_adc = 2**bit - 1
    if np.percentile(peaks / e_qe, 99.5) > max_adc:
        logging.warning(
            f"Peak intensities exceed the {bit}-bit range: simulated decays will clip"
        )
    m, v = m_a / max_adc, var_a / max_adc**2
    v = min(v, m * (1 - m) * 0.95)
    return BetaPrior.from_mean_conc(m, m * (1 - m) / v - 1), bit


# --------------------------------------------------------------------- lifetime priors
def tau_bounds(tau: np.ndarray, period: float) -> tuple[float, float]:
    """(offset, scale) for the simulator's tau2_beta_range, with a 20 % margin."""
    tau = tau[np.isfinite(tau) & (tau > 0)]
    lo, hi = np.percentile(tau, [1, 99])
    offset = max(0.05, 0.7 * lo)
    upper = min(1.3 * hi, 1.5 * period)
    return offset, max(upper - offset, 0.1)


def lifetime_priors(
    values: dict[str, np.ndarray], is_bi: bool, period: float, inflation: float
):
    t2 = values["tau2"] if is_bi else values.get("tau", values.get("tau2"))
    t2 = t2[np.isfinite(t2) & (t2 > 0)]
    offset, scale = tau_bounds(t2, period)
    tau2 = fit_beta(t2, offset, scale, inflation)
    eff = a1 = None
    moments = {"tau2": (float(np.mean(t2)), float(np.std(t2)))}
    if is_bi:
        e = values["efficiency"]
        e = e[np.isfinite(e)]
        a = values.get("alpha1", np.array([]))
        a = a[np.isfinite(a)]
        eff = fit_beta(e, SIM_FRACTION_OFFSET, SIM_FRACTION_SCALE, inflation)
        a1 = (
            fit_beta(a, SIM_FRACTION_OFFSET, SIM_FRACTION_SCALE, inflation)
            if len(a)
            else BetaPrior(2, 2, SIM_FRACTION_OFFSET, SIM_FRACTION_SCALE)
        )
        for name, x in (("efficiency", e), ("alpha1", a)):
            if len(x):
                moments[name] = (float(np.mean(x)), float(np.std(x)))
    return tau2, eff, a1, moments
