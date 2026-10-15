"""IRF template and shift range for the simulator, from a measured IRF or the data."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from pyfli import logging
from pyfli.irf_deconvolution.fli_solver import fourier_shift
from pyfli.solver.shared_metrics import poisson_deviance

from .expected_decay import DecayForwardModel
from .summary_stats import rising_edge


@dataclass
class IRFEstimate:
    source: str  # "experimental" | "simulated"
    template: np.ndarray  # (T,) peak-normalised IRF before the OffsetGen shift
    a_center: float  # bins; OffsetGen a ~ U(a_center - a_half, a_center + a_half)
    a_half: float
    b_max: float  # OffsetGen b ~ U(0, b_max), as a fraction of the IRF peak
    params: dict = field(
        default_factory=dict
    )  # mu / sigma_ns / tail_ns for a simulated IRF
    pooled_fit: dict = field(
        default_factory=dict
    )  # lifetimes and background from the pooled fit
    library: np.ndarray | None = (
        None  # (K, T) peak-normalised IRFs; one is drawn per simulated pixel
    )
    library_b_max: np.ndarray | None = (
        None  # (K,) baseline range per library IRF (background is per channel)
    )

    def b_ranges(self) -> list[tuple[float, float]] | None:
        if self.library is None or self.library_b_max is None:
            return None
        return [(0.0, float(b)) for b in self.library_b_max]

    @property
    def a_range(self) -> tuple[float, float]:
        return (self.a_center - self.a_half, self.a_center + self.a_half)

    @property
    def b_range(self) -> tuple[float, float]:
        return (0.0, self.b_max)

    def irf_cube(self) -> np.ndarray:
        """
        (1, 1, T) cube in the layout ``SimGenerator`` / ``OffsetGen`` expect (pixel=(0, 0)),
        or (1, K, T) for an IRF library (see ``sim_bridge.MultiIRFSimGenerator``).
        """
        if self.library is not None:
            return self.library[None, :, :].astype(np.float64)
        return self.template[None, None, :].astype(np.float64)

    def effective_irf(self) -> np.ndarray:
        """Peak-normalised IRF at the centre of the shift range (what a typical pixel sees)."""
        return (
            _fourier_shift(self.template, self.a_center)
            if self.a_center
            else self.template
        )


def _fourier_shift(y: np.ndarray, shift: float) -> np.ndarray:
    """Sub-bin circular delay of ``y`` by ``shift`` bins (pyfli's ``fourier_shift``), negative ringing clipped."""
    return np.clip(fourier_shift(np.asarray(y, float), shift), 0, None)


def _peak_position(irf: np.ndarray) -> np.ndarray:
    """Sub-bin peak position (centroid of the 5 bins around the maximum) per trace."""
    T = irf.shape[-1]
    pk = irf.argmax(-1)
    idx = (pk[:, None] + np.arange(-2, 3)[None, :]) % T
    w = np.take_along_axis(irf, idx, -1)
    w = np.clip(w - w.min(-1, keepdims=True), 0, None)
    return pk + (w * np.arange(-2, 3)).sum(-1) / np.maximum(w.sum(-1), 1e-12)


def experimental_template(
    irf: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, float]:
    """Peak-normalised IRF template and the half-width (bins) of the per-pixel IRF shift range."""
    if irf.ndim == 1:
        return irf / irf.max(), 0.5
    traces = (
        irf[mask]
        if mask is not None and mask.shape == irf.shape[:2]
        else irf.reshape(-1, irf.shape[-1])
    )
    peaks = traces.max(-1)
    traces = traces[peaks >= 0.2 * np.percentile(peaks, 99)]
    pos = _peak_position(traces)
    ref = np.median(pos)
    aligned = np.stack(
        [np.roll(tr, int(np.rint(ref - p))) / tr.max() for tr, p in zip(traces, pos)]
    )
    template = aligned.mean(0)
    lo, hi = np.percentile(pos, [2.5, 97.5])
    return template / template.max(), max(0.5, (hi - lo) / 2)


def _fit_pooled(
    fm: DecayForwardModel,
    pooled: np.ndarray,
    irf_from_theta,
    theta0: list,
    theta_bounds: list,
    edge: float,
) -> tuple[np.ndarray, dict]:
    """
    Fit the pooled decay with: IRF(theta) + baseline b, two lifetimes, photon fraction f,
    and a background fraction beta. Several starts; returns the best parameter vector.
    For a circular (TCSPC) model the baseline b only adds a flat floor, indistinguishable
    from beta in a pooled decay, so b is held at 0 there and set later by
    :func:`split_background`.
    """
    N = pooled.sum()
    T = len(pooled)

    def unpack(x):
        n = len(theta0)
        theta = x[:n]
        b, tau_s, dtau, f, beta = x[n:]
        return theta, b, tau_s, tau_s + dtau, f, beta

    def loss(x):
        theta, b, tau_s, tau_l, f, beta = unpack(x)
        irf = fm.with_baseline(irf_from_theta(theta), b)
        mu = N * (
            (1 - beta) * fm.shape(irf, [tau_s, tau_l], [f, 1 - f]) + beta * fm.flat()
        )
        return float(poisson_deviance(mu, pooled)) / T

    b_hi, b0 = (0.0, 0.0) if fm.circular else (0.2, 1e-3)
    bounds = [
        *theta_bounds,
        (0.0, b_hi),
        (0.03, fm.period / 2),
        (0.05, fm.period),
        (0.0, 1.0),
        (0.0, 0.95),
    ]
    best = None
    for tau_s0, dtau0, f0 in (
        (0.4, 2.0, 0.4),
        (1.0, 2.5, 0.5),
        (0.2, 3.0, 0.3),
        (1.5, 1.5, 0.6),
    ):
        x0 = [*theta0, b0, tau_s0, dtau0, f0, 0.02]
        res = minimize(
            loss, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500}
        )
        if best is None or res.fun < best.fun:
            best = res
    theta, b, tau_s, tau_l, f, beta = unpack(best.x)
    info = {
        "tau_short_ns": float(tau_s),
        "tau_long_ns": float(tau_l),
        "frac_short": float(f),
        "background_fraction": float(beta),
        "irf_baseline_b": float(b),
        "deviance_per_bin": float(best.fun),
        "edge_bin": float(edge),
    }
    return np.asarray(theta), info


def estimate_irf(
    fm: DecayForwardModel,
    decay: np.ndarray,
    mask: np.ndarray,
    irf: np.ndarray | None = None,
    mode: str = "auto",
) -> IRFEstimate:
    """
    ``mode``: "experimental" uses the given IRF, "simulated" fits an EMG IRF, "auto" does
    both when an IRF is given and keeps the one whose pooled fit has the lower deviance.
    A measured IRF can be the wrong one for the data (other day, wavelength, detector
    setting); "auto" catches that.
    """
    pooled = decay[mask].sum(0).astype(float)
    n_pix = int(mask.sum())
    if irf is None and mode == "experimental":
        raise ValueError("irf_mode='experimental' needs an IRF")
    candidates = []
    if irf is not None and mode in ("experimental", "auto"):
        candidates.append(_estimate(fm, pooled, mask, irf))
    if irf is None or mode in ("simulated", "auto"):
        candidates.append(_estimate(fm, pooled, mask, None))
    est = min(candidates, key=lambda e: e.pooled_fit["deviance_per_bin"])
    if len(candidates) > 1:
        est.pooled_fit["irf_choice"] = {
            c.source: c.pooled_fit["deviance_per_bin"] for c in candidates
        }
        logging.info(
            f"IRF auto mode: pooled deviance per bin {est.pooled_fit['irf_choice']} -> {est.source}"
        )
    # background fraction of the pooled decay -> dark counts per bin per pixel (initial guess)
    est.pooled_fit["dcr_per_bin_per_pixel"] = float(
        est.pooled_fit["background_fraction"] * pooled.sum() / fm.n_active / n_pix
    )
    logging.info(
        f"IRF ({est.source}) initial estimate: {est.params}; pooled fit: {est.pooled_fit}"
    )
    return est


def split_background(
    est: IRFEstimate,
    fm: DecayForwardModel,
    bg_bins: np.ndarray,
    slope: float,
    intercept: float,
) -> float:
    """
    Split the pre-edge background of a circular (TCSPC) model into dark counts and the
    ``OffsetGen`` IRF baseline ``b``, using the per-pixel regression
    ``pre-edge level = intercept + slope * counts``.

    A baseline ``b`` on the peak-normalised IRF adds ``b / (sum(IRF) + b T)`` to every bin
    per signal photon, the same flat floor as dark counts in a pooled decay, so the pooled
    fit cannot separate them. Per pixel they differ: dark counts give a constant level (the
    intercept), the baseline a level proportional to the signal (the slope left over after
    the previous-pulse tail of the pooled-fit lifetimes). Sets ``est.b_max`` (``b ~ U(0,
    b_max)``, so ``b_max = 2 b``) and ``pooled_fit["irf_baseline_b"]``; returns the dark-count
    rate per bin per pixel (the pooled background when the intercept is not positive). The
    counts include the dark counts themselves, so ``intercept = dcr (1 - slope n_active)``.
    """
    pf = est.pooled_fit
    shape = fm.shape(
        fm.with_baseline(est.effective_irf(), 0.0),
        [pf["tau_short_ns"], pf["tau_long_ns"]],
        [pf["frac_short"], 1.0 - pf["frac_short"]],
    )
    tail = float(shape[np.asarray(bg_bins, dtype=int)].mean())
    excess = max(float(slope) - tail, 0.0)
    irf_sum = float(est.effective_irf().sum())
    b = excess * irf_sum / max(1.0 - excess * fm.n_bins, 1e-6)
    pf["tail_slope"] = tail
    pf["irf_baseline_b"] = float(b)
    est.b_max = max(2.0 * b, 1e-4)
    if intercept <= 0:
        return pf["dcr_per_bin_per_pixel"]
    return float(intercept) / max(1.0 - float(slope) * fm.n_active, 1e-3)


def _estimate(
    fm: DecayForwardModel, pooled: np.ndarray, mask: np.ndarray, irf: np.ndarray | None
) -> IRFEstimate:
    edge = float(np.median(rising_edge(pooled[None, :])))
    peak = int(np.argmax(pooled))

    if irf is not None:
        template, a_half = experimental_template(irf, mask)
        tpk = float(_peak_position(template[None, :])[0])

        def irf_from_theta(th):
            return _fourier_shift(template, th[0])

        theta, info = _fit_pooled(
            fm,
            pooled,
            irf_from_theta,
            [peak - tpk - 1.0],
            [(-fm.n_bins / 4, fm.n_bins / 4)],
            edge,
        )
        est = IRFEstimate(
            source="experimental",
            template=template,
            a_center=float(theta[0]),
            a_half=float(a_half),
            b_max=max(2 * info["irf_baseline_b"], 1e-4),
            params={"global_shift_bins": float(theta[0])},
            pooled_fit=info,
        )
    else:

        def irf_from_theta(th):
            return fm.simulated_irf(th[0], th[1], th[2])

        mu0 = edge + 1.0
        best = None
        for dmu in (-2.0, 0.0, 2.0):
            theta, info = _fit_pooled(
                fm,
                pooled,
                irf_from_theta,
                [mu0 + dmu, 0.1, 0.05],
                [(edge - 15, peak + 3), (0.3 * fm.dt, 1.0), (0.005, 0.6)],
                edge,
            )
            if best is None or info["deviance_per_bin"] < best[1]["deviance_per_bin"]:
                best = (theta, info)
        theta, info = best
        est = IRFEstimate(
            source="simulated",
            template=irf_from_theta(theta),
            a_center=0.0,
            a_half=0.5,
            b_max=max(2 * info["irf_baseline_b"], 1e-4),
            params={
                "mu_bin": float(theta[0]),
                "sigma_ns": float(theta[1]),
                "tail_ns": float(theta[2]),
            },
            pooled_fit=info,
        )
    return est


def irf_theta_names(source: str) -> list[str]:
    return (
        ["mu_bin", "log_sigma_ns", "log_tail_ns"]
        if source == "simulated"
        else ["a_center"]
    )


def irf_theta_from_estimate(est: IRFEstimate) -> np.ndarray:
    if est.source == "simulated":
        p = est.params
        return np.array([p["mu_bin"], np.log(p["sigma_ns"]), np.log(p["tail_ns"])])
    return np.array([est.a_center])


def apply_irf_theta(
    est: IRFEstimate, fm: DecayForwardModel, theta: np.ndarray
) -> IRFEstimate:
    """Return a copy of ``est`` with the optimiser's IRF parameters applied."""
    if est.source == "simulated":
        mu, sigma, tail = (
            float(theta[0]),
            float(np.exp(theta[1])),
            float(np.exp(theta[2])),
        )
        return IRFEstimate(
            "simulated",
            fm.simulated_irf(mu, sigma, tail),
            est.a_center,
            est.a_half,
            est.b_max,
            {"mu_bin": mu, "sigma_ns": sigma, "tail_ns": tail},
            est.pooled_fit,
        )
    return IRFEstimate(
        "experimental",
        est.template,
        float(theta[0]),
        est.a_half,
        est.b_max,
        {"global_shift_bins": float(theta[0])},
        est.pooled_fit,
    )
