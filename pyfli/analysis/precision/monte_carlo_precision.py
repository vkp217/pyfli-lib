"""Monte Carlo lifetime precision: repeated simulation of one pixel across photon counts."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyfli.analyticalWorkflow import AnalyticalHelpers
from pyfli.simulator import PhotonCountSimulator
from pyfli.simulator.calibration.config import SIM_MIN_CYCLES, SIM_MU_PER_CYCLE
from pyfli.solver import BaseFLIFitter, FLICPUProcessor, MLEFLIFitter

from .crb_precision import crb_lifetime_cv
from .fisher_crlb import estimator_efficiency, f_value
from .fit_free_lifetime import mean_arrival_lifetime, phasor_lifetime

MU_PER_CYCLE = SIM_MU_PER_CYCLE
"""Expected photons per laser cycle used by ``PhotonCountSimulator``."""

FIT_METHODS = {
    "NLSF": (BaseFLIFitter, "least_squares"),
    "MLE": (MLEFLIFitter, "poisson"),
}
FIT_FREE_METHODS = ("mean_arrival", "phasor", "phasor_bin_corrected")


def simulate_fixed_lifetime_decays(
    irf: np.ndarray,
    tau: float,
    n_photons: float,
    n_repeats: int,
    laser_feq: float = 80.0,
    dcr: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate `n_repeats` independent photon histograms of one pixel with lifetime
    `tau` and on average `n_photons` detected signal photons, using
    ``PhotonCountSimulator``.

    Parameters
    ----------
    irf : np.ndarray
        IRF trace ``(T,)`` (or a cube, from which the simulator picks one pixel).
    tau : float
        Fixed lifetime in ns.
    n_photons : float
        Expected signal photons per decay; must be >= 10 (the simulator needs at
        least 1000 laser cycles).
    n_repeats : int
        Number of independent copies.
    laser_feq : float
        Laser repetition rate in MHz (period ``1000 / laser_feq`` ns).
    dcr : float
        Dark counts per gate added by the simulator (0 disables them).
    seed : int | None
        Seeds the simulator engine and NumPy's global RNG (used for dark counts).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(decays, irf_used)``: decays ``(n_repeats, T)`` and the normalized IRF the
        simulator convolved with ``(T,)``.
    """
    n_cycles = round(n_photons / MU_PER_CYCLE)
    if n_cycles < SIM_MIN_CYCLES:
        raise ValueError(
            f"n_photons={n_photons} is below the simulator minimum of "
            f"{SIM_MIN_CYCLES * MU_PER_CYCLE:g} photons ({SIM_MIN_CYCLES} laser cycles)."
        )
    if seed is not None:
        np.random.seed(seed)
    simulator = PhotonCountSimulator(
        irf,
        sensor_type="discrete",
        tau2=(1, 1),
        tau2_dist="beta",
        tau2_beta_range=(0.0, float(tau)),
        mono_fraction=1.0,
        n_cycles=(n_cycles, n_cycles),
        photo_count=(1, 1),
        laser_feq=laser_feq,
        jitter=False,
        qe_on=False,
        dcr_on=dcr > 0,
        dcr=dcr,
        clip_on=False,
        seed=seed,
    )
    runs = [simulator() for _ in range(n_repeats)]
    decays = np.stack([run["raw_data"]["decay"] for run in runs])
    return decays, np.asarray(runs[0]["raw_data"]["irf"], dtype=float)


def fit_lifetimes(
    decays: np.ndarray,
    irf: np.ndarray,
    freq: Any,
    method: str,
    max_iter: int = 1000,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Fit every decay of `decays` ``(M, T)`` with a mono-exponential model using
    ``FLICPUProcessor`` and the fitter/estimator of `method` (``"NLSF"`` or
    ``"MLE"``). Returns the fitted lifetimes ``(M,)``; failed fits are NaN.
    """
    fitter_class, estimator = FIT_METHODS[method]
    m, t = decays.shape
    cube = np.asarray(decays, dtype=float)[None]
    irf_cube = np.ascontiguousarray(np.broadcast_to(irf, (1, m, t)))
    result = FLICPUProcessor(freq, fitter_class).process_image(
        cube,
        irf_cube,
        mask=np.ones((1, m), dtype=bool),
        data_name=f"monte_carlo_{method}",
        model_type="mono-exponential",
        estimator=estimator,
        n_jobs=n_jobs,
        max_iter=max_iter,
    )
    if result is None:
        return np.full(m, np.nan)
    tau = result["results"]["maps"]["tau_map"][0].astype(float)
    health = result["results"]["maps"]["pixel_health_map"][0]
    return np.where((health > 0) & (tau > 0), tau, np.nan)


def estimate_lifetimes(
    decays: np.ndarray,
    irf: np.ndarray,
    laser_period: float,
    method: str,
    freq: Any = None,
    background_per_bin: float = 0.0,
    max_iter: int = 1000,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Lifetime of every decay with one method: ``"NLSF"``, ``"MLE"`` (fits),
    ``"mean_arrival"``, ``"phasor"`` or ``"phasor_bin_corrected"`` (fit-free,
    background-subtracted).
    """
    if method in FIT_METHODS:
        return fit_lifetimes(decays, irf, freq, method, max_iter, n_jobs)
    if method == "mean_arrival":
        return mean_arrival_lifetime(decays, irf, laser_period, background_per_bin)
    if method == "phasor":
        return phasor_lifetime(decays, irf, laser_period, background_per_bin)
    if method == "phasor_bin_corrected":
        return phasor_lifetime(
            decays, irf, laser_period, background_per_bin, bin_corrected=True
        )
    raise ValueError(
        f"Unknown method {method!r}; use one of {list(FIT_METHODS) + list(FIT_FREE_METHODS)}."
    )


def run_monte_carlo(
    irf: np.ndarray,
    tau: float,
    photon_counts: Any,
    n_repeats: int = 200,
    methods: Any = ("NLSF", "MLE", "mean_arrival", "phasor"),
    laser_feq: float = 80.0,
    dcr: float = 0.0,
    max_iter: int = 1000,
    n_jobs: int = -1,
    seed: int | None = None,
    crb_fit_background: bool | None = None,
) -> pd.DataFrame:
    """
    Monte Carlo precision of each method against photon count, for one fixed
    lifetime.

    For every photon count ``N`` in `photon_counts`, `n_repeats` decays are
    simulated, every method estimates the lifetime of every decay, and the spread
    of the estimates is summarized. The Cramér-Rao bound for the same ``tau``,
    ``N``, IRF, window and background is added for reference.

    Parameters
    ----------
    irf : np.ndarray
        IRF trace ``(T,)``.
    tau : float
        True lifetime (ns) shared by all copies.
    photon_counts : iterable of float
        Expected signal photons per decay (each >= 10).
    n_repeats : int
        Copies per photon count.
    methods : iterable of str
        Any of ``"NLSF"``, ``"MLE"``, ``"mean_arrival"``, ``"phasor"``,
        ``"phasor_bin_corrected"``.
    laser_feq : float
        Laser repetition rate in MHz.
    dcr : float
        Dark counts per gate (0 disables them).
    max_iter : int
        Optimizer iteration budget for the fits.
    n_jobs : int
        Parallel fitting jobs.
    seed : int | None
        Base seed; photon count ``i`` uses ``seed + i``.
    crb_fit_background : bool | None
        Whether the bound treats the background as unknown (as fits with a free
        offset do). Defaults to ``dcr > 0``.

    Returns
    -------
    pd.DataFrame
        One row per (method, N): ``method``, ``n_photons``, ``n_detected`` (mean
        total counts), ``n_valid``, ``tau_true``, ``tau_mean``, ``tau_std``,
        ``bias``, ``rel_bias``, ``cv`` (std / mean), ``F`` (``cv * sqrt(N)``),
        ``cv_crb``, ``F_crb`` and ``efficiency`` (``(cv_crb / cv) ** 2``, see
        :func:`.fisher_crlb.estimator_efficiency`; 1 means the method reaches the
        bound).
    """
    laser_period = 1000.0 / laser_feq
    fit_background = dcr > 0 if crb_fit_background is None else crb_fit_background
    rows = []
    for i, n_photons in enumerate(photon_counts):
        decays, irf_used = simulate_fixed_lifetime_decays(
            irf,
            tau,
            n_photons,
            n_repeats,
            laser_feq=laser_feq,
            dcr=dcr,
            seed=None if seed is None else seed + i,
        )
        n_gates = decays.shape[1]
        freq = AnalyticalHelpers(
            laser_period=laser_period,
            gate_delay=laser_period / n_gates,
            num_gate=n_gates,
        ).freq_computation()
        cv_crb = float(
            crb_lifetime_cv(tau, n_photons, irf_used, laser_period, dcr, fit_background)
        )
        for method in methods:
            est = estimate_lifetimes(
                decays,
                irf_used,
                laser_period,
                method,
                freq=freq,
                background_per_bin=dcr,
                max_iter=max_iter,
                n_jobs=n_jobs,
            )
            est = est[np.isfinite(est) & (est > 0)]
            mean = float(np.mean(est)) if est.size else np.nan
            std = float(np.std(est, ddof=1)) if est.size > 1 else np.nan
            cv = std / mean if mean else np.nan
            rows.append(
                {
                    "method": method,
                    "n_photons": float(n_photons),
                    "n_detected": float(decays.sum(axis=1).mean()),
                    "n_valid": int(est.size),
                    "tau_true": float(tau),
                    "tau_mean": mean,
                    "tau_std": std,
                    "bias": mean - tau,
                    "rel_bias": (mean - tau) / tau,
                    "cv": cv,
                    "F": float(f_value(cv, n_photons)),
                    "cv_crb": cv_crb,
                    "F_crb": float(f_value(cv_crb, n_photons)),
                    "efficiency": float(estimator_efficiency(cv, cv_crb)),
                }
            )
    return pd.DataFrame(rows)


def plot_monte_carlo(
    df: pd.DataFrame, figsize: tuple[float, float] = (13, 4.5)
) -> tuple[Any, Any]:
    """
    Plot a :func:`run_monte_carlo` table: CV vs. photon count (log-log) with the
    Cramér-Rao bound, and the F-value vs. photon count with the bound and the ideal
    ``F = 1``.
    """
    fig, (ax_cv, ax_f) = plt.subplots(1, 2, figsize=figsize)
    for method, sub in df.groupby("method", sort=False):
        sub = sub.sort_values("n_photons")
        ax_cv.plot(sub["n_photons"], sub["cv"], marker="o", label=method)
        ax_f.plot(sub["n_photons"], sub["F"], marker="o", label=method)
    crb = df.drop_duplicates("n_photons").sort_values("n_photons")
    ax_cv.plot(crb["n_photons"], crb["cv_crb"], "k--", label="Cramér-Rao bound")
    ax_f.plot(crb["n_photons"], crb["F_crb"], "k--", label="Cramér-Rao bound")
    ax_f.axhline(1.0, color="grey", linestyle=":", label="ideal (F = 1)")
    ax_cv.set_xscale("log")
    ax_cv.set_yscale("log")
    ax_f.set_xscale("log")
    ax_cv.set_xlabel("signal photons per decay, N")
    ax_f.set_xlabel("signal photons per decay, N")
    ax_cv.set_ylabel(r"CV  $\sigma_\tau / \bar\tau$")
    ax_f.set_ylabel(r"F-value  CV$\cdot\sqrt{N}$")
    tau_true = df["tau_true"].iloc[0]
    ax_cv.set_title(f"Monte Carlo precision (tau = {tau_true:g} ns)", fontweight="bold")
    ax_f.set_title("Distance from the ideal limit", fontweight="bold")
    ax_f.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    return fig, (ax_cv, ax_f)
