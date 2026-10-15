"""
Fit-free mono-exponential lifetimes from the photon histogram: mean arrival time
and phasor lifetime.
"""

from typing import Any

import numpy as np

from pyfli.phasor.phasorS import PhasorAnalyzer, discrete_phasor_lifetime
from pyfli.simulator.calibration.expected_decay import gated_bin_probabilities


def gate_times(n_gates: int, laser_period: float) -> np.ndarray:
    """Gate start times ``k * laser_period / n_gates`` in ns (endpoint excluded)."""
    return np.arange(n_gates) * (laser_period / n_gates)


def _signal(decay: np.ndarray, background_per_bin: float) -> np.ndarray:
    """Background-subtracted counts, deliberately not clipped at zero: clipping would
    discard the negative noise excursions and bias the moments at low counts."""
    return np.asarray(decay, dtype=float) - background_per_bin


def _shared_irf(irf: np.ndarray) -> np.ndarray:
    irf = np.asarray(irf, dtype=float)
    if irf.ndim != 1:
        raise ValueError(
            "Pass a shared 1-D IRF of shape (T,), e.g. sim_irf[r, c] or "
            "sim_irf[mask].mean(axis=0)."
        )
    return irf / irf.sum()


def mean_arrival_time_naive(
    decay: np.ndarray,
    irf: np.ndarray,
    laser_period: float,
    background_per_bin: float = 0.0,
) -> np.ndarray:
    """
    ``<t>_decay - <t>_IRF`` per decay (ns) -- the textbook first-moment lifetime,
    exact only for an unlimited window without wrap-around or gating.
    """
    irf = _shared_irf(irf)
    t = gate_times(irf.size, laser_period)
    d = _signal(decay, background_per_bin)
    total = d.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_t = (d * t).sum(axis=-1) / total
    return np.where(total > 0, mean_t - np.sum(irf * t), np.nan)


def _mean_time_lookup(
    irf: np.ndarray, laser_period: float, tau_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    t = gate_times(irf.size, laser_period)
    model_mean = gated_bin_probabilities(tau_grid, irf, laser_period) @ t
    keep = np.concatenate([[True], np.diff(np.maximum.accumulate(model_mean)) > 0])
    return model_mean[keep], tau_grid[keep]


def mean_arrival_lifetime(
    decay: np.ndarray,
    irf: np.ndarray,
    laser_period: float,
    background_per_bin: float = 0.0,
    tau_grid: np.ndarray | None = None,
) -> np.ndarray:
    """
    Lifetime whose exact gated-model first moment equals the measured mean arrival
    time (method of moments with the wrap-around / gate / IRF model).

    Parameters
    ----------
    decay : np.ndarray
        Photon histograms, time axis last.
    irf : np.ndarray
        Shared IRF ``(T,)``.
    laser_period : float
        Laser repetition period in ns.
    background_per_bin : float
        Expected background counts per gate, subtracted before the moment.
    tau_grid : np.ndarray | None
        Lifetimes (ns) of the lookup table; defaults to 2000 log-spaced values in
        ``[0.01, 5 * laser_period]``.

    Returns
    -------
    np.ndarray
        Lifetime per decay (ns); NaN for empty decays or means outside the table.
    """
    irf = _shared_irf(irf)
    if tau_grid is None:
        tau_grid = np.geomspace(0.01, 5 * laser_period, 2000)
    model_mean, tau_tab = _mean_time_lookup(irf, laser_period, np.asarray(tau_grid))
    t = gate_times(irf.size, laser_period)
    d = _signal(decay, background_per_bin)
    total = d.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_t = (d * t).sum(axis=-1) / total
    tau = np.interp(mean_t, model_mean, tau_tab, left=np.nan, right=np.nan)
    return np.where(total > 0, tau, np.nan)


def phasor_lifetime(
    decay: np.ndarray,
    irf: np.ndarray,
    laser_period: float,
    background_per_bin: float = 0.0,
    bin_corrected: bool = False,
    device: Any | None = "cpu",
) -> np.ndarray:
    """
    First-harmonic phasor lifetime at the laser repetition frequency, calibrated
    with the IRF, via :class:`~pyfli.phasor.phasorS.PhasorAnalyzer`.

    Parameters
    ----------
    decay : np.ndarray
        Photon histograms, time axis last.
    irf : np.ndarray
        Shared IRF ``(T,)``.
    laser_period : float
        Laser repetition period in ns (the phasor frequency is ``1 / laser_period``).
    background_per_bin : float
        Expected background counts per gate, subtracted before the phasor.
    bin_corrected : bool
        False (default): the standard phase lifetime ``S / (omega * G)`` from
        ``PhasorAnalyzer.compute_lifetime``. True: invert the calibrated phasor
        ``z = (1 - q) / (1 - q * exp(i*omega*dt))`` of the exact wrapped, gated
        exponential for ``q = exp(-dt / tau)``, which removes the gate-width bias.
    device : Any | None
        Torch device for ``PhasorAnalyzer`` (calibration runs in torch).

    Returns
    -------
    np.ndarray
        Lifetime per decay (ns), shape ``decay.shape[:-1]``.
    """
    irf = _shared_irf(irf)
    d = _signal(decay, background_per_bin)
    spatial = d.shape[:-1]
    n = irf.size
    d_flat = d.reshape(1, -1, n)
    t = gate_times(n, laser_period)
    phasor = PhasorAnalyzer(
        frequency_hz=1e9 / laser_period, time_axis_ns=t, n_harmonics=1, device=device
    )
    g, s = phasor.create_phasor_cpu(d_flat)
    irf_cube = np.broadcast_to(irf, d_flat.shape)
    gc, sc = phasor.calibrate_pixelwise(g, s, np.ascontiguousarray(irf_cube))
    gc, sc = gc[0].astype(float), sc[0].astype(float)
    if bin_corrected:
        tau = discrete_phasor_lifetime(gc, sc, n, laser_period)
    else:
        tau = phasor.compute_lifetime(gc, sc)
    empty = d_flat.sum(axis=-1) <= 0
    tau = np.where(empty, np.nan, tau)
    return tau.reshape(spatial)


def fit_free_lifetime_maps(
    decay: np.ndarray,
    irf: np.ndarray,
    laser_period: float,
    mask: np.ndarray | None = None,
    background_per_bin: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    All fit-free lifetime maps for an image cube, ready for ``CVPlot`` /
    ``FactorAnalysis``: ``{"tau_mean_arrival", "tau_mean_arrival_naive",
    "tau_phasor", "tau_phasor_bin_corrected"}``, each ``(H, W)`` with NaN outside
    `mask`.

    `irf` may be a shared ``(T,)`` IRF or the ``(H, W, T)`` cube, in which case the
    mean IRF over `mask` is used.
    """
    decay = np.asarray(decay, dtype=float)
    mask = decay.sum(axis=-1) > 0 if mask is None else np.asarray(mask, dtype=bool)
    irf = np.asarray(irf, dtype=float)
    if irf.ndim == 3:
        irf = irf[mask].mean(axis=0)
    sel = decay[mask]
    estimates = {
        "tau_mean_arrival": mean_arrival_lifetime(
            sel, irf, laser_period, background_per_bin
        ),
        "tau_mean_arrival_naive": mean_arrival_time_naive(
            sel, irf, laser_period, background_per_bin
        ),
        "tau_phasor": phasor_lifetime(sel, irf, laser_period, background_per_bin),
        "tau_phasor_bin_corrected": phasor_lifetime(
            sel, irf, laser_period, background_per_bin, bin_corrected=True
        ),
    }
    maps = {}
    for key, values in estimates.items():
        full = np.full(mask.shape, np.nan)
        full[mask] = values
        maps[key] = full
    return maps
