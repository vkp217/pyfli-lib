"""Cramér-Rao lower bound on mono-exponential lifetime precision for gated / TCSPC data."""

from typing import Any

import numpy as np
import pandas as pd

from pyfli.simulator.calibration.expected_decay import gated_bin_probabilities

from .fisher_crlb import crlb_cv, f_value, fisher_information, select_parameters

_CHUNK = 4096

PARAMS = ("N", "tau", "b")
"""Parameter order of :func:`gated_decay_model`: signal photons, lifetime, background."""
TAU_INDEX = PARAMS.index("tau")


def gated_decay_model(
    params: np.ndarray, irf: np.ndarray, laser_period: float
) -> np.ndarray:
    """
    Expected counts per gate ``mu_k = N * p_k(tau) + b`` for parameter vectors
    ``params`` ``(P, 3)`` ordered as :data:`PARAMS` (``N``, ``tau``, ``b``); `irf` is
    ``(T,)`` or ``(P, T)``. This is the model handed to the generic
    :func:`.fisher_crlb.fisher_information`.
    """
    params = np.atleast_2d(np.asarray(params, dtype=float))
    n, tau, b = params[:, 0], params[:, 1], params[:, 2]
    p = gated_bin_probabilities(tau, irf, laser_period)
    return n[:, None] * p + b[:, None]


def _crb_cv_chunk(
    tau: np.ndarray,
    n: np.ndarray,
    b: np.ndarray,
    irf: np.ndarray,
    laser_period: float,
    fit_background: bool,
) -> np.ndarray:
    info = fisher_information(
        lambda theta: gated_decay_model(theta, irf, laser_period),
        np.column_stack([n, tau, b]),
    )
    estimated = [PARAMS.index("N"), TAU_INDEX]
    if fit_background:
        estimated.append(PARAMS.index("b"))
    return crlb_cv(select_parameters(info, estimated), tau, estimated.index(TAU_INDEX))


def crb_lifetime_cv(
    tau: Any,
    n_photons: Any,
    irf: np.ndarray,
    laser_period: float,
    background_per_bin: Any = 0.0,
    fit_background: bool = False,
) -> np.ndarray:
    """
    Ideal (Cramér-Rao) coefficient of variation of the lifetime, ``sigma_tau / tau``.

    The unknowns are the photon count ``N`` and ``tau`` (plus the background level
    when `fit_background` is True, which is what NLSF / MLE fits with a free offset
    ``v_shift`` do). The background level itself always enters the Poisson noise.

    Parameters
    ----------
    tau, n_photons : float | np.ndarray
        True lifetime (ns) and expected number of detected *signal* photons;
        broadcast against each other.
    irf : np.ndarray
        IRF of shape ``(T,)``, or one IRF per value with shape ``S + (T,)``.
    laser_period : float
        Laser repetition period in ns.
    background_per_bin : float | np.ndarray
        Expected background (dark) counts per gate, e.g. the simulator's ``dcr``.
    fit_background : bool
        Treat the background level as an extra unknown parameter.

    Returns
    -------
    np.ndarray
        Ideal CV with the broadcast shape of the inputs; NaN where ``tau <= 0``,
        ``N <= 0`` or an input is not finite. ``CV * sqrt(N)`` is the F-value.
    """
    tau, n, b = np.broadcast_arrays(
        np.asarray(tau, dtype=float),
        np.asarray(n_photons, dtype=float),
        np.asarray(background_per_bin, dtype=float),
    )
    irf = np.asarray(irf, dtype=float)
    shape = tau.shape
    tau_f, n_f, b_f = tau.ravel(), n.ravel(), b.ravel()
    irf_f = irf if irf.ndim == 1 else irf.reshape(-1, irf.shape[-1])
    out = np.full(tau_f.shape, np.nan)
    valid = (
        np.isfinite(tau_f)
        & np.isfinite(n_f)
        & np.isfinite(b_f)
        & (tau_f > 0)
        & (n_f > 0)
    )
    idx = np.flatnonzero(valid)
    for start in range(0, idx.size, _CHUNK):
        sel = idx[start : start + _CHUNK]
        irf_sel = irf_f if irf_f.ndim == 1 else irf_f[sel]
        out[sel] = _crb_cv_chunk(
            tau_f[sel], n_f[sel], b_f[sel], irf_sel, laser_period, fit_background
        )
    return out.reshape(shape)


def crb_f_value(
    tau: Any,
    irf: np.ndarray,
    laser_period: float,
    n_photons: Any = 1e4,
    background_per_bin: Any = 0.0,
    fit_background: bool = False,
) -> np.ndarray:
    """
    Ideal F-value ``CV_ideal * sqrt(N)``. Without background it does not depend on
    `n_photons`; with background it rises as the signal-to-background ratio drops.
    """
    cv = crb_lifetime_cv(
        tau, n_photons, irf, laser_period, background_per_bin, fit_background
    )
    return f_value(cv, n_photons)


def crb_cv_map(
    tau_map: np.ndarray,
    n_photons_map: np.ndarray,
    irf: np.ndarray,
    laser_period: float,
    mask: np.ndarray | None = None,
    background_per_bin: float = 0.0,
    fit_background: bool = False,
) -> np.ndarray:
    """
    Per-pixel ideal CV map.

    Parameters
    ----------
    tau_map : np.ndarray
        True lifetime per pixel, shape ``(H, W)`` (e.g. the simulator's ground-truth
        ``maps["tau_map"]``).
    n_photons_map : np.ndarray
        Expected signal photons per pixel, shape ``(H, W)`` (e.g. the simulator's
        ``maps["photon_count_map"]``, or the decay sum minus ``T * background``).
    irf : np.ndarray
        Shared IRF ``(T,)`` or per-pixel IRF cube ``(H, W, T)``.
    laser_period : float
        Laser repetition period in ns.
    mask : np.ndarray | None
        Pixels to compute; others are NaN. Defaults to ``tau_map > 0``.
    background_per_bin, fit_background
        See :func:`crb_lifetime_cv`.

    Returns
    -------
    np.ndarray
        ``(H, W)`` ideal CV map; multiply by ``sqrt(n_photons_map)`` for the F-value.
    """
    tau_map = np.asarray(tau_map, dtype=float)
    mask = (tau_map > 0) if mask is None else np.asarray(mask, dtype=bool)
    irf = np.asarray(irf, dtype=float)
    out = np.full(tau_map.shape, np.nan)
    irf_sel = irf if irf.ndim == 1 else irf[mask]
    out[mask] = crb_lifetime_cv(
        tau_map[mask],
        np.asarray(n_photons_map, dtype=float)[mask],
        irf_sel,
        laser_period,
        background_per_bin,
        fit_background,
    )
    return out


def crb_for_cv_bins(
    cv_df: pd.DataFrame,
    tau_map: np.ndarray,
    photon_map: np.ndarray,
    irf: np.ndarray,
    laser_period: float,
    mask: np.ndarray | None = None,
    cluster_mask: np.ndarray | None = None,
    cluster_names: dict[int, str] | None = None,
    n_photons_map: np.ndarray | None = None,
    background_per_bin: float = 0.0,
    fit_background: bool = False,
) -> pd.DataFrame:
    """
    Ideal CV for every photon-count bin of a :meth:`pyfli.data_vnp.CVPlot.compute`
    table, so it can be plotted against (or merged with) the measured curves.

    Bin membership is rebuilt from the table's ``bin_low`` / ``bin_high`` edges on
    `photon_map` (the same photon map given to ``CVPlot``), per ``cluster`` when the
    table has clusters. Within a bin, the per-pixel CRB variances are averaged, so
    ``cv_crb = sqrt(mean((cv_i * tau_i) ** 2)) / mean(tau_i)`` -- the CV an ideal
    estimator would show across those pixels, excluding the true lifetime spread.

    Parameters
    ----------
    cv_df : pd.DataFrame
        Output of ``CVPlot.compute`` (columns ``cluster``, ``bin_rank``, ``bin_low``,
        ``bin_high``, ``bin_center``).
    tau_map : np.ndarray
        True lifetime per pixel (ground truth), ``(H, W)``.
    photon_map : np.ndarray
        Photon map used for binning in ``CVPlot`` (e.g. ``decay.sum(axis=-1)``).
    irf, laser_period
        See :func:`crb_lifetime_cv`.
    mask : np.ndarray | None
        Pixel mask used in ``CVPlot``.
    cluster_mask, cluster_names
        The same cluster labels and names given to ``CVPlot``, if any.
    n_photons_map : np.ndarray | None
        Expected *signal* photons per pixel for the bound; defaults to `photon_map`.
        Pass the ground-truth ``photon_count_map`` to exclude dark counts.
    background_per_bin, fit_background
        See :func:`crb_lifetime_cv`.

    Returns
    -------
    pd.DataFrame
        One row per (cluster, bin): ``cluster``, ``bin_rank``, ``bin_center``,
        ``tau_true``, ``n_signal``, ``cv_crb`` and ``F_crb`` (``cv_crb *
        sqrt(bin_center)``, directly comparable with a measured ``cv *
        sqrt(bin_center)``). Merge with the measured table on
        ``["cluster", "bin_rank"]``.
    """
    tau_map = np.asarray(tau_map, dtype=float)
    photon_map = np.asarray(photon_map, dtype=float)
    base = (tau_map > 0) if mask is None else np.asarray(mask, dtype=bool)
    n_map = photon_map if n_photons_map is None else np.asarray(n_photons_map, float)
    cv_map = crb_cv_map(
        tau_map, n_map, irf, laser_period, base, background_per_bin, fit_background
    )
    name_to_id = {v: k for k, v in (cluster_names or {}).items()}

    bins = cv_df.drop_duplicates(subset=["cluster", "bin_rank"])
    rows = []
    for _, r in bins.iterrows():
        sel = base & np.isfinite(photon_map)
        cluster = r["cluster"]
        if cluster is not None and not pd.isna(cluster) and cluster_mask is not None:
            cid = name_to_id.get(cluster, cluster)
            sel = sel & (np.asarray(cluster_mask) == cid)
        in_bin = (photon_map > r["bin_low"]) & (photon_map <= r["bin_high"])
        if r["bin_rank"] == 0:
            in_bin |= photon_map == r["bin_low"]
        sel = sel & in_bin & np.isfinite(cv_map)
        if not np.any(sel):
            continue
        tau_i, cv_i = tau_map[sel], cv_map[sel]
        cv_bin = np.sqrt(np.mean((cv_i * tau_i) ** 2)) / np.mean(tau_i)
        rows.append(
            {
                "cluster": cluster,
                "bin_rank": r["bin_rank"],
                "bin_center": r["bin_center"],
                "tau_true": float(np.mean(tau_i)),
                "n_signal": float(np.mean(n_map[sel])),
                "cv_crb": float(cv_bin),
                "F_crb": float(f_value(cv_bin, r["bin_center"])),
            }
        )
    return pd.DataFrame(rows)
