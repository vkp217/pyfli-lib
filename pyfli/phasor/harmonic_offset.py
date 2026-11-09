"""
Fit-free pixelwise IRF-decay timing offset ``t0`` from multi-harmonic phasors: mono
estimate, bracket for any completely monotone decay, chi-square selection and correction.
"""

from typing import Any

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

from pyfli.cm_stats.chi_square import degrees_of_freedom
from pyfli.reconstruction.common_reconstruct import gate_starts

from .phasorS.phasor_locus_tools import _omega_rad_per_ns, modulation_lifetime
from .phasorS.phasor_simple import PhasorAnalyzer

_EPS = 1e-12


def _harmonic_axis(z: np.ndarray) -> np.ndarray:
    return np.arange(1, z.shape[0] + 1).reshape((-1,) + (1,) * (z.ndim - 1))


def _clipped_modulation(z: np.ndarray) -> np.ndarray:
    return np.clip(np.abs(z), 0.0, 1.0)


def harmonic_phasors(
    analyzer: PhasorAnalyzer, decay: ArrayLike, n_harmonics: int | None = None
) -> np.ndarray:
    """
    Raw complex phasors ``G + iS`` of `decay` at harmonics ``1..n_harmonics``, from
    :meth:`PhasorAnalyzer.create_phasor_cpu`.

    Returns
    -------
    np.ndarray
        ``(n_harmonics,) + decay.shape[:-1]``; `n_harmonics` defaults to the analyzer's.
    """
    if n_harmonics is not None and n_harmonics != analyzer.n_harmonics:
        analyzer = PhasorAnalyzer(
            analyzer.frequency, analyzer.time_axis_ns, n_harmonics, device="cpu"
        )
    G, S = analyzer.create_phasor_cpu(decay)
    return np.asarray(G) + 1j * np.asarray(S)


def calibrate_phasors(z_raw: ArrayLike, z_irf: ArrayLike) -> np.ndarray:
    """
    Calibrated phasors ``z_k = z~_k / z~^I_k`` (as :meth:`PhasorAnalyzer.calibrate`) for
    the harmonics of `z_raw` ``(K, ...)``; `z_irf` is ``(>= K,)`` or ``(>= K, ...)``.
    """
    z_raw = np.asarray(z_raw, dtype=complex)
    z_irf = np.asarray(z_irf, dtype=complex)[: z_raw.shape[0]]
    if z_irf.ndim == 1:
        z_irf = z_irf.reshape((-1,) + (1,) * (z_raw.ndim - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return z_raw / z_irf


def unwrap_phases(
    z: ArrayLike, frequency_hz: float, t0_guess_ns: ArrayLike = 0.0
) -> np.ndarray:
    """
    Phases ``phi_k`` of calibrated phasors ``z`` ``(K, ...)``, unwrapped beyond the
    principal value. Since ``phi_k`` lies in ``[k omega t0, k omega t0 + pi/2)``, each
    harmonic is taken within ``pi`` of ``k omega (L + U) / 2 + pi/4``, the centre of the
    bracket ``[L, U]`` of eq. 7 built from the lower harmonics; this is unique while
    ``k omega (U - L) < 3 pi / 2``. Unwrapping against ``k phi_1`` (paper section 9)
    instead fails for long lifetimes at ``k >= 4``. Harmonic 1 is centred on
    `t0_guess_ns` and needs ``|t0 - t0_guess| < 3T/8``.
    """
    z = np.asarray(z, dtype=complex)
    w = _omega_rad_per_ns(frequency_hz, 1)
    m = _clipped_modulation(z)
    phi = np.empty(z.shape)
    lower = np.full(z.shape[1:], -np.inf)
    upper = np.full(z.shape[1:], np.inf)
    ref = w * np.asarray(t0_guess_ns, dtype=float) + np.pi / 4
    for h in range(z.shape[0]):
        kw = (h + 1) * w
        if h:
            ref = kw * (lower + upper) / 2 + np.pi / 4
        phi[h] = np.angle(z[h] * np.exp(-1j * ref)) + ref
        lower = np.maximum(lower, (phi[h] - np.arccos(m[h])) / kw)
        upper = np.minimum(upper, phi[h] / kw)
    return phi


def mono_offset(
    z: ArrayLike, frequency_hz: float, t0_guess_ns: ArrayLike = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-harmonic mono-exponential offset and lifetime (paper eq. 6),
    ``t0_k = (phi_k - arccos m_k) / (k omega)`` and ``tau_k = sqrt(m_k^-2 - 1) / (k omega)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(t0_k, tau_k)`` in ns, each ``(K, ...)``. Moduli above 1 are clipped to 1.
    """
    z = np.asarray(z, dtype=complex)
    k = _harmonic_axis(z)
    w = _omega_rad_per_ns(frequency_hz, 1)
    phi = unwrap_phases(z, frequency_hz, t0_guess_ns)
    t0_k = (phi - np.arccos(_clipped_modulation(z))) / (k * w)
    tau_k = np.stack(
        [
            modulation_lifetime(zk.real, zk.imag, frequency_hz, h + 1)
            for h, zk in enumerate(z)
        ]
    )
    return t0_k, tau_k


def offset_bracket(
    z: ArrayLike, frequency_hz: float, t0_guess_ns: ArrayLike = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-sided bracket ``max_k t0_k <= t0 <= min_k phi_k / (k omega)`` (paper eq. 7),
    valid for every completely monotone decay when ``c = 1``. The upper bound holds for
    any ``c > 0``.
    """
    z = np.asarray(z, dtype=complex)
    k = _harmonic_axis(z)
    w = _omega_rad_per_ns(frequency_hz, 1)
    t0_k, _ = mono_offset(z, frequency_hz, t0_guess_ns)
    upper = unwrap_phases(z, frequency_hz, t0_guess_ns) / (k * w)
    return t0_k.max(axis=0), upper.min(axis=0)


def phasor_noise(
    z_raw: ArrayLike,
    z_irf: ArrayLike,
    n_photons: ArrayLike,
    n_harmonics: int,
    excess_noise: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Shot-noise covariance of the calibrated phasors in each phasor's own rotated frame
    ``X + iY = e^{-i phi_k} z_k`` (paper eq. 8), including the cross-harmonic terms that
    involve ``z_{k+l}`` and ``z_{k-l}``.

    Parameters
    ----------
    z_raw : array_like
        Raw (uncalibrated) sample phasors ``(>= 2K, ...)``.
    z_irf : array_like
        Raw IRF phasors ``(>= K,)`` or ``(>= K, ...)``.
    n_photons : array_like
        Detected counts per pixel (background included), shape ``(...)``.
    n_harmonics : int
        ``K``.
    excess_noise : float
        Variance multiplier: ICCD excess-noise factor times counts per photon.

    Returns
    -------
    dict[str, np.ndarray]
        ``"XX"``, ``"YY"``, ``"XY"`` (``Cov(X_k, Y_l)``), ``"YX"``, each ``(..., K, K)``,
        and ``"m"``, the calibrated moduli ``(..., K)``.
    """
    K = int(n_harmonics)
    z_raw = np.asarray(z_raw, dtype=complex)
    if z_raw.shape[0] < 2 * K:
        raise ValueError(
            f"z_raw needs at least {2 * K} harmonics, got {z_raw.shape[0]}"
        )
    p = np.moveaxis(z_raw[: 2 * K], 0, -1)
    full = np.concatenate([np.conj(p[..., ::-1]), np.ones_like(p[..., :1]), p], axis=-1)
    k = np.arange(1, K + 1)
    pk = p[..., :K]
    rot = np.exp(-1j * np.angle(pk))
    uv = (
        full[..., 2 * K + k[:, None] + k[None, :]] - pk[..., :, None] * pk[..., None, :]
    )
    uv = uv * rot[..., :, None] * rot[..., None, :]
    uvc = full[..., 2 * K + k[:, None] - k[None, :]] - pk[..., :, None] * np.conj(
        pk[..., None, :]
    )
    uvc = uvc * rot[..., :, None] * np.conj(rot[..., None, :])
    irf_mod = np.moveaxis(np.abs(np.asarray(z_irf, dtype=complex)[:K]), 0, -1)
    scale = excess_noise / (
        np.asarray(n_photons, dtype=float)[..., None, None]
        * irf_mod[..., :, None]
        * irf_mod[..., None, :]
    )
    return {
        "XX": 0.5 * (uv + uvc).real * scale,
        "YY": 0.5 * (uvc - uv).real * scale,
        "XY": 0.5 * (uv.imag - uvc.imag) * scale,
        "YX": 0.5 * (uv.imag + uvc.imag) * scale,
        "m": np.abs(pk) / irf_mod,
    }


def offset_covariance(
    noise: dict[str, np.ndarray], frequency_hz: float, include_lifetime: bool = False
) -> np.ndarray:
    """
    Delta-method covariance of the per-harmonic offsets ``t0_k`` (paper eq. 9, with the
    full ``K x K`` cross-harmonic form), from :func:`phasor_noise`. With
    `include_lifetime`, the joint covariance of ``(t0_1..t0_K, tau_1..tau_K)``, using
    ``d tau_k = -dX_k / (k omega m_k^2 sqrt(1 - m_k^2))``.

    Returns
    -------
    np.ndarray
        ``(..., K, K)``, or ``(..., 2K, 2K)`` with `include_lifetime`, in ns^2.
    """
    m = np.clip(noise["m"], _EPS, 1.0 - _EPS)
    K = m.shape[-1]
    kw = np.arange(1, K + 1) * _omega_rad_per_ns(frequency_hz, 1)
    s = np.sqrt(1.0 - m**2)
    eye = np.eye(K)
    jac = np.concatenate(
        [eye / (kw * s)[..., None, :], eye / (kw * m)[..., None, :]], axis=-1
    )
    if include_lifetime:
        tau_x = -eye / (kw * m**2 * s)[..., None, :]
        tau_rows = np.concatenate([tau_x, np.zeros(tau_x.shape)], axis=-1)
        jac = np.concatenate([jac, tau_rows], axis=-2)
    sigma = np.concatenate(
        [
            np.concatenate([noise["XX"], noise["XY"]], axis=-1),
            np.concatenate([noise["YX"], noise["YY"]], axis=-1),
        ],
        axis=-2,
    )
    return jac @ sigma @ np.swapaxes(jac, -1, -2)


def combine_offsets(
    t0_k: ArrayLike,
    cov: ArrayLike,
    full_covariance: bool = True,
    tau_k: ArrayLike | None = None,
) -> dict[str, Any]:
    """
    Generalised least-squares mono-exponential test across harmonics (paper eq. 10).
    With `t0_k` only, one parameter ``t0`` on ``K`` observations (``dof = K - 1``); with
    `tau_k` and a joint `cov` from ``offset_covariance(..., include_lifetime=True)``,
    two parameters ``(t0, tau)`` on ``2K`` observations (``dof = 2K - 2``), which also
    tests that the modulation lifetimes agree. ``full_covariance=False`` drops the
    cross terms.

    Returns
    -------
    dict[str, Any]
        ``"t0"``, ``"t0_se"``, ``"tau"`` and ``"tau_se"`` (``None`` without `tau_k`),
        ``"chi2"``, ``"reduced_chi2"`` (``chi2 / dof``), ``"dof"`` and ``"n_params"``.
        Pixels with non-finite inputs are ``NaN``.
    """
    y = np.moveaxis(np.asarray(t0_k, dtype=float), 0, -1)
    K = y.shape[-1]
    n_params = 1 if tau_k is None else 2
    if tau_k is not None:
        y = np.concatenate([y, np.moveaxis(np.asarray(tau_k, dtype=float), 0, -1)], -1)
    n = y.shape[-1]
    C = np.asarray(cov, dtype=float)
    if C.shape[-1] != n:
        raise ValueError(f"cov is {C.shape[-1]}x{C.shape[-1]}, expected {n}x{n}")
    if not full_covariance:
        C = np.eye(n) * np.diagonal(C, axis1=-2, axis2=-1)[..., None, :]
    bad = ~(np.isfinite(y).all(-1) & np.isfinite(C).all((-2, -1)))
    C = np.where(bad[..., None, None], np.eye(n), C)
    y = np.where(bad[..., None], 0.0, y)
    design = np.zeros((n, n_params))
    design[:K, 0] = 1.0
    design[K:, n_params - 1] = 1.0
    cinv_a = np.linalg.solve(C, np.broadcast_to(design, C.shape[:-2] + design.shape))
    fisher_inv = np.linalg.inv(design.T @ cinv_a)
    beta = (fisher_inv @ (np.swapaxes(cinv_a, -1, -2) @ y[..., None]))[..., 0]
    r = y - beta @ design.T
    chi2 = (r * np.linalg.solve(C, r[..., None])[..., 0]).sum(-1)
    nan = np.where(bad, np.nan, 1.0)
    se = np.sqrt(np.diagonal(fisher_inv, axis1=-2, axis2=-1)) * nan[..., None]
    dof = degrees_of_freedom(n, n_params, None)
    return {
        "t0": beta[..., 0] * nan,
        "t0_se": se[..., 0],
        "tau": None if tau_k is None else beta[..., 1] * nan,
        "tau_se": None if tau_k is None else se[..., 1],
        "chi2": chi2 * nan,
        "reduced_chi2": chi2 * nan / dof,
        "dof": dof,
        "n_params": n_params,
    }


def select_mono(chi2: ArrayLike, dof: int, alpha: float = 0.05) -> np.ndarray:
    """
    Pixels consistent with the offset model: ``chi2 < chi2_{dof, 1-alpha}`` (``dof`` from
    :func:`combine_offsets` or :func:`bi_exponential_offset`).
    """
    with np.errstate(invalid="ignore"):
        return np.asarray(chi2) < chi2_dist.ppf(1.0 - alpha, dof)


def _bi_pixel(zk, rot, whiten, w, lo, up, margin):
    k = np.arange(1, zk.size + 1) * w

    def residual(p):
        t0, l1, l2, a = p
        model = np.exp(1j * k * t0) * (
            a / (1 - 1j * k * np.exp(l1)) + (1 - a) / (1 - 1j * k * np.exp(l2))
        )
        d = (zk - model) / rot
        return whiten @ np.concatenate([d.real, d.imag])

    low = [lo - margin, np.log(0.02), np.log(0.05), 0.0]
    high = [up + margin, np.log(30.0), np.log(50.0), 1.0]
    best = None
    for t0 in (lo, 0.5 * (lo + up)):
        for l1, l2 in ((np.log(0.3), np.log(3.0)), (np.log(0.8), np.log(4.0))):
            res = least_squares(residual, [t0, l1, l2, 0.5], bounds=(low, high))
            if best is None or res.cost < best.cost:
                best = res
    t0, l1, l2, a = best.x
    if l1 > l2:
        l1, l2, a = l2, l1, 1.0 - a
    cov = np.linalg.pinv(best.jac.T @ best.jac)
    return t0, np.sqrt(max(cov[0, 0], 0.0)), np.exp(l1), np.exp(l2), a, 2.0 * best.cost


def bi_exponential_offset(
    z: ArrayLike,
    noise: dict[str, np.ndarray],
    frequency_hz: float,
    t0_guess_ns: ArrayLike = 0.0,
    margin_ns: float = 0.1,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """
    Per-pixel offset of a bi-exponential decay from ``K >= 3`` harmonics. Theorem 1 makes
    ``t0`` unidentifiable for an arbitrary completely monotone decay, but under the
    bi-exponential model ``z_k = e^{i k omega t0} (A / (1 - i k omega tau1) + (1 - A) /
    (1 - i k omega tau2))`` the ``2K`` real data determine the four unknowns
    ``(t0, tau1, tau2, A)``. Each pixel is fitted by weighted least squares, whitened by the
    shot-noise covariance of :func:`phasor_noise`, with ``t0`` inside its bracket
    (:func:`offset_bracket`) widened by `margin_ns`. ``A = 1`` is the mono estimate.

    Returns
    -------
    dict[str, Any]
        ``"t0"``, ``"t0_se"``, ``"tau1"``, ``"tau2"``, ``"alpha1"`` (photon fraction of
        ``tau1``), ``"chi2"`` and ``"reduced_chi2"``, each ``(...)`` (``NaN`` where the
        noise is undefined), ``"dof"`` = ``2K - 4`` and ``"n_params"`` = 4.
    """
    z = np.asarray(z, dtype=complex)
    K = z.shape[0]
    if K < 3:
        raise ValueError("the bi-exponential offset needs at least 3 harmonics")
    w = _omega_rad_per_ns(frequency_hz, 1)
    lower, upper = offset_bracket(z, frequency_hz, t0_guess_ns)
    sigma = np.concatenate(
        [
            np.concatenate([noise["XX"], noise["XY"]], axis=-1),
            np.concatenate([noise["YX"], noise["YY"]], axis=-1),
        ],
        axis=-2,
    )
    spatial = z.shape[1:]
    zf = np.moveaxis(z, 0, -1).reshape(-1, K)
    sf = sigma.reshape(-1, 2 * K, 2 * K)
    lo, up = lower.ravel(), upper.ravel()
    ok = np.isfinite(zf).all(-1) & np.isfinite(sf).all((-2, -1)) & np.isfinite(lo + up)
    idx = np.flatnonzero(ok)

    def run(i):
        try:
            whiten = np.linalg.cholesky(np.linalg.inv(sf[i])).T
        except np.linalg.LinAlgError:
            return (np.nan,) * 6
        return _bi_pixel(
            zf[i], np.exp(1j * np.angle(zf[i])), whiten, w, lo[i], up[i], margin_ns
        )

    rows = Parallel(n_jobs=n_jobs)(delayed(run)(i) for i in idx)
    out = np.full((zf.shape[0], 6), np.nan)
    if rows:
        out[idx] = np.asarray(rows)
    dof = degrees_of_freedom(2 * K, 4, None)
    keys = ("t0", "t0_se", "tau1", "tau2", "alpha1", "chi2")
    result = {key: out[:, i].reshape(spatial) for i, key in enumerate(keys)}
    result.update({"reduced_chi2": result["chi2"] / dof, "dof": dof, "n_params": 4})
    return result


def imbalance_offset(
    z: ArrayLike,
    frequency_hz: float,
    noise: dict[str, np.ndarray] | None = None,
    t0_guess_ns: ArrayLike = 0.0,
) -> dict[str, np.ndarray]:
    """
    Closed-form ``c``, ``tau`` and ``t0`` under modulation imbalance ``c != 1`` (paper
    eq. 12): weighted least squares of ``1 / m_k^2 = P + Q k^2``.

    Parameters
    ----------
    z : array_like
        Calibrated phasors ``(K >= 2, ...)``.
    frequency_hz : float
        Laser repetition frequency in hertz.
    noise : dict | None
        :func:`phasor_noise` output; gives inverse-variance weights and makes ``chi2``
        (``K - 2`` degrees of freedom) a test of constant ``c``. Unweighted if ``None``.
    t0_guess_ns : array_like
        See :func:`unwrap_phases`.

    Returns
    -------
    dict[str, np.ndarray]
        ``"c"``, ``"tau"`` (ns), ``"t0"`` (ns, weighted mean of ``"t0_k"``), ``"t0_k"``
        ``(K, ...)``, ``"chi2"``, ``"reduced_chi2"`` (``NaN`` for ``K = 2``), ``"dof"``
        and ``"n_params"`` (``P``, ``Q``).
    """
    z = np.asarray(z, dtype=complex)
    k = _harmonic_axis(z)
    kw = k * _omega_rad_per_ns(frequency_hz, 1)
    m = np.clip(np.abs(z), _EPS, None)
    y = 1.0 / m**2
    x = (k**2).astype(float) * np.ones(z.shape)
    if noise is None:
        w_fit = np.ones(z.shape)
        w_t0 = np.ones(z.shape)
    else:
        var_x = np.moveaxis(np.diagonal(noise["XX"], axis1=-2, axis2=-1), -1, 0)
        var_y = np.moveaxis(np.diagonal(noise["YY"], axis1=-2, axis2=-1), -1, 0)
        w_fit = m**6 / (4.0 * var_x)
        w_t0 = (kw * m) ** 2 / var_y
    s0, s1, s2 = w_fit.sum(0), (w_fit * x).sum(0), (w_fit * x**2).sum(0)
    sy, sxy = (w_fit * y).sum(0), (w_fit * x * y).sum(0)
    det = s0 * s2 - s1**2
    P = (s2 * sy - s1 * sxy) / det
    Q = (s0 * sxy - s1 * sy) / det
    with np.errstate(invalid="ignore"):
        c = P**-0.5
        tau = np.sqrt(Q / P) / _omega_rad_per_ns(frequency_hz, 1)
    t0_k = (unwrap_phases(z, frequency_hz, t0_guess_ns) - np.arctan(kw * tau)) / kw
    chi2 = (w_fit * (y - P - Q * x) ** 2).sum(0)
    dof = degrees_of_freedom(z.shape[0], 2, None)
    return {
        "c": c,
        "tau": tau,
        "t0": (w_t0 * t0_k).sum(0) / w_t0.sum(0),
        "t0_k": t0_k,
        "chi2": chi2,
        "reduced_chi2": chi2 / dof if dof > 0 else np.full(chi2.shape, np.nan),
        "dof": dof,
        "n_params": 2,
    }


def modulation_factor(
    n_counts: ArrayLike,
    v_shift: ArrayLike,
    n_gates: int,
    irf_counts: ArrayLike | None = None,
    irf_v_shift: ArrayLike = 0.0,
) -> np.ndarray:
    """
    Modulation imbalance ``c = (1 - beta) / (1 - beta_I)`` of paper eq. 3 from the
    constant per-gate backgrounds (the solver's ``v_shift``): ``beta = n_gates v_shift /
    n_counts`` for the sample and ``beta_I = n_gates irf_v_shift / irf_counts`` for the
    IRF trace. Dividing calibrated phasors by ``c`` restores ``c = 1`` exactly.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (
            n_gates
            * np.asarray(v_shift, dtype=float)
            / np.asarray(n_counts, dtype=float)
        )
        if irf_counts is None:
            beta_i = 0.0
        else:
            beta_i = (
                n_gates
                * np.asarray(irf_v_shift, dtype=float)
                / np.asarray(irf_counts, dtype=float)
            )
        return (1.0 - beta) / (1.0 - beta_i)


def h_shift_from_t0(t0_ns: ArrayLike, time_axis_ns: ArrayLike) -> np.ndarray:
    """
    Solver/simulator ``h_shift`` for a harmonic offset ``t0``: ``t0 + dt/2``. The forward
    model (:func:`pyfli.cm_funcs.forward_model.model_numpy`,
    :class:`pyfli.reconstruction.ParamToDecay`) and the simulators excite at the start of
    each IRF gate and integrate the decay over gates ``[t, t + dt)``, so the calibrated
    phasor phase is ``k omega (h_shift - dt/2)``.
    """
    t = np.asarray(time_axis_ns, dtype=float)
    return np.asarray(t0_ns, dtype=float) + 0.5 * (t[1] - t[0])


def t0_from_h_shift(h_shift_ns: ArrayLike, time_axis_ns: ArrayLike) -> np.ndarray:
    """Inverse of :func:`h_shift_from_t0`."""
    t = np.asarray(time_axis_ns, dtype=float)
    return np.asarray(h_shift_ns, dtype=float) - 0.5 * (t[1] - t[0])


def correct_phasors(
    G: ArrayLike, S: ArrayLike, t0_ns: ArrayLike, frequency_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove the offset from calibrated phasor stacks ``(K, ...)``:
    ``z_k <- e^{-i k omega t0} z_k`` (paper eq. 13), exact for every decay.
    """
    z = np.asarray(G, dtype=float) + 1j * np.asarray(S, dtype=float)
    k = _harmonic_axis(z)
    z = z * np.exp(-1j * k * _omega_rad_per_ns(frequency_hz, 1) * np.asarray(t0_ns))
    return z.real, z.imag


def fit_offset_surface(
    t0_hat: ArrayLike,
    accepted: ArrayLike,
    weights: ArrayLike | None = None,
    order: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spatial model ``t0(r)`` from accepted pixels of a 2-D ``t0_hat`` map: the median for
    ``order=0``, otherwise a weighted polynomial ``sum_{i+j<=order} theta_ij x^i y^j``
    on pixel coordinates scaled to ``[-1, 1]``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(t0_map, theta)``; ``theta`` is ``(median,)`` for ``order=0``.
    """
    t0_hat = np.asarray(t0_hat, dtype=float)
    use = np.asarray(accepted, dtype=bool) & np.isfinite(t0_hat)
    if weights is not None:
        use &= np.isfinite(weights) & (np.asarray(weights) > 0)
    if not use.any():
        raise ValueError("no accepted pixels with a finite offset")
    if order == 0:
        med = np.median(t0_hat[use])
        return np.full(t0_hat.shape, med), np.array([med])
    H, W = t0_hat.shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    basis = np.stack(
        [xx**i * yy**j for i in range(order + 1) for j in range(order + 1 - i)], -1
    )
    sw = np.ones(use.sum()) if weights is None else np.sqrt(np.asarray(weights)[use])
    theta, *_ = np.linalg.lstsq(basis[use] * sw[:, None], t0_hat[use] * sw, rcond=None)
    return basis @ theta, theta


def bracket_violations(
    t0_map: ArrayLike, lower: ArrayLike, upper: ArrayLike, tol_ns: ArrayLike = 0.0
) -> np.ndarray:
    """Pixels whose modelled ``t0`` lies outside ``[lower - tol, upper + tol]``."""
    t0_map = np.asarray(t0_map)
    return (t0_map < np.asarray(lower) - tol_ns) | (t0_map > np.asarray(upper) + tol_ns)


class HarmonicOffset:
    """
    Pixelwise ``t0`` estimation and correction on top of a :class:`PhasorAnalyzer`
    (``n_harmonics >= 2``; its time axis must be :func:`gate_starts` of the period
    ``1 / frequency`` over which the phasors are computed).
    """

    def __init__(self, analyzer: PhasorAnalyzer) -> None:
        if analyzer.n_harmonics < 2:
            raise ValueError("HarmonicOffset needs an analyzer with n_harmonics >= 2")
        t = np.asarray(analyzer.time_axis_ns, dtype=float)
        expected = t[0] + gate_starts(1e9 / analyzer.frequency, t.size)
        if not np.allclose(t, expected, rtol=0.0, atol=1e-6 * (expected[1] - t[0])):
            raise ValueError(
                "analyzer.time_axis_ns must be the gate starts of one period 1/frequency "
                "(gate_starts(period, n), i.e. linspace(0, period, n, endpoint=False)); "
                f"got spacing {t[1] - t[0]} ns for period/n = {expected[1] - t[0]} ns"
            )
        self.analyzer = analyzer
        self.frequency = analyzer.frequency
        self.n_harmonics = analyzer.n_harmonics

    def estimate(
        self,
        decay: ArrayLike,
        irf: ArrayLike,
        alpha: float = 0.05,
        order: int = 0,
        excess_noise: float = 1.0,
        full_covariance: bool = True,
        include_lifetime: bool = False,
        t0_guess_ns: ArrayLike = 0.0,
        v_shift: ArrayLike | None = None,
        irf_v_shift: ArrayLike = 0.0,
        model: str = "mono-exponential",
        n_jobs: int = -1,
    ) -> dict[str, Any]:
        """
        Run the full pipeline on a decay cube ``(H, W, T)`` and a 1-D or per-pixel IRF.
        `include_lifetime` tests ``(t0_k, tau_k)`` jointly (``dof = 2K - 2``) instead of
        ``t0_k`` alone (``dof = K - 1``; see :func:`combine_offsets`). A known
        per-gate background `v_shift` (scalar or ``(H, W)``, e.g. the solver's
        ``v_shift_map``) and `irf_v_shift` give ``c`` (:func:`modulation_factor`), which
        is divided out of the calibrated phasors and their noise.
        ``model="bi-exponential"`` (``n_harmonics >= 3``) estimates each pixel's offset
        under the bi-exponential model (:func:`bi_exponential_offset`), so bi-exponential
        samples need no mono-exponential pixels; the chi-square test then has
        ``dof = 2K - 4``.

        Returns
        -------
        dict[str, Any]
            ``z`` (calibrated phasors ``(K, H, W)``), ``t0_k``, ``tau_k``, ``lower``,
            ``upper``, ``cov``, ``c``, the :func:`combine_offsets` keys with ``t0`` as
            ``t0_hat`` (and ``tau`` as ``tau_hat``), ``accepted``, ``t0_map``,
            ``h_shift_map`` (solver convention, :func:`h_shift_from_t0`), ``theta`` and
            ``violations`` (rejected pixels whose bracket excludes ``t0_map``). For the
            bi-exponential model, the :func:`bi_exponential_offset` keys with ``t0`` as
            ``t0_hat`` and ``tau1``, ``tau2``, ``alpha1`` as ``tau1_hat``, ``tau2_hat``,
            ``alpha1_hat``, and ``cov`` = ``None``.
        """
        if model not in ("mono-exponential", "bi-exponential"):
            raise ValueError(
                f"model must be 'mono-exponential' or 'bi-exponential', got {model!r}"
            )
        K = self.n_harmonics
        decay = np.asarray(decay, dtype=float)
        z_raw = harmonic_phasors(self.analyzer, decay, 2 * K)
        z_irf = harmonic_phasors(self.analyzer, irf, 2 * K)
        c = np.ones(())
        if v_shift is not None:
            irf_arr = np.asarray(irf, dtype=float)
            c = modulation_factor(
                decay.sum(-1), v_shift, decay.shape[-1], irf_arr.sum(-1), irf_v_shift
            )
            if z_irf.ndim == 1:
                z_irf = z_irf.reshape((-1,) + (1,) * c.ndim)
            z_irf = z_irf * c
        z = calibrate_phasors(z_raw[:K], z_irf)
        t0_k, tau_k = mono_offset(z, self.frequency, t0_guess_ns)
        lower, upper = offset_bracket(z, self.frequency, t0_guess_ns)
        noise = phasor_noise(z_raw, z_irf, decay.sum(-1), K, excess_noise)
        if model == "bi-exponential":
            cov = None
            test = bi_exponential_offset(
                z, noise, self.frequency, t0_guess_ns, n_jobs=n_jobs
            )
            test = {
                f"{key}_hat" if key in ("tau1", "tau2", "alpha1") else key: value
                for key, value in test.items()
            }
            test["tau"] = None
        else:
            cov = offset_covariance(noise, self.frequency, include_lifetime)
            test = combine_offsets(
                t0_k, cov, full_covariance, tau_k if include_lifetime else None
            )
        accepted = select_mono(test["chi2"], test["dof"], alpha)
        t0_map, theta = fit_offset_surface(
            test["t0"], accepted, test["t0_se"] ** -2.0, order
        )
        return {
            "z": z,
            "t0_k": t0_k,
            "tau_k": tau_k,
            "lower": lower,
            "upper": upper,
            "cov": cov,
            "c": c,
            "t0_hat": test.pop("t0"),
            "tau_hat": test.pop("tau"),
            **test,
            "accepted": accepted,
            "t0_map": t0_map,
            "h_shift_map": h_shift_from_t0(t0_map, self.analyzer.time_axis_ns),
            "theta": theta,
            "violations": ~accepted & bracket_violations(t0_map, lower, upper),
        }

    def correct(
        self, G: ArrayLike, S: ArrayLike, t0_ns: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rotate calibrated phasor stacks by the offset (:func:`correct_phasors`)."""
        return correct_phasors(G, S, t0_ns, self.frequency)
