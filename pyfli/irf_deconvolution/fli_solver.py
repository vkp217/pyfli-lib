"""
Solve FLI lifetimes and instrument response functions with detector-aware weighting.

This module belongs to :mod:`pyfli.irf_deconvolution` and is part of PyFLI detector-
aware IRF deconvolution and joint FLI fitting utilities. Public API includes classes
:class:`SolverConfig`; functions :func:`cyclic_conv`, :func:`cyclic_corr`,
:func:`decay_basis`, :func:`build_gate_matrix`, :func:`project_simplex`,
:func:`huber_tv_grad`, :func:`spatial_laplacian`, :func:`fourier_shift`,
:func:`pin_barycenter`, and :func:`fit_decay_pixel`.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

from pyfli import logging
from pyfli.reconstruction.common_reconstruct import circular_convolve

try:
    from .detector_weights import make_observation
except ImportError:
    from detector_weights import make_observation

EPS = 1e-9


def cyclic_conv(h: np.ndarray, f: np.ndarray) -> Any:
    """
    Run the cyclic conv routine.

    Parameters
    ----------
    h : np.ndarray
        IRF, image height, or temporal kernel used by the routine.
    f : np.ndarray
        Decay basis, distribution, or signal function used by the calculation.

    Returns
    -------
    Any
        Object produced by cyclic conv.
    """
    return circular_convolve(h, f)


def cyclic_corr(u: np.ndarray, f: np.ndarray) -> Any:
    """
    Run the cyclic corr routine.

    Parameters
    ----------
    u : np.ndarray
        Signal vector used by cyclic correlation.
    f : np.ndarray
        Decay basis, distribution, or signal function used by the calculation.

    Returns
    -------
    Any
        Object produced by cyclic corr.
    """
    N = u.shape[-1]
    return np.fft.irfft(
        np.conj(np.fft.rfft(f, axis=-1)) * np.fft.rfft(u, axis=-1), n=N, axis=-1
    )


def decay_basis(taus: np.ndarray, t: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Run the decay basis routine.

    Parameters
    ----------
    taus : np.ndarray
        Lifetime grid or lifetime vector in nanoseconds.
    t : np.ndarray
        Time axis or acquisition period used by the calculation.
    T : np.ndarray
        Time axis or acquisition period used by the calculation.

    Returns
    -------
    np.ndarray
        Exponential decay basis evaluated on the time grid.
    """
    taus = np.atleast_1d(np.asarray(taus, float))
    return np.stack([np.exp(-t / tau) / (1.0 - np.exp(-T / tau)) for tau in taus], 0)


def build_gate_matrix(
    t: np.ndarray,
    T: np.ndarray,
    n_gates: int,
    width: float,
    edge: float = 0.0,
    eta: float | None = None,
) -> np.ndarray:
    """
    Build gate matrix.

    Parameters
    ----------
    t : np.ndarray
        Time axis or acquisition period used by the calculation.
    T : np.ndarray
        Time axis or acquisition period used by the calculation.
    n_gates : int
        Number of acquisition gates.
    width : float
        Gate width used by the gate matrix.
    edge : float
        Gate edge offset used when building the gate matrix.
    eta : float | None
        Optional gate efficiency profile.

    Returns
    -------
    np.ndarray
        Gate-integration matrix mapping decay samples to gates.
    """
    N = t.size
    dt = T / N
    centers = np.linspace(0, T, n_gates, endpoint=False)
    G = np.zeros((n_gates, N))
    for g, t0 in enumerate(centers):
        if edge <= 0:
            rel = (t - t0) % T
            G[g] = ((rel >= 0) & (rel < width)).astype(float)
        else:
            rel = t - t0
            prof = 0.5 * (
                erf(rel / (np.sqrt(2) * edge))
                - erf((rel - width) / (np.sqrt(2) * edge))
            )
            G[g] = np.clip(prof, 0, 1)
    G *= dt
    if eta is not None:
        G *= np.asarray(eta)[:, None]
    return G


def project_simplex(V: np.ndarray) -> Any:
    """
    Run the project simplex routine.

    Parameters
    ----------
    V : np.ndarray
        Vector or matrix evaluated by the simplex projection.

    Returns
    -------
    Any
        Object produced by project simplex.
    """
    V = np.atleast_2d(V)
    n = V.shape[1]
    U = np.sort(V, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, n + 1)
    cond = U - cssv / ind > 0
    rho = cond.sum(axis=1)
    theta = cssv[np.arange(V.shape[0]), rho - 1] / rho
    return np.maximum(V - theta[:, None], 0.0)


def huber_tv_grad(h: np.ndarray, delta: np.ndarray) -> Any:
    """
    Run the huber TV grad routine.

    Parameters
    ----------
    h : np.ndarray
        IRF, image height, or temporal kernel used by the routine.
    delta : np.ndarray
        Huber transition value used by the TV gradient.

    Returns
    -------
    Any
        Object produced by huber TV grad.
    """
    d = h - np.roll(h, 1, axis=-1)
    psi = np.where(np.abs(d) <= delta, d / delta, np.sign(d))
    return psi - np.roll(psi, -1, axis=-1)


def spatial_laplacian(H: np.ndarray, ny: np.ndarray, nx: np.ndarray) -> Any:
    """
    Run the spatial laplacian routine.

    Parameters
    ----------
    H : np.ndarray
        IRF estimate, image stack, or convolution kernel used by the solver.
    ny : np.ndarray
        Image height used for reshaping flattened arrays.
    nx : np.ndarray
        Image width used for reshaping flattened arrays.

    Returns
    -------
    Any
        Object produced by spatial laplacian.
    """
    N = H.shape[-1]
    Himg = H.reshape(ny, nx, N)
    lap = (
        4 * Himg
        - np.roll(Himg, 1, 0)
        - np.roll(Himg, -1, 0)
        - np.roll(Himg, 1, 1)
        - np.roll(Himg, -1, 1)
    )
    return lap.reshape(ny * nx, N)


def fourier_shift(H: np.ndarray, s: np.ndarray) -> Any:
    """
    Run the fourier shift routine.

    Parameters
    ----------
    H : np.ndarray
        IRF estimate, image stack, or convolution kernel used by the solver.
    s : np.ndarray
        Phasor imaginary coordinate or shift amount.

    Returns
    -------
    Any
        Object produced by fourier shift.
    """
    N = H.shape[-1]
    k = np.fft.rfftfreq(N, d=1.0 / N)
    phase = np.exp(-2j * np.pi * k * s / N)
    return np.fft.irfft(np.fft.rfft(H, axis=-1) * phase, n=N, axis=-1)


def pin_barycenter(H: np.ndarray, c_target: np.ndarray) -> Any:
    """
    Run the pin barycenter routine.

    Parameters
    ----------
    H : np.ndarray
        IRF estimate, image stack, or convolution kernel used by the solver.
    c_target : np.ndarray
        Target barycenter used to pin the IRF shift.

    Returns
    -------
    Any
        Object produced by pin barycenter.
    """
    idx = np.arange(H.shape[-1])
    cbar = float(np.mean((H * idx).sum(-1) / np.maximum(H.sum(-1), EPS)))
    return np.maximum(fourier_shift(H, c_target - cbar), 0.0)


@dataclass
class SolverConfig:
    """
    Run the solver config routine.
    controls model count, lifetime bounds, regularization, IRF update iterations, and
    logging behavior.

    Parameters
    ----------
    T : float
        Time axis or acquisition period used by the calculation.
    n_models : int
        Number of mixture models or candidate components to fit.
    tau_init : tuple
        Initial lifetime vector for pixel-wise exponential fitting.
    tau_bounds : tuple
        Lower and upper lifetime bounds for fitted exponential components.
    tau_sep : float
        Minimum separation enforced between fitted lifetimes.
    rho1 : float
        Penalty weight for the first regularized optimization term.
    rho2 : float
        Penalty weight for the second regularized optimization term.
    outer_iters : int
        Number of outer optimization iterations.
    irf_inner_iters : int
        Number of inner iterations used when updating the IRF estimate.
    irf_step : float
        Step size for IRF updates.
    estimate_irf : bool
        If ``True``, update the IRF during optimization.
    pin_global_shift : bool
        If ``True``, keep the global IRF shift fixed during optimization.
    verbose : bool
        If ``True``, report progress and diagnostic messages during processing.
    """

    T: float = 12.5
    n_models: int = 2
    tau_init: tuple = (0.5, 2.0)
    tau_bounds: tuple = (0.05, 8.0)
    tau_sep: float = 1.4
    rho1: float = 0.02
    rho2: float = 0.10
    outer_iters: int = 8
    irf_inner_iters: int = 250
    irf_step: float = 0.5
    estimate_irf: bool = True
    pin_global_shift: bool = False
    verbose: bool = True


def _phi(
    taus: np.ndarray, h: np.ndarray, t: np.ndarray, T: np.ndarray, G: np.ndarray
) -> Any:
    """
    Run the phi routine.

    Parameters
    ----------
    taus : np.ndarray
        Lifetime grid or lifetime vector in nanoseconds.
    h : np.ndarray
        IRF, image height, or temporal kernel used by the routine.
    t : np.ndarray
        Time axis or acquisition period used by the calculation.
    T : np.ndarray
        Time axis or acquisition period used by the calculation.
    G : np.ndarray
        Phasor real coordinate.

    Returns
    -------
    Any
        Object produced by phi.
    """
    B = decay_basis(taus, t, T)
    M = cyclic_conv(h[None, :], B)
    return G @ M.T


def fit_decay_pixel(
    lam_obs: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    G: np.ndarray,
    cfg: Any,
) -> Any:
    """
    Fit decay pixel.

    Parameters
    ----------
    lam_obs : np.ndarray
        Observed photon-rate array after detector conversion.
    w : np.ndarray
        Weight vector, image width, or basis vector used by the routine.
    h : np.ndarray
        IRF, image height, or temporal kernel used by the routine.
    t : np.ndarray
        Time axis or acquisition period used by the calculation.
    T : np.ndarray
        Time axis or acquisition period used by the calculation.
    G : np.ndarray
        Phasor real coordinate.
    cfg : Any
        Configuration object or keyword dictionary used by the algorithm.

    Returns
    -------
    Any
        Object produced by fit decay pixel.
    """
    sw = np.sqrt(w)
    lo, hi = cfg.tau_bounds

    def resid(log_taus: np.ndarray) -> Any:
        """
        Run the resid routine.

        Parameters
        ----------
        log_taus : np.ndarray
            Log-transformed lifetimes optimized by the residual function.

        Returns
        -------
        Any
            Object produced by resid.
        """
        taus = np.exp(log_taus)
        Phi = _phi(taus, h, t, T, G)
        A, *_ = np.linalg.lstsq(sw[:, None] * Phi, sw * lam_obs, rcond=None)
        A = np.maximum(A, 0.0)
        return sw * (Phi @ A - lam_obs)

    x0 = np.log(np.clip(cfg.tau_init[: cfg.n_models], lo, hi))
    sol = least_squares(
        resid, x0, method="trf", bounds=(np.log(lo), np.log(hi)), max_nfev=200
    )
    taus = np.sort(np.exp(sol.x))
    Phi = _phi(taus, h, t, T, G)
    A, *_ = np.linalg.lstsq(sw[:, None] * Phi, sw * lam_obs, rcond=None)
    A = np.maximum(A, 0.0)
    return taus, A, Phi @ A


def update_irf(
    H: np.ndarray,
    F: np.ndarray,
    lam_obs: np.ndarray,
    W: np.ndarray,
    G: np.ndarray,
    mu1: np.ndarray,
    mu2: np.ndarray,
    ny: np.ndarray,
    nx: np.ndarray,
    cfg: Any,
) -> Any:
    """
    Update IRF.

    Parameters
    ----------
    H : np.ndarray
        IRF estimate, image stack, or convolution kernel used by the solver.
    F : np.ndarray
        Forward model matrix or decay estimate used by the solver.
    lam_obs : np.ndarray
        Observed photon-rate array after detector conversion.
    W : np.ndarray
        Weight matrix or vector applied in the optimization objective.
    G : np.ndarray
        Phasor real coordinate.
    mu1 : np.ndarray
        Auxiliary optimization variable for the first regularized update.
    mu2 : np.ndarray
        Auxiliary optimization variable for the second regularized update.
    ny : np.ndarray
        Image height used for reshaping flattened arrays.
    nx : np.ndarray
        Image width used for reshaping flattened arrays.
    cfg : Any
        Configuration object or keyword dictionary used by the algorithm.

    Returns
    -------
    Any
        Object produced by update IRF.
    """
    P, N = H.shape

    def data_grad(Hx: np.ndarray) -> Any:
        """
        Run the data grad routine.

        Parameters
        ----------
        Hx : np.ndarray
            Candidate IRF vector evaluated by the gradient or Hessian helper.

        Returns
        -------
        Any
            Object produced by data grad.
        """
        lam = cyclic_conv(Hx, F) @ G.T
        resid = W * (lam - lam_obs)
        back = resid @ G
        return 2.0 * cyclic_corr(back, F)

    def data_hess(V: np.ndarray) -> Any:
        """
        Run the data hess routine.

        Parameters
        ----------
        V : np.ndarray
            Vector or matrix evaluated by the simplex projection.

        Returns
        -------
        Any
            Object produced by data hess.
        """
        lam = cyclic_conv(V, F) @ G.T
        return 2.0 * cyclic_corr((W * lam) @ G, F)

    v = np.random.default_rng(0).standard_normal((P, N))
    v /= np.linalg.norm(v) + EPS
    L_data = 1.0
    for _ in range(12):
        Av = data_hess(v)
        L_data = np.linalg.norm(Av)
        v = Av / (L_data + EPS)

    delta = 0.05 * float(H.max())
    L = L_data + mu1 / max(delta, EPS) + 8.0 * mu2 + EPS
    step = 1.0 / L

    for _ in range(cfg.irf_inner_iters):
        g = data_grad(H)
        g += mu1 * huber_tv_grad(H, delta)
        g += mu2 * 2.0 * spatial_laplacian(H, ny, nx)
        H = project_simplex(H - step * g)
    return H


def solve_flim(
    y: np.ndarray,
    detector: str,
    det_params: np.ndarray,
    ny: np.ndarray,
    nx: np.ndarray,
    gate_spec: np.ndarray,
    cfg: SolverConfig,
    h_init: np.ndarray | None = None,
) -> Any:
    """
    Run the solve FLI routine.

    Parameters
    ----------
    y : np.ndarray
        Observed signal, target data, or coordinate array.
    detector : str
        Detector model name used to select weighting or conversion logic.
    det_params : np.ndarray
        Detector model parameters used for observation weighting.
    ny : np.ndarray
        Image height used for reshaping flattened arrays.
    nx : np.ndarray
        Image width used for reshaping flattened arrays.
    gate_spec : np.ndarray
        Gate timing specification used by the detector model.
    cfg : SolverConfig
        Configuration object or keyword dictionary used by the algorithm.
    h_init : np.ndarray | None
        Initial IRF estimate supplied to the solver.

    Returns
    -------
    Any
        Object produced by solve FLI.
    """
    P = y.shape[0]
    N = gate_spec.get("N", 256)
    t = np.linspace(0, cfg.T, N, endpoint=False)
    G = build_gate_matrix(
        t,
        cfg.T,
        gate_spec["n_gates"],
        gate_spec["width"],
        gate_spec.get("edge", 0.0),
        gate_spec.get("eta", None),
    )

    lam_obs, W = make_observation(y, detector, det_params)

    if h_init is None:
        sig = max(2.0, N / 64)
        c = N // 8
        h0 = np.exp(-0.5 * ((np.arange(N) - c) / sig) ** 2)
        H = np.tile(h0 / h0.sum(), (P, 1))
    else:
        H = h_init.copy()

    taus = np.tile(cfg.tau_init[: cfg.n_models], (P, 1)).astype(float)
    amps = np.zeros((P, cfg.n_models))
    F = np.zeros((P, N))

    def assemble_F(taus: np.ndarray, amps: np.ndarray) -> np.ndarray:
        """
        Run the assemble f routine.

        Parameters
        ----------
        taus : np.ndarray
            Lifetime grid or lifetime vector in nanoseconds.
        amps : np.ndarray
            Amplitude vector estimated for fixed lifetimes.

        Returns
        -------
        np.ndarray
            Assembled forward-model matrix.
        """
        out = np.zeros((P, N))
        for k in range(P):
            Bk = decay_basis(taus[k], t, cfg.T)
            out[k] = amps[k] @ Bk
        return out

    for k in range(P):
        tk, ak, _ = fit_decay_pixel(lam_obs[k], W[k], H[k], t, cfg.T, G, cfg)
        taus[k], amps[k] = tk, ak
    F = assemble_F(taus, amps)
    lam_model = cyclic_conv(H, F) @ G.T
    D0 = float(np.sum(W * (lam_model - lam_obs) ** 2))
    TV0 = float(np.sum(np.abs(H - np.roll(H, 1, -1))))
    E_h = float(np.sum(H**2))
    mu1 = cfg.rho1 * D0 / (TV0 + EPS)
    mu2 = cfg.rho2 * D0 / (E_h + EPS)
    if cfg.verbose and cfg.estimate_irf:
        logging.info(
            f"[init] data={D0:.3g}  TV0={TV0:.3g}  E_h={E_h:.3g}  ->  mu1={mu1:.4g}  mu2={mu2:.4g}"
        )

    idx = np.arange(N)
    c_target = float(np.mean((H * idx).sum(-1) / np.maximum(H.sum(-1), EPS)))
    for it in range(cfg.outer_iters):
        for k in range(P):
            taus[k], amps[k], _ = fit_decay_pixel(
                lam_obs[k], W[k], H[k], t, cfg.T, G, cfg
            )
        F = assemble_F(taus, amps)
        lam_model = cyclic_conv(H, F) @ G.T
        _, W = make_observation(y, detector, det_params)
        if cfg.estimate_irf:
            H = update_irf(H, F, lam_obs, W, G, mu1, mu2, ny, nx, cfg)
            if cfg.pin_global_shift:
                H = pin_barycenter(H, c_target)
        if cfg.verbose:
            misfit = float(np.sum(W * (cyclic_conv(H, F) @ G.T - lam_obs) ** 2))
            logging.info(
                f"[iter {it + 1:2d}] weighted misfit = {misfit:.5g}   mean tau = {taus.mean(0)}"
            )
        if not cfg.estimate_irf:
            break

    return dict(taus=taus, amps=amps, irf=H, mu1=mu1, mu2=mu2, gate=G, t=t)
