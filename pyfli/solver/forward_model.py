"""
Evaluate exponential decay kernels and convolved NumPy forward models.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLI fitting routines. Public API includes
functions :func:`gate_integrated_kernel`, :func:`decay_kernel` and
:func:`model_numpy`.

The decay is modelled as it is measured: photon counts per time gate. For a decay with
onset ``h_shift`` (ns), the expected counts in the gate ``[a, b)`` are the integral of
the normalized exponential over that gate,

    mono:  S * [exp(-(max(a, h) - h) / tau) - exp(-(max(b, h) - h) / tau)]
    bi:    S * a1 * [...tau1...] + S * (1 - a1) * [...tau2...]

which is zero before the onset, continuous (piecewise smooth) in ``h_shift`` so a
sub-gate delay can be fitted with gradient-based optimizers, and sums to ``S`` over all
gates -- ``S`` is the total number of photons of the decay (``photon_count_map``).
"""

from typing import Any

import numpy as np

_EPS = 1e-8


def _gate_integral(a: np.ndarray, b: np.ndarray, tau: Any, h_shift: Any) -> np.ndarray:
    """Fraction of a unit-area exponential with onset `h_shift` inside ``[a, b)``."""
    tau_safe = np.clip(tau, _EPS, None)
    a_eff = np.maximum(a, h_shift) - h_shift
    b_eff = np.maximum(b, h_shift) - h_shift
    return np.exp(-a_eff / tau_safe) - np.exp(-b_eff / tau_safe)


def gate_integrated_kernel(
    gate_start: np.ndarray,
    dt: float,
    params: Any,
    model_type: str,
    h_shift: Any = 0.0,
) -> tuple[np.ndarray, Any]:
    """
    Expected counts per gate of the un-convolved decay, integrated over each gate.

    Parameters
    ----------
    gate_start : np.ndarray
        Start time (ns) of every gate; gate ``k`` covers ``[gate_start[k],
        gate_start[k] + dt)``. Any shape that broadcasts with the parameters (e.g.
        ``(T,)``, or ``(1, 1, T)`` against ``(H, W, 1)`` parameter maps).
    dt : float
        Gate width in ns.
    params : Any
        ``(S, tau, v_shift)`` for ``"mono-exponential"`` or
        ``(S, a1, tau1, tau2, v_shift)`` for ``"bi-exponential"``; scalars or arrays
        broadcastable against `gate_start`.
    model_type : str
        ``"mono-exponential"`` or ``"bi-exponential"``.
    h_shift : Any
        Decay onset in ns (scalar or broadcastable array).

    Returns
    -------
    tuple[np.ndarray, Any]
        ``(kernel, v_shift)``: counts per gate (summing to ``S`` over all gates) and
        the constant offset, which is added after the IRF convolution.
    """
    a = np.asarray(gate_start, dtype=float)
    b = a + dt
    if model_type == "mono-exponential":
        S, tau, v_shift = params
        kernel = S * _gate_integral(a, b, tau, h_shift)
    else:
        S, a1, tau1, tau2, v_shift = params
        kernel = S * (
            a1 * _gate_integral(a, b, tau1, h_shift)
            + (1.0 - a1) * _gate_integral(a, b, tau2, h_shift)
        )
    return kernel, v_shift


def _gate_width(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    return float(t[1] - t[0]) if t.size > 1 else 1.0


def decay_kernel(
    t: np.ndarray, params: Any, model_type: str, h_shift: float = 0.0
) -> tuple:
    """Return (kernel, v_shift) on the gates starting at `t`.

    The kernel is the gate-integrated decay of :func:`gate_integrated_kernel` (zero
    before the onset `h_shift`, in ns, and summing to ``S``). Gates are
    ``[t_k, t_k + dt)`` with ``dt = t[1] - t[0]``.
    """
    kernel, v_shift = gate_integrated_kernel(
        t, _gate_width(t), params, model_type, h_shift=h_shift
    )
    return kernel, float(v_shift)


def negative_lag_gates(h_shift: Any, dt: float) -> int:
    """
    Number of gates before ``t = 0`` the kernel must cover so that a decay with onset
    `h_shift` < 0 (earlier than the IRF) is shifted, not truncated, by the IRF
    convolution. Zero for ``h_shift >= 0``.
    """
    h = np.asarray(h_shift, dtype=float)
    h = h[np.isfinite(h)]
    earliest = float(h.min()) if h.size else 0.0
    return int(np.ceil(max(-earliest, 0.0) / dt)) if dt > 0 else 0


def model_numpy(
    t: np.ndarray,
    irf: np.ndarray,
    params: Any,
    model_type: str,
) -> np.ndarray:
    """
    Evaluate the NumPy FLI forward model: the gate-integrated decay (onset
    ``h_shift``) convolved with the normalized IRF, plus the constant ``v_shift``.

    Parameters
    ----------
    t : np.ndarray
        Gate start times (ns), uniformly spaced.
    irf : np.ndarray
        Instrument response function aligned with the decay signal.
    params : Any
        ``[S, tau, v_shift, h_shift]`` (mono) or
        ``[S, a1, tau1, tau2, v_shift, h_shift]`` (bi); ``S`` is the total photon
        count of the decay.
    model_type : str
        FLI model family, such as mono- or bi-exponential.

    Returns
    -------
    np.ndarray
        Expected counts per gate, same length as `t`.
    """
    params = np.asarray(params, dtype=float)

    h_shift = float(params[-1])
    kernel_params = params[:-1]

    t = np.asarray(t, dtype=float)
    n = t.size
    dt = _gate_width(t)
    n_neg = negative_lag_gates(h_shift, dt)
    lags = np.arange(-n_neg, n) * dt
    kernel, v_shift = gate_integrated_kernel(
        lags, dt, kernel_params, model_type, h_shift=h_shift
    )

    irf = np.asarray(irf, dtype=float)
    irf_sum = irf.sum()
    irf_norm = irf / irf_sum if irf_sum > 0 else irf

    convolved = np.convolve(kernel, irf_norm, mode="full")[n_neg : n_neg + n]
    return convolved + float(v_shift)
