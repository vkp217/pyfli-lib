"""
Evaluate exponential decay kernels and convolved NumPy forward models.

This module belongs to :mod:`pyfli.cm_funcs` and is the single forward model shared by the
PyFLI solver, reconstruction, Laguerre, simulator and phasor packages. Public API includes
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

``shift_method`` selects the convolution: ``"zero_pad"`` (default) convolves linearly,
with the decay zero before the onset and after the window; ``"circular"`` is the
steady-state periodic model for a window equal to one laser period (TCSPC/SPAD), where
earlier pulses' tails wrap around
(:func:`pyfli.reconstruction.common_reconstruct.periodic_gate_integral` convolved by
:func:`pyfli.reconstruction.common_reconstruct.circular_convolve`).

``excitation`` places the excitation of each IRF gate: ``"bin_start"`` (default) at the
gate start, as pyfli's simulators do; ``"uniform"`` spread uniformly over the gate
``[h, h + dt)``, as in a real instrument, i.e. the gate integral averaged over the onset.
``gate_width`` (default ``dt``) is the width of the detection gates that start every
``dt``; set it only when the IRF was not itself recorded through the same gate (a gated
IRF already contains the gate), e.g. for overlapping ICCD gates with a laser-pulse IRF.
"""

from typing import Any

import numpy as np

from pyfli.reconstruction.common_reconstruct import (
    circular_convolve,
    periodic_gate_integral,
)

SHIFT_METHODS = ("zero_pad", "circular")
MODEL_TYPES = ("mono-exponential", "bi-exponential")
EXCITATIONS = ("bin_start", "uniform")


def validate_model_type(model_type: Any) -> str:
    """Check `model_type` is one of :data:`MODEL_TYPES` (catches e.g. a ``p0`` passed in its place)."""
    if not isinstance(model_type, str) or model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")
    return model_type


_EPS = 1e-8


def validate_excitation(excitation: Any) -> str:
    """Check `excitation` is one of :data:`EXCITATIONS`."""
    if excitation not in EXCITATIONS:
        raise ValueError(f"excitation must be one of {EXCITATIONS}, got {excitation!r}")
    return excitation


def validate_gate_width(gate_width: Any) -> float | None:
    """Check `gate_width` is ``None`` or a positive finite width in ns."""
    if gate_width is None:
        return None
    width = float(gate_width)
    if not np.isfinite(width) or width <= 0:
        raise ValueError(f"gate_width must be positive and finite, got {gate_width!r}")
    return width


def _xp(x: Any) -> Any:
    if type(x).__module__.split(".")[0] == "torch":
        import torch

        return torch
    return np


def _positive(v: Any, xp: Any) -> Any:
    return xp.clamp(v, min=0.0) if xp is not np else np.maximum(v, 0.0)


def _uniform_gate_integral(a: Any, b: Any, tau: Any, h_shift: Any, dt: float) -> Any:
    """
    Fraction of a unit-area exponential inside ``[a, b)`` with its onset uniform over
    ``[h_shift, h_shift + dt)``: with ``Q(v) = v+ + tau expm1(-v+ / tau)``,
    ``[Q(b') - Q(b' - dt) - Q(a') + Q(a' - dt)] / dt`` (``x' = x - h_shift``); for
    ``a' >= dt`` the closed form ``tau / dt (1 - e^{-dt/tau}) (e^{-(a'-dt)/tau} - e^{-(b'-dt)/tau})``.
    """
    xp = _xp(tau)
    a1 = a - h_shift
    b1 = b - h_shift

    def q(v: Any) -> Any:
        vp = _positive(v, xp)
        return vp + tau * xp.expm1(-vp / tau)

    overlap = (q(b1) - q(b1 - dt) - q(a1) + q(a1 - dt)) / dt
    x = _positive(a1 - dt, xp)
    y = _positive(b1 - dt, xp)
    tail = (tau / dt) * (-xp.expm1(-dt / tau)) * (xp.exp(-x / tau) - xp.exp(-y / tau))
    return xp.where(a1 >= dt, tail, overlap)


def _periodic_cdf_integral(v: Any, tau: Any, period: float, xp: Any) -> Any:
    """``R(v) = int_0^v Psi(u) du`` of the steady-state cumulative count ``Psi``."""
    r = v % period
    n = (v - r) / period
    c = -xp.expm1(-period / tau)
    return (
        period * n * (n - 1.0) / 2.0
        + n * (period / c - tau)
        + n * r
        + (r + tau * xp.expm1(-r / tau)) / c
    )


def _uniform_periodic_gate_integral(
    a: Any, b: Any, tau: Any, h_shift: Any, dt: float, period: float
) -> Any:
    """
    Steady-state periodic counterpart of :func:`_uniform_gate_integral`:
    ``[R(b') - R(b' - dt) - R(a') + R(a' - dt)] / dt`` with ``a'`` reduced to
    ``[0, period)``; a gate inside ``[dt, period]`` uses the closed form divided by
    ``1 - e^{-period/tau}``.
    """
    xp = _xp(tau)
    a1 = a - h_shift
    b1 = b - h_shift
    offset = a1 - a1 % period
    a1 = a1 - offset
    b1 = b1 - offset

    def r(v: Any) -> Any:
        return _periodic_cdf_integral(v, tau, period, xp)

    general = (r(b1) - r(b1 - dt) - r(a1) + r(a1 - dt)) / dt
    x = _positive(a1 - dt, xp)
    y = _positive(b1 - dt, xp)
    inside = (
        (tau / dt)
        * (-xp.expm1(-dt / tau))
        * (xp.exp(-x / tau) - xp.exp(-y / tau))
        / (-xp.expm1(-period / tau))
    )
    return xp.where((a1 >= dt) & (b1 <= period), inside, general)


def gate_fraction(
    a: Any,
    b: Any,
    tau: Any,
    h_shift: Any,
    dt: float,
    period: float | None = None,
    excitation: str = "bin_start",
) -> Any:
    """
    Fraction of the photons of a unit-area exponential (lifetime `tau`, onset `h_shift`)
    in the gate ``[a, b)``: single decay (``period=None``) or periodic steady state, with
    the excitation at the IRF-gate start or uniform over ``[h_shift, h_shift + dt)``.
    NumPy arrays or torch tensors.
    """
    xp = _xp(tau)
    tau_safe = xp.clamp(tau, min=_EPS) if xp is not np else np.clip(tau, _EPS, None)
    if excitation == "uniform":
        if period is None:
            return _uniform_gate_integral(a, b, tau_safe, h_shift, dt)
        return _uniform_periodic_gate_integral(a, b, tau_safe, h_shift, dt, period)
    if period is None:
        if xp is np:
            return _gate_integral(a, b, tau, h_shift)
        a_eff = xp.maximum(a, h_shift) - h_shift
        b_eff = xp.maximum(b, h_shift) - h_shift
        return xp.exp(-a_eff / tau_safe) - xp.exp(-b_eff / tau_safe)
    return periodic_gate_integral(a, b, tau_safe, h_shift, period, expm1=xp.expm1)


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
    period: float | None = None,
    excitation: str = "bin_start",
    gate_width: float | None = None,
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
        Gate spacing (and IRF bin width) in ns.
    params : Any
        ``(S, tau, v_shift)`` for ``"mono-exponential"`` or
        ``(S, a1, tau1, tau2, v_shift)`` for ``"bi-exponential"``; scalars or arrays
        broadcastable against `gate_start`.
    model_type : str
        ``"mono-exponential"`` or ``"bi-exponential"``.
    h_shift : Any
        Decay onset in ns (scalar or broadcastable array).
    period : float | None
        Laser period in ns for the steady-state periodic decay
        (:func:`~pyfli.reconstruction.common_reconstruct.periodic_gate_integral`);
        ``None`` for a single decay that is zero before the onset.
    excitation : str
        ``"bin_start"`` or ``"uniform"`` (see the module docstring).
    gate_width : float | None
        Detection-gate width in ns; ``None`` means ``dt`` (gates tile the window).

    Returns
    -------
    tuple[np.ndarray, Any]
        ``(kernel, v_shift)``: counts per gate (summing to ``S`` over all gates, or over
        gates tiling one `period`, times ``gate_width / dt``) and the constant offset,
        which is added after the IRF convolution.
    """
    validate_excitation(excitation)
    width = validate_gate_width(gate_width)
    a = np.asarray(gate_start, dtype=float)
    b = a + dt if width is None else a + width

    def fraction(tau: Any) -> np.ndarray:
        if excitation == "uniform":
            return gate_fraction(a, b, tau, h_shift, dt, period, "uniform")
        if period is None:
            return _gate_integral(a, b, tau, h_shift)
        return periodic_gate_integral(a, b, np.clip(tau, _EPS, None), h_shift, period)

    if model_type == "mono-exponential":
        S, tau, v_shift = params
        kernel = S * fraction(tau)
    else:
        S, a1, tau1, tau2, v_shift = params
        kernel = S * (a1 * fraction(tau1) + (1.0 - a1) * fraction(tau2))
    return kernel, v_shift


def _gate_width(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    return float(t[1] - t[0]) if t.size > 1 else 1.0


def validate_shift_method(
    shift_method: str, t_acq: float | None = None, t_laser: float | None = None
) -> str:
    """
    Check `shift_method` is in :data:`SHIFT_METHODS` and, for ``"circular"`` with both
    periods given, that the acquisition window equals the laser period (ns).
    """
    if shift_method not in SHIFT_METHODS:
        raise ValueError(
            f"shift_method must be one of {SHIFT_METHODS}, got {shift_method!r}"
        )
    if (
        shift_method == "circular"
        and t_acq is not None
        and t_laser is not None
        and not np.isclose(t_acq, t_laser)
    ):
        raise ValueError(
            "shift_method='circular' needs the acquisition window to equal the laser "
            f"period, got {t_acq} ns vs {t_laser} ns"
        )
    return shift_method


def _period(t: np.ndarray, shift_method: str) -> float | None:
    validate_shift_method(shift_method)
    return np.asarray(t).size * _gate_width(t) if shift_method == "circular" else None


def decay_kernel(
    t: np.ndarray,
    params: Any,
    model_type: str,
    h_shift: float = 0.0,
    shift_method: str = "zero_pad",
    excitation: str = "bin_start",
    gate_width: float | None = None,
) -> tuple:
    """Return (kernel, v_shift) on the gates starting at `t`.

    The kernel is the gate-integrated decay of :func:`gate_integrated_kernel` (zero
    before the onset `h_shift`, in ns, and summing to ``S``; periodic over the window
    ``len(t) * dt`` for ``shift_method="circular"``). Gates are ``[t_k, t_k + dt)``
    with ``dt = t[1] - t[0]`` (``[t_k, t_k + gate_width)`` when given); `excitation` as
    in :func:`gate_integrated_kernel`.
    """
    kernel, v_shift = gate_integrated_kernel(
        t,
        _gate_width(t),
        params,
        model_type,
        h_shift=h_shift,
        period=_period(t, shift_method),
        excitation=excitation,
        gate_width=gate_width,
    )
    return kernel, float(v_shift)


def negative_lag_gates(h_shift: Any, dt: float, gate_width: float | None = None) -> int:
    """
    Number of gates before ``t = 0`` the kernel must cover so that a decay with onset
    `h_shift` < 0 (earlier than the IRF), or seen by gates wider than ``dt``, is shifted,
    not truncated, by the IRF convolution: lags ``m`` with ``m dt + gate_width > h``.
    Zero for ``h_shift >= 0`` and ``gate_width <= dt``.
    """
    h = np.asarray(h_shift, dtype=float)
    h = h[np.isfinite(h)]
    earliest = float(h.min()) if h.size else 0.0
    if dt <= 0:
        return 0
    width = validate_gate_width(gate_width)
    if width is None:
        return int(np.ceil(max(-earliest, 0.0) / dt))
    return int(max(0.0, -(np.floor((earliest - width) / dt) + 1.0)))


def model_numpy(
    t: np.ndarray,
    irf: np.ndarray,
    params: Any,
    model_type: str,
    shift_method: str = "zero_pad",
    excitation: str = "bin_start",
    gate_width: float | None = None,
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
    shift_method : str
        ``"zero_pad"`` (linear convolution) or ``"circular"`` (periodic steady state;
        the window ``len(t) * dt`` must be one laser period).
    excitation : str
        ``"bin_start"`` (default) or ``"uniform"`` excitation within each IRF gate.
    gate_width : float | None
        Detection-gate width in ns (``None``: ``dt``); see the module docstring.

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
    irf = np.asarray(irf, dtype=float)
    irf_sum = irf.sum()
    irf_norm = irf / irf_sum if irf_sum > 0 else irf
    period = _period(t, shift_method)
    options = {"excitation": excitation, "gate_width": gate_width}
    if period is not None:
        kernel, v_shift = gate_integrated_kernel(
            np.arange(n) * dt, dt, kernel_params, model_type, h_shift, period, **options
        )
        return circular_convolve(kernel, irf_norm) + float(v_shift)

    n_neg = negative_lag_gates(h_shift, dt, gate_width)
    lags = np.arange(-n_neg, n) * dt
    kernel, v_shift = gate_integrated_kernel(
        lags, dt, kernel_params, model_type, h_shift=h_shift, **options
    )
    convolved = np.convolve(kernel, irf_norm, mode="full")[n_neg : n_neg + n]
    return convolved + float(v_shift)
