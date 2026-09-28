"""
Provide static initial-guess, bounds, convolution, and parameter-resolution helpers for
solvers.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLI fitting routines. Public API includes
functions :func:`resolve_params_and_bounds`, :func:`photon_amplitude_guess`,
:func:`moment_based_guess`, and :func:`rld_based_guess`.
"""

from typing import Any

import numpy as np


def resolve_params_and_bounds(
    user_p0: np.ndarray,
    user_bounds: np.ndarray,
    model_type: str,
    t: np.ndarray,
    decay: np.ndarray,
    T_laser: np.ndarray,
    guess_plugin: np.ndarray,
    T_acq: np.ndarray,
) -> tuple[Any, ...]:
    """
    Run the resolve params and bounds routine.

    Parameters
    ----------
    user_p0 : np.ndarray
        User-supplied initial parameter vector.
    user_bounds : np.ndarray
        User-supplied parameter bounds.
    model_type : str
        FLI model family, such as mono- or bi-exponential.
    t : np.ndarray
        Time axis or acquisition period used by the calculation.
    decay : np.ndarray
        Time-resolved decay signal or decay cube.
    T_laser : np.ndarray
        Laser repetition period used by the initial-guess routine.
    guess_plugin : np.ndarray
        Optional callable that supplies initial parameter guesses.
    T_acq : np.ndarray
        Acquisition window length used by the initial-guess routine.

    Returns
    -------
    tuple[Any, ...]
        Tuple containing initial parameter guesses and optimization bounds.
    """
    smart_dict = guess_plugin(t, decay, T_acq, T_laser, model_type)

    t_arr = np.asarray(t, dtype=float)
    half_gate = 0.5 * float(t_arr[1] - t_arr[0]) if t_arr.size > 1 else 0.0
    smart_dict.setdefault("h_shift", half_gate)

    if isinstance(user_p0, dict):
        smart_dict.update(user_p0)
    elif isinstance(user_p0, (list, np.ndarray)):
        keys = (
            ["amp", "tau", "v_shift", "h_shift"]
            if model_type == "mono-exponential"
            else ["amp", "alpha1", "tau1", "tau2", "v_shift", "h_shift"]
        )
        for i, val in enumerate(user_p0):
            if i < len(keys):
                smart_dict[keys[i]] = val

    if model_type == "mono-exponential":
        p0_vec = np.array(
            [
                smart_dict["amp"],
                smart_dict["tau"],
                smart_dict["v_shift"],
                smart_dict.get("h_shift", half_gate),
            ]
        )
    else:
        p0_vec = np.array(
            [
                smart_dict["amp"],
                smart_dict["alpha1"],
                smart_dict["tau1"],
                smart_dict["tau2"],
                smart_dict["v_shift"],
                smart_dict.get("h_shift", half_gate),
            ]
        )

    n_params = len(p0_vec)

    # h_shift is now in ns (same units as t), so bound by T_acq/4
    shift_bound = T_acq / 4.0

    low_vec = np.zeros(n_params)
    high_vec = np.full(n_params, np.inf)

    if model_type == "bi-exponential":
        low_vec[1], high_vec[1] = 0.0, 1.0
        low_vec[2], high_vec[2] = 1e-4, T_laser
        low_vec[3], high_vec[3] = max(float(p0_vec[2]), 1e-4), T_laser
        low_vec[5], high_vec[5] = -shift_bound, shift_bound
    else:
        low_vec[1], high_vec[1] = 1e-4, T_laser
        low_vec[3], high_vec[3] = -shift_bound, shift_bound

    if isinstance(user_bounds, dict):
        key_map = (
            {
                "amp": 0,
                "tau": 1,
                "v_shift": 2,
                "h_shift": 3,
            }
            if model_type == "mono-exponential"
            else {
                "amp": 0,
                "alpha1": 1,
                "tau1": 2,
                "tau2": 3,
                "v_shift": 4,
                "h_shift": 5,
            }
        )
        for k, v in user_bounds.items():
            if k in key_map:
                low_vec[key_map[k]], high_vec[key_map[k]] = v
    elif isinstance(user_bounds, (list, np.ndarray)):
        for i, b in enumerate(user_bounds):
            if b is not None and i < n_params:
                low_vec[i], high_vec[i] = b

    high_vec = np.maximum(high_vec, low_vec + 1e-6)

    p0_safe = np.clip(p0_vec, low_vec + 1e-7, high_vec - 1e-7)

    return p0_safe, (low_vec, high_vec)


def photon_amplitude_guess(clean_decay: np.ndarray, tau: float, T_acq: float) -> float:
    """
    Initial guess of the forward model's amplitude ``S`` -- the total photon count of
    the decay: the background-subtracted counts in the window, divided by the
    fraction ``1 - exp(-T_acq / tau)`` of a decay with lifetime `tau` that falls
    inside the acquisition window.
    """
    inside = -np.expm1(-T_acq / tau) if tau > 0 else 1.0
    return float(np.sum(clean_decay) / max(inside, 1e-12))


def moment_based_guess(
    t: np.ndarray,
    decay: np.ndarray,
    T_acq: np.ndarray,
    T_laser: np.ndarray,
    model_type: str = "mono-exponential",
) -> dict[Any, Any]:
    """
    Run the moment based guess routine.

    Parameters
    ----------
    t : np.ndarray
        Time axis or acquisition period used by the calculation.
    decay : np.ndarray
        Time-resolved decay signal or decay cube.
    T_acq : np.ndarray
        Acquisition window length used by the initial-guess routine.
    T_laser : np.ndarray
        Laser repetition period used by the initial-guess routine.
    model_type : str
        FLI model family, such as mono- or bi-exponential.

    Returns
    -------
    dict[Any, Any]
        Dictionary containing the data produced by moment based guess.
    """
    offset_guess = np.percentile(decay, 5)
    clean_d = np.clip(decay - offset_guess, 1e-6, None)

    idx_max = np.argmax(clean_d)
    t_decay = t[idx_max:] - t[idx_max]
    d_decay = clean_d[idx_max:]

    m0 = np.trapezoid(d_decay, t_decay)
    if m0 > 0:
        m1 = np.trapezoid(t_decay * d_decay, t_decay)
        tau_mean = m1 / m0
    else:
        tau_mean = T_laser / 10.0

    tau_g = np.clip(tau_mean, 0.05, T_laser * 0.8)

    s_guess = photon_amplitude_guess(clean_d, tau_g, T_acq)

    if model_type == "mono-exponential":
        return {
            "amp": float(s_guess),
            "tau": float(tau_g),
            "v_shift": float(offset_guess),
        }
    else:
        if len(d_decay) > 2:
            half = max(len(d_decay) // 2, 1)
            a_early = float(np.trapezoid(d_decay[:half], t_decay[:half])) + 1e-9
            a_late = float(np.trapezoid(d_decay[half:], t_decay[half:])) + 1e-9
            alpha1_guess = float(np.clip(a_early / (a_early + a_late), 0.001, 0.999))
        else:
            alpha1_guess = 0.5
        return {
            "amp": float(s_guess),
            "alpha1": alpha1_guess,
            "tau1": float(tau_g * 0.5),
            "tau2": float(tau_g * 1.5),
            "v_shift": float(offset_guess),
        }


def rld_based_guess(
    t: np.ndarray,
    decay: np.ndarray,
    T_acq: np.ndarray,
    T_laser: np.ndarray,
    model_type: str = "mono-exponential",
) -> dict[Any, Any]:
    """
    Run the RLD based guess routine.

    Parameters
    ----------
    t : np.ndarray
        Time axis or acquisition period used by the calculation.
    decay : np.ndarray
        Time-resolved decay signal or decay cube.
    T_acq : np.ndarray
        Acquisition window length used by the initial-guess routine.
    T_laser : np.ndarray
        Laser repetition period used by the initial-guess routine.
    model_type : str
        FLI model family, such as mono- or bi-exponential.

    Returns
    -------
    dict[Any, Any]
        Dictionary containing the data produced by RLD based guess.
    """
    offset_guess = np.percentile(decay, 5)
    clean_d = np.clip(decay - offset_guess, 1e-6, None)

    idx_max = np.argmax(clean_d)
    t_fit = t[idx_max:] - t[idx_max]
    y_fit = clean_d[idx_max:]

    num_bins = len(y_fit)
    dt = t_fit[1] - t_fit[0] if num_bins > 1 else 1.0

    if model_type == "mono-exponential":
        mid = num_bins // 2
        a0 = np.sum(y_fit[:mid])
        a1 = np.sum(y_fit[mid : 2 * mid])

        tau_g = (dt * mid) / np.log(a0 / a1) if (a1 > 0 and a0 > a1) else T_laser / 10.0
        tau_g = np.clip(tau_g, 0.05, T_laser * 0.8)

        return {
            "amp": photon_amplitude_guess(clean_d, tau_g, T_acq),
            "tau": float(tau_g),
            "v_shift": float(offset_guess),
        }
    else:
        q = num_bins // 4
        a0, a1 = np.sum(y_fit[:q]), np.sum(y_fit[q : 2 * q])
        a2, a3 = np.sum(y_fit[2 * q : 3 * q]), np.sum(y_fit[3 * q : 4 * q])

        t1 = (dt * q) / np.log(a0 / a1) if (a1 > 0 and a0 > a1) else T_laser * 0.05
        t2 = (dt * q) / np.log(a2 / a3) if (a3 > 0 and a2 > a3) else T_laser * 0.15

        tau1_g = np.clip(min(t1, t2), 0.05, T_laser * 0.4)
        tau2_g = np.clip(max(t1, t2), tau1_g * 1.1, T_laser * 0.8)

        a_early, a_late = a0 + a1 + 1e-9, a2 + a3 + 1e-9
        alpha1_guess = float(np.clip(a_early / (a_early + a_late), 0.001, 0.999))

        return {
            "amp": photon_amplitude_guess(clean_d, tau2_g, T_acq),
            "alpha1": alpha1_guess,
            "tau1": float(tau1_g),
            "tau2": float(tau2_g),
            "v_shift": float(offset_guess),
        }
