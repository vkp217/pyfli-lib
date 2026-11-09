"""
Implement Laguerre-basis deconvolution for FLI decay reconstruction.

This module belongs to :mod:`pyfli.laguerre` and is part of PyFLI's Laguerre-basis
deconvolution and fitting method. Public API includes classes :class:`LaguerreFLI`.
"""

import os
from typing import Any

import h5py
import numpy as np
from scipy.optimize import least_squares, minimize_scalar, nnls
from scipy.signal import fftconvolve, lfilter
from tqdm.auto import tqdm

from pyfli import logging
from pyfli.reconstruction.common_reconstruct import (
    NLSF_WEIGHTINGS,
    circular_convolve,
    fourier_shift,
    irls,
    residual_variance,
    weighted_linear_lstsq,
)

from ..cm_funcs.forward_model import (
    gate_integrated_kernel,
    negative_lag_gates,
    validate_excitation,
    validate_gate_width,
    validate_shift_method,
)
from ..cm_stats.chi_square import degrees_of_freedom
from ..cm_stats.goodness_of_fit import r_squared
from ..cm_stats.shared_metrics import pearson_chi_square, reduced_poisson_deviance
from ..solver.base_static import moment_based_guess


class LaguerreFLI:
    """
    Run the laguerre FLI routine.
    Laguerre basis, projects decays into coefficient space, reconstructs denoised
    decays, and supports lifetime estimation from the reconstructed signal.

    Parameters
    ----------
    n_components : int
        Number of exponential lifetime components to fit.
    n_laguerre : Optional[int]
        Number of Laguerre basis functions used for reconstruction.
    alpha : float
        Regularization strength or statistical threshold value, depending on context.
    dt : float
        Sampling interval between adjacent decay bins.
    auto_alpha : bool
        If ``True``, estimate the Laguerre alpha parameter from the data.
    taus_init : Optional[np.ndarray]
        Initial lifetime estimates used to seed exponential fitting.
    laser_period_ns : Optional[float]
        Laser repetition period in nanoseconds.
    reg_strength : float
        Regularization weight applied to higher-order Laguerre coefficients.
    reg_power : float
        Exponent controlling how regularization increases across coefficient order.
    nonneg : bool
        If ``True``, constrain fitted coefficients or amplitudes to be non-negative.
    verbose : bool
        If ``True``, report progress and diagnostic messages during processing.
    shift_method : str
        IRF convolution, as in :func:`pyfli.cm_funcs.forward_model.model_numpy`:
        ``"zero_pad"`` (linear) or ``"circular"`` (periodic steady state; the window
        ``T * dt`` must be one laser period).
    fit_background : bool
        If ``True``, fit a constant per-gate background (the solver's ``v_shift``). With
        ``shift_method="zero_pad"`` it is an extra linear term next to the Laguerre
        coefficients; with ``"circular"`` the circularly convolved Laguerre functions span
        a constant, so it is fitted in the exponential stage instead (a constant ``b`` in
        the deconvolved kernel is the same ``b`` in the data, the IRF summing to 1).
    weighting : str
        Weights of the coefficient least squares, as the solver's NLSF
        (:func:`~pyfli.reconstruction.common_reconstruct.residual_variance`):
        ``"irls"`` (Poisson, iteratively reweighted;
        :func:`~pyfli.reconstruction.common_reconstruct.irls`), ``"none"`` or
        ``"neyman"``.
    variance_floor, irls_max_rounds, irls_tol : float, int, float
        IRLS variance floor (counts), maximum rounds and per-pixel relative tolerance.
    excitation : str
        Excitation within each IRF gate, ``"bin_start"`` (default) or ``"uniform"``, as
        in :func:`pyfli.cm_funcs.forward_model.model_numpy`.
    gate_width : float | None
        Detection-gate width in ns (``None``: `dt`); only for an IRF not recorded
        through the same gate.

    Notes
    -----
    Each pixel's exponential model is the solver's gate-integrated kernel
    (:func:`pyfli.cm_funcs.forward_model.gate_integrated_kernel`, periodic for
    ``"circular"``), ``h[n] = sum_i (S a_i) g_n(tau_i, onset)``, so the amplitudes are
    photons and ``photon_count_map``, the ``alpha`` maps (photon fractions) and
    ``tau_mean_map`` (photon-weighted) follow the solver's definitions; the onset is the
    fractional part of ``irf_shift`` (see :meth:`fit`). ``chi2_map``,
    ``reduced_chi2_map``, ``R2_map`` and ``fit_map`` describe that exponential model
    convolved with the IRF (``n_params = 2N``, plus 1 with `fit_background`);
    ``laguerre_*`` maps describe the Laguerre expansion fit (``n_params`` = the number of
    basis functions, plus 1 with `fit_background` for ``"zero_pad"``); both add 1 for a
    per-pixel ``irf_shift`` map (see :meth:`fit`).
    """

    def __init__(
        self,
        n_components: int = 2,
        n_laguerre: int | None = None,
        alpha: float = 0.85,
        dt: float = 1.0,
        auto_alpha: bool = False,
        taus_init: np.ndarray | None = None,
        laser_period_ns: float | None = None,
        reg_strength: float = 0.0,
        reg_power: float = 2.0,
        nonneg: bool = True,
        verbose: bool = True,
        shift_method: str = "zero_pad",
        fit_background: bool = False,
        weighting: str = "irls",
        variance_floor: float = 1.0,
        irls_max_rounds: int = 20,
        irls_tol: float = 1e-6,
        excitation: str = "bin_start",
        gate_width: float | None = None,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie strictly in (0, 1).")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if laser_period_ns is not None and laser_period_ns <= 0:
            raise ValueError("laser_period_ns must be positive.")

        self.n_components = int(n_components)
        self.n_laguerre = (
            int(n_laguerre) if n_laguerre is not None else max(4, 2 * n_components)
        )
        if self.n_laguerre < self.n_components:
            raise ValueError("n_laguerre must be >= n_components.")

        self.alpha = float(alpha)
        self.dt = float(dt)
        self.auto_alpha = bool(auto_alpha)
        self.laser_period_ns = (
            float(laser_period_ns) if laser_period_ns is not None else None
        )
        self.taus_init = np.asarray(taus_init, float) if taus_init is not None else None
        self.reg_strength = float(reg_strength)
        self.reg_power = float(reg_power)
        self.nonneg = bool(nonneg)
        self.verbose = bool(verbose)
        self.shift_method = validate_shift_method(shift_method)
        self.excitation = validate_excitation(excitation)
        self.gate_width = validate_gate_width(gate_width)
        self.fit_background = bool(fit_background)
        if weighting not in NLSF_WEIGHTINGS:
            raise ValueError(
                f"weighting must be one of {NLSF_WEIGHTINGS}, got {weighting!r}"
            )
        self.weighting = weighting
        self.variance_floor = float(variance_floor)
        self.irls_max_rounds = int(irls_max_rounds)
        self.irls_tol = float(irls_tol)
        self._bg_column = self.fit_background and self.shift_method == "zero_pad"
        self._bg_in_exp = self.fit_background and self.shift_method == "circular"

        self.basis_: np.ndarray | None = None
        self.V_: np.ndarray | None = None
        self.coeffs_: np.ndarray | None = None
        self.taus_: np.ndarray | None = None
        self.n_unique_irf_: int | None = None
        self.amplitudes_: np.ndarray | None = None
        self.fractions_: np.ndarray | None = None
        self.tau_mean_: np.ndarray | None = None
        self.converged_: np.ndarray | None = None
        self.reconstructed_: np.ndarray | None = None
        self.residuals_: np.ndarray | None = None
        self.fit_curve_: np.ndarray | None = None
        self.residual_curve_: np.ndarray | None = None
        self.decay_: np.ndarray | None = None
        self.irf_lead_: int = 0
        self.onset_: np.ndarray = np.zeros((1, 1))
        self.background_: np.ndarray | None = None
        self.photons_: np.ndarray | None = None
        self.photon_fractions_: np.ndarray | None = None
        self.exp_fit_curve_: np.ndarray | None = None

    @staticmethod
    def _discrete_laguerre_basis(T: int, alpha: float, L: int) -> np.ndarray:
        """
        Build the discrete Laguerre basis matrix.

        Parameters
        ----------
        T : int
            Time axis or acquisition period used by the calculation.
        alpha : float
            Regularization strength, fraction value, or significance threshold used by the
        routine.
        L : int
            Number of Laguerre basis functions or coefficient dimension.

        Returns
        -------
        np.ndarray
            Discrete Laguerre basis matrix with basis functions along rows.
        """
        b = np.zeros((L, T), dtype=np.float64)
        n = np.arange(T)
        b[0] = np.sqrt(1.0 - alpha) * alpha ** (n / 2.0)
        sa = np.sqrt(alpha)
        a_coef = [1.0, -sa]
        for j in range(1, L):
            prev = b[j - 1]
            shifted = np.empty_like(prev)
            shifted[0] = 0.0
            shifted[1:] = prev[:-1]
            u = sa * prev - shifted
            b[j] = lfilter([1.0], a_coef, u)
        return b

    @staticmethod
    def _convolve_with_irf(
        basis: np.ndarray, irf: np.ndarray, circular: bool = False, lead: int = 0
    ) -> np.ndarray:
        """
        Convolve each Laguerre basis function with the IRF.

        Parameters
        ----------
        basis : np.ndarray
            Laguerre basis matrix before or after IRF convolution.
        irf : np.ndarray
            Instrument response function aligned with the decay signal.
        circular : bool
            Circular instead of linear (truncated) convolution.
        lead : int
            Linear convolution only: index of time 0 in `irf`, which may extend
            ``lead`` gates before it (an IRF advanced by a negative shift).

        Returns
        -------
        np.ndarray
            Laguerre design matrix after convolution with the IRF.
        """
        _, T = basis.shape
        irf = np.asarray(irf, float).ravel()
        s = irf.sum()
        if s > 0:
            irf = irf / s
        if circular:
            return circular_convolve(basis, irf[None, :]).T
        full = fftconvolve(basis, irf[None, :], mode="full", axes=1)
        return full[:, lead : lead + T].T

    def _design(self, basis: np.ndarray, irf: np.ndarray) -> np.ndarray:
        """IRF-convolved Laguerre design matrix ``(T, L)``, plus a constant column with `fit_background`."""
        V = self._convolve_with_irf(
            basis, irf, self.shift_method == "circular", self.irf_lead_
        )
        if self._bg_column:
            V = np.hstack([V, np.ones((V.shape[0], 1))])
        return V

    def _design_batch(self, irf_2d: np.ndarray) -> np.ndarray:
        """Per-pixel designs ``(P, T, K)`` for IRFs ``(P, T')``, as :meth:`_design` row by row."""
        T = self.basis_.shape[-1]
        lead = self.irf_lead_
        s = irf_2d.sum(axis=-1, keepdims=True)
        irf = np.divide(irf_2d, s, out=irf_2d.astype(float).copy(), where=s > 0)
        basis = self.basis_[None]
        if self.shift_method == "circular":
            conv = circular_convolve(basis, irf[:, None, :])
        else:
            conv = fftconvolve(basis, irf[:, None, :], mode="full", axes=-1)
            conv = conv[..., lead : lead + T]
        V = np.swapaxes(conv, -1, -2)
        if self._bg_column:
            V = np.concatenate([V, np.ones((*V.shape[:-1], 1))], axis=-1)
        return V

    def _convolve_stack(self, kernel: np.ndarray, irf: np.ndarray) -> np.ndarray:
        """Convolve ``(X, Y, T)`` kernels with a ``(T',)`` or ``(X, Y, T')`` IRF, normalised per pixel."""
        T = kernel.shape[-1]
        lead = self.irf_lead_
        irf = np.asarray(irf, float)
        irf = irf[None, None, :] if irf.ndim == 1 else irf
        s = irf.sum(-1, keepdims=True)
        irf = np.divide(irf, s, out=np.zeros_like(irf), where=s > 0)
        if self.shift_method == "circular":
            return circular_convolve(kernel, irf)
        return fftconvolve(kernel, irf, mode="full", axes=-1)[..., lead : lead + T]

    @staticmethod
    def _unique_irf_groups(irf_2d: np.ndarray, decimals: int = 6) -> tuple[Any, ...]:
        """
        Group pixels that share numerically identical IRFs.

        Parameters
        ----------
        irf_2d : np.ndarray
            Two-dimensional array of per-pixel IRFs, flattened over pixels by time.
        decimals : int
            Decimal precision used when grouping IRFs.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing the IRF group index for each pixel and representative group
            indices.
        """
        _, _ = irf_2d.shape
        s = irf_2d.sum(axis=1, keepdims=True)
        norm = np.divide(irf_2d, s, out=np.zeros_like(irf_2d), where=s > 0)
        keys = np.round(norm, decimals)
        _, first_idx, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        inverse = inverse.ravel()
        rep_idx = first_idx.tolist()
        return inverse, rep_idx

    def _penalty(self, L: int) -> np.ndarray:
        """
        Build the coefficient regularization penalty matrix.

        Parameters
        ----------
        L : int
            Number of Laguerre basis functions or coefficient dimension.

        Returns
        -------
        np.ndarray
            Regularization weights for the Laguerre coefficients.
        """
        return (np.arange(L, dtype=float) + 1.0) ** self.reg_power

    def _solve_coefficients(self, V: np.ndarray, Y2d: np.ndarray) -> np.ndarray:
        """
        Solve Laguerre coefficients for all valid decays.

        Parameters
        ----------
        V : np.ndarray
            Design ``(T, K)`` shared by all decays, or ``(P, T, K)`` per decay.
        Y2d : np.ndarray
            Decays ``(T, P)``.

        Returns
        -------
        np.ndarray
            Coefficients ``(K, P)``, from
            :func:`~pyfli.reconstruction.common_reconstruct.weighted_linear_lstsq`
            with :attr:`weighting` (IRLS converged per decay).
        """
        d = np.asarray(Y2d, dtype=float).T
        penalty = None
        if self.reg_strength > 0.0:
            penalty = self._penalty(self.n_laguerre)
            if self.basis_.shape[0] > self.n_laguerre:
                penalty = np.append(penalty, 0.0)
            if self._bg_column:
                penalty = np.append(penalty, 0.0)

        def solve(_: Any, variance: np.ndarray | None) -> tuple[np.ndarray, None]:
            return (
                weighted_linear_lstsq(V, d, variance, penalty, self.reg_strength),
                None,
            )

        def predict(C: np.ndarray) -> np.ndarray:
            return C @ V.T if V.ndim == 2 else np.einsum("ptk,pk->pt", V, C)

        def row_change(old: np.ndarray, new: np.ndarray) -> np.ndarray:
            scale = np.maximum(np.max(np.abs(old), axis=-1), 1e-12)
            return np.max(np.abs(new - old), axis=-1) / scale

        start = (
            residual_variance(None, d, "neyman") if self.weighting == "neyman" else None
        )
        C, _ = solve(None, start)
        if self.weighting == "irls":
            C, _ = irls(
                solve,
                predict,
                C,
                self.variance_floor,
                self.irls_max_rounds,
                self.irls_tol,
                change=row_change,
            )
        return C.T

    def _optimize_alpha(
        self, avg_decay: np.ndarray, avg_irf: np.ndarray, T: int
    ) -> float:
        """
        Optimize the Laguerre alpha value against an average decay.

        Parameters
        ----------
        avg_decay : np.ndarray
            Average decay trace used during alpha optimization.
        avg_irf : np.ndarray
            Average IRF used as the representative response for global fitting.
        T : int
            Time axis or acquisition period used by the calculation.

        Returns
        -------
        float
            Floating-point result computed by optimize alpha.
        """

        def obj(a: np.ndarray) -> float:
            """
            Run the obj routine.

            Parameters
            ----------
            a : np.ndarray
                Lower integration or interval bound.

            Returns
            -------
            float
                Floating-point result computed by obj.
            """
            if not (1e-3 < a < 0.999):
                return 1e30
            B = self._discrete_laguerre_basis(T, float(a), self.n_laguerre)
            V = self._design(B, avg_irf)
            c, *_ = np.linalg.lstsq(V, avg_decay, rcond=None)
            return float(((V @ c - avg_decay) ** 2).sum())

        res = minimize_scalar(
            obj, bounds=(0.05, 0.98), method="bounded", options={"xatol": 1e-3}
        )
        return float(res.x)

    @staticmethod
    def _nnls_safe(E: np.ndarray, h: np.ndarray, maxiter: int) -> np.ndarray:
        """
        Solve a non-negative least-squares problem with a fallback path.

        Parameters
        ----------
        E : np.ndarray
            GUI or plotting event object supplied by the framework.
        h : np.ndarray
            IRF, image height, or temporal kernel used by the routine.
        maxiter : int
            Maximum number of optimization iterations.

        Returns
        -------
        np.ndarray
            Non-negative least-squares solution, with zeros when fitting fails.
        """
        try:
            a, _ = nnls(E, h, maxiter=maxiter)
            return a
        except RuntimeError:
            a, *_ = np.linalg.lstsq(E, h, rcond=None)
            return np.clip(a, 0.0, None)

    def _solve_amps(self, E: np.ndarray, h: np.ndarray, maxiter: int) -> np.ndarray:
        """
        Estimate exponential amplitudes for a fixed lifetime set.

        Parameters
        ----------
        E : np.ndarray
            GUI or plotting event object supplied by the framework.
        h : np.ndarray
            IRF, image height, or temporal kernel used by the routine.
        maxiter : int
            Maximum number of optimization iterations.

        Returns
        -------
        np.ndarray
            Estimated exponential amplitudes for the supplied lifetimes.
        """
        if self.nonneg:
            return self._nnls_safe(E, h, maxiter)
        a, *_ = np.linalg.lstsq(E, h, rcond=None)
        return a

    def _tau_bounds(self, T: int) -> tuple[Any, ...]:
        """
        Build lifetime bounds for exponential fitting.

        Parameters
        ----------
        T : int
            Time axis or acquisition period used by the calculation.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing lower and upper lifetime bounds.
        """
        tau_lo = self.dt
        tau_hi = (
            self.laser_period_ns if self.laser_period_ns is not None else T * self.dt
        )
        tau_hi = max(tau_hi, tau_lo + 1e-6)
        return tau_lo, tau_hi

    def _safe_tau0(self, tau0: np.ndarray, tau_lo: float, tau_hi: float) -> np.ndarray:
        """
        Clip an initial lifetime guess into valid bounds.

        Parameters
        ----------
        tau0 : np.ndarray
            Initial lifetime guess before clipping to valid bounds.
        tau_lo : float
            Lower lifetime bound.
        tau_hi : float
            Upper lifetime bound.

        Returns
        -------
        np.ndarray
            Initial lifetime estimates clipped to the allowed tau bounds.
        """
        return np.clip(tau0, tau_lo + 1e-7, tau_hi - 1e-7)

    def _exp_basis(self, T: int, taus: np.ndarray, onset: Any) -> np.ndarray:
        """
        Unit-photon gate-integrated exponentials ``(..., N, T)`` with lifetimes `taus`
        ``(..., N)`` and onset `onset` (ns, broadcast against ``(...)``), as the solver's
        :func:`~pyfli.cm_funcs.forward_model.gate_integrated_kernel`.
        """
        period = T * self.dt if self.shift_method == "circular" else None
        kernel, _ = gate_integrated_kernel(
            np.arange(T) * self.dt,
            self.dt,
            (1.0, np.asarray(taus, dtype=float)[..., None], 0.0),
            "mono-exponential",
            h_shift=np.asarray(onset, dtype=float)[..., None, None],
            period=period,
            excitation=self.excitation,
            gate_width=self.gate_width,
        )
        return kernel

    def _spike_gates(self, T: int) -> list[int]:
        """
        Gates where the exponential kernel is not geometric in the gate index, each given
        a unit spike in the Laguerre basis: gates starting before the end of the
        excitation window (``onset`` plus ``dt`` for uniform excitation) and, for
        ``"circular"``, gates reaching the next pulse's onset.
        """
        width = self.dt if self.gate_width is None else self.gate_width
        reach = float(np.max(self.onset_)) + (
            self.dt if self.excitation == "uniform" else 0.0
        )
        gates = {m for m in range(T) if m * self.dt < reach}
        if self.shift_method == "circular" and width > self.dt:
            nxt = T * self.dt + float(np.min(self.onset_))
            gates |= {m for m in range(T) if m * self.dt + width > nxt}
        return sorted(gates)

    def _exp_design(self, T: int, taus: np.ndarray, onset: Any) -> np.ndarray:
        """``(T, N)`` exponential design of :meth:`_exp_basis`, plus a constant column when the background is fitted in the exponential stage."""
        E = self._exp_basis(T, taus, onset).T
        if self._bg_in_exp:
            E = np.hstack([E, np.ones((T, 1))])
        return E

    def _estimate_global_taus(self, h_avg: np.ndarray) -> np.ndarray:
        """
        Estimate global taus.

        Parameters
        ----------
        h_avg : np.ndarray
            Average reconstructed decay used to estimate global lifetimes.

        Returns
        -------
        np.ndarray
            Estimated global lifetime values shared across pixels.
        """
        T = h_avg.shape[0]
        N = self.n_components
        tau_lo, tau_hi = self._tau_bounds(T)

        if self.taus_init is not None and self.taus_init.size == N:
            tau0 = self._safe_tau0(self.taus_init.astype(float), tau_lo, tau_hi)
        else:
            T_acq = T * self.dt
            T_laser = (
                self.laser_period_ns if self.laser_period_ns is not None else T_acq
            )
            t_axis = np.arange(T, dtype=float) * self.dt
            model_str = "mono-exponential" if N == 1 else "bi-exponential"
            guess = moment_based_guess(t_axis, h_avg, T_acq, T_laser, model_str)

            if N == 1:
                tau0 = np.array([guess["tau"]])
            elif N == 2:
                tau0 = np.array([guess["tau1"], guess["tau2"]])
            else:
                tau_start = guess.get("tau1", 0.05 * T_acq)
                tau_end = guess.get("tau2", 0.5 * T_acq)
                tau0 = np.geomspace(
                    max(tau_start, tau_lo), min(tau_end, tau_hi * 0.9), N
                )
            tau0 = self._safe_tau0(tau0, tau_lo, tau_hi)

        def residual(params: Any) -> Any:
            """
            Run the residual routine.

            Parameters
            ----------
            params : Any
                Model, detector, or plotting parameters used by the routine.

            Returns
            -------
            Any
                Object produced by residual.
            """
            E = self._exp_design(T, params, np.median(self.onset_))
            a = self._solve_amps(E, h_avg, 200 * N)
            return E @ a - h_avg

        res = least_squares(
            residual,
            tau0,
            method="trf",
            bounds=([tau_lo] * N, [tau_hi] * N),
            max_nfev=2000,
        )
        return np.sort(np.clip(np.abs(res.x), tau_lo, tau_hi))

    def _fit_pixel_exponentials(
        self,
        h_stack: np.ndarray,
        tau_init: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit pixel exponentials.

        Parameters
        ----------
        h_stack : np.ndarray
            Stack of reconstructed decays fitted pixel by pixel.
        tau_init : np.ndarray
            Initial lifetime vector for pixel-wise exponential fitting.
        mask : Optional[np.ndarray]
            Boolean or labeled mask selecting pixels for the operation.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Per-pixel fitted amplitudes, fractions, lifetimes, and convergence flags.
        """
        X, Y, T = h_stack.shape
        N = self.n_components
        tau_lo, tau_hi = self._tau_bounds(T)

        tau_init_safe = self._safe_tau0(tau_init, tau_lo, tau_hi)
        bounds = ([tau_lo] * N, [tau_hi] * N)

        taus_map = np.zeros((X, Y, N), dtype=np.float64)
        amps_map = np.zeros((X, Y, N + int(self._bg_in_exp)), dtype=np.float64)
        converged_map = np.zeros((X, Y), dtype=np.float32)

        total_px = int(mask.sum()) if mask is not None else X * Y
        with tqdm(
            total=total_px,
            desc="  Pixels",
            unit="px",
            leave=False,
            disable=not self.verbose,
        ) as pbar:
            for x in range(X):
                for y in range(Y):
                    if mask is not None and not mask[x, y]:
                        continue
                    h = h_stack[x, y, :]
                    if h.sum() >= 1e-10:
                        onset = self.onset_[x, y]

                        def residual(
                            params: Any, h: np.ndarray = h, onset: float = onset
                        ) -> Any:
                            E = self._exp_design(T, params, onset)
                            a = self._solve_amps(E, h, 200 * N)
                            return E @ a - h

                        try:
                            res = least_squares(
                                residual,
                                tau_init_safe.copy(),
                                method="trf",
                                bounds=bounds,
                                max_nfev=1000,
                            )
                            taus_px = np.sort(np.clip(np.abs(res.x), tau_lo, tau_hi))
                            converged_map[x, y] = float(res.success)
                        except Exception:
                            taus_px = tau_init_safe.copy()
                            converged_map[x, y] = 0.0

                        E_px = self._exp_design(T, taus_px, onset)
                        taus_map[x, y, :] = taus_px
                        amps_map[x, y, :] = self._solve_amps(E_px, h, 200 * N)
                    pbar.update(1)

        return taus_map, amps_map, converged_map

    def fit(
        self,
        decay: np.ndarray,
        irf: np.ndarray,
        mask: np.ndarray | None = None,
        irf_shift: float | np.ndarray | None = None,
        h_shift: float | np.ndarray | None = None,
    ) -> "LaguerreFLI":
        """
        Fit the model to decay and IRF data.

        Parameters
        ----------
        decay : np.ndarray
            Time-resolved decay signal or decay cube.
        irf : np.ndarray
            Instrument response function aligned with the decay signal.
        mask : Optional[np.ndarray]
            Boolean or labeled mask selecting pixels for the operation.
        irf_shift : float | np.ndarray | None
            IRF delay in gates, scalar or ``(X, Y)``, e.g. from
            :meth:`pyfli.data_cc.irfAligner.IRFAligner.estimate_shift_debiased` or
            :meth:`~pyfli.data_cc.irfAligner.IRFAligner.estimate_shift_rmse` (positive
            when the decay starts later). The IRF is delayed by it
            by its whole gates (floor) and the remaining fraction ``f`` in ``[0, 1)``
            is the onset of the exponential kernel, which together are exactly the
            solver's onset ``h_shift = irf_shift * dt``, reported as ``h_shift_map``.
            With ``f > 0`` the kernel is geometric from gate 1 and differs only in gate
            0, so a unit spike at gate 0 is added to the Laguerre basis (one more
            expansion coefficient) to represent it exactly. A per-pixel map counts as
            one fitted parameter.
        h_shift : float | np.ndarray | None
            The same offset in the solver's convention, in ns (scalar or ``(X, Y)``):
            ``HarmonicOffset.estimate()["h_shift_map"]``
            (:mod:`pyfli.phasor.harmonic_offset`), a solver fit's ``h_shift_map``, or
            any other estimate; used as ``irf_shift = h_shift / dt``. Give at most one
            of `irf_shift` and `h_shift`.

        Returns
        -------
        'LaguerreFLI'
            Object produced by fit.
        """
        decay = np.asarray(decay, dtype=np.float64)
        irf = np.asarray(irf, dtype=np.float64)

        if decay.ndim == 1:
            decay = decay[None, None, :]
        if decay.ndim != 3:
            raise ValueError("decay must have shape (X, Y, T) or (T,).")
        X, Y, T = decay.shape
        self.decay_ = decay.astype(np.float32)
        if (
            irf.shape != decay.shape
            if irf.ndim == 3
            else not (irf.ndim == 1 and irf.shape[0] == T)
        ):
            raise ValueError("irf must be (T,) or match the decay shape (X, Y, T).")
        self.h_shift_ = np.zeros((X, Y))
        self.n_shift_params_ = 0
        self.irf_lead_ = 0
        self.onset_ = np.zeros((X, Y))
        if h_shift is not None:
            if irf_shift is not None:
                raise ValueError("give at most one of irf_shift and h_shift.")
            irf_shift = np.asarray(h_shift, dtype=float) / self.dt
        wide_lags = (
            negative_lag_gates(0.0, self.dt, self.gate_width)
            if self.shift_method == "zero_pad"
            else 0
        )
        if wide_lags and irf_shift is None:
            irf_shift = 0.0
        if irf_shift is not None:
            shift = np.asarray(irf_shift, dtype=float)
            if shift.ndim not in (0, 2) or (shift.ndim == 2 and shift.shape != (X, Y)):
                raise ValueError(f"irf_shift must be a scalar or ({X}, {Y}) map.")
            if shift.ndim == 2 and irf.ndim == 1:
                irf = np.broadcast_to(irf, (X, Y, T))
            fraction = shift - np.floor(shift)
            whole = np.floor(shift) - wide_lags
            if self.shift_method == "zero_pad":
                self.irf_lead_ = int(max(-whole.min(), 0.0))
                trail = int(max(whole.max(), 0.0))
                pad = [(0, 0)] * (irf.ndim - 1) + [(self.irf_lead_, trail)]
                irf = np.pad(irf, pad)
            irf = fourier_shift(irf, whole)
            self.onset_ = np.broadcast_to(
                (fraction + wide_lags) * self.dt, (X, Y)
            ).astype(float)
            self.h_shift_ = np.broadcast_to(shift * self.dt, (X, Y)).astype(float)
            self.n_shift_params_ = int(shift.ndim == 2)
        self.irf_ = irf
        if (
            self.shift_method == "circular"
            and self.laser_period_ns is not None
            and not np.isclose(T * self.dt, self.laser_period_ns)
        ):
            raise ValueError(
                "shift_method='circular' needs T * dt to equal laser_period_ns, got "
                f"{T * self.dt} ns vs {self.laser_period_ns} ns"
            )

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (X, Y):
                raise ValueError(
                    f"mask shape {mask.shape} must match image shape ({X}, {Y})."
                )
        self.mask_ = mask

        ppirf = irf.ndim == 3

        decay_flat = decay.reshape(-1, T)
        if mask is not None:
            avg_decay = decay_flat[mask.ravel()].mean(0)
        else:
            avg_decay = decay_flat.mean(0)

        if ppirf:
            irf_2d = irf.reshape(-1, irf.shape[-1])
            _, rep_idx = self._unique_irf_groups(irf_2d)
            self.n_unique_irf_ = len(rep_idx)
        else:
            self.n_unique_irf_ = 1

        single_irf = (not ppirf) or self.n_unique_irf_ == 1
        if not ppirf:
            alpha_irf = irf
        elif self.n_unique_irf_ == 1:
            alpha_irf = irf_2d[rep_idx[0]]
        else:
            alpha_irf = irf_2d.mean(0)

        with tqdm(
            total=5, desc="LaguerreFLI", unit="stage", disable=not self.verbose
        ) as pbar:
            pbar.set_description("Building Laguerre basis")
            if self.auto_alpha:
                self.alpha = self._optimize_alpha(avg_decay, alpha_irf, T)
            self.basis_ = self._discrete_laguerre_basis(T, self.alpha, self.n_laguerre)
            spikes = self._spike_gates(T)
            if spikes:
                spike = np.zeros((len(spikes), T))
                spike[np.arange(len(spikes)), spikes] = 1.0
                self.basis_ = np.vstack([self.basis_, spike])
            n_basis = self.basis_.shape[0]
            pbar.update(1)

            pbar.set_description("Solving Laguerre coefficients")
            Y2d = decay_flat.T
            if single_irf:
                self.V_ = self._design(self.basis_, alpha_irf)
                C = self._solve_coefficients(self.V_, Y2d)
                model_y = (self.V_ @ C).T.reshape(X, Y, T)
            else:
                P = Y2d.shape[1]
                self.V_ = None
                C = np.zeros((n_basis + self._bg_column, P), dtype=np.float64)
                fit_2d = np.zeros((P, T), dtype=np.float64)
                for start in range(0, P, 1024):
                    cols = slice(start, start + 1024)
                    Vb = self._design_batch(irf_2d[cols])
                    Cb = self._solve_coefficients(Vb, Y2d[:, cols])
                    C[:, cols] = Cb
                    fit_2d[cols] = np.einsum("ptk,kp->pt", Vb, Cb)
                model_y = fit_2d.reshape(X, Y, T)
            self.background_ = (
                C[-1].reshape(X, Y) if self._bg_column else np.zeros((X, Y))
            )
            C = C[:n_basis]
            self.coeffs_ = C.T.reshape(X, Y, n_basis)
            self.fit_curve_ = model_y
            self.residual_curve_ = decay - model_y
            self.residuals_ = (self.residual_curve_**2).sum(-1)
            h_stack = (self.basis_.T @ C).T.reshape(X, Y, T)
            self.reconstructed_ = h_stack
            pbar.update(1)

            pbar.set_description("Estimating global lifetimes")
            h_flat = h_stack.reshape(-1, T)
            h_avg = h_flat[mask.ravel()].mean(0) if mask is not None else h_flat.mean(0)
            taus_init = self._estimate_global_taus(h_avg)
            pbar.update(1)

            pbar.set_description("Fitting per-pixel exponentials")
            self.taus_, A, self.converged_ = self._fit_pixel_exponentials(
                h_stack, taus_init, mask=mask
            )
            if self._bg_in_exp:
                self.background_ = A[..., -1]
                A = A[..., :-1]
            self.amplitudes_ = A
            tau_safe = np.where(self.taus_ > 0, self.taus_, 1.0)
            density = A / tau_safe
            total_density = density.sum(axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.fractions_ = np.where(
                    total_density > 0, density / total_density, 0.0
                )
            pbar.update(1)

            pbar.set_description("Computing lifetime maps")
            self.photons_ = A
            total_photons = self.photons_.sum(axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.photon_fractions_ = np.where(
                    total_photons > 0, self.photons_ / total_photons, 0.0
                )
            self.tau_mean_ = (self.photon_fractions_ * self.taus_).sum(axis=-1)
            kernel = np.einsum(
                "xyi,xyin->xyn", A, self._exp_basis(T, tau_safe, self.onset_)
            )
            self.exp_fit_curve_ = (
                self._convolve_stack(kernel, irf) + self.background_[..., None]
            )
            pbar.update(1)

        return self

    def get_parameters(self, data_name: str = "LaguerreFLI_Dataset") -> dict:
        """
        Return parameters.

        Parameters
        ----------
        data_name : str
            Label assigned to the fitted or processed dataset.

        Returns
        -------
        dict
            Dictionary containing the data produced by get parameters.
        """
        if self.coeffs_ is None:
            raise RuntimeError("Call .fit(decay, irf) first.")

        N = self.n_components
        X, Y = self.tau_mean_.shape
        T = self.reconstructed_.shape[-1]

        decay_d = self.decay_.astype(np.float64)
        exp_fit = self.exp_fit_curve_.astype(np.float64)
        lag_fit = self.fit_curve_.astype(np.float64)
        sdf_map = self.reconstructed_.astype(np.float32)
        photon_count = self.photons_.sum(axis=-1).astype(np.float32)

        n_params = 2 * N + int(self.fit_background) + self.n_shift_params_
        n_params_laguerre = (
            self.basis_.shape[0] + int(self._bg_column) + self.n_shift_params_
        )
        chi_sq_raw, chi_sq_reduced = reduced_poisson_deviance(
            exp_fit, decay_d, n_params
        )
        lag_chi_raw, lag_chi_reduced = reduced_poisson_deviance(
            lag_fit, decay_d, n_params_laguerre
        )
        pearson_raw = pearson_chi_square(exp_fit, decay_d).astype(np.float32)
        dof = degrees_of_freedom(T, n_params)

        residuals_d = decay_d - exp_fit
        r2_map = r_squared(residuals_d, decay_d).astype(np.float32)

        pixel_health = (photon_count > 0).astype(np.float32)

        if N == 1:
            tau_maps = {"tau_map": self.taus_[..., 0].astype(np.float32)}
            alpha_maps = {
                "alpha_map": self.photon_fractions_[..., 0].astype(np.float32)
            }
        else:
            tau_maps = {
                f"tau{i + 1}_map": self.taus_[..., i].astype(np.float32)
                for i in range(N)
            }
            alpha_maps = {
                f"alpha{i + 1}_map": self.photon_fractions_[..., i].astype(np.float32)
                for i in range(N)
            }

        if N >= 2:
            tau1_m = self.taus_[..., 0]
            tau2_m = self.taus_[..., 1]
            fret_eff = np.where(tau2_m > 0, 1.0 - tau1_m / tau2_m, 0.0).astype(
                np.float32
            )
        else:
            fret_eff = np.zeros((X, Y), dtype=np.float32)

        convergence = (
            self.converged_.astype(np.float32)
            if self.converged_ is not None
            else pixel_health.copy()
        )

        maps = {
            **tau_maps,
            **alpha_maps,
            "photon_count_map": photon_count,
            "tau_mean_map": self.tau_mean_.astype(np.float32),
            "v_shift_map": self.background_.astype(np.float32),
            "h_shift_map": self.h_shift_.astype(np.float32),
            "fret_efficiency_map": fret_eff,
            "R2_map": r2_map,
            "chi2_map": chi_sq_raw.astype(np.float32),
            "reduced_chi2_map": chi_sq_reduced.astype(np.float32),
            "pearson_chi2_map": pearson_raw,
            "pearson_reduced_chi2_map": (pearson_raw / dof).astype(np.float32),
            "laguerre_chi2_map": lag_chi_raw.astype(np.float32),
            "laguerre_reduced_chi2_map": lag_chi_reduced.astype(np.float32),
            "convergence_map": convergence,
            "pixel_health_map": pixel_health,
        }

        internal_popt_len = 2 * N + 2
        error_maps = np.zeros((X, Y, internal_popt_len), dtype=np.float32)

        tr_maps = {
            "fit_map": exp_fit.astype(np.float32),
            "residual_map": residuals_d.astype(np.float32),
            "laguerre_fit_map": lag_fit.astype(np.float32),
            "laguerre_residual_map": (decay_d - lag_fit).astype(np.float32),
            "sdf_map": sdf_map,
        }

        mask = photon_count > 0
        mean_chi_sq = float(chi_sq_reduced[mask].mean()) if mask.any() else float("nan")
        logging.info(f"Mean Reduced Chi-Squared (Active Pixels): {mean_chi_sq:.4f}")

        return {
            "name": data_name,
            "method": f"LaguerreFLI_{N}exp",
            "results": {
                "maps": maps,
                "error_maps": error_maps,
                "TR_maps": tr_maps,
            },
        }

    def save_results(self, dataset: dict, folder: str = "results") -> None:
        """
        Save results.

        Parameters
        ----------
        dataset : dict
            Dataset dictionary or fit result collection to save.
        folder : str
            Output directory used when saving results.

        Returns
        -------
        None
            No object is returned; the function save results.
        """

        if dataset is None:
            return
        os.makedirs(folder, exist_ok=True)
        h5_path = os.path.join(folder, f"{dataset['name']}_results.h5")
        with h5py.File(h5_path, "w") as f:
            f.attrs["method"] = dataset["method"]
            res_grp = f.create_group("results")
            maps_grp = res_grp.create_group("maps")
            for k, v in dataset["results"]["maps"].items():
                maps_grp.create_dataset(
                    k, data=v, compression="gzip", compression_opts=4
                )
            res_grp.create_group("error_maps").create_dataset(
                "errors",
                data=dataset["results"]["error_maps"],
                compression="gzip",
                compression_opts=4,
            )
            tr_grp = res_grp.create_group("TR_maps")
            for k, v in dataset["results"]["TR_maps"].items():
                tr_grp.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        logging.info(f"Analysis complete. Results saved to: {h5_path}")

    def load_map(self, h5_path: str, map_name: str = "tau1_map") -> np.ndarray | None:
        """
        Load a map from a .h5 file.

        Parameters
        ----------
        h5_path : str
            Filesystem path used by the routine.
        map_name : str
            Name of the saved parameter map to load.

        Returns
        -------
        Optional[np.ndarray]
            Map array loaded from disk.
        """
        import h5py

        with h5py.File(h5_path, "r") as f:
            key = f"results/maps/{map_name}"
            if key in f:
                return f[key][()]
            logging.warning(f"Map '{map_name}' not found in {h5_path}")
            return None

    def predict(self) -> np.ndarray:
        """
        Return reconstructed value.

        Returns
        -------
        np.ndarray
            Reconstructed decay array predicted from the fitted Laguerre model.
        """
        if self.reconstructed_ is None:
            raise RuntimeError("Call .fit(decay, irf) first.")
        return self.reconstructed_

    def __repr__(self) -> str:
        period = (
            f"{self.laser_period_ns} ns"
            if self.laser_period_ns is not None
            else "not set"
        )
        return (
            f"LaguerreFLI(n_components={self.n_components}, "
            f"n_laguerre={self.n_laguerre}, alpha={self.alpha:.3f}, "
            f"dt={self.dt} ns, laser_period={period}, "
            f"reg_strength={self.reg_strength})"
        )
