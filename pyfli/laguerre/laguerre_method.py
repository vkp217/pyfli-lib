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

from ..solver.base_static import moment_based_guess
from ..solver.shared_metrics import reduced_poisson_deviance


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
    def _convolve_with_irf(basis: np.ndarray, irf: np.ndarray) -> np.ndarray:
        """
        Convolve each Laguerre basis function with the IRF.

        Parameters
        ----------
        basis : np.ndarray
            Laguerre basis matrix before or after IRF convolution.
        irf : np.ndarray
            Instrument response function aligned with the decay signal.

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
        full = fftconvolve(basis, irf[None, :], mode="full", axes=1)
        return full[:, :T].T

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
            Vector or matrix evaluated by the simplex projection.
        Y2d : np.ndarray
            Flattened decay matrix solved for Laguerre coefficients.

        Returns
        -------
        np.ndarray
            Coefficient matrix fitted for each decay trace.
        """
        if self.reg_strength > 0.0:
            L = V.shape[1]
            lam = self.reg_strength * float(np.mean(np.diag(V.T @ V)))
            VtV = V.T @ V + lam * np.diag(self._penalty(L))
            return np.linalg.solve(VtV, V.T @ Y2d)
        C, *_ = np.linalg.lstsq(V, Y2d, rcond=None)
        return C

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
            V = self._convolve_with_irf(B, avg_irf)
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
        n = np.arange(T)
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
            E = np.exp(-n[:, None] * self.dt / params[None, :])
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
        n = np.arange(T)
        tau_lo, tau_hi = self._tau_bounds(T)

        tau_init_safe = self._safe_tau0(tau_init, tau_lo, tau_hi)
        bounds = ([tau_lo] * N, [tau_hi] * N)

        taus_map = np.zeros((X, Y, N), dtype=np.float64)
        amps_map = np.zeros((X, Y, N), dtype=np.float64)
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

                        def residual(params: Any, h: np.ndarray = h) -> Any:
                            E = np.exp(-n[:, None] * self.dt / params[None, :])
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

                        E_px = np.exp(-n[:, None] * self.dt / taus_px[None, :])
                        taus_map[x, y, :] = taus_px
                        amps_map[x, y, :] = self._solve_amps(E_px, h, 200 * N)
                    pbar.update(1)

        return taus_map, amps_map, converged_map

    def fit(
        self,
        decay: np.ndarray,
        irf: np.ndarray,
        mask: np.ndarray | None = None,
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

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (X, Y):
                raise ValueError(
                    f"mask shape {mask.shape} must match image shape ({X}, {Y})."
                )
        self.mask_ = mask

        ppirf = irf.ndim == 3
        if ppirf and irf.shape != decay.shape:
            raise ValueError("per-pixel IRF must match decay shape.")
        if not ppirf and not (irf.ndim == 1 and irf.shape[0] == T):
            raise ValueError("irf must be (T,) or (X, Y, T).")

        decay_flat = decay.reshape(-1, T)
        if mask is not None:
            avg_decay = decay_flat[mask.ravel()].mean(0)
        else:
            avg_decay = decay_flat.mean(0)

        if ppirf:
            irf_2d = irf.reshape(-1, T)
            labels, rep_idx = self._unique_irf_groups(irf_2d)
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
            pbar.update(1)

            pbar.set_description("Solving Laguerre coefficients")
            Y2d = decay_flat.T
            if single_irf:
                self.V_ = self._convolve_with_irf(self.basis_, alpha_irf)
                C = self._solve_coefficients(self.V_, Y2d)
                model_y = (self.V_ @ C).T.reshape(X, Y, T)
            else:
                P = Y2d.shape[1]
                self.V_ = None
                C = np.zeros((self.n_laguerre, P), dtype=np.float64)
                fit_2d = np.zeros((P, T), dtype=np.float64)
                for g, rep in enumerate(rep_idx):
                    cols = np.flatnonzero(labels == g)
                    Vg = self._convolve_with_irf(self.basis_, irf_2d[rep])
                    Cg = self._solve_coefficients(Vg, Y2d[:, cols])
                    C[:, cols] = Cg
                    fit_2d[cols] = (Vg @ Cg).T
                model_y = fit_2d.reshape(X, Y, T)
            self.coeffs_ = C.T.reshape(X, Y, self.n_laguerre)
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
            self.amplitudes_ = A
            total_amp = A.sum(axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.fractions_ = np.where(total_amp > 0, A / total_amp, 0.0)
            pbar.update(1)

            pbar.set_description("Computing lifetime maps")
            # Intensity-weighted mean: <τ> = Σ αᵢτᵢ² / Σ αᵢτᵢ
            # (fractions_ are amplitude fractions; αᵢτᵢ is proportional to photon count of component i)
            num = (self.fractions_ * self.taus_**2).sum(axis=-1)
            den = (self.fractions_ * self.taus_).sum(axis=-1)
            has_signal = total_amp.squeeze(-1) > 0
            with np.errstate(invalid="ignore", divide="ignore"):
                self.tau_mean_ = np.where(has_signal, num / np.maximum(den, 1e-10), 0.0)
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
        eps = 1e-8

        fit_map = (
            self.fit_curve_ if self.fit_curve_ is not None else self.reconstructed_
        ).astype(np.float32)
        res_map = (
            self.residual_curve_
            if self.residual_curve_ is not None
            else np.zeros_like(fit_map)
        ).astype(np.float32)
        sdf_map = self.reconstructed_.astype(np.float32)

        if self.decay_ is not None:
            photon_count = self.decay_.sum(axis=-1).astype(np.float32)
        else:
            photon_count = self.amplitudes_.sum(axis=-1).astype(np.float32)

        scaled_fit = fit_map.astype(np.float64)
        decay_d = (
            self.decay_.astype(np.float64) if self.decay_ is not None else scaled_fit
        )
        variance = scaled_fit.copy()
        variance[variance <= 0] = 1.0
        dof = max(T - self.n_laguerre, 1)
        residuals_d = decay_d - scaled_fit
        chi_sq_raw, chi_sq_reduced = reduced_poisson_deviance(
            scaled_fit, decay_d, self.n_laguerre
        )
        chi_sq_raw = chi_sq_raw.astype(np.float32)
        chi_sq_reduced = chi_sq_reduced.astype(np.float32)
        pearson_raw = np.sum((residuals_d**2) / variance, axis=-1).astype(np.float32)

        ss_res = np.sum(residuals_d**2, axis=-1)
        ss_tot = np.sum((decay_d - decay_d.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
        r2_map = (
            1.0
            - np.divide(ss_res, ss_tot, out=np.zeros_like(ss_res), where=ss_tot > eps)
        ).astype(np.float32)

        pixel_health = (photon_count > 0).astype(np.float32)

        if N == 1:
            tau_maps = {"tau_map": self.taus_[..., 0].astype(np.float32)}
            alpha_maps = {"alpha_map": self.fractions_[..., 0].astype(np.float32)}
        else:
            tau_maps = {
                f"tau{i + 1}_map": self.taus_[..., i].astype(np.float32)
                for i in range(N)
            }
            photon_weight = self.fractions_ * self.taus_
            total_photon_weight = photon_weight.sum(axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                photon_fractions = np.where(
                    total_photon_weight > 0, photon_weight / total_photon_weight, 0.0
                )
            alpha_maps = {
                f"alpha{i + 1}_map": photon_fractions[..., i].astype(np.float32)
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
            "v_shift_map": np.zeros((X, Y), dtype=np.float32),
            "h_shift_map": np.zeros((X, Y), dtype=np.float32),
            "fret_efficiency_map": fret_eff,
            "R2_map": r2_map,
            "chi2_map": chi_sq_raw,
            "reduced_chi2_map": chi_sq_reduced,
            "pearson_chi2_map": pearson_raw,
            "pearson_reduced_chi2_map": (pearson_raw / dof).astype(np.float32),
            "convergence_map": convergence,
            "pixel_health_map": pixel_health,
        }

        internal_popt_len = 2 * N + 1
        error_maps = np.zeros((X, Y, internal_popt_len), dtype=np.float32)

        tr_maps = {
            "fit_map": fit_map,
            "residual_map": res_map,
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
