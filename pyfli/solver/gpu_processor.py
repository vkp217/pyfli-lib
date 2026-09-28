"""
Fit FLI image cubes with Torch-based GPU optimization and optional CRLB estimates.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLI fitting routines. Public API includes
classes :class:`FLIGPUProcessor`.
"""

import math
import os
import time
from typing import Any

import h5py
import numpy as np
import torch
from tqdm import tqdm

from pyfli import logging

from .shared_metrics import pearson_chi_square, reduced_poisson_deviance


class FLIGPUProcessor:
    """
    Fit FLI image cubes with Torch on GPU or CPU fallback. It vectorizes parameter
    transforms, model evaluation, optimization, CRLB estimation, reconstruction, and
    result saving.

    Parameters
    ----------
    freq : float
        Acquisition frequency information used to derive timing constants.
    fitter_class : Any | None
        Fitter class instantiated by the processor.
    device : Any | None
        Execution device, such as a Torch device or device string.
    """

    def __init__(
        self, freq: float, fitter_class: Any | None = None, device: Any | None = None
    ) -> None:
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.freq = freq
        self.fitter_class = fitter_class
        self.T_acq = 1000.0 / freq[1]
        self.T_laser = 1000.0 / freq[0]
        logging.info(f"Using Device: {self.device}")

    def _transform_params(self, raw_p: np.ndarray, model_type: str) -> Any:
        """
        Run the transform params routine.

        Parameters
        ----------
        raw_p : np.ndarray
            Unconstrained optimizer parameters before physical transformation.
        model_type : str
            FLI model family, such as mono- or bi-exponential.

        Returns
        -------
        Any
            Object produced by transform params.
        """
        shift_bound = self.T_acq / 4.0
        S = torch.exp(raw_p[:, 0:1])
        b = torch.exp(raw_p[:, -2:-1])
        h_shift = torch.tanh(raw_p[:, -1:]) * shift_bound

        if model_type == "bi-exponential":
            a1 = torch.sigmoid(raw_p[:, 1:2])
            t1 = torch.exp(raw_p[:, 2:3])
            t2 = t1 + torch.exp(raw_p[:, 3:4])
            return torch.cat([S, a1, t1, t2, b, h_shift], dim=1)
        else:
            tau = torch.exp(raw_p[:, 1:2])
            return torch.cat([S, tau, b, h_shift], dim=1)

    def _model_kernel(
        self, params: Any, t: np.ndarray, irf: np.ndarray, model_type: str
    ) -> Any:
        """
        Batched forward model, identical to :func:`pyfli.solver.model_numpy`: the
        gate-integrated decay with onset ``h_shift`` (evaluated on extra negative-lag
        gates so an earlier onset shifts the curve instead of truncating it),
        convolved with the normalized IRF, plus ``v_shift``. ``S`` is the total
        photon count of the decay.

        Parameters
        ----------
        params : Any
            Model, detector, or plotting parameters used by the routine.
        t : np.ndarray
            Time axis or acquisition period used by the calculation.
        irf : np.ndarray
            Instrument response function aligned with the decay signal.
        model_type : str
            FLI model family, such as mono- or bi-exponential.

        Returns
        -------
        Any
            Object produced by model kernel.
        """
        T = t.shape[-1]
        dt = self.T_acq / T
        n_neg = math.ceil((self.T_acq / 4.0) / dt) + 1
        lags = torch.arange(-n_neg, T, device=t.device, dtype=params.dtype) * dt
        a = lags[None, :]
        b_edge = a + dt
        h_shift = params[:, -1:]
        a_eff = torch.maximum(a, h_shift) - h_shift
        b_eff = torch.maximum(b_edge, h_shift) - h_shift

        def gate_fraction(tau: Any) -> Any:
            return torch.exp(-a_eff / tau) - torch.exp(-b_eff / tau)

        if model_type == "mono-exponential":
            S, tau, b = params[:, 0:1], params[:, 1:2], params[:, 2:3]
            decay = S * gate_fraction(tau)
        else:
            S, a1, t1, t2, b = (
                params[:, 0:1],
                params[:, 1:2],
                params[:, 2:3],
                params[:, 3:4],
                params[:, 4:5],
            )
            decay = S * (a1 * gate_fraction(t1) + (1.0 - a1) * gate_fraction(t2))

        irf_norm = irf / irf.sum(dim=1, keepdim=True).clamp(min=1e-9)

        n_fft = 2 * (T + n_neg)
        decay_fft = torch.fft.rfft(decay, n=n_fft)
        irf_fft = torch.fft.rfft(irf_norm, n=n_fft)
        convolved = torch.fft.irfft(decay_fft * irf_fft, n=n_fft)[
            ..., n_neg : n_neg + T
        ]
        return convolved + b

    def _compute_crlb_errors(
        self, p_phys: np.ndarray, t: np.ndarray, irf: np.ndarray, model_type: str
    ) -> Any:
        """
        Compute crlb errors.

        Parameters
        ----------
        p_phys : np.ndarray
            Physical parameter tensor after constrained transformation.
        t : np.ndarray
            Time axis or acquisition period used by the calculation.
        irf : np.ndarray
            Instrument response function aligned with the decay signal.
        model_type : str
            FLI model family, such as mono- or bi-exponential.

        Returns
        -------
        Any
            Object produced by compute CRLB errors.
        """
        p_phys = p_phys.detach().clone().requires_grad_(True)

        def model_func(p: Any) -> Any:
            """
            Run the model func routine.

            Parameters
            ----------
            p : Any
                Detector parameter object or fitted parameter vector.

            Returns
            -------
            Any
                Object produced by model func.
            """
            return self._model_kernel(p, t, irf, model_type)

        jac = torch.autograd.functional.jacobian(model_func, p_phys, vectorize=True)
        jac = torch.diagonal(jac, dim1=0, dim2=2).permute(2, 0, 1)

        with torch.no_grad():
            pred = model_func(p_phys)
            W = 1.0 / torch.clamp(pred, min=1.0)
            jt_w = jac.transpose(1, 2) * W.unsqueeze(1)
            fim = torch.bmm(jt_w, jac)
            trace = (
                torch.diagonal(fim, dim1=1, dim2=2)
                .sum(dim=1, keepdim=True)
                .unsqueeze(-1)
            )
            eps_mat = (1e-6 * trace / fim.shape[-1]) * torch.eye(
                fim.shape[-1], device=self.device
            )
            try:
                cov = torch.inverse(fim + eps_mat)
                return torch.sqrt(torch.abs(torch.diagonal(cov, dim1=1, dim2=2)))
            except RuntimeError:
                return torch.zeros_like(p_phys)

    def fit_image(
        self,
        image_cube: np.ndarray,
        irf_cube: np.ndarray,
        mask: np.ndarray | None = None,
        mode: str = "MLE",
        model_type: str = "bi-exponential",
        max_iter: int = 500,
        CRLB: bool = False,
        data_name: str = "Torch_Fit",
        p0: Any | None = None,
        fit_indices: tuple[int, int] | None = None,
        weighting: str = "irls",
        **kwargs: Any,
    ) -> Any:
        # Normalise mode tag: NLSF/LSE variants → 'NLSF', everything else → 'MLE'
        """
        Fit image.

        Parameters
        ----------
        image_cube : np.ndarray
            Time-resolved decay image cube.
        irf_cube : np.ndarray
            Instrument response cube aligned with the decay image cube.
        mask : np.ndarray | None
            Boolean or labeled mask selecting pixels for the operation.
        mode : str
            Mode selector used by the fitting, loading, or plotting routine.
        model_type : str
            FLI model family, such as mono- or bi-exponential.
        max_iter : int
            Maximum number of optimization iterations.
        CRLB : bool
            If ``True``, compute Cramer-Rao lower-bound uncertainty estimates.
        data_name : str
            Label assigned to the fitted or processed dataset.
        p0 : Any | None
            Initial parameter vector supplied to the optimizer.
        fit_indices : tuple[int, int] | None
            Optional (gate_num_start, gate_num_end) gate range to fit over, e.g. to
            focus on the tail of the decay. The forward model is still evaluated over
            the full trace (needed for correct IRF convolution); only the loss and fit
            statistics are restricted to this gate range. ``None`` fits the full trace.
        weighting : str
            Residual weights for ``mode="NLSF"`` (ignored for MLE, which uses the
            Poisson deviance): ``"irls"`` (default) divides each squared residual by
            the current model value held constant in the gradient -- the batched
            counterpart of iteratively reweighted least squares, whose converged
            solution solves the Poisson likelihood equations; ``"none"`` is
            unweighted; ``"neyman"`` divides by the measured counts (former default,
            biased towards short lifetimes at low counts). ``mode="NEYMAN"`` implies
            ``"neyman"``. ``variance_floor`` (kwarg, default 1.0) floors the IRLS
            variance.
        **kwargs : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        Any
            Object produced by fit image.
        """
        _NLSF_MODES = {"NLSF", "LSE", "WLS", "NEYMAN", "LEAST_SQUARES"}
        if mode.upper() == "NEYMAN":
            weighting = "neyman"
        if weighting not in ("irls", "none", "neyman"):
            raise ValueError(
                f"weighting must be 'irls', 'none' or 'neyman', got {weighting!r}"
            )
        mode = "NLSF" if mode.upper() in _NLSF_MODES else "MLE"
        variance_floor = kwargs.get("variance_floor", 1.0)

        start_time = time.time()
        H, W, T = image_cube.shape
        t_axis = torch.arange(T, device=self.device) * (self.T_acq / T)

        if fit_indices is not None:
            gate_start, gate_end = fit_indices
            gate_start, gate_end = max(gate_start, 0), min(gate_end, T)
        else:
            gate_start, gate_end = 0, T

        irf_tensor = torch.tensor(
            irf_cube, device=self.device, dtype=torch.float32
        ).reshape(-1, T)

        if mask is None:
            mask = np.sum(image_cube, axis=2) > 20

        valid_idx = np.where(mask.flatten())[0]
        if len(valid_idx) == 0:
            logging.warning("No valid pixels found.")
            return None

        flat_data = torch.tensor(
            image_cube.reshape(-1, T)[valid_idx],
            device=self.device,
            dtype=torch.float32,
        )
        flat_irf = irf_tensor[valid_idx]

        if p0 is not None:
            p_guess = self._p0_to_tensor(p0, len(valid_idx), model_type)
        else:
            p_guess = self._get_sophisticated_guess(flat_data, flat_irf, model_type)
        raw_p = torch.zeros_like(p_guess)

        with torch.no_grad():
            raw_p[:, 0] = torch.log(torch.clamp(p_guess[:, 0], min=1e-3))  # log(S)
            raw_p[:, -2] = torch.log(
                torch.clamp(p_guess[:, -2], min=1e-6)
            )  # log(v_shift)
            raw_p[:, -1] = math.atanh(0.5 * (self.T_acq / T) / (self.T_acq / 4.0))
            if model_type == "bi-exponential":
                raw_p[:, 1] = torch.logit(torch.clamp(p_guess[:, 1], 0.001, 0.999))
                raw_p[:, 2] = torch.log(torch.clamp(p_guess[:, 2], min=1e-3))
                raw_p[:, 3] = torch.log(
                    torch.clamp(p_guess[:, 3] - p_guess[:, 2], min=1e-3)
                )
            else:
                raw_p[:, 1] = torch.log(torch.clamp(p_guess[:, 1], min=0.1))

        raw_p.requires_grad_(True)
        pixel_health_map = np.ones(H * W, dtype=np.float32)

        if mode == "NLSF":

            def objective_fn(p_raw: np.ndarray) -> Any:
                """
                Run the objective fn routine.

                Parameters
                ----------
                p_raw : np.ndarray
                    Raw unconstrained parameter tensor optimized by the objective.

                Returns
                -------
                Any
                    Object produced by objective fn.
                """
                p_phys = self._transform_params(p_raw, model_type)
                pred = self._model_kernel(p_phys, t_axis, flat_irf, model_type)
                pred_sel = pred[:, gate_start:gate_end]
                data_sel = flat_data[:, gate_start:gate_end]
                if weighting == "neyman":
                    variance = torch.clamp(data_sel, min=1.0)
                elif weighting == "none":
                    variance = torch.ones_like(data_sel)
                else:
                    variance = torch.clamp(pred_sel.detach(), min=variance_floor)
                per_px = torch.sum((pred_sel - data_sel) ** 2 / variance, dim=1)
                return per_px[torch.isfinite(per_px)].sum()
        else:
            # Poisson MLE (C-statistic): matches CPU MLEFLIFitter
            def objective_fn(p_raw: np.ndarray) -> Any:
                """
                Run the objective fn routine.

                Parameters
                ----------
                p_raw : np.ndarray
                    Raw unconstrained parameter tensor optimized by the objective.

                Returns
                -------
                Any
                    Object produced by objective fn.
                """
                p_phys = self._transform_params(p_raw, model_type)
                pred = self._model_kernel(p_phys, t_axis, flat_irf, model_type)
                pred_sel = pred[:, gate_start:gate_end]
                data_sel = flat_data[:, gate_start:gate_end]
                pred_safe = torch.clamp(pred_sel, min=1e-9)
                per_px = 2.0 * torch.sum(
                    pred_safe
                    - data_sel
                    + data_sel * torch.log(data_sel.clamp(min=1e-9) / pred_safe),
                    dim=1,
                )
                return per_px[torch.isfinite(per_px)].sum()

        # Adam operates per-parameter independently — unlike LBFGS it does not maintain
        # a single global Hessian across all pixels, correctly handling the joint space.
        optimizer = torch.optim.Adam([raw_p], lr=kwargs.get("lr", 0.05))

        logging.info(f"--- GPU {mode} Processing ({len(valid_idx)} pixels) ---")
        pbar = tqdm(total=max_iter, desc=f"Optimizing ({mode})")

        prev_loss = float("inf")
        patience_count = 0
        patience = kwargs.get("patience", 50)

        try:
            for step in range(max_iter):
                optimizer.zero_grad()
                loss = objective_fn(raw_p)
                loss.backward()
                optimizer.step()
                pbar.update(1)

                cur = loss.item()
                if abs(prev_loss - cur) < 1e-7 * (abs(prev_loss) + 1e-10):
                    patience_count += 1
                    if patience_count >= patience:
                        pbar.update(max_iter - pbar.n)
                        break
                else:
                    patience_count = 0
                prev_loss = cur
        except Exception as e:
            logging.warning(f"Optimization interrupted: {e}")
            pixel_health_map[valid_idx] = 0

        pbar.close()

        with torch.no_grad():
            p_final = self._transform_params(raw_p, model_type)
            fit_flat = self._model_kernel(p_final, t_axis, flat_irf, model_type)
            res_flat = flat_data - fit_flat
            dof = max((gate_end - gate_start) - p_final.shape[1], 1)

            fit_sel = fit_flat[:, gate_start:gate_end]
            data_sel = flat_data[:, gate_start:gate_end]
            res_sel = res_flat[:, gate_start:gate_end]

            fit_sel_np = fit_sel.detach().cpu().numpy().astype(np.float64)
            data_sel_np = data_sel.detach().cpu().numpy().astype(np.float64)
            chi2_raw_flat, chi2_red_flat = reduced_poisson_deviance(
                fit_sel_np, data_sel_np, p_final.shape[1]
            )
            pearson_flat = pearson_chi_square(fit_sel_np, data_sel_np)

            ss_tot = torch.sum(
                (data_sel - data_sel.mean(dim=1, keepdim=True)) ** 2, dim=1
            )
            ss_res = torch.sum(res_sel**2, dim=1)
            r2_flat = torch.where(
                ss_tot > 0, 1.0 - ss_res / ss_tot, torch.zeros_like(ss_tot)
            )
            rmse_flat = torch.sqrt(torch.mean(res_sel**2, dim=1))

            perr_flat = torch.zeros_like(p_final)
            if CRLB:
                perr_flat = self._compute_crlb_errors(
                    p_final, t_axis, flat_irf, model_type
                )

        full_popt = np.zeros((H * W, p_final.shape[1]))
        full_perr = np.zeros((H * W, p_final.shape[1]))
        full_fit = np.zeros((H * W, T))
        full_res = np.zeros((H * W, T))
        full_chi2_raw = np.zeros(H * W)
        full_chi2_red = np.zeros(H * W)
        full_r2 = np.zeros(H * W)
        full_rmse = np.zeros(H * W)

        full_popt[valid_idx] = p_final.detach().cpu().numpy()
        full_perr[valid_idx] = perr_flat.detach().cpu().numpy()
        full_fit[valid_idx] = fit_flat.detach().cpu().numpy()
        full_res[valid_idx] = res_flat.detach().cpu().numpy()
        full_chi2_raw[valid_idx] = chi2_raw_flat
        full_chi2_red[valid_idx] = chi2_red_flat
        full_pearson = np.zeros(H * W)
        full_pearson[valid_idx] = pearson_flat
        full_r2[valid_idx] = r2_flat.detach().cpu().numpy()
        full_rmse[valid_idx] = rmse_flat.detach().cpu().numpy()

        logging.info(f"Fit Finished in {time.time() - start_time:.2f}s")

        health_mask = np.zeros(H * W)
        health_mask[valid_idx] = 1.0
        tau_lo, tau_hi = 1e-4, self.T_laser
        p_np = p_final.detach().cpu().numpy()
        chi2_red_np = full_chi2_red[valid_idx]

        if model_type == "bi-exponential":
            at_bound = (
                (p_np[:, 2] <= tau_lo * 1.01)
                | (p_np[:, 2] >= tau_hi * 0.99)
                | (p_np[:, 3] <= tau_lo * 1.01)
                | (p_np[:, 3] >= tau_hi * 0.99)
            )
        else:
            at_bound = (p_np[:, 1] <= tau_lo * 1.01) | (p_np[:, 1] >= tau_hi * 0.99)
        health_mask[valid_idx] = np.where(at_bound | (chi2_red_np > 5.0), 0.0, 1.0)

        dataset = self._reconstruct_dataset(
            full_popt.reshape(H, W, -1),
            full_perr.reshape(H, W, -1),
            full_fit.reshape(H, W, T),
            full_res.reshape(H, W, T),
            full_chi2_raw.reshape(H, W),
            full_chi2_red.reshape(H, W),
            full_r2.reshape(H, W),
            full_rmse.reshape(H, W),
            health_mask.reshape(H, W),
            model_type,
            mode,
            data_name,
        )
        dataset["results"]["maps"]["pearson_chi2_map"] = full_pearson.reshape(
            H, W
        ).astype(np.float32)
        dataset["results"]["maps"]["pearson_reduced_chi2_map"] = (
            full_pearson.reshape(H, W) / dof
        ).astype(np.float32)
        return dataset

    def _reconstruct_dataset(
        self,
        p_maps: np.ndarray,
        e_maps: np.ndarray,
        fit_map: np.ndarray,
        res_map: np.ndarray,
        chi2_raw: np.ndarray,
        chi2_reduced: np.ndarray,
        r2_map: np.ndarray,
        rmse_map: np.ndarray,
        health_map: np.ndarray,
        model_type: str,
        mode: str,
        name: str,
    ) -> dict[Any, Any]:
        """
        Reconstruct dataset.

        Parameters
        ----------
        p_maps : np.ndarray
            Physical parameter maps reconstructed from flattened fit results.
        e_maps : np.ndarray
            Parameter-error maps reconstructed from flattened fit results.
        fit_map : np.ndarray
            Parameter or mask map processed by the routine.
        res_map : np.ndarray
            Parameter or mask map processed by the routine.
        chi2_raw : np.ndarray
            Raw chi-square map from the fit reconstruction.
        chi2_reduced : np.ndarray
            Reduced chi-square map from the fit reconstruction.
        r2_map : np.ndarray
            Parameter or mask map processed by the routine.
        rmse_map : np.ndarray
            Root-mean-square-error map from the fit reconstruction.
        health_map : np.ndarray
            Parameter or mask map processed by the routine.
        model_type : str
            FLI model family, such as mono- or bi-exponential.
        mode : str
            Mode selector used by the fitting, loading, or plotting routine.
        name : str
            Dataset, experiment, figure, or output name.

        Returns
        -------
        dict[Any, Any]
            Dictionary containing the data produced by reconstruct dataset.
        """
        S = p_maps[..., 0]
        common = {
            "chi2_map": chi2_raw,
            "reduced_chi2_map": chi2_reduced,
            "R2_map": r2_map,
            "rmse_map": rmse_map,
            "pixel_health_map": health_map,
            "convergence_map": health_map,
        }
        if model_type == "bi-exponential":
            tau1_m, tau2_m = p_maps[..., 2], p_maps[..., 3]
            alpha1_m = p_maps[..., 1]
            maps = {
                "photon_count_map": S,
                "alpha1_map": alpha1_m,
                "tau1_map": tau1_m,
                "tau2_map": tau2_m,
                "tau_mean_map": (alpha1_m * tau1_m + (1.0 - alpha1_m) * tau2_m).astype(
                    np.float32
                ),
                "v_shift_map": p_maps[..., 4].astype(np.float32),
                "h_shift_map": p_maps[..., 5].astype(np.float32),
                "fret_efficiency_map": (
                    1.0
                    - np.divide(
                        tau1_m,
                        tau2_m,
                        out=np.zeros_like(tau2_m, dtype=np.float32),
                        where=tau2_m > 0,
                    )
                ).astype(np.float32),
                **common,
            }
        else:
            maps = {
                "photon_count_map": S,
                "tau_map": p_maps[..., 1],
                "v_shift_map": p_maps[..., 2].astype(np.float32),
                "h_shift_map": p_maps[..., 3].astype(np.float32),
                **common,
            }
        return {
            "name": name,
            "method": f"GPU_{mode}",
            "results": {
                "maps": maps,
                "error_maps": e_maps,
                "TR_maps": {
                    "fit_map": fit_map.astype(np.float32),
                    "residual_map": res_map.astype(np.float32),
                },
            },
        }

    def _p0_to_tensor(self, p0: Any, n_pixels: int, model_type: str) -> Any:
        """
        Run the p0 to tensor routine.

        Parameters
        ----------
        p0 : Any
            Initial parameter vector supplied to the optimizer.
        n_pixels : int
            Number of samples, components, gates, or iterations used by the routine.
        model_type : str
            FLI model family, such as mono- or bi-exponential.

        Returns
        -------
        Any
            Object produced by p0 to tensor.
        """
        if isinstance(p0, dict):
            if model_type == "bi-exponential":
                vals = [
                    p0.get("amp", 1000.0),
                    p0.get("alpha1", 0.2),
                    p0.get("tau1", 0.5),
                    p0.get("tau2", 1.1),
                    p0.get("v_shift", 10.0),
                    p0.get("h_shift", 0.0),
                ]
            else:
                vals = [
                    p0.get("amp", 1000.0),
                    p0.get("tau", 0.9),
                    p0.get("v_shift", 10.0),
                    p0.get("h_shift", 0.0),
                ]
            arr = np.tile(vals, (n_pixels, 1)).astype(np.float32)
        else:
            row = np.asarray(p0, dtype=np.float32).ravel()
            arr = np.tile(row, (n_pixels, 1))
        return torch.tensor(arr, device=self.device, dtype=torch.float32)

    def _get_sophisticated_guess(
        self, data: np.ndarray, irf: np.ndarray, model_type: str
    ) -> Any:
        """
        Return sophisticated guess.

        Parameters
        ----------
        data : np.ndarray
            Data array or mapping processed by the routine.
        irf : np.ndarray
            Instrument response function aligned with the decay signal.
        model_type : str
            FLI model family, such as mono- or bi-exponential.

        Returns
        -------
        Any
            Object produced by get sophisticated guess.
        """
        cpu_data = data.detach().cpu().numpy().astype(np.float64)
        P, T = cpu_data.shape
        t_axis = np.linspace(0, self.T_acq, T, endpoint=False)
        dt = (t_axis[1] - t_axis[0]) if T > 1 else 1.0

        offset_guess = np.percentile(cpu_data, 5, axis=1)
        clean_d = np.clip(cpu_data - offset_guess[:, None], 1e-6, None)

        idx_max = np.argmax(clean_d, axis=1)
        col_idx = np.arange(T)[None, :]
        post_peak = col_idx >= idx_max[:, None]
        d_post = clean_d * post_peak

        t_peak = t_axis[idx_max]
        t_rel = np.maximum(t_axis[None, :] - t_peak[:, None], 0.0) * post_peak

        m0 = np.trapezoid(d_post, dx=dt, axis=1).clip(min=1e-12)
        m1 = np.trapezoid(t_rel * d_post, dx=dt, axis=1)

        tau_mean = np.clip(m1 / m0, 0.05, self.T_laser * 0.8)
        inside = -np.expm1(-self.T_acq / tau_mean)
        s_guess = np.clip(clean_d.sum(axis=1) / inside, 1e-3, None)
        offset_safe = np.clip(offset_guess, 0.0, None)

        h_shift_guess = np.zeros(P)

        if model_type == "mono-exponential":
            guesses = np.stack([s_guess, tau_mean, offset_safe, h_shift_guess], axis=1)
        else:
            tau1 = np.clip(tau_mean * 0.5, 1e-4, self.T_laser * 0.99)
            tau2 = np.clip(tau_mean * 1.5, tau1 * 1.01, self.T_laser)

            # Neutral alpha1=0.5: the area-ratio estimate is unreliable when one
            # lifetime is near the IRF width (fast component area ≈ slow area in
            # any early/late split), so starting at 0.5 is always safer.
            alpha1 = np.full(P, 0.5)

            guesses = np.stack(
                [s_guess, alpha1, tau1, tau2, offset_safe, h_shift_guess], axis=1
            )

        return torch.tensor(guesses.astype(np.float32), device=self.device)

    def save_results(self, dataset: np.ndarray, folder: str = "results") -> None:
        """
        Save results.

        Parameters
        ----------
        dataset : np.ndarray
            Dataset dictionary or fit result collection to save.
        folder : str
            Output directory used when saving results.

        Returns
        -------
        None
            No object is returned; the function save results.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        h5_path = os.path.join(folder, f"{dataset['name']}_GPU_results.h5")

        with h5py.File(h5_path, "w") as f:
            f.attrs["method"] = dataset["method"]
            res_grp = f.create_group("results")

            maps_grp = res_grp.create_group("maps")
            for k, v in dataset["results"]["maps"].items():
                maps_grp.create_dataset(
                    k, data=v.astype(np.float32), compression="gzip"
                )

            err_grp = res_grp.create_group("error_maps")
            err_grp.create_dataset(
                "errors", data=dataset["results"]["error_maps"], compression="gzip"
            )

            tr_grp = res_grp.create_group("TR_maps")
            for k, v in dataset["results"]["TR_maps"].items():
                tr_grp.create_dataset(k, data=v, compression="gzip")

        logging.info(f"Dataset successfully saved to: {h5_path}")
