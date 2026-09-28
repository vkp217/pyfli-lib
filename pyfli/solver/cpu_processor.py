"""
Process FLI image cubes on CPU with parallel pixel-level fitting.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLI fitting routines. Public API includes
classes :class:`FLICPUProcessor`.
"""

import os
from typing import Any

import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from pyfli import logging

from .shared_metrics import pearson_chi_square

try:
    from .global_fitter import GlobalFLIFitter as _GlobalFLIFitter
except ImportError:
    _GlobalFLIFitter = None


class FLICPUProcessor:
    """
    Run pixel-wise FLI fitting on CPU. The processor parallelizes fitting across image
    pixels, reconstructs parameter maps, saves results, and loads saved maps.

    Parameters
    ----------
    freq : float
        Acquisition frequency information used to derive timing constants.
    fitter_class : Any
        Fitter class instantiated by the processor.
    """

    def __init__(self, freq: float, fitter_class: Any) -> None:
        self.freq = freq
        self.fitter_class = fitter_class

    def _fit_task(
        self,
        y_data: np.ndarray,
        irf_p: np.ndarray,
        coords: Any,
        model_type: str,
        estimator: np.ndarray,
        p0: Any,
        bounds: np.ndarray,
        fit_indices: tuple[int, int] | None,
        kwargs: np.ndarray,
    ) -> tuple[Any, ...]:
        """
        Fit task.

        Parameters
        ----------
        y_data : np.ndarray
            Observed decay data passed to the fitter.
        irf_p : np.ndarray
            Per-pixel IRF passed to a worker fitting task.
        coords : Any
            Pixel coordinates associated with the fit task.
        model_type : str
            FLI model family, such as mono- or bi-exponential.
        estimator : np.ndarray
            Estimator name used to choose a fitting objective.
        p0 : Any
            Initial parameter vector supplied to the optimizer.
        bounds : np.ndarray
            Lower and upper parameter bounds supplied to the optimizer.
        fit_indices : tuple[int, int] | None
            Optional (gate_num_start, gate_num_end) gate range to fit over.
        kwargs : np.ndarray
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing worker fit results and pixel index metadata.
        """
        y_data = y_data.astype(np.float32)
        irf_p = irf_p.astype(np.float32)

        try:
            shift_method = kwargs.get("shift_method", "zero_pad")
            fit_kwargs = {k: v for k, v in kwargs.items() if k != "shift_method"}

            fitter = self.fitter_class(
                self.freq,
                y_data,
                irf_p,
                shift_method=shift_method,
                fit_indices=fit_indices,
            )

            res = fitter.fit_with_estimator(
                estimator_type=estimator,
                model_type=model_type,
                p0=p0,
                bounds=bounds,
                **fit_kwargs,
            )

            popt = res[0]

            fit_curve = fitter.model_fit(fitter.t, popt, model_type=model_type).astype(
                np.float32
            )
            residual = (y_data - fit_curve).astype(np.float32)

            health = 1
            return coords, res, fit_curve, residual, health

        except Exception:
            n_params = 6 if model_type == "bi-exponential" else 4
            dummy_res = (
                np.zeros(n_params),
                np.zeros(n_params),
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0.0,
            )
            dummy_curve = np.zeros_like(y_data)
            health = 0
            return coords, dummy_res, dummy_curve, dummy_curve, health

    def process_image(
        self,
        image_cube: np.ndarray,
        irf_cube: np.ndarray,
        mask: np.ndarray | None = None,
        data_name: str = "FLIM_Dataset",
        model_type: str = "bi-exponential",
        estimator: str = "least_squares",
        p0: Any | None = None,
        bounds: np.ndarray | None = None,
        n_jobs: int = -1,
        backend: str = "loky",
        fit_indices: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Process image.

        Parameters
        ----------
        image_cube : np.ndarray
            Time-resolved decay image cube.
        irf_cube : np.ndarray
            Instrument response cube aligned with the decay image cube.
        mask : np.ndarray | None
            Boolean or labeled mask selecting pixels for the operation.
        data_name : str
            Label assigned to the fitted or processed dataset.
        model_type : str
            FLI model family, such as mono- or bi-exponential.
        estimator : str
            Estimator name used to choose a fitting objective.
        p0 : Any | None
            Initial parameter vector supplied to the optimizer.
        bounds : np.ndarray | None
            Lower and upper parameter bounds supplied to the optimizer.
        n_jobs : int
            Number of parallel jobs used for CPU fitting.
        backend : str
            Joblib execution backend used for parallel CPU fitting.
        fit_indices : tuple[int, int] | None
            Optional (gate_num_start, gate_num_end) gate range to fit over, e.g. to
            focus on the tail of the decay. ``None`` fits the full trace.
        **kwargs : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        Any
            Object produced by process image.
        """
        H, W, T = image_cube.shape

        if (
            mask is not None
            and np.max(mask) > 1
            and _GlobalFLIFitter is not None
            and issubclass(self.fitter_class, _GlobalFLIFitter)
        ):
            logging.info(
                "Multi-label mask detected. Switching to Global Cluster Fitting..."
            )
            g_fitter = self.fitter_class(
                self.freq, image_cube[0, 0, :], irf_cube[0, 0, :]
            )
            return g_fitter.process_clusters(
                image_cube,
                irf_cube,
                mask,
                estimator=estimator,
                model_type=model_type,
                fit_indices=fit_indices,
                **kwargs,
            )

        if mask is None:
            mask = np.sum(image_cube, axis=2) > 20

        final_coords = [
            (r, c) for r, c in np.argwhere(mask) if np.sum(image_cube[r, c, :]) > 0
        ]

        if not final_coords:
            logging.warning("No valid pixels found in mask.")
            return None

        tasks = (
            delayed(self._fit_task)(
                image_cube[r, c, :],
                irf_cube[r, c, :],
                (r, c),
                model_type,
                estimator,
                p0,
                bounds,
                fit_indices,
                kwargs,
            )
            for r, c in final_coords
        )

        results = Parallel(n_jobs=n_jobs, backend=backend)(
            tqdm(
                tasks,
                total=len(final_coords),
                desc=f"Fitting Pixels ({estimator})",
                unit="px",
            )
        )

        internal_popt_len = 6 if model_type == "bi-exponential" else 4
        p_maps = np.zeros((H, W, internal_popt_len), dtype=np.float32)
        e_maps = np.zeros((H, W, internal_popt_len), dtype=np.float32)

        r2_map = np.zeros((H, W), dtype=np.float32)
        stat_map = np.zeros((H, W), dtype=np.float32)
        red_stat_map = np.zeros((H, W), dtype=np.float32)
        conv_map = np.zeros((H, W), dtype=np.float32)
        pixel_health_map = np.zeros((H, W), dtype=np.float32)
        rmse_map = np.zeros((H, W), dtype=np.float32)

        fit_map = np.zeros((H, W, T), dtype=np.float32)
        res_map = np.zeros((H, W, T), dtype=np.float32)

        for coords, res, f_curve, r_curve, health in results:
            r, c = coords
            pixel_health_map[r, c] = health
            p_maps[r, c, :] = res[0]
            e_maps[r, c, :] = res[1]
            r2_map[r, c] = res[2]
            stat_map[r, c] = res[3]
            red_stat_map[r, c] = res[4]
            conv_map[r, c] = res[6]
            rmse_map[r, c] = res[7]
            fit_map[r, c, :] = f_curve
            res_map[r, c, :] = r_curve

        S_map = p_maps[..., 0]
        if model_type == "bi-exponential":
            tau1_m, tau2_m = p_maps[..., 2], p_maps[..., 3]
            alpha1_m = p_maps[..., 1]
            param_maps = {
                "photon_count_map": S_map,
                "alpha1_map": alpha1_m,
                "tau1_map": tau1_m,
                "tau2_map": tau2_m,
                "tau_mean_map": (alpha1_m * tau1_m + (1.0 - alpha1_m) * tau2_m).astype(
                    np.float32
                ),
                "v_shift_map": p_maps[..., 4],
                "fret_efficiency_map": (
                    1.0
                    - np.divide(
                        tau1_m,
                        tau2_m,
                        out=np.zeros_like(tau2_m, dtype=np.float32),
                        where=tau2_m > 0,
                    )
                ).astype(np.float32),
                "h_shift_map": p_maps[..., 5].astype(np.float32),
            }
        else:
            param_maps = {
                "photon_count_map": S_map,
                "tau_map": p_maps[..., 1],
                "v_shift_map": p_maps[..., 2],
                "h_shift_map": p_maps[..., 3].astype(np.float32),
            }

        gate_lo, gate_hi = (0, T) if fit_indices is None else fit_indices
        gate_lo, gate_hi = max(gate_lo, 0), min(gate_hi, T)
        pearson_map = np.zeros((H, W), dtype=np.float32)
        fitted = pixel_health_map > 0
        pearson_map[fitted] = pearson_chi_square(
            fit_map[fitted, gate_lo:gate_hi], image_cube[fitted, gate_lo:gate_hi]
        )
        pearson_dof = max((gate_hi - gate_lo) - internal_popt_len, 1)

        param_maps.update(
            {
                "R2_map": r2_map,
                "chi2_map": stat_map,
                "reduced_chi2_map": red_stat_map,
                "pearson_chi2_map": pearson_map,
                "pearson_reduced_chi2_map": (pearson_map / pearson_dof).astype(
                    np.float32
                ),
                "rmse_map": rmse_map,
                "convergence_map": conv_map,
                "pixel_health_map": pixel_health_map,
            }
        )

        return {
            "name": data_name,
            "method": f"CPU_{estimator}",
            "results": {
                "maps": param_maps,
                "error_maps": e_maps,
                "TR_maps": {"fit_map": fit_map, "residual_map": res_map},
            },
        }

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
        if dataset is None:
            return
        if not os.path.exists(folder):
            os.makedirs(folder)

        h5_path = os.path.join(folder, f"{dataset['name']}_results.h5")
        with h5py.File(h5_path, "w") as f:
            res_grp = f.create_group("results")

            maps_grp = res_grp.create_group("maps")
            for k, v in dataset["results"]["maps"].items():
                maps_grp.create_dataset(
                    k, data=v, compression="gzip", compression_opts=4
                )

            err_grp = res_grp.create_group("error_maps")
            err_grp.create_dataset(
                "errors",
                data=dataset["results"]["error_maps"],
                compression="gzip",
                compression_opts=4,
            )

            tr_grp = res_grp.create_group("TR_maps")
            for k, v in dataset["results"]["TR_maps"].items():
                tr_grp.create_dataset(k, data=v, compression="gzip", compression_opts=4)

        logging.info(f"Analysis complete. Results saved to: {h5_path}")

    def load_map(self, h5_path: str, map_name: str = "tau1_map") -> Any:
        """
        Load map.

        Parameters
        ----------
        h5_path : str
            Filesystem path used by the routine.
        map_name : str
            Name of the saved parameter map to load.

        Returns
        -------
        Any
            Object produced by load map.
        """
        with h5py.File(h5_path, "r") as f:
            if f"results/maps/{map_name}" in f:
                return f[f"results/maps/{map_name}"][()]
            else:
                logging.warning(f"Map {map_name} not found in {h5_path}")
                return None
