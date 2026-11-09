"""
Provide ROI mask loading, hot-pixel detection, and assorted data I/O helpers.

This module belongs to :mod:`pyfli.io` and is part of PyFLI detector importers, file
readers, saving helpers, and processed-data loaders. Public API includes classes
:class:`DataIOUtils`.
"""

import json

# dataIO_utils.py
import os
from typing import Any

import h5py
import numpy as np

from pyfli import logging
from pyfli.cm_stats.robust import MAD_TO_SIGMA, median_absolute_deviation


class DataIOUtils:
    """
    Collect utility methods for ROI loading, mask handling, and hot-pixel detection.
    These helpers support detector import and preprocessing workflows.
    """

    def __init__(self) -> None:
        pass

    def load_phasors_hdf5(self, file_path: str) -> tuple[Any, ...]:
        """
        Load phasors hdf5.

        Parameters
        ----------
        file_path : str
            Path to the file being loaded.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing phasor arrays and metadata read from HDF5.
        """
        with h5py.File(file_path, "r") as f:
            Gc = f["Gc"][:]
            Sc = f["Sc"][:]
            tau = f["tau"][:] if "tau" in f else None
        if tau is not None:
            if Gc.shape[1:] != tau.shape:
                raise ValueError(
                    f"Dimension mismatch: Phasor spatial size {Gc.shape[1:]} "
                    f"does not match Tau size {tau.shape}."
                )
            if Gc.shape != Sc.shape:
                raise ValueError("Critical Error: Gc and Sc dimensions do not match.")
        return Gc, Sc, tau

    def roiNloader(
        self, map_array: np.ndarray, file_path: str, visualize: bool = True
    ) -> np.ndarray:
        """
        Run the ROI nloader routine.

        Parameters
        ----------
        map_array : np.ndarray
            Parameter map or mask array being loaded or displayed.
        file_path : str
            Path to the file being loaded.
        visualize : bool
            Whether to display the loaded ROI mask.

        Returns
        -------
        np.ndarray
            ROI mask or coordinate array loaded from file.
        """
        if map_array.ndim == 3:
            H, W, _ = map_array.shape
        elif map_array.ndim == 2:
            H, W = map_array.shape
        else:
            raise ValueError("Correct data map is not provided")
        mask = np.zeros((H, W), dtype=bool)
        with open(file_path) as fid:
            J = json.load(fid)
        p = J.get("Named ROI Descriptions", [])
        for roi in p:
            try:
                contours = roi["ROI Descriptor"]["Contours"]
                for contour in contours:
                    coords = contour["Coordinates"]
                    if len(coords) >= 2:
                        x = int(coords[0])
                        y = int(coords[1])
                        if 0 <= y < H and 0 <= x < W:
                            mask[y, x] = True
            except KeyError:
                continue
        return mask

    @staticmethod
    def hot_pixels(
        background: np.ndarray,
        threshold_sigma: float = 5.0,
        save_path: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect hot pixels from loaded SPAD background data.

        Parameters
        ----------
        background : np.ndarray
            Background cube (H, W, T) or stack (N, H, W, T).
        threshold_sigma : float
            MAD threshold used for hot-pixel detection.
        save_path : str | None
            Optional path used to save the resulting mask.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Hot-pixel map, sum, mean, and median background cubes.
        """
        if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
            raise ValueError(
                f"threshold_sigma must be positive, got {threshold_sigma}."
            )

        array = np.asarray(background)

        if array.ndim == 3:
            accumulated = array[np.newaxis, ...]

        elif array.ndim == 4:
            accumulated = array

        else:
            raise ValueError(
                "SPAD background data must have shape (H, W, T) "
                f"or (N, H, W, T), got {array.shape}."
            )

        if any(size < 1 for size in accumulated.shape):
            raise ValueError(
                f"SPAD background data cannot contain empty dimensions: {array.shape}."
            )

        if not np.issubdtype(
            accumulated.dtype,
            np.number,
        ):
            raise TypeError(
                f"SPAD background data must be numeric, got dtype {accumulated.dtype}."
            )

        if not np.all(np.isfinite(accumulated)):
            raise ValueError("SPAD background data contains non-finite values.")

        bg_sum = np.sum(
            accumulated,
            axis=0,
        )

        bg_mean = np.mean(
            accumulated,
            axis=0,
        )

        bg_median = np.median(
            accumulated,
            axis=0,
        )

        total_counts = np.sum(
            bg_sum,
            axis=-1,
        )

        median, mad = median_absolute_deviation(total_counts)

        threshold = median + threshold_sigma * MAD_TO_SIGMA * mad

        hot_pixel_map = total_counts > threshold

        n_hot = int(np.sum(hot_pixel_map))

        logging.info(
            f"Detected {n_hot} hot pixels "
            f"({100.0 * n_hot / total_counts.size:.3f}% "
            f"of {total_counts.shape})"
        )

        logging.info(
            f"Threshold: {threshold:.1f}  "
            f"(median={median:.1f}, "
            f"MAD={mad:.1f}, "
            f"σ={threshold_sigma})"
        )

        if save_path:
            import matplotlib.pyplot as plt

            plt.imsave(
                save_path,
                hot_pixel_map.astype(np.uint8) * 255,
                cmap="gray",
            )

            logging.info(f"Hot pixel mask saved to: {save_path}")

        return (
            hot_pixel_map,
            bg_sum,
            bg_mean,
            bg_median,
        )

    def detect_hot_pixels(
        self,
        bg_path: str,
        threshold_sigma: float = 5.0,
        save_path: str | None = None,
        gate_group_path: str | None = None,
        gate_prefix: str | None = None,
    ) -> Any:
        """
        Detect hot pixels from a SPAD HDF5 file or directory.

        Parameters
        ----------
        bg_path : str
            Background HDF5 file or directory.
        threshold_sigma : float
            MAD threshold used for hot-pixel detection.
        save_path : str | None
            Optional path used to save the resulting mask.
        gate_group_path : str | None
            Optional HDF5 gate group.
        gate_prefix : str | None
            Optional split-gate dataset prefix.

        Returns
        -------
        Any
            Hot-pixel map and background statistics.
        """
        from .spad_hdf5 import read_spad_hdf5

        if not bg_path:
            raise ValueError("Background path must be provided.")

        absolute_path = os.path.abspath(bg_path)

        if os.path.isfile(absolute_path):
            if not absolute_path.lower().endswith(
                (
                    ".h5",
                    ".hdf5",
                )
            ):
                raise ValueError(f"Expected an HDF5 file (.h5 / .hdf5), got: {bg_path}")

            files = [absolute_path]

        elif os.path.isdir(absolute_path):
            files = [
                os.path.join(
                    absolute_path,
                    filename,
                )
                for filename in sorted(os.listdir(absolute_path))
                if filename.lower().endswith(
                    (
                        ".h5",
                        ".hdf5",
                    )
                )
            ]

            if not files:
                raise FileNotFoundError(f"No HDF5 files found in: {absolute_path}")

        else:
            raise FileNotFoundError(f"Path not found: {absolute_path}")

        cubes = []
        reference_shape = None

        for file_path in files:
            result = read_spad_hdf5(
                file_path,
                gate_group_path=gate_group_path,
                gate_prefix=gate_prefix,
            )

            if reference_shape is None:
                reference_shape = result.data.shape

            elif result.data.shape != reference_shape:
                raise ValueError(
                    "Background HDF5 files contain mismatched "
                    f"SPAD shapes: {reference_shape} and "
                    f"{result.data.shape} in '{file_path}'."
                )

            cubes.append(result.data)

        return self.hot_pixels(
            np.stack(
                cubes,
                axis=0,
            ),
            threshold_sigma=threshold_sigma,
            save_path=save_path,
        )
