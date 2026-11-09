# simulator/noise_models.py

"""
Apply Poisson, dark-count, read-noise, jitter, and pile-up models to simulated decays.

This module belongs to :mod:`pyfli.simulator` and is part of PyFLI synthetic FLI/FLIM
data generation, hardware noise modeling, calibration, and validation tools. Public API
includes classes :class:`NoiseEngine`.
"""

from typing import Any

import numpy as np


class NoiseEngine:
    """
    Apply detector noise transformations to clean simulated decays. Static methods model
    Poisson counting, dark counts, read noise, temporal jitter, and simple TCSPC pile-up
    filtering.
    """

    @staticmethod
    def apply_poisson(clean_signal: np.ndarray) -> Any:
        """
        Apply poisson.

        Parameters
        ----------
        clean_signal : np.ndarray
            Noise-free simulated signal before detector noise is applied.

        Returns
        -------
        Any
            Object produced by apply poisson.
        """
        return np.random.poisson(np.clip(clean_signal, 0, None)).astype(np.float64)

    @staticmethod
    def apply_dcr(decay: np.ndarray, dcr_level: float = 0.5) -> Any:
        """
        Simulates Dark Count Rate (thermal noise).
        dcr_level: average dark photons per bin per measurement.
        """
        dark_noise = np.random.poisson(dcr_level, size=decay.shape)
        return decay + dark_noise

    @staticmethod
    def apply_read_noise(decay: np.ndarray, sigma_read: float = 1.5) -> Any:
        """
        Simulates electronic read noise (Gaussian).
        Common in ICCD sensors during CCD readout.
        """
        read_noise = np.random.normal(0, sigma_read, size=decay.shape)
        return decay + read_noise

    @staticmethod
    def apply_jitter(decay: np.ndarray, max_shift: int = 2) -> np.ndarray:
        """
        Apply jitter.

        Parameters
        ----------
        decay : np.ndarray
            Time-resolved decay signal or decay cube.
        max_shift : int
            Maximum random temporal jitter shift in bins.

        Returns
        -------
        np.ndarray
            Decay array after timing jitter has been applied.
        """
        n = len(decay)
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return decay
        if shift > 0:
            return np.concatenate([np.zeros(shift), decay[: n - shift]])
        return np.concatenate([decay[-shift:], np.zeros(-shift)])

    @staticmethod
    def tcspc_pileup_filter(
        arrival_times: np.ndarray, t_rep: float, mode: str = "wrap"
    ) -> Any:
        # Simplistic pile-up: real systems might only take the first photon per cycle
        """
        Run the TCSPC pileup filter routine.

        Parameters
        ----------
        arrival_times : np.ndarray
            Photon arrival times passed to the TCSPC pile-up filter.
        t_rep : float
            Laser repetition period used for TCSPC pile-up filtering.
        mode : str
            'wrap' (default) folds photons back via modulo; 'truncate' drops them.

        Returns
        -------
        Any
            Object produced by TCSPC pileup filter.
        """
        if mode == "truncate":
            return arrival_times[(arrival_times >= 0) & (arrival_times < t_rep)]
        if mode == "wrap":
            return arrival_times % t_rep
        raise ValueError(f"mode must be 'truncate' or 'wrap', got {mode!r}")
