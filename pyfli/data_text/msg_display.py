"""
Format fitting parameters, session settings, and pixel summaries for display or logging.

This module belongs to :mod:`pyfli.data_text` and is part of PyFLI text display helpers
used by interactive fitting workflows. Public API includes classes
:class:`MessageDisplay`.
"""

from typing import Any, ClassVar

import numpy as np

from pyfli import logging


class MessageDisplay:
    """
    Format fitting parameters, session settings, and pixel summaries for notebook or
    console display. An optional saver can persist the same messages alongside analysis
    outputs.

    Parameters
    ----------
    saver : Any | None
        Optional object responsible for persisting display text or outputs.
    """

    def __init__(self, saver: Any | None = None) -> None:
        self.saver = saver

    def _internal_log(self, message: Any) -> None:
        """
        Run the internal log routine.

        Parameters
        ----------
        message : Any
            Message text displayed to the user.

        Returns
        -------
        None
            No object is returned; the function perform internal log.
        """
        if self.saver:
            self.saver.log(message)
        else:
            logging.info(message)

    def disp_params(
        self, res_px: np.ndarray, model_type: str = "bi-exponential"
    ) -> None:
        """
        Run the disp params routine.

        Parameters
        ----------
        res_px : np.ndarray
            Fit result dictionary for one pixel.
        model_type : str
            FLI model family, such as mono- or bi-exponential.

        Returns
        -------
        None
            No object is returned; the function perform disp params.
        """
        if not res_px:
            raise ValueError("Data was not provided (res_px is empty or None)")

        try:
            p, err = res_px[0], res_px[1]
            r2, chi2, red_chi2 = res_px[2], res_px[3], res_px[4]
            conv = res_px[6]
        except IndexError:
            raise IndexError("res_px does not have the expected number of elements.")

        # Build output string
        output = []
        output.append("\n" + "=" * 30)
        output.append(f"FIT PARAMETERS ({model_type.upper()})")
        output.append("-" * 30)

        labels = (
            ["photon_counts", "alpha1", "tau1", "tau2", "v-shift"]
            if model_type == "bi-exponential"
            else ["photon_counts", "tau", "v-shift"]
        )

        for i, label in enumerate(labels):
            output.append(f"{label:8}: {p[i]:.4f} \u00b1 {err[i]:.4f}")

        output.append("-" * 30)
        output.append(f"R2           : {r2:.4f}")
        output.append(f"chi2         : {chi2:.4f}")
        output.append(f"Reduced chi2 : {red_chi2:.4f}")
        output.append(f"Convergence  : {conv}")
        output.append("=" * 30 + "\n")

        # Display and Log
        full_msg = "\n".join(output)
        self._internal_log(full_msg)

    def fit_session(self, **kwargs: Any) -> None:
        """
        Fit session.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function fit session.
        """
        pretty_labels = {
            "model_type": "Decay Model",
            "processor_name": "Processor",
            "fitter_name": "Fitting Method",
            "p0": "Initial Guesses (p0)",
            "use_initial_guess": "Using Guess",
            "use_bounds": "Using Bounds",
        }

        header = "\n" + "-" * 60 + f"\n{'SESSION CONFIGURATION':^60}\n" + "-" * 60
        self._internal_log(header)

        # Log parameters via save_params if saver exists for structured logging
        if self.saver:
            self.saver.save_params(**kwargs)

        for key, value in kwargs.items():
            label = pretty_labels.get(key, key.replace("_", " ").capitalize())
            self._internal_log(f"{label:25}: {value}")

        footer = "-" * 60 + f"\n{'Session Initialized':^60}\n" + "-" * 60 + "\n"
        self._internal_log(footer)

    # Fixed display order: label → candidate map keys (first match wins)
    _PIXEL_FIELDS: ClassVar[list[tuple[str, list[str]]]] = [
        ("A", ["photon_count_map"]),
        ("α", ["alpha1_map", "alpha_map"]),
        ("τ₁", ["tau1_map", "tau_map"]),
        ("τ₂", ["tau2_map"]),
        ("R²", ["R2_map"]),
        ("Red.χ²", ["reduced_chi2_map"]),
        ("Raw.χ²", ["chi2_map"]),
        ("Pearson", ["pearson_reduced_chi2_map"]),
        ("v-shift", ["v_shift_map"]),
        ("h-shift", ["h_shift_map"]),
    ]

    def get_pixel_summary(self, data_maps: np.ndarray, px: np.ndarray) -> np.ndarray:
        """
        Return pixel summary.

        Parameters
        ----------
        data_maps : np.ndarray
            Dictionary of parameter maps used to summarize a pixel.
        px : np.ndarray
            Pixel column coordinate.

        Returns
        -------
        np.ndarray
            Per-pixel summary values for the requested coordinate.
        """
        x, y = px
        rows = []
        for label, candidates in self._PIXEL_FIELDS:
            val = "—"
            for key in candidates:
                m = data_maps.get(key)
                if isinstance(m, np.ndarray) and m.ndim == 2:
                    try:
                        v = m[x, y]
                        val = f"{float(v):.4f}"
                    except Exception:
                        val = "error"
                    break
            rows.append((label, val))

        label_w = max(len(lbl) for lbl, _ in rows)
        rule = "─" * (label_w + 14)
        lines = [f"\n  Pixel {px}", f"  {rule}"]
        for label, val in rows:
            lines.append(f"  {label:<{label_w}}   {val}")
        lines.append(f"  {rule}\n")

        output = "\n".join(lines)
        logging.info(output)

        if self.saver:
            self.saver.log(output)

        return rows
