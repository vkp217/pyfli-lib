# solver/comparison.py
"""
Compare least-squares and maximum-likelihood fitters across selected pixels and
datasets.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLI fitting routines. Public API includes
classes :class:`FittingComparator`.
"""

import contextlib
import io
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class FittingComparator:
    """
    Compare fitting methods on selected pixels or whole datasets. It runs base and MLE
    fitters, summarizes model statistics, plots comparisons, and saves comparison
    outputs.

    Parameters
    ----------
    freq : float
        Acquisition frequency information used to derive timing constants.
    base_fitter_class : Any
        Least-squares fitter class used as a fitting backend.
    mle_fitter_class : Any
        Maximum-likelihood fitter class used as a fitting backend.
    """

    def __init__(
        self, freq: float, base_fitter_class: Any, mle_fitter_class: Any
    ) -> None:
        self.freq = freq
        self.BaseClass = base_fitter_class
        self.MLEClass = mle_fitter_class

        self.method_mapping = {
            "least_squares": ("NLSF", self.BaseClass),
            "trust_region": ("NLSF", self.BaseClass),
            "unconstrained": ("NLSF", self.BaseClass),
            "poisson": ("MLE", self.MLEClass),
            "pearson": ("MLE", self.MLEClass),
            "neyman": ("MLE", self.MLEClass),
        }

    @staticmethod
    def _print_summary_table(rows: np.ndarray, model_type: str) -> Any:
        """rows: list of [method, category, success, elapsed, r2, stat, red_stat, popt | None]"""
        is_bi = model_type == "bi-exponential"
        if is_bi:
            cols = [
                ("Method", 14, "<"),
                ("Type", 4, "^"),
                ("A", 7, ">"),
                ("α", 7, ">"),
                ("τ₁", 8, ">"),
                ("τ₂", 8, ">"),
                ("R²", 7, ">"),
                ("Red.χ²", 8, ">"),
                ("Raw.χ²", 8, ">"),
                ("v-shift", 7, ">"),
                ("h-shift", 8, ">"),
            ]
        else:
            cols = [
                ("Method", 14, "<"),
                ("Type", 4, "^"),
                ("A", 7, ">"),
                ("τ", 8, ">"),
                ("R²", 7, ">"),
                ("Red.χ²", 8, ">"),
                ("Raw.χ²", 8, ">"),
                ("v-shift", 7, ">"),
                ("h-shift", 8, ">"),
            ]

        def fmt_cell(text: np.ndarray, width: float, align: Any) -> Any:
            """
            Run the fmt cell routine.

            Parameters
            ----------
            text : np.ndarray
                Text rendered into a UI label or formatted table cell.
            width : float
                Gate width used by the gate matrix.
            align : Any
                Text alignment used when formatting a table cell.

            Returns
            -------
            Any
                Object produced by fmt cell.
            """
            s = str(text)
            if align == "<":
                return f" {s:<{width}} "
            if align == ">":
                return f" {s:>{width}} "
            return f" {s:^{width}} "

        def row_str(cells: Any) -> Any:
            """
            Run the row str routine.

            Parameters
            ----------
            cells : Any
                Rendered table cells for one formatted row.

            Returns
            -------
            Any
                Object produced by row str.
            """
            return (
                "│"
                + "│".join(fmt_cell(c, w, a) for c, (_, w, a) in zip(cells, cols))
                + "│"
            )

        def sep(left: np.ndarray, mid: Any, right: np.ndarray) -> Any:
            """
            Run the sep routine.

            Parameters
            ----------
            left : np.ndarray
                Left border character used by the table separator.
            mid : Any
                Middle separator character used by the table formatter.
            right : np.ndarray
                Right border character used by the table separator.

            Returns
            -------
            Any
                Object produced by sep.
            """
            return left + mid.join("─" * (w + 2) for _, w, _ in cols) + right

        # print(), not logging.info(): save_results() captures this table via
        # contextlib.redirect_stdout, which only intercepts stdout -- logging
        # records go to the logging handler (stderr by default) and would be
        # silently dropped from the saved log otherwise.
        print("")
        print(sep("┌", "┬", "┐"))
        print(row_str([c[0] for c in cols]))
        print(sep("├", "┼", "┤"))

        for method, category, success, elapsed, r2, stat, red_stat, popt in rows:
            if popt is None:
                filler = ["—"] * (len(cols) - 2)
                cells = [method, category, *filler]
            elif is_bi and len(popt) >= 6:
                cells = [
                    method,
                    category,
                    f"{popt[0]:.2f}",
                    f"{popt[1]:.4f}",
                    f"{popt[2]:.3f}",
                    f"{popt[3]:.3f}",
                    f"{r2:.4f}",
                    f"{red_stat:.4f}",
                    f"{stat:.2f}",
                    f"{popt[4]:.2f}",
                    f"{popt[5]:.3f}",
                ]
            elif is_bi and len(popt) >= 5:
                cells = [
                    method,
                    category,
                    f"{popt[0]:.2f}",
                    f"{popt[1]:.4f}",
                    f"{popt[2]:.3f}",
                    f"{popt[3]:.3f}",
                    f"{r2:.4f}",
                    f"{red_stat:.4f}",
                    f"{stat:.2f}",
                    f"{popt[4]:.2f}",
                    "—",
                ]
            else:
                cells = [
                    method,
                    category,
                    f"{popt[0]:.2f}",
                    f"{popt[1]:.3f}",
                    f"{r2:.4f}",
                    f"{red_stat:.4f}",
                    f"{stat:.2f}",
                    f"{popt[2]:.2f}",
                    f"{popt[3]:.3f}" if len(popt) > 3 else "—",
                ]
            print(row_str(cells))

        print(sep("└", "┴", "┘"))
        print("")

    @staticmethod
    def _weighted_residual(method: str, y: np.ndarray, model: Any) -> Any:
        """Return normalised residuals appropriate for each estimator.

        Poisson MLE  → signed deviance residual  sign(y−m)·√(2(m−y+y·ln(y/m)))
        Pearson χ²   → (y − m) / √m
        Neyman  χ²   → (y − m) / √max(y, 1)
        NLSF         → (y − m) / √max(m, 1)   (approx Poisson weight)
        """
        m = np.clip(model, 1e-9, None)
        if method == "poisson":
            safe_y = np.where(y > 0, y, 1e-9)
            dev = 2.0 * (m - y + y * np.log(safe_y / m))
            return np.sign(y - m) * np.sqrt(np.maximum(dev, 0.0))
        elif method == "pearson":
            return (y - m) / np.sqrt(m)
        elif method == "neyman":
            return (y - m) / np.sqrt(np.maximum(y, 1.0))
        else:  # least_squares, trust_region, unconstrained
            return (y - m) / np.sqrt(np.maximum(m, 1.0))

    def compare_selected(
        self,
        methods: np.ndarray,
        y_data: np.ndarray,
        irf_data: np.ndarray,
        model_type: str = "bi-exponential",
        p0: Any | None = None,
        bounds: np.ndarray | None = None,
        yscale: str = "log",
        plot: bool = True,
        fit_indices: tuple[int, int] | None = None,
        shift_method: str = "zero_pad",
        excitation: str = "bin_start",
        gate_width: float | None = None,
    ) -> tuple[Any, ...]:
        """
        Compare selected.

        Parameters
        ----------
        methods : np.ndarray
            Names of fitting methods to include in the comparison.
        y_data : np.ndarray
            Observed decay data passed to the fitter.
        irf_data : np.ndarray
            Instrument response data used to convolve or simulate decays.
        model_type : str
            FLI model family, such as mono- or bi-exponential.
        p0 : Any | None
            Initial parameter vector supplied to the optimizer.
        bounds : np.ndarray | None
            Lower and upper parameter bounds supplied to the optimizer.
        yscale : str
            Scale used for the y-axis.
        plot : bool
            Whether diagnostic plots should be generated.
        fit_indices : tuple[int, int] | None
            Optional (gate_num_start, gate_num_end) gate range to fit over.
        shift_method : str
            ``"zero_pad"`` or ``"circular"`` forward model (see
            :func:`pyfli.cm_funcs.forward_model.model_numpy`).
        excitation : str
            ``"bin_start"`` or ``"uniform"`` excitation within each IRF gate.
        gate_width : float | None
            Detection-gate width in ns (``None``: the gate spacing).

        Returns
        -------
        tuple[Any, ...]
            Tuple containing comparison metrics for the selected fitting methods.
        """
        results_table = []
        if y_data.ndim != 1 or irf_data.ndim != 1:
            raise ValueError("compare_selected expects 1D decay and IRF traces")
        y_in = y_data.astype(np.float32)
        irf_in = irf_data.astype(np.float32)

        plot_data = {"y": y_in, "irf": irf_in, "fits": {}, "residuals": {}, "t": None}

        W = 62
        n_methods = len([m for m in methods if m in self.method_mapping])
        title = f"FLI Fitting Results  |  {model_type.upper()}"
        subtitle = f"{n_methods} method{'s' if n_methods != 1 else ''} queued"
        print(f"\n┌{'─' * W}┐")
        print(f"│  {title:<{W - 2}}│")
        print(f"│  {subtitle:<{W - 2}}│")
        print(f"└{'─' * W}┘\n")

        for method in methods:
            if method not in self.method_mapping:
                continue
            category, Fitter = self.method_mapping[method]

            fitter_inst = Fitter(
                self.freq,
                y_in,
                irf_in,
                shift_method=shift_method,
                fit_indices=fit_indices,
                excitation=excitation,
                gate_width=gate_width,
            )
            if plot_data["t"] is None:
                plot_data["t"] = fitter_inst.t  # physical time axis (ns)
            start_time = time.perf_counter()

            try:
                res = fitter_inst.fit_with_estimator(
                    estimator_type=method, model_type=model_type, p0=p0, bounds=bounds
                )
                elapsed = (time.perf_counter() - start_time) * 1000

                popt = res[0]
                fit_full = fitter_inst.model_fit(
                    fitter_inst.t, popt, model_type=model_type
                ).astype(np.float32)

                # Normalised residuals only over the fitted region
                idx = fitter_inst.fit_indices
                resid = np.full_like(y_in, np.nan)
                resid[idx] = self._weighted_residual(method, y_in[idx], fit_full[idx])

                r2 = res[2]
                stat = res[3]
                red_stat = res[4]
                success = "YES" if res[6] == 1 else "NO"

                if plot:
                    plot_data["fits"][method] = fit_full
                    plot_data["residuals"][method] = resid

                results_table.append(
                    [
                        method.upper(),
                        category,
                        success,
                        f"{elapsed:.2f} ms",
                        r2,
                        stat,
                        red_stat,
                        popt,
                    ]
                )

            except Exception:
                results_table.append(
                    [method.upper(), category, "FAIL", "N/A", 0.0, 0.0, 0.0, None]
                )

        self._print_summary_table(results_table, model_type)

        fig = None
        if plot and plot_data["fits"]:
            fig = self._plot_comparison(plot_data, yscale, model_type)

        return results_table, fig

    def run_all(
        self,
        y_data: np.ndarray,
        irf_data: np.ndarray,
        model_type: str = "bi-exponential",
        p0: Any | None = None,
        bounds: np.ndarray | None = None,
        yscale: str = "log",
        plot: bool = True,
        fit_indices: tuple[int, int] | None = None,
    ) -> Any:
        """
        Run all.

        Parameters
        ----------
        y_data : np.ndarray
            Observed decay data passed to the fitter.
        irf_data : np.ndarray
            Instrument response data used to convolve or simulate decays.
        model_type : str
            FLI model family, such as mono- or bi-exponential.
        p0 : Any | None
            Initial parameter vector supplied to the optimizer.
        bounds : np.ndarray | None
            Lower and upper parameter bounds supplied to the optimizer.
        yscale : str
            Scale used for the y-axis.
        plot : bool
            Whether diagnostic plots should be generated.
        fit_indices : tuple[int, int] | None
            Optional (gate_num_start, gate_num_end) gate range to fit over.

        Returns
        -------
        Any
            Object produced by run all.
        """
        return self.compare_selected(
            list(self.method_mapping.keys()),
            y_data,
            irf_data,
            model_type,
            p0,
            bounds,
            yscale=yscale,
            plot=plot,
            fit_indices=fit_indices,
        )

    def _plot_comparison(
        self, data: np.ndarray, yscale: np.ndarray, model_type: str
    ) -> np.ndarray:
        """
        Plot comparison.

        Parameters
        ----------
        data : np.ndarray
            Data array or mapping processed by the routine.
        yscale : np.ndarray
            Scale used for the y-axis.
        model_type : str
            FLI model family, such as mono- or bi-exponential.

        Returns
        -------
        np.ndarray
            Matplotlib figure or axes containing the method comparison.
        """
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
        )

        t = data["t"] if data.get("t") is not None else np.arange(len(data["y"]))
        x_label = "Time (ns)" if data.get("t") is not None else "Sample Index"

        ax1.step(t, data["y"], where="mid", color="gray", alpha=0.3, label="Raw Data")

        colors = plt.cm.tab10(np.linspace(0, 1, len(data["fits"])))
        for i, (name, fit) in enumerate(data["fits"].items()):
            c = colors[i]
            ax1.plot(t, fit, label=f"Fit: {name.upper()}", color=c, linewidth=1.5)
            resid = data["residuals"][name]
            valid = ~np.isnan(resid)
            ax2.plot(t[valid], resid[valid], color=c, alpha=0.7, label=name.upper())

        ax1.set_yscale(yscale)
        ax1.set_ylabel("Photon Counts")
        ax1.set_title(f"FLI Diagnostic Comparison ({model_type.upper()})")
        ax1.legend(loc="upper right", fontsize="x-small", ncol=2)
        ax1.grid(True, which="both", ls="-", alpha=0.05)

        ax2.axhline(0, color="black", linewidth=1.2, alpha=0.8)
        ax2.set_ylabel("Normalised Residuals")
        ax2.set_xlabel(x_label)
        ax2.legend(loc="upper right", fontsize="x-small", ncol=2)
        ax2.grid(True, alpha=0.05)

        plt.tight_layout()
        plt.show()
        return fig

    def save_results(
        self,
        saver: Any,
        results_table: np.ndarray,
        fig: Any | None = None,
        model_type: str = "bi-exponential",
        name: str = "fitting_comparison",
    ) -> None:
        """
        Save results.

        Parameters
        ----------
        saver : Any
            Optional saver used to persist messages or figures.
        results_table : np.ndarray
            Rows of fit comparison results written to disk.
        fig : Any | None
            Matplotlib figure object to update or save.
        model_type : str
            FLI model family, such as mono- or bi-exponential.
        name : str
            Dataset, experiment, figure, or output name.

        Returns
        -------
        None
            No object is returned; the function save results.
        """
        # 1. Write the table to the log FILE only -- saver.log() would also
        # echo every line through logging.info, which is noisy for a
        # multi-row table (unlike single status messages such as
        # "IMAGE SAVED >> ...").
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._print_summary_table(results_table, model_type)
        for line in buf.getvalue().splitlines():
            saver.log_to_file(line)

        # 2. Save the diagnostic figure if one was produced
        if fig is not None:
            saver.save_plot(name, fig=fig, dpi=300, close=False)
