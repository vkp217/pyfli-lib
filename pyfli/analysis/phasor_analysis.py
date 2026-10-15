"""Compute phasor coordinates from binned decay data and save phasor lifetime maps."""

import os
from typing import Any

import numpy as np

from pyfli import logging

from ..analyticalWorkflow import AnalyticalHelpers, PhasorAnalyzer
from ..data_vnp import ColorProcessor


def compute_freq_axis(
    binned_irf: np.ndarray, laser_period_ns: float = 12.5
) -> tuple[Any, ...]:
    """Derive frequency and time axis from IRF array shape.

    Parameters
    ----------
    binned_irf      : np.ndarray  (H, W, T)
    laser_period_ns : float       default 12.5 ns → 80 MHz

    Returns
    -------
    freq         : tuple from AnalyticalHelpers.freq_computation()
    freq_hz      : float
    time_axis_ns : np.ndarray  length T, gate start times ``k * T / num_gates``
                   covering exactly one laser period (endpoint excluded), so the
                   phasor sums are the discrete Fourier transform at the laser frequency
    """
    num_gates = binned_irf.shape[2]
    gate_delay = laser_period_ns / num_gates
    freq = AnalyticalHelpers(
        laser_period=laser_period_ns, gate_delay=gate_delay, num_gate=num_gates
    ).freq_computation()
    freq_hz = freq[1] * 1e6
    time_axis_ns = np.arange(num_gates) * gate_delay
    return freq, freq_hz, time_axis_ns


def compute_phasor(
    binned_decay: np.ndarray,
    binned_irf: np.ndarray,
    freq_hz: np.ndarray,
    time_axis_ns: np.ndarray,
    n_harmonics: int = 3,
) -> tuple[Any, ...]:
    """Compute and calibrate phasor coordinates.

    Parameters
    ----------
    binned_decay : np.ndarray  (H, W, T)
    binned_irf   : np.ndarray  (H, W, T)
    freq_hz      : float
    time_axis_ns : np.ndarray
    n_harmonics  : int

    Returns
    -------
    phasor_obj  : PhasorAnalyzer   reuse this instance for all plotting calls
    Gc, Sc      : np.ndarray  (n_harmonics, H, W)  calibrated phasor coordinates
    tau_map_ns  : np.ndarray  (H, W)               apparent lifetime from 1st harmonic
    """
    phasor_obj = PhasorAnalyzer(
        frequency_hz=freq_hz, time_axis_ns=time_axis_ns, n_harmonics=n_harmonics
    )
    G, S = phasor_obj.create_phasor_gpu(binned_decay)
    Gc, Sc = phasor_obj.calibrate_pixelwise(G, S, binned_irf)
    tau_map_ns = phasor_obj.compute_lifetime(Gc[0], Sc[0])
    return phasor_obj, Gc, Sc, tau_map_ns


def plot_phasor_figures(
    phasor_obj: np.ndarray,
    Gc: Any,
    Sc: Any,
    binned_decay: np.ndarray,
    mask: np.ndarray,
    colorset: str = "jet",
    saver: Any | None = None,
) -> np.ndarray:
    """Generate all standard phasor figures.

    Produces: phasor diagram, overlay subplots, harmonics plot (if ≥ 3 harmonics),
    pure phasor map, and traceable analysis.

    Parameters
    ----------
    phasor_obj  : PhasorAnalyzer   returned by compute_phasor()
    Gc, Sc      : np.ndarray  (n_harmonics, H, W)
    binned_decay: np.ndarray  (H, W, T)
    mask        : np.ndarray  (H, W) bool
    colorset    : str         colormap name for hexbin density plots
    saver       : DataSaver or None

    Returns
    -------
    figs : dict[str, Figure]   keyed by figure name
    """
    plasma_m = ColorProcessor().lowest_zero("plasma")
    viridis_m = ColorProcessor().lowest_zero("viridis")

    figs = {}

    figs["phasor_diagram"] = phasor_obj.plot_phasor_diagram(
        Gc[0],
        Sc[0],
        mask=mask,
        colors=None,
        hexbin_color=colorset,
        ax=None,
        figsize=(8, 3),
        half_circle=True,
        title="Phasor Diagram",
        xlim=(-0.1, 1.1),
        ylim=(0.0, 0.6),
        kdeplot=False,
        kde_color="white",
        kde_levels=5,
        kde_linewidths=1,
        kde_alpha=0.5,
    )
    if saver:
        saver.save_plot("Phasor_Plot", fig=figs["phasor_diagram"], close=False)

    figs["phasor_subplots"] = phasor_obj.plot_overlay_subplots(
        binned_decay,
        Gc[0],
        Sc[0],
        mask=mask,
        colormaps=[plasma_m, viridis_m],
        hexbin_color="jet",
        noise_removed=True,
        figsize=(10, 8),
    )
    if saver:
        saver.save_plot("Phasor_subplots", fig=figs["phasor_subplots"], close=False)

    n_harmonics = Gc.shape[0]
    if n_harmonics >= 2:
        figs["phasor_harmonics"] = phasor_obj.plot_phasor_harmonics(
            Gc,
            Sc,
            harmonics=tuple(range(1, n_harmonics + 1)),
            mask=mask,
            hexbin_color=colorset,
            figsize=(14, 4),
        )
        if saver:
            saver.save_plot(
                "phasor_harmonics_Plot", fig=figs["phasor_harmonics"], close=False
            )

    figs["phasor_pure_map"] = phasor_obj.plot_pure_phasor_map(
        Gc[0], Sc[0], binned_decay, noise_removed=True, colormap="viridis"
    )

    figs["phasor_traceable"] = phasor_obj.plot_traceable_analysis(
        Gc[0], Sc[0], binned_decay, mask=mask, colormap="viridis"
    )

    return figs


def save_phasor_result(
    save_dir: str, tau_map_ns: np.ndarray, saver: Any | None = None
) -> np.ndarray:
    """
    Save phasor result.

    Parameters
    ----------
    save_dir : str
        Directory where outputs are saved.
    tau_map_ns : np.ndarray
        Lifetime map in nanoseconds.
    saver : Any | None
        Optional saver used to persist messages or figures.

    Returns
    -------
    np.ndarray
        Saved phasor result arrays or metadata.
    """
    result = {
        "results": {
            "maps": {
                "tau_map": tau_map_ns,
                "mean_lifetime": tau_map_ns,
            },
            "TR_maps": {},
        }
    }
    path = os.path.join(save_dir, "phasor_tau_map.npy")
    np.save(path, result)
    if saver:
        saver.log("Phasor tau map saved as phasor_tau_map.npy")
    logging.info(f"Phasor result saved → {path}")
    return path
