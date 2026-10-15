"""Compact phasor analyzer for CPU and optional GPU workflows."""

from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from pyfli import logging
from pyfli.reconstruction.common_reconstruct import (
    periodic_bin_exponential,
    periodic_instrument_exponential,
)

from .phasor_lifetime import normal_lifetime
from .phasor_simple_plots import PhasorPlotsMixin


class PhasorAnalyzer(PhasorPlotsMixin):
    """
    Compute, calibrate, and interpret FLI phasors from decay data. The analyzer
    supports NumPy and optional Torch execution, multi-harmonic phasors, IRF
    calibration, lifetime conversion, fractional component estimates, and plotting
    through the phasor mixin.

    Parameters
    ----------
    frequency_hz : float
        Excitation frequency in hertz.
    time_axis_ns : np.ndarray
        Time axis for decay samples in nanoseconds.
    n_harmonics : int
        Number of phasor harmonics to compute.
    device : Any | None
        Execution device, such as a Torch device or device string.
    """

    def __init__(
        self,
        frequency_hz: float,
        time_axis_ns: np.ndarray,
        n_harmonics: int = 1,
        device: Any | None = None,
    ) -> None:
        self.frequency = float(frequency_hz)
        self.time_axis_ns = np.asarray(time_axis_ns)
        self.n_harmonics = int(n_harmonics)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.omega = 2 * np.pi * self.frequency
        self.t_s_np = self.time_axis_ns * 1e-9
        self.t_s_torch = torch.tensor(
            self.t_s_np, dtype=torch.float32, device=self.device
        )
        self.eps = 1e-12

    # ── phasor computation ────────────────────────────────────────────────────

    def _phasor_numpy(self, decay: np.ndarray) -> tuple[Any, ...]:
        """
        Run the phasor numpy routine.

        Parameters
        ----------
        decay : np.ndarray
            Time-resolved decay signal or decay cube.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing NumPy-computed phasor coordinates and intensity values.
        """
        decay = np.asarray(decay, dtype=np.float64)
        *spatial, T = decay.shape
        decay_flat = decay.reshape(-1, T)

        I_sum = np.clip(np.sum(decay_flat, axis=1), self.eps, None)
        G_all, S_all = [], []

        for k in range(1, self.n_harmonics + 1):
            omega_k = k * self.omega
            cos_k = np.cos(omega_k * self.t_s_np)
            sin_k = np.sin(omega_k * self.t_s_np)
            G_all.append((np.sum(decay_flat * cos_k, axis=1) / I_sum).reshape(spatial))
            S_all.append((np.sum(decay_flat * sin_k, axis=1) / I_sum).reshape(spatial))

        return np.stack(G_all), np.stack(S_all)

    def _phasor_torch(self, decay: np.ndarray) -> tuple[Any, ...]:
        """
        Run the phasor torch routine.

        Parameters
        ----------
        decay : np.ndarray
            Time-resolved decay signal or decay cube.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing Torch-computed phasor coordinates and intensity values.
        """
        decay_t = torch.tensor(
            np.asarray(decay), dtype=torch.float32, device=self.device
        )
        *spatial, T = decay_t.shape
        decay_flat = decay_t.reshape(-1, T)

        I_sum = torch.clamp(torch.sum(decay_flat, dim=1), min=self.eps)
        G_all, S_all = [], []

        for k in range(1, self.n_harmonics + 1):
            omega_k = k * self.omega
            cos_k = torch.cos(omega_k * self.t_s_torch)
            sin_k = torch.sin(omega_k * self.t_s_torch)
            G_all.append(
                (torch.sum(decay_flat * cos_k, dim=1) / I_sum).reshape(spatial)
            )
            S_all.append(
                (torch.sum(decay_flat * sin_k, dim=1) / I_sum).reshape(spatial)
            )

        return torch.stack(G_all), torch.stack(S_all)

    def create_phasor_cpu(self, decay: np.ndarray) -> Any:
        """
        Create phasor cpu.

        Parameters
        ----------
        decay : np.ndarray
            Time-resolved decay signal or decay cube.

        Returns
        -------
        Any
            Object produced by create phasor CPU.
        """
        return self._phasor_numpy(decay)

    def create_phasor_gpu(self, decay: np.ndarray) -> tuple[Any, ...]:
        """
        Create phasor gpu.

        Parameters
        ----------
        decay : np.ndarray
            Time-resolved decay signal or decay cube.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing GPU-computed phasor coordinates and intensity values.
        """
        G, S = self._phasor_torch(decay)
        return G.cpu().numpy(), S.cpu().numpy()

    # ── calibration ───────────────────────────────────────────────────────────

    def calibrate(
        self, G: np.ndarray, S: np.ndarray, irf: np.ndarray
    ) -> tuple[Any, ...]:
        """
        Run the calibrate routine.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.
        irf : np.ndarray
            Instrument response function aligned with the decay signal.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing calibrated phasor coordinates and calibration factors.
        """
        G = np.asarray(G)
        S = np.asarray(S)
        irf = np.asarray(irf)
        if irf.ndim == 3:
            irf = irf.mean(axis=(0, 1))

        denom = np.clip(np.sum(irf), self.eps, None)
        G_irf, S_irf = [], []
        for k in range(1, self.n_harmonics + 1):
            omega_k = k * self.omega
            G_irf.append(np.sum(irf * np.cos(omega_k * self.t_s_np)) / denom)
            S_irf.append(np.sum(irf * np.sin(omega_k * self.t_s_np)) / denom)

        G_irf = np.array(G_irf)
        S_irf = np.array(S_irf)

        P = G + 1j * S
        P_irf = G_irf[:, None, None] + 1j * S_irf[:, None, None]
        P_irf_abs_sq = np.clip(
            G_irf[:, None, None] ** 2 + S_irf[:, None, None] ** 2, self.eps, None
        )
        P_true = P * np.conj(P_irf) / P_irf_abs_sq

        return np.real(P_true), np.imag(P_true)

    def calibrate_pixelwise(
        self, G: np.ndarray, S: np.ndarray, irf: np.ndarray
    ) -> tuple[Any, ...]:
        """
        Run the calibrate pixelwise routine.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.
        irf : np.ndarray
            Instrument response function aligned with the decay signal.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing per-pixel calibrated phasor coordinates and factors.
        """
        G = np.asarray(G, dtype=np.float32)
        S = np.asarray(S, dtype=np.float32)
        irf = np.asarray(irf, dtype=np.float32)

        H, W, T = irf.shape
        K = self.n_harmonics
        irf_flat = torch.tensor(irf.reshape(-1, T), device=self.device)
        I_sum = irf_flat.sum(dim=1, keepdim=True).clamp(min=self.eps)
        irf_norm = irf_flat / I_sum
        t_s = self.t_s_torch

        Gc_list, Sc_list = [], []

        for k in range(1, K + 1):
            omega_k = k * self.omega
            cos_k = torch.cos(
                torch.tensor(omega_k, dtype=torch.float32, device=self.device) * t_s
            )
            sin_k = torch.sin(
                torch.tensor(omega_k, dtype=torch.float32, device=self.device) * t_s
            )

            G_irf_flat = (irf_norm * cos_k).sum(dim=1)
            S_irf_flat = (irf_norm * sin_k).sum(dim=1)
            G_irf = G_irf_flat.reshape(H, W)
            S_irf = S_irf_flat.reshape(H, W)

            G_meas = torch.tensor(G[k - 1], device=self.device)
            S_meas = torch.tensor(S[k - 1], device=self.device)
            denom = (G_irf**2 + S_irf**2).clamp(min=self.eps)

            Gc_k = (G_meas * G_irf + S_meas * S_irf) / denom
            Sc_k = (S_meas * G_irf - G_meas * S_irf) / denom

            Gc_list.append(Gc_k.cpu().numpy())
            Sc_list.append(Sc_k.cpu().numpy())

        return np.stack(Gc_list), np.stack(Sc_list)

    def calibratre_reference(
        self,
        G: np.ndarray,
        S: np.ndarray,
        ref_data: np.ndarray,
        ref_lifetime_ns: float | None = None,
    ) -> tuple[Any, ...]:
        """
        Run the calibratre reference routine.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.
        ref_data : np.ndarray
            Decay trace of a reference sample used for calibration.
        ref_lifetime_ns : float | None
            Known lifetime of the reference sample in nanoseconds. When None, the
            reference is treated as a zero-lifetime instrument response, matching
            :meth:`calibrate`.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing calibrated phasor coordinates and calibration factors.
        """
        G = np.asarray(G)
        S = np.asarray(S)
        ref_data = np.asarray(ref_data)
        if ref_data.ndim == 3:
            ref_data = ref_data.mean(axis=(0, 1))

        denom = np.clip(np.sum(ref_data), self.eps, None)
        G_ref, S_ref = [], []
        for k in range(1, self.n_harmonics + 1):
            omega_k = k * self.omega
            G_ref.append(np.sum(ref_data * np.cos(omega_k * self.t_s_np)) / denom)
            S_ref.append(np.sum(ref_data * np.sin(omega_k * self.t_s_np)) / denom)

        G_ref = np.array(G_ref)
        S_ref = np.array(S_ref)

        if ref_lifetime_ns is not None:
            harmonic_freqs = self.frequency * np.arange(1, self.n_harmonics + 1)
            G_theory, S_theory = self.lifetime_to_phasor(
                ref_lifetime_ns, harmonic_freqs
            )
        else:
            G_theory = np.ones(self.n_harmonics)
            S_theory = np.zeros(self.n_harmonics)

        P = G + 1j * S
        P_ref = G_ref[:, None, None] + 1j * S_ref[:, None, None]
        P_theory = G_theory[:, None, None] + 1j * S_theory[:, None, None]
        P_ref_abs_sq = np.clip(
            G_ref[:, None, None] ** 2 + S_ref[:, None, None] ** 2, self.eps, None
        )
        P_true = P * np.conj(P_ref) * P_theory / P_ref_abs_sq

        return np.real(P_true), np.imag(P_true)

    def calibrate_reference(
        self,
        G: np.ndarray,
        S: np.ndarray,
        ref_data: np.ndarray,
        ref_lifetime_ns: float | None = None,
    ) -> tuple[Any, ...]:
        """Correctly spelled name of :meth:`calibratre_reference` (same arguments and result)."""
        return self.calibratre_reference(G, S, ref_data, ref_lifetime_ns)

    def calibrate_reference_pixelwise(
        self,
        G: np.ndarray,
        S: np.ndarray,
        ref_data: np.ndarray,
        ref_lifetime_ns: float | None = None,
    ) -> tuple[Any, ...]:
        """Correctly spelled name of :meth:`calibratre_reference_pixelwise` (same arguments and result)."""
        return self.calibratre_reference_pixelwise(G, S, ref_data, ref_lifetime_ns)

    def calibratre_reference_pixelwise(
        self,
        G: np.ndarray,
        S: np.ndarray,
        ref_data: np.ndarray,
        ref_lifetime_ns: float | None = None,
    ) -> tuple[Any, ...]:
        """
        Run the calibratre reference pixelwise routine.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.
        ref_data : np.ndarray
            Per-pixel decay cube of a reference sample used for calibration.
        ref_lifetime_ns : float | None
            Known lifetime of the reference sample in nanoseconds. When None, the
            reference is treated as a zero-lifetime instrument response, matching
            :meth:`calibrate_pixelwise`.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing per-pixel calibrated phasor coordinates and factors.
        """
        G = np.asarray(G, dtype=np.float32)
        S = np.asarray(S, dtype=np.float32)
        ref_data = np.asarray(ref_data, dtype=np.float32)

        H, W, T = ref_data.shape
        K = self.n_harmonics
        ref_flat = torch.tensor(ref_data.reshape(-1, T), device=self.device)
        I_sum = ref_flat.sum(dim=1, keepdim=True).clamp(min=self.eps)
        ref_norm = ref_flat / I_sum
        t_s = self.t_s_torch

        Gc_list, Sc_list = [], []

        for k in range(1, K + 1):
            omega_k = k * self.omega
            cos_k = torch.cos(
                torch.tensor(omega_k, dtype=torch.float32, device=self.device) * t_s
            )
            sin_k = torch.sin(
                torch.tensor(omega_k, dtype=torch.float32, device=self.device) * t_s
            )

            G_ref_flat = (ref_norm * cos_k).sum(dim=1)
            S_ref_flat = (ref_norm * sin_k).sum(dim=1)
            G_ref = G_ref_flat.reshape(H, W)
            S_ref = S_ref_flat.reshape(H, W)

            G_meas = torch.tensor(G[k - 1], device=self.device)
            S_meas = torch.tensor(S[k - 1], device=self.device)
            denom = (G_ref**2 + S_ref**2).clamp(min=self.eps)

            Gc_k = (G_meas * G_ref + S_meas * S_ref) / denom
            Sc_k = (S_meas * G_ref - G_meas * S_ref) / denom

            if ref_lifetime_ns is not None:
                G_theory, S_theory = self.lifetime_to_phasor(
                    ref_lifetime_ns, k * self.frequency
                )
                G_theory = float(G_theory)
                S_theory = float(S_theory)
                Gc_k, Sc_k = (
                    Gc_k * G_theory - Sc_k * S_theory,
                    Gc_k * S_theory + Sc_k * G_theory,
                )

            Gc_list.append(Gc_k.cpu().numpy())
            Sc_list.append(Sc_k.cpu().numpy())

        return np.stack(Gc_list), np.stack(Sc_list)

    # ── lifetime conversion ───────────────────────────────────────────────────

    def lifetime_to_phasor(
        self, tau_ns: np.ndarray, frequency_hz: float
    ) -> tuple[Any, ...]:
        """
        Run the lifetime to phasor routine.

        Parameters
        ----------
        tau_ns : np.ndarray
            Lifetime value in nanoseconds.
        frequency_hz : float
            Excitation frequency in hertz.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing phasor coordinates for the supplied lifetime values.
        """
        tau_s = np.asarray(tau_ns) * 1e-9
        omega = 2 * np.pi * frequency_hz
        denom = 1 + (omega * tau_s) ** 2
        return 1 / denom, (omega * tau_s) / denom

    def compute_lifetime(self, G: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        Compute lifetime.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.

        Returns
        -------
        np.ndarray
            Phase lifetime ``S / (omega G)`` in ns, with ``omega`` the laser (first
            harmonic) angular frequency: pass first-harmonic phasors, or divide by
            ``k`` for harmonic ``k``. For binned data this standard phase lifetime
            carries the gate-width bias of the binned locus (a few % at 256 bins);
            :func:`~pyfli.phasor.phasorS.discrete_phasor_lifetime` inverts the binned
            locus exactly.
        """
        G = np.asarray(G, dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)
        safe_denom = np.where(np.abs(G) > 1e-4, G * self.omega, np.inf)
        return np.where(np.abs(G) > 1e-4, S / safe_denom * 1e9, np.nan)

    def compute_modulation_lifetime(self, G: np.ndarray, S: np.ndarray) -> Any:
        """
        Compute modulation lifetime.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.

        Returns
        -------
        Any
            Modulation lifetime ``sqrt(1 / M^2 - 1) / omega`` in ns, ``M^2 = G^2 + S^2``,
            with ``omega`` the laser (first harmonic) angular frequency: pass
            first-harmonic phasors, or divide by ``k`` for harmonic ``k``.
        """
        G = np.asarray(G, dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)
        M_sq = np.clip(G**2 + S**2, self.eps, 1.0 - self.eps)
        return np.sqrt(1.0 / M_sq - 1.0) / self.omega * 1e9

    def compute_normal_lifetime(self, G: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        Compute the normal lifetime ``tau_N``: the lifetime of the universal-semicircle
        point nearest to each phasor
        (:func:`~pyfli.phasor.phasorS.phasor_lifetime.normal_lifetime`).

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate (IRF-calibrated, first harmonic).
        S : np.ndarray
            Phasor imaginary coordinate.

        Returns
        -------
        np.ndarray
            Normal lifetime map in nanoseconds.
        """
        return normal_lifetime(G, S, self.frequency)

    def compute_apparent_lifetimes(
        self, G: np.ndarray, S: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Compute the phase, modulation and normal lifetimes of each phasor.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate (IRF-calibrated, first harmonic).
        S : np.ndarray
            Phasor imaginary coordinate.

        Returns
        -------
        dict[str, np.ndarray]
            ``{"tau_phi", "tau_m", "tau_n"}`` maps in nanoseconds, from
            :meth:`compute_lifetime`, :meth:`compute_modulation_lifetime` and
            :meth:`compute_normal_lifetime`.
        """
        return {
            "tau_phi": self.compute_lifetime(G, S),
            "tau_m": self.compute_modulation_lifetime(G, S),
            "tau_n": self.compute_normal_lifetime(G, S),
        }

    # ── two-component analysis ────────────────────────────────────────────────

    def component_kernel(self, tau_ns: Any, model: str = "continuous") -> np.ndarray:
        """
        Per-bin photon distribution of a pure periodically excited mono-exponential
        component over this analyzer's time axis (one laser period), before IRF
        convolution.

        Parameters
        ----------
        tau_ns : Any
            Lifetime(s) in nanoseconds (any shape ``S``).
        model : str
            ``"continuous"``: excitation photons arrive anywhere inside their bin, as in
            a real instrument
            (:func:`~pyfli.reconstruction.common_reconstruct.periodic_instrument_exponential`).
            ``"bin_start"``: every excitation at a bin start, as in pyfli's simulators
            (:func:`~pyfli.reconstruction.common_reconstruct.periodic_bin_exponential`).

        Returns
        -------
        np.ndarray
            ``S + (T,)``, summing to 1 along the last axis.
        """
        n_bins, period_ns = len(self.t_s_np), 1e9 / self.frequency
        if model == "continuous":
            return periodic_instrument_exponential(tau_ns, n_bins, period_ns)
        if model == "bin_start":
            return periodic_bin_exponential(tau_ns, n_bins, period_ns)
        raise ValueError(f"model must be 'continuous' or 'bin_start', got {model!r}")

    def component_phasor(
        self, tau_ns: Any, harmonic: int = 1, model: str = "continuous"
    ) -> tuple[Any, ...]:
        """
        Phasor of a pure mono-exponential component as it appears in IRF-calibrated
        binned data on this analyzer's time axis.

        ``model="continuous"`` (real instruments) gives the universal-semicircle point
        :meth:`lifetime_to_phasor`, exactly. ``model="bin_start"`` (data simulated by
        pyfli's ``PhotonCountSimulator`` / ``ContinuousSimulator``, whose excitations
        sit at bin starts) gives the binned locus, which is rotated by about half a bin
        (phase lifetime a few % lower at 256 bins).

        Parameters
        ----------
        tau_ns : Any
            Lifetime(s) in nanoseconds.
        harmonic : int
            Harmonic (1 = laser frequency).
        model : str
            ``"continuous"`` or ``"bin_start"`` (see :meth:`component_kernel`).

        Returns
        -------
        tuple[Any, ...]
            ``(g, s)`` with the shape of `tau_ns`.
        """
        if model == "continuous":
            return self.lifetime_to_phasor(tau_ns, harmonic * self.frequency)
        p = self.component_kernel(tau_ns, model)
        w = harmonic * self.omega * self.t_s_np
        return p @ np.cos(w), p @ np.sin(w)

    def compute_fractions(
        self,
        G: np.ndarray,
        S: np.ndarray,
        tau1_ns: np.ndarray,
        tau2_ns: np.ndarray,
        mask: np.ndarray | None = None,
        hexbin_color: np.ndarray | None = None,
        plot_graph: bool = True,
        ax: Any | None = None,
        half_circle: bool = False,
        component_model: str = "continuous",
    ) -> tuple[Any, ...]:
        """
        Compute fractions.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.
        tau1_ns : np.ndarray
            Short lifetime component in nanoseconds.
        tau2_ns : np.ndarray
            Long lifetime component in nanoseconds.
        mask : np.ndarray | None
            Boolean or labeled mask selecting pixels for the operation.
        hexbin_color : np.ndarray | None
            Optional values used to color phasor hexbin density.
        plot_graph : bool
            Whether the phasor graph should be drawn.
        ax : Any | None
            Matplotlib axes object on which the plot is drawn.
        half_circle : bool
            Whether to draw only the upper half of the universal phasor circle.
        component_model : str
            Where a pure component's calibrated phasor lies (see
            :meth:`component_phasor`): ``"continuous"`` (default, the universal
            semicircle -- real instruments) or ``"bin_start"`` (pyfli's
            ``PhotonCountSimulator`` / ``ContinuousSimulator`` data).

        Returns
        -------
        tuple[Any, ...]
            ``(A1, A2)``: intensity (photon) fractions of the two components, from the
            projection of each phasor onto the line between the component phasors.
        """
        g1, s1 = self.component_phasor(tau1_ns, model=component_model)
        g2, s2 = self.component_phasor(tau2_ns, model=component_model)

        if plot_graph:
            created_fig = ax is None
            if created_fig:
                fig, ax = plt.subplots(figsize=(8, 6))
            self.plot_phasor_diagram(
                G,
                S,
                mask=mask,
                colors=None,
                hexbin_color="jet_r",
                ax=ax,
                figsize=(8, 3),
                half_circle=half_circle,
                title="Phasor Diagram",
                xlim=(-0.1, 1.1),
                ylim=(0.0, 0.6),
                kdeplot=False,
                kde_color="white",
                kde_levels=5,
                kde_linewidths=1,
                kde_alpha=0.5,
            )
            ax.plot(
                [g1, g2], [s1, s2], color="#2C0F02", linestyle="--", lw=2, zorder=10
            )
            ax.plot(g1, s1, "o", color="#E5D16E", markersize=8, label="...", zorder=11)
            ax.plot(g2, s2, "o", color="#363D45", markersize=8, label="...", zorder=11)
            ax.legend(loc="upper right")
            if created_fig:
                plt.tight_layout()

        line_vec_g = g1 - g2
        line_vec_s = s1 - s2
        line_mag_sq = line_vec_g**2 + line_vec_s**2 + self.eps

        A1 = np.clip(
            ((G - g2) * line_vec_g + (S - s2) * line_vec_s) / line_mag_sq, 0, 1
        )
        return A1, 1 - A1

    # ── biexponential reconstruction ──────────────────────────────────────────

    def _convolve_batch(self, signal: np.ndarray, kernel: np.ndarray) -> Any:
        """
        Circular (periodic) convolution of a batch of decays with IRFs.

        The time axis spans one laser period and the excitation repeats, so the IRF
        spread of photons near the end of the period wraps to its start, as in the
        measured decays (and in the phasor model, which assumes periodic excitation).

        Parameters
        ----------
        signal : np.ndarray
            Signal batch convolved with the supplied kernel, ``(N, T)``.
        kernel : np.ndarray
            Convolution kernel applied to the signal batch, ``(N, T)`` or ``(1, T)``.

        Returns
        -------
        Any
            Convolved batch, ``(N, T)``.
        """
        T = signal.shape[1]
        S_fft = torch.fft.rfft(signal, n=T, dim=1)
        K_fft = torch.fft.rfft(kernel, n=T, dim=1)
        return torch.fft.irfft(S_fft * K_fft, n=T, dim=1)

    def _build_model_decay(
        self,
        A1: Any,
        A2: Any,
        tau1_ns: np.ndarray,
        tau2_ns: np.ndarray,
        component_model: str = "continuous",
    ) -> Any:
        """
        Build the per-bin photon distribution of a bi-exponential mixture.

        ``A1`` / ``A2`` are intensity (photon) fractions, as returned by
        :meth:`compute_fractions`; each component is the periodic per-bin distribution
        of :meth:`component_kernel` for `component_model`, so the model is consistent
        with the phasor model used for the fractions.

        Parameters
        ----------
        A1 : Any
            Intensity fraction of the first exponential component, per pixel.
        A2 : Any
            Intensity fraction of the second exponential component, per pixel.
        tau1_ns : np.ndarray
            Short lifetime component in nanoseconds (scalar or one per pixel).
        tau2_ns : np.ndarray
            Long lifetime component in nanoseconds (scalar or one per pixel).
        component_model : str
            ``"continuous"`` or ``"bin_start"`` (see :meth:`component_kernel`).

        Returns
        -------
        Any
            Model decays ``(n_pixels, T)``, each summing to ``A1 + A2``.
        """
        a1 = np.asarray(A1, dtype=float).ravel()[:, None]
        a2 = np.asarray(A2, dtype=float).ravel()[:, None]
        p1 = self.component_kernel(
            np.asarray(tau1_ns, dtype=float).ravel(), component_model
        )
        p2 = self.component_kernel(
            np.asarray(tau2_ns, dtype=float).ravel(), component_model
        )
        model = a1 * p1 + a2 * p2
        return torch.tensor(model, dtype=torch.float32, device=self.device)

    def _normalize_irf(self, irf: np.ndarray) -> Any:
        """
        Normalize irf.

        Parameters
        ----------
        irf : np.ndarray
            Instrument response function aligned with the decay signal.

        Returns
        -------
        Any
            Object produced by normalize IRF.
        """
        irf_flat = np.asarray(irf, dtype=np.float32).reshape(-1, irf.shape[2])
        irf_t = torch.tensor(irf_flat, dtype=torch.float32, device=self.device)
        norms = irf_t.sum(dim=1, keepdim=True).clamp(min=self.eps)
        return irf_t / norms

    def analyze_biexponential_and_reconstruct(
        self,
        G: np.ndarray,
        S: np.ndarray,
        irf: np.ndarray,
        tau1_ns: np.ndarray | None = None,
        tau2_ns: np.ndarray | None = None,
        plot: bool = True,
        axes: Any | None = None,
        component_model: str = "continuous",
    ) -> Any:
        """
        Run the analyze biexponential and reconstruct routine.

        Parameters
        ----------
        G : np.ndarray
            Phasor real coordinate.
        S : np.ndarray
            Phasor imaginary coordinate or shift amount.
        irf : np.ndarray
            Instrument response function aligned with the decay signal.
        tau1_ns : np.ndarray | None
            Short lifetime component in nanoseconds.
        tau2_ns : np.ndarray | None
            Long lifetime component in nanoseconds.
        plot : bool
            Whether diagnostic plots should be generated.
        axes : Any | None
            Matplotlib axes collection used for drawing subplots.
        component_model : str
            ``"continuous"`` (default, real instruments) or ``"bin_start"`` (pyfli
            simulated data); used for both the fractions and the reconstruction.

        Returns
        -------
        Any
            Reconstructed decays ``(H, W, T)``: the periodic bi-exponential with the
            phasor fractions, circularly convolved with each pixel's IRF; every pixel
            sums to 1 (multiply by the pixel's photon count to compare with the data).
        """
        if tau1_ns is None or tau2_ns is None:
            return None

        A1, A2 = self.compute_fractions(
            G, S, tau1_ns, tau2_ns, plot_graph=False, component_model=component_model
        )
        tau_map_ns = self.compute_lifetime(G, S)

        if plot:
            created_fig = axes is None
            if created_fig:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            else:
                fig = axes[0].get_figure()

            im1 = axes[0].imshow(A1, origin="upper", cmap="viridis")
            axes[0].set_title(f"A1 Map (Fraction of {tau1_ns} ns)")
            fig.colorbar(im1, ax=axes[0])

            im2 = axes[1].imshow(A2, origin="upper", cmap="plasma")
            axes[1].set_title(f"A2 Map (Fraction of {tau2_ns} ns)")
            fig.colorbar(im2, ax=axes[1])

            im3 = axes[2].imshow(
                np.clip(tau_map_ns, 0, 5), origin="upper", cmap="magma"
            )
            axes[2].set_title("Phase Lifetime Map (ns)")
            fig.colorbar(im3, ax=axes[2])

            for ax in axes:
                ax.axis("off")
            if created_fig:
                plt.tight_layout()

        H, W = A1.shape
        T = irf.shape[2]

        model_t = self._build_model_decay(A1, A2, tau1_ns, tau2_ns, component_model)
        irf_t = self._normalize_irf(irf)
        recon_t = self._convolve_batch(model_t, irf_t)
        reconstructed_decay = recon_t.cpu().numpy().reshape(H, W, T)

        return reconstructed_decay

    # ── utilities ─────────────────────────────────────────────────────────────

    def generate_intensity_image(self, decay: np.ndarray) -> np.ndarray:
        """
        Generate intensity image.

        Parameters
        ----------
        decay : np.ndarray
            Time-resolved decay signal or decay cube.

        Returns
        -------
        np.ndarray
            Intensity image obtained by integrating the decay along the time axis.
        """
        return np.sum(decay, axis=2)

    def save_phasors_hdf5(
        self, Gc: Any, Sc: Any, tau_phasor: np.ndarray, save_file: np.ndarray
    ) -> None:
        """
        Save phasors hdf5.

        Parameters
        ----------
        Gc : Any
            Calibrated phasor real coordinate map.
        Sc : Any
            Calibrated phasor imaginary coordinate map.
        tau_phasor : np.ndarray
            Lifetime map estimated from phasor coordinates.
        save_file : np.ndarray
            HDF5 path where phasor results are saved.

        Returns
        -------
        None
            No object is returned; the function save phasors hdf5.
        """
        try:
            with h5py.File(save_file, "w") as hf:
                hf.create_dataset("Gc", data=Gc, compression="gzip", chunks=True)
                hf.create_dataset("Sc", data=Sc, compression="gzip", chunks=True)
                hf.create_dataset(
                    "tau_phasor", data=tau_phasor, compression="gzip", chunks=True
                )
                hf.attrs["n_harmonics"] = Gc.shape[0]
                hf.attrs["resolution"] = f"{Gc.shape[1]}x{Gc.shape[2]}"
            logging.info(f"Successfully saved data to {save_file}")
        except Exception as e:
            logging.error(f"An error occurred while saving: {e}")
