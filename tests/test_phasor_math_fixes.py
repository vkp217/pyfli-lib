"""Regression tests for the phasor mathematics."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyfli.analysis import compute_freq_axis
from pyfli.phasor.phasorS import (
    MonoLocus,
    PhasorAdditionalPlots,
    PhasorAnalyzer,
    cluster_summary,
    phase_lifetime_gated,
)
from pyfli.phasor.phasorS.phasor_simple_utils import _TAU_MARKS_NS, _draw_lifetime_ticks
from pyfli.reconstruction.common_reconstruct import (
    periodic_bin_exponential,
    periodic_instrument_exponential,
)
from pyfli.simulator.calibration.expected_decay import gated_bin_probabilities

FREQ_HZ = 80e6
T = 1e9 / FREQ_HZ
N = 256
OMEGA = 2 * np.pi / T
TAUS = np.array([0.3, 1.0, 2.5, 6.0])


def psed(t, tau):
    return np.exp(-np.mod(t, T) / tau) / (1 - np.exp(-T / tau))


def phasor_of(signal, times):
    return np.sum(signal * np.cos(OMEGA * times)) / signal.sum(), np.sum(
        signal * np.sin(OMEGA * times)
    ) / signal.sum()


def gated_reference(tau, w_frac, n_gates, sub=20000):
    W = w_frac * T
    counts = [
        np.sum(psed(k * T / n_gates + (np.arange(sub) + 0.5) * W / sub, tau))
        for k in range(n_gates)
    ]
    return phasor_of(np.array(counts), np.arange(n_gates) * T / n_gates)


@pytest.fixture
def analyzer():
    return PhasorAnalyzer(FREQ_HZ, np.arange(N) * T / N, 1, device="cpu")


class TestLoci:
    @pytest.mark.parametrize(
        "w_frac,n_gates", [(1 / 64, 64), (0.04, 64), (0.25, 32), (0.5, 4)]
    )
    def test_gated_n_matches_periodic_integration(self, w_frac, n_gates):
        ref = np.array([gated_reference(tau, w_frac, n_gates) for tau in TAUS]).T
        got = MonoLocus(FREQ_HZ)._gated_n_gs(TAUS, w_frac, n_gates)
        np.testing.assert_allclose(got, ref, atol=2e-6)

    def test_offset_is_a_delay(self):
        t = (np.arange(100000) + 0.5) * T / 100000
        ref = np.array([phasor_of(psed(t - 0.2 * T, tau), t) for tau in TAUS]).T
        np.testing.assert_allclose(
            MonoLocus(FREQ_HZ)._offset_gs(TAUS, 0.2), ref, atol=1e-8
        )

    @pytest.mark.parametrize("w_frac", [0.3, 0.5, 1.0])
    def test_gated_single_is_normalised_truncated_window(self, w_frac):
        loc = MonoLocus(FREQ_HZ)
        t = (np.arange(100000) + 0.5) * T / 100000
        m = t < w_frac * T
        ref = np.array([phasor_of(psed(t[m], tau), t[m]) for tau in TAUS]).T
        np.testing.assert_allclose(loc._gated_single_gs(TAUS, w_frac), ref, atol=1e-8)
        g, s, _, _ = loc.gated_single_locus(w_frac, tau_ns=TAUS, draw=False)
        np.testing.assert_allclose(
            phase_lifetime_gated(g, s, FREQ_HZ, w_frac), TAUS, rtol=1e-6
        )

    def test_full_gate_is_universal_circle(self):
        g, s = MonoLocus(FREQ_HZ)._gated_single_gs(TAUS, 1.0)
        np.testing.assert_allclose(np.hypot(g - 0.5, s), 0.5, atol=1e-12)

    def test_locus_ticks_only_inside_traced_range(self):
        fig, ax = plt.subplots()
        MonoLocus(FREQ_HZ).continuous_locus(tau_ns=np.linspace(1.2, 4.0, 50), ax=ax)
        labels = [
            t.get_text()
            for t in ax.texts
            if t.get_text().endswith("ns") and "MHz" not in t.get_text()
        ]
        assert labels == ["1.5 ns", "2.0 ns", "3.0 ns"]
        plt.close(fig)

    def test_draw_ticks_custom_labels(self):
        fig, ax = plt.subplots()
        _draw_lifetime_ticks(ax, [0.5, 0.6], [0.4, 0.4], taus=[7.0, 9.0])
        assert [t.get_text() for t in ax.texts] == ["7.0", "9.0"]
        _draw_lifetime_ticks(ax, [0.5], [0.4])
        assert ax.texts[-1].get_text() == f"{_TAU_MARKS_NS[0]:.1f}"
        plt.close(fig)


class TestColourBarAndHarmonics:
    def test_traceable_colorbar_labels_match_pixel_colours(self, analyzer):
        tau = np.array([[0.5, 1.0, 2.0], [3.0, 4.0, 6.0]])
        x = OMEGA * tau
        G, S = 1 / (1 + x**2), x / (1 + x**2)
        fig = analyzer.plot_traceable_analysis(G, S, colormap="plasma")
        ax_cb = next(
            a for a in fig.axes if a.get_ylabel().startswith("Normal lifetime")
        )
        ticks = ax_cb.get_yticks()
        labels = [float(t.get_text()) for t in ax_cb.get_yticklabels()]
        np.testing.assert_allclose(
            ticks, 2 * np.arctan(OMEGA * np.array(labels)) / np.pi, atol=1e-9
        )
        colors = analyzer.phasor_radial_color(G, S, colormap="plasma")
        cmap = plt.colormaps["plasma"]
        for t, c in zip(tau.ravel(), colors.reshape(-1, 3)):
            np.testing.assert_allclose(
                c, cmap(2 * np.arctan(OMEGA * t) / np.pi)[:3], atol=1e-9
            )
        plt.close(fig)

    def test_harmonic_phase_lifetime_map(self):
        pa = PhasorAdditionalPlots(FREQ_HZ, np.arange(N) * T / N, 3, device="cpu")
        tau = np.full((3, 3), 2.5)
        G = np.stack([1 / (1 + (h * OMEGA * tau) ** 2) for h in (1, 2, 3)])
        S = np.stack([h * OMEGA * tau * G[h - 1] for h in (1, 2, 3)])
        for k in range(3):
            fig = pa.phasorlifetime_to_phasor(G, S, harmonic=k, cmap_scale=(0, 10))
            np.testing.assert_allclose(
                np.asarray(fig.axes[0].get_images()[0].get_array()), 2.5, rtol=1e-9
            )
            plt.close(fig)


class TestKernels:
    @pytest.mark.parametrize(
        "kernel", [periodic_bin_exponential, periodic_instrument_exponential]
    )
    def test_kernels_sum_to_one(self, kernel):
        np.testing.assert_allclose(
            kernel(np.array([0.05, 1.0, 8.0]), N, T).sum(-1), 1.0
        )

    def test_instrument_kernel_phasor_is_continuous(self):
        for tau in (0.05, 0.5, 3.0, 6.0):
            h = periodic_instrument_exponential(tau, N, T)
            z = np.sum(h * np.exp(1j * OMEGA * np.arange(N) * T / N))
            assert abs(z - 1 / (1 - 1j * OMEGA * tau)) < 1e-4

    def test_instrument_kernel_matches_monte_carlo(self):
        rng = np.random.default_rng(0)
        dt = T / N
        u = rng.random(1_000_000) * dt
        arrivals = u + rng.exponential(2.0, u.size)
        mc = np.bincount((np.mod(arrivals, T) / dt).astype(int), minlength=N) / u.size
        np.testing.assert_allclose(
            periodic_instrument_exponential(2.0, N, T), mc, atol=1e-3
        )

    def test_bin_start_kernel_matches_simulator_model(self):
        delta = np.zeros(N)
        delta[0] = 1.0
        np.testing.assert_allclose(
            periodic_bin_exponential(2.0, N, T),
            gated_bin_probabilities(2.0, delta, T),
            atol=1e-12,
        )


class TestTwoComponent:
    @pytest.fixture
    def irf(self):
        irf = np.exp(-0.5 * ((np.arange(N) - 37) / 1.5) ** 2)
        return irf / irf.sum()

    def _calibrated(self, analyzer, decay, irf):
        cube = decay[None, None, :] * 1e5
        g, s = analyzer.create_phasor_cpu(cube)
        ic = np.broadcast_to(irf, cube.shape).copy()
        gc, sc = analyzer.calibrate_pixelwise(g, s, ic)
        return gc[0], sc[0], ic

    def test_component_models(self, analyzer):
        g, s = analyzer.component_phasor(np.array([0.5, 2.0]))
        np.testing.assert_allclose(
            (g, s), analyzer.lifetime_to_phasor(np.array([0.5, 2.0]), FREQ_HZ)
        )
        gb, sb = analyzer.component_phasor(2.0, model="bin_start")
        assert gb > g[1] and sb > 0
        with pytest.raises(ValueError, match="model"):
            analyzer.component_kernel(1.0, model="gaussian")

    @pytest.mark.parametrize("a1,tau2", [(0.2, 2.0), (0.6, 6.0)])
    def test_bin_start_data_exact(self, analyzer, irf, a1, tau2):
        decay = a1 * gated_bin_probabilities(0.5, irf, T) + (
            1 - a1
        ) * gated_bin_probabilities(tau2, irf, T)
        gc, sc, ic = self._calibrated(analyzer, decay, irf)
        A1, A2 = analyzer.compute_fractions(
            gc, sc, 0.5, tau2, plot_graph=False, component_model="bin_start"
        )
        assert float(A1[0, 0]) == pytest.approx(a1, abs=1e-5) and float(
            A2[0, 0]
        ) == pytest.approx(1 - a1, abs=1e-5)
        rec = analyzer.analyze_biexponential_and_reconstruct(
            gc, sc, ic, 0.5, tau2, plot=False, component_model="bin_start"
        )
        assert rec.shape == (1, 1, N) and rec.sum() == pytest.approx(1.0, abs=1e-5)
        np.testing.assert_allclose(rec[0, 0], decay / decay.sum(), atol=1e-6)

    @pytest.mark.parametrize("a1,tau2", [(0.3, 2.0), (0.7, 6.0)])
    def test_real_instrument_data(self, analyzer, irf, a1, tau2):
        decay = np.real(
            np.fft.ifft(
                np.fft.fft(irf)
                * np.fft.fft(
                    a1 * periodic_instrument_exponential(0.5, N, T)
                    + (1 - a1) * periodic_instrument_exponential(tau2, N, T)
                )
            )
        )
        gc, sc, ic = self._calibrated(analyzer, decay, irf)
        A1, _ = analyzer.compute_fractions(gc, sc, 0.5, tau2, plot_graph=False)
        assert float(A1[0, 0]) == pytest.approx(a1, abs=2e-3)
        rec = analyzer.analyze_biexponential_and_reconstruct(
            gc, sc, ic, 0.5, tau2, plot=False
        )
        np.testing.assert_allclose(
            rec[0, 0], decay / decay.sum(), atol=2e-3 * decay.max() / decay.sum()
        )

    def test_reconstruction_keeps_previous_pulse_tail(self, analyzer, irf):
        tau2 = 6.0
        decay = gated_bin_probabilities(tau2, irf, T)
        gc, sc, ic = self._calibrated(analyzer, decay, irf)
        rec = analyzer.analyze_biexponential_and_reconstruct(
            gc, sc, ic, 0.5, tau2, plot=False, component_model="bin_start"
        )[0, 0]
        assert rec[:30].mean() > 0.02 * rec.max()

    def test_calibrate_reference_aliases(self, analyzer, irf):
        sample = gated_bin_probabilities(3.0, irf, T)[None, None, :] * 1e4
        ref = gated_bin_probabilities(1.0, irf, T)[None, None, :] * 1e4
        g, s = analyzer.create_phasor_cpu(sample)
        np.testing.assert_array_equal(
            analyzer.calibrate_reference(g, s, ref, 1.0),
            analyzer.calibratre_reference(g, s, ref, 1.0),
        )
        np.testing.assert_array_equal(
            analyzer.calibrate_reference_pixelwise(g, s, ref, 1.0),
            analyzer.calibratre_reference_pixelwise(g, s, ref, 1.0),
        )


def test_cluster_summary_intensity_weighting(analyzer):
    irf = np.zeros(N)
    irf[0] = 1.0
    d1, d2 = (
        100 * periodic_instrument_exponential(1.0, N, T),
        900 * periodic_instrument_exponential(4.0, N, T),
    )
    cube = np.stack([d1, d2])[None]
    g, s = analyzer.create_phasor_cpu(cube)
    labels = np.array([[1, 1]])
    out = cluster_summary(g[0], s[0], labels, analyzer, intensity=cube.sum(-1))
    gp, sp = analyzer.create_phasor_cpu((d1 + d2)[None, None, :])
    assert out.loc[1, "G_mean"] == pytest.approx(float(gp[0, 0, 0]), abs=1e-10)
    assert out.loc[1, "S_mean"] == pytest.approx(float(sp[0, 0, 0]), abs=1e-10)
    plain = cluster_summary(g[0], s[0], labels, analyzer)
    assert plain.loc[1, "G_mean"] == pytest.approx(float(g[0].mean()))


def test_compute_freq_axis_covers_one_period():
    _, freq_hz, t = compute_freq_axis(np.zeros((1, 1, N)), laser_period_ns=T)
    np.testing.assert_allclose(t, np.arange(N) * T / N)
    assert freq_hz == pytest.approx(FREQ_HZ)
