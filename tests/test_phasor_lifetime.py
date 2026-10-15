"""
Tests for the apparent phasor lifetimes (pyfli.phasor.phasorS.phasor_lifetime) and their
PhasorAnalyzer integration: phase tau_phi, modulation tau_M and normal tau_N.
"""

import numpy as np
import pytest

from pyfli.phasor.phasorS import (
    APPARENT_LIFETIMES,
    PhasorAnalyzer,
    apparent_lifetimes,
    modulation_lifetime,
    normal_lifetime,
    phase_lifetime,
    semicircle_nearest_point,
)

FREQ_HZ = 80e6
OMEGA = 2 * np.pi * FREQ_HZ * 1e-9
TAU = np.array([0.05, 0.5, 1.0, 2.5, 6.0, 30.0])


def on_semicircle(tau, harmonic=1):
    x = harmonic * OMEGA * np.asarray(tau, dtype=float)
    g = 1.0 / (1.0 + x**2)
    return g, x * g


@pytest.fixture
def analyzer():
    return PhasorAnalyzer(
        frequency_hz=FREQ_HZ, time_axis_ns=np.arange(256) * 12.5 / 256, device="cpu"
    )


class TestOnSemicircle:
    @pytest.mark.parametrize("harmonic", [1, 2, 3])
    def test_all_three_recover_true_lifetime(self, harmonic):
        g, s = on_semicircle(TAU, harmonic)
        out = apparent_lifetimes(g, s, FREQ_HZ, harmonic)
        assert tuple(out) == APPARENT_LIFETIMES
        for key in APPARENT_LIFETIMES:
            np.testing.assert_allclose(out[key], TAU, rtol=1e-9)

    def test_nearest_point_of_a_semicircle_point_is_itself(self):
        g, s = on_semicircle(TAU)
        gn, sn = semicircle_nearest_point(g, s)
        np.testing.assert_allclose(gn, g, atol=1e-12)
        np.testing.assert_allclose(sn, s, atol=1e-12)


class TestGeometry:
    def test_nearest_point_is_on_semicircle_and_closest(self):
        rng = np.random.default_rng(0)
        g, s = rng.uniform(-0.2, 1.2, 500), rng.uniform(0.0, 0.7, 500)
        gn, sn = semicircle_nearest_point(g, s)
        np.testing.assert_allclose(np.hypot(gn - 0.5, sn), 0.5, atol=1e-12)
        assert np.all(sn >= 0)
        phi = np.linspace(0, np.pi, 20001)
        circle = np.stack([0.5 + 0.5 * np.cos(phi), 0.5 * np.sin(phi)], axis=1)
        brute = np.min(
            np.hypot(g[:, None] - circle[None, :, 0], s[:, None] - circle[None, :, 1]),
            axis=1,
        )
        np.testing.assert_allclose(np.hypot(g - gn, s - sn), brute, atol=1e-6)

    def test_nearest_point_lies_on_line_through_centre(self):
        g, s = np.array([0.7, 0.2]), np.array([0.2, 0.1])
        gn, sn = semicircle_nearest_point(g, s)
        cross = (g - 0.5) * sn - s * (gn - 0.5)
        np.testing.assert_allclose(cross, 0.0, atol=1e-12)

    def test_normal_lifetime_matches_lifetime_of_nearest_point(self):
        g, s = np.array([0.6, 0.3, 0.8]), np.array([0.2, 0.3, 0.1])
        gn, sn = semicircle_nearest_point(g, s)
        np.testing.assert_allclose(
            normal_lifetime(g, s, FREQ_HZ), phase_lifetime(gn, sn, FREQ_HZ), rtol=1e-10
        )
        np.testing.assert_allclose(
            normal_lifetime(g, s, FREQ_HZ),
            modulation_lifetime(gn, sn, FREQ_HZ),
            rtol=1e-9,
        )

    def test_scaling_along_the_radius_keeps_tau_n(self):
        g, s = on_semicircle(np.array([1.0, 4.0]))
        for scale in (0.3, 0.8, 1.5):
            np.testing.assert_allclose(
                normal_lifetime(0.5 + scale * (g - 0.5), scale * s, FREQ_HZ),
                [1.0, 4.0],
                rtol=1e-9,
            )


class TestMixtures:
    def test_two_component_ordering(self):
        g, s = on_semicircle(np.array([0.5, 6.0]))
        for a in (0.2, 0.5, 0.8):
            out = apparent_lifetimes(
                a * g[0] + (1 - a) * g[1], a * s[0] + (1 - a) * s[1], FREQ_HZ
            )
            assert 0.5 < out["tau_phi"] < out["tau_n"] < out["tau_m"] < 6.0

    def test_maps_keep_shape(self):
        g, s = on_semicircle(np.full((3, 4), 2.0))
        out = apparent_lifetimes(g * 0.9, s * 0.9, FREQ_HZ)
        assert all(v.shape == (3, 4) for v in out.values())


class TestEdgeCases:
    def test_semicircle_ends_centre_and_nan(self):
        tau = normal_lifetime(
            np.array([1.0, 0.0, 0.5, np.nan, 0.5]),
            np.array([0.0, 0.0, 0.0, 0.1, np.inf]),
            FREQ_HZ,
        )
        assert (
            tau[0] == 0.0
            and np.isinf(tau[1])
            and np.isnan(tau[2])
            and np.isnan(tau[3])
            and np.isnan(tau[4])
        )

    def test_below_axis_maps_to_nearest_end(self):
        gn, sn = semicircle_nearest_point(np.array([0.8, 0.2]), np.array([-0.1, -0.1]))
        np.testing.assert_allclose(gn, [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(sn, [0.0, 0.0], atol=1e-12)
        tau = normal_lifetime(np.array([0.8, 0.2]), np.array([-0.1, -0.1]), FREQ_HZ)
        assert tau[0] == 0.0 and np.isinf(tau[1])

    def test_scalars(self):
        g, s = on_semicircle(2.0)
        assert float(normal_lifetime(g, s, FREQ_HZ)) == pytest.approx(2.0)
        gn, sn = semicircle_nearest_point(float(g), float(s))
        assert np.ndim(gn) == 0 and float(gn) == pytest.approx(float(g))


class TestPhasorAnalyzerIntegration:
    def test_compute_normal_lifetime(self, analyzer):
        g, s = on_semicircle(TAU)
        np.testing.assert_allclose(
            analyzer.compute_normal_lifetime(g, s), TAU, rtol=1e-9
        )
        np.testing.assert_allclose(
            analyzer.compute_normal_lifetime(g * 0.7, s * 0.7),
            normal_lifetime(g * 0.7, s * 0.7, FREQ_HZ),
        )

    def test_compute_apparent_lifetimes_uses_existing_methods(self, analyzer):
        g, s = on_semicircle(np.array([[0.5, 2.0], [4.0, 8.0]]))
        g, s = 0.95 * g, 0.95 * s
        out = analyzer.compute_apparent_lifetimes(g, s)
        assert tuple(out) == APPARENT_LIFETIMES
        np.testing.assert_array_equal(out["tau_phi"], analyzer.compute_lifetime(g, s))
        np.testing.assert_array_equal(
            out["tau_m"], analyzer.compute_modulation_lifetime(g, s)
        )
        np.testing.assert_array_equal(
            out["tau_n"], analyzer.compute_normal_lifetime(g, s)
        )

    def test_existing_methods_agree_with_module_functions(self, analyzer):
        g, s = on_semicircle(TAU)
        g, s = 0.9 * g, 0.9 * s
        np.testing.assert_allclose(
            analyzer.compute_lifetime(g, s), phase_lifetime(g, s, FREQ_HZ), rtol=1e-9
        )
        np.testing.assert_allclose(
            analyzer.compute_modulation_lifetime(g, s),
            modulation_lifetime(g, s, FREQ_HZ),
            rtol=1e-6,
        )
