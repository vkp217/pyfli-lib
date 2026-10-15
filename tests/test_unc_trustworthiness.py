"""Tests for pyfli.bayes_utils.unc_trustworthiness."""

import numpy as np
import pytest
from scipy.stats import median_abs_deviation

from pyfli.bayes_utils import trustworthiness_CrI, trustworthiness_gof, unc


@pytest.fixture
def samples():
    rng = np.random.default_rng(0)
    loc = rng.uniform(1.0, 3.0, (4, 5, 1))
    sd = rng.uniform(0.05, 0.5, (4, 5, 1))
    return loc + sd * rng.standard_normal((4, 5, 2000))


class TestUnc:
    def test_matches_scipy_raw_mad(self, samples):
        np.testing.assert_allclose(unc(samples), median_abs_deviation(samples, axis=-1))
        assert unc(samples).shape == (4, 5)

    def test_known_values(self):
        assert unc(np.array([1.0, 2.0, 3.0, 4.0, 100.0])) == pytest.approx(1.0)
        assert unc(np.full(10, 2.5)) == 0.0

    def test_normal_scale_estimates_sd(self):
        x = np.random.default_rng(1).normal(0, 2.0, 200_000)
        assert unc(x, scale="normal") == pytest.approx(2.0, rel=0.02)

    def test_axis(self, samples):
        np.testing.assert_allclose(
            unc(np.moveaxis(samples, -1, 0), axis=0), unc(samples)
        )


class TestTrustworthinessCrI:
    def test_formula(self, samples):
        med = np.median(samples, axis=-1)
        q5, q95 = np.percentile(samples, [5, 95], axis=-1)
        expected = 1.0 / (1.0 + (q95 - q5) / np.abs(med))
        np.testing.assert_allclose(trustworthiness_CrI(samples, med), expected)
        np.testing.assert_allclose(trustworthiness_CrI(samples), expected)

    def test_range_and_ordering(self):
        rng = np.random.default_rng(2)
        narrow = 2.0 + 0.01 * rng.standard_normal(1000)
        wide = 2.0 + 1.0 * rng.standard_normal(1000)
        t = trustworthiness_CrI(np.stack([narrow, wide]))
        assert np.all((t > 0) & (t <= 1))
        assert t[0] > t[1]
        assert trustworthiness_CrI(np.full((1, 50), 3.0))[0] == 1.0

    def test_ci_width(self, samples):
        assert np.all(
            trustworthiness_CrI(samples, ci=50) > trustworthiness_CrI(samples, ci=95)
        )
        with pytest.raises(ValueError):
            trustworthiness_CrI(samples, ci=100)

    def test_zero_median_is_nan(self):
        s = np.zeros((2, 3, 100))
        s[0, 0] = np.linspace(1, 2, 100)
        t = trustworthiness_CrI(s)
        assert np.isfinite(t[0, 0])
        assert np.isnan(t).sum() == 5

    def test_median_shape_mismatch(self, samples):
        with pytest.raises(ValueError, match="median_map shape"):
            trustworthiness_CrI(samples, np.ones((5, 4)))


def _decay(n_photons, K=128, tau=2.5, bg=0.2):
    t = (np.arange(K) + 0.5) * 12.5 / K
    shape = np.exp(-t / tau)
    return n_photons * shape / shape.sum() + bg


class TestTrustworthinessGof:
    def test_validation(self):
        with pytest.raises(ValueError, match="mode"):
            trustworthiness_gof(2, mode="both")
        with pytest.raises(ValueError, match="expectation"):
            trustworthiness_gof(2, expectation="approx")
        with pytest.raises(ValueError, match="positive"):
            trustworthiness_gof(2, z_plus=0)

    def test_deviance_terms(self):
        y = np.array([0.0, 3.0, 10.0])
        mu = np.array([2.0, 3.0, 7.0])
        expected = [4.0, 0.0, 2 * (10 * np.log(10 / 7) - 3)]
        np.testing.assert_allclose(trustworthiness_gof.deviance_terms(y, mu), expected)

    def test_expected_moments_limits(self):
        E, V = trustworthiness_gof.expected_moments(np.array([1e5, 500.0, 1e-7]))
        assert E[0] == 1.0 and V[0] == 2.0
        assert E[1] == pytest.approx(1.0, abs=0.01)
        assert V[1] == pytest.approx(2.0, abs=0.05)
        assert E[2] == pytest.approx(2e-6 * np.log(1e6), rel=0.05)

    def test_expected_moments_match_monte_carlo(self):
        rng = np.random.default_rng(3)
        for mu in (0.3, 2.0):
            y = rng.poisson(mu, 400_000)
            d = trustworthiness_gof.deviance_terms(y, np.full(y.shape, mu))
            E, V = trustworthiness_gof.expected_moments(np.array(mu))
            assert E == pytest.approx(d.mean(), rel=0.01)
            assert V == pytest.approx(d.var(), rel=0.03)

    @pytest.mark.parametrize("n_photons", [100, 300, 5e4])
    def test_exact_z_is_standard_normal_for_true_model(self, n_photons):
        rng = np.random.default_rng(4)
        mu = np.broadcast_to(_decay(n_photons, bg=0.01), (4000, 128))
        y = rng.poisson(mu).astype(float)
        z, _, _ = trustworthiness_gof(0).zscore(y, mu)
        assert abs(z.mean()) < 0.1
        assert z.std() == pytest.approx(1.0, abs=0.1)

    def test_asymptotic_z_biased_for_dim_pixels(self):
        rng = np.random.default_rng(5)
        mu = np.broadcast_to(_decay(100, bg=0.005), (2000, 128))
        y = rng.poisson(mu).astype(float)
        z_exact, _, _ = trustworthiness_gof(0).zscore(y, mu)
        z_asym, D, nu = trustworthiness_gof(0, expectation="asymptotic").zscore(y, mu)
        np.testing.assert_allclose(z_asym, (D - nu) / np.sqrt(2 * nu))
        assert np.median(z_asym) < -1
        assert abs(np.median(z_exact)) < 0.3

    def test_score_modes(self):
        z = np.array([-10.0, -1.0, 0.0, 3.0])
        one = trustworthiness_gof(2, z_plus=3.0).score(z)
        two = trustworthiness_gof(2, mode="two-sided", z_plus=3.0, z_minus=5.0).score(z)
        np.testing.assert_allclose(one, [1.0, 1.0, 1.0, np.exp(-0.5)])
        np.testing.assert_allclose(
            two, [np.exp(-2.0), np.exp(-1 / 50), 1.0, np.exp(-0.5)]
        )

    def test_call_shapes_and_keys(self):
        rng = np.random.default_rng(6)
        mu = np.broadcast_to(_decay(1000), (3, 4, 128))
        y = rng.poisson(mu).astype(float)
        out = trustworthiness_gof(2)(y, mu)
        assert set(out) == {"D", "nu", "chi2_nu", "z", "G"}
        assert all(v.shape == (3, 4) for v in out.values())
        assert np.all(out["nu"] == 126)
        np.testing.assert_allclose(out["chi2_nu"], out["D"] / 126)
        assert np.all((out["G"] > 0) & (out["G"] <= 1))

    def test_misfit_scores_low(self):
        rng = np.random.default_rng(7)
        mu_true = _decay(5e4, tau=0.6) / 2 + _decay(5e4, tau=3.5) / 2
        y = rng.poisson(np.broadcast_to(mu_true, (200, 128))).astype(float)
        good = trustworthiness_gof(0)(y, mu_true)["G"]
        bad = trustworthiness_gof(0)(y, _decay(5e4, tau=2.0))["G"]
        assert np.median(good) > 0.8
        assert np.all(bad < 1e-3)

    def test_mask_and_axis(self):
        rng = np.random.default_rng(8)
        mu = np.broadcast_to(_decay(1000), (5, 128))
        y = rng.poisson(mu).astype(float)
        mask = np.arange(128) < 64
        gof = trustworthiness_gof(1)
        masked = gof(y, mu, mask=mask)
        sliced = gof(y[:, :64], mu[:, :64])
        for key in ("D", "nu", "z"):
            np.testing.assert_allclose(masked[key], sliced[key])
        on_axis0 = trustworthiness_gof(1, axis=0)(y.T, mu.T)
        np.testing.assert_allclose(on_axis0["z"], gof(y, mu)["z"])
