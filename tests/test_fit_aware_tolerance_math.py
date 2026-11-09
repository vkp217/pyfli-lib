"""Mathematical checks of pyfli.bayes_utils.fit_aware_tolerance against the
trustworthiness note (Sections 3 and 7): limits, identities and worked numbers."""

import numpy as np
import pytest
from scipy.stats import norm

from pyfli.bayes_utils.fit_aware_tolerance import (
    FitAwareTolerance,
    acceptance_band,
    combine_fit_aware_factors,
    dispersion_adjusted_tolerance,
    gate,
    gaussian_tolerance_probability,
    joint_tolerance_probability,
    reduced_chi2,
    residual_noise_ratio,
    tolerance_probability,
    tolerance_standard_error,
    weighted_r2,
)

K = 256
T_NS = np.arange(K) * 0.05


def _decay(n_photons, tau=2.0, bg=2.0):
    irf = np.exp(-0.5 * ((T_NS - 1.0) / 0.1) ** 2)
    c = np.convolve(irf, np.exp(-T_NS / tau))[:K]
    return n_photons * c / c.sum() + bg


def _noisy(lam, n_pixels, seed):
    rng = np.random.default_rng(seed)
    return rng.poisson(np.broadcast_to(lam, (n_pixels, K))).astype(float)


class TestToleranceProbability:
    @pytest.mark.parametrize("ratio", [0.5, 1.0, 1.6449, 2.5])
    def test_gaussian_limit(self, ratio):
        sigma = 0.3
        draws = np.random.default_rng(0).normal(1.0, sigma, 400_000)
        T = tolerance_probability(draws, ratio * sigma, center=1.0)
        assert T == pytest.approx(2 * norm.cdf(ratio) - 1, abs=3e-3)
        assert gaussian_tolerance_probability(ratio * sigma, sigma) == pytest.approx(
            2 * norm.cdf(ratio) - 1
        )

    def test_relative_tolerance_is_log_scale(self):
        draws = np.exp(np.random.default_rng(1).normal(np.log(2.0), 0.04, 200_000))
        eps = 0.05
        T_rel = tolerance_probability(draws, eps, relative=True)
        d = np.log(draws) - np.log(np.median(draws))
        exact = np.mean((d > np.log1p(-eps)) & (d < np.log1p(eps)))
        assert T_rel == pytest.approx(exact, abs=1e-12)
        T_log = tolerance_probability(np.log(draws), np.log1p(eps))
        assert T_rel == pytest.approx(T_log, abs=0.02)

    def test_absolute_tolerance_does_not_penalise_small_values(self):
        rng = np.random.default_rng(2)
        small = 0.18 + 0.03 * rng.standard_normal(100_000)
        large = 2.3 + 0.03 * rng.standard_normal(100_000)
        assert tolerance_probability(small, 0.05) == pytest.approx(
            tolerance_probability(large, 0.05), abs=0.01
        )

    def test_monte_carlo_standard_error_worst_case(self):
        assert tolerance_standard_error(0.5, 1000) == pytest.approx(0.0158, abs=1e-4)
        assert tolerance_standard_error(1.0, 1000) == 0.0

    def test_joint_tolerance_bounds(self):
        rng = np.random.default_rng(3)
        a, b = rng.standard_normal((2, 300_000))
        indep = {"a": a, "b": b}
        deltas = {"a": 1.0, "b": 1.0}
        p = 2 * norm.cdf(1.0) - 1
        assert joint_tolerance_probability(indep, deltas) == pytest.approx(
            p * p, abs=5e-3
        )
        same = {"a": a, "b": a}
        assert joint_tolerance_probability(same, deltas) == pytest.approx(
            tolerance_probability(a, 1.0)
        )
        anti = {"a": a, "b": -0.9 * a + 0.3 * b}
        joint = joint_tolerance_probability(anti, deltas)
        assert joint <= min(tolerance_probability(v, 1.0) for v in anti.values())


class TestReducedChi2:
    def test_band_numbers(self):
        nu = 256 - 5
        lower, upper = acceptance_band(nu)
        assert np.sqrt(2 / nu) == pytest.approx(0.089, abs=1e-3)
        assert lower == pytest.approx(0.82, abs=5e-3)
        assert 1 + 2 * np.sqrt(2 / nu) == pytest.approx(1.18, abs=5e-3)
        assert upper == pytest.approx(1.20)
        _, upper_small_nu = acceptance_band(20)
        assert upper_small_nu == pytest.approx(1 + 2 * np.sqrt(0.1))

    @pytest.mark.parametrize("statistic", ["pearson", "deviance"])
    def test_reference_distribution_under_true_model(self, statistic):
        lam = _decay(2e5, bg=60.0)
        y = _noisy(lam, 4000, seed=4)
        chi2_nu, nu = reduced_chi2(y, lam, 0, statistic=statistic)
        assert np.all(nu == K)
        assert chi2_nu.mean() == pytest.approx(1.0, abs=0.01)
        assert chi2_nu.std() == pytest.approx(np.sqrt(2 / K), rel=0.08)

    def test_pearson_and_deviance_agree_at_high_counts(self):
        lam = _decay(1e6, bg=200.0)
        y = _noisy(lam, 200, seed=5)
        p, _ = reduced_chi2(y, lam, 4, statistic="pearson")
        d, _ = reduced_chi2(y, lam, 4, statistic="deviance")
        np.testing.assert_allclose(p, d, atol=0.02)

    def test_degrees_of_freedom_from_window(self):
        lam = _decay(1e4)
        mask = np.arange(K) < 100
        _, nu = reduced_chi2(lam, lam, 5, mask=mask)
        assert nu == 95


class TestNoiseNormalisedR2:
    def test_rho_identity(self):
        lam = _decay(5e3)
        y = _noisy(lam, 50, seed=6)
        r = residual_noise_ratio(y, lam, 4)
        np.testing.assert_allclose(r["rho_R"], (1 - r["R2"]) / (1 - r["R2_ceil"]))
        np.testing.assert_allclose(r["E_RSS"], (K - 4) / K * lam.sum())

    def test_rho_is_photon_count_independent_but_raw_r2_is_not(self):
        rho, r2 = [], []
        for i, n in enumerate((1e3, 1e4, 1e5)):
            lam = _decay(n, bg=0.5)
            out = residual_noise_ratio(_noisy(lam, 2000, seed=7 + i), lam, 0)
            rho.append(out["rho_R"].mean())
            r2.append(out["R2"].mean())
        np.testing.assert_allclose(rho, 1.0, atol=0.02)
        assert r2[0] < 0.95 < r2[2]

    def test_weighted_r2_normalised_is_reduced_chi2(self):
        lam = _decay(2e4)
        y = _noisy(lam, 30, seed=10)
        r2w, r2w_ceil = weighted_r2(y, lam, 4)
        chi2_nu, _ = reduced_chi2(y, lam, 4)
        np.testing.assert_allclose((1 - r2w) / (1 - r2w_ceil), chi2_nu, rtol=1e-10)

    def test_peak_misfit_seen_by_rho_not_by_chi2(self):
        lam_true = _decay(3e5)
        bump = np.exp(-0.5 * ((np.arange(K) - np.argmax(lam_true)) / 2.0) ** 2)
        lam_fit = lam_true * (1 - 0.03 * bump)
        y = _noisy(lam_true, 2000, seed=11)
        chi2_nu, nu = reduced_chi2(y, lam_fit, 4)
        rho = residual_noise_ratio(y, lam_fit, 4)["rho_R"]
        lower, upper = acceptance_band(nu[0])
        assert lower < np.median(chi2_nu) < upper
        assert np.median(rho) > upper
        a_chi = gate(np.median(chi2_nu), lower, upper)
        a_r = gate(np.median(rho), lower, upper)
        assert a_chi == 1.0 and a_r < 0.6


class TestGate:
    def test_shape(self):
        lo, up, s = 0.8, 1.2, 0.2
        x = np.array([lo - s, lo, 1.0, up, up + s, up + 2 * s])
        np.testing.assert_allclose(
            gate(x, lo, up, s),
            [np.exp(-0.5), 1, 1, 1, np.exp(-0.5), np.exp(-2.0)],
        )

    def test_monotone_outside_band(self):
        x = np.linspace(1.2, 3.0, 50)
        assert np.all(np.diff(gate(x, 0.8, 1.2)) < 0)
        x = np.linspace(0.0, 0.8, 50)
        assert np.all(np.diff(gate(x, 0.8, 1.2)) > 0)


class TestWorkedIllustration:
    def test_section_7_8_table(self):
        chi2 = np.array([1.05, 1.45, 1.10, 1.08])
        rho = np.array([1.08, 1.30, 1.60, 1.10])
        a_m = np.array([0.93, 0.90, 0.85, 0.20])
        out = combine_fit_aware_factors(0.90, chi2, rho, a_m, nu=251)
        assert out["lower"] == pytest.approx(0.82, abs=5e-3)
        assert out["upper"] == pytest.approx(1.20)
        np.testing.assert_allclose(out["A_chi"], [1.00, 0.46, 1.00, 1.00], atol=5e-3)
        np.testing.assert_allclose(
            out["A_R"] * out["A_M_tilde"], [1.00, 0.88, 0.14, 0.40], atol=5e-3
        )
        np.testing.assert_allclose(out["T_gof"], [0.90, 0.36, 0.12, 0.36], atol=5e-3)

    def test_adequate_pixel_keeps_T_delta(self):
        out = combine_fit_aware_factors(0.73, 1.0, 1.0, 0.6, nu=251)
        assert out["T_gof"] == pytest.approx(0.73)

    def test_early_window_takes_minimum(self):
        full = combine_fit_aware_factors(1.0, 1.0, 1.0, 1.0, nu=251)
        both = combine_fit_aware_factors(
            1.0, 1.0, 1.0, 1.0, nu=251, chi2_nu_early=1.6, rho_R_early=1.0, nu_early=50
        )
        lo, up = acceptance_band(50)
        assert full["A_chi"] == 1.0
        assert both["A_chi"] == pytest.approx(gate(1.6, lo, up))
        undefined = combine_fit_aware_factors(
            1.0,
            1.0,
            1.0,
            1.0,
            nu=251,
            chi2_nu_early=np.nan,
            rho_R_early=np.nan,
            nu_early=0,
        )
        assert undefined["A_chi"] == 1.0


class TestDispersionAdjusted:
    def test_section_7_7_numbers(self):
        delta = norm.ppf(0.95)
        assert gaussian_tolerance_probability(delta, 1.0) == pytest.approx(
            0.90, abs=1e-3
        )
        assert gaussian_tolerance_probability(delta, 1.0, 1.45) == pytest.approx(
            0.83, abs=5e-3
        )
        assert gaussian_tolerance_probability(delta, 1.0, 2.0) == pytest.approx(
            0.76, abs=5e-3
        )

    @pytest.mark.parametrize("phi", [0.5, 1.0, 1.45, 2.0, 4.0])
    def test_matches_gaussian_closed_form(self, phi):
        sigma, delta = 0.2, 0.25
        draws = np.random.default_rng(12).normal(3.0, sigma, 400_000)
        T = dispersion_adjusted_tolerance(draws, delta, phi, center=3.0)
        assert T == pytest.approx(
            gaussian_tolerance_probability(delta, sigma, max(1.0, phi)), abs=3e-3
        )

    def test_inflating_draws_is_equivalent(self):
        draws = np.random.default_rng(13).gamma(4.0, 0.5, 50_000)
        m = np.median(draws)
        phi = 1.7
        inflated = m + np.sqrt(phi) * (draws - m)
        assert dispersion_adjusted_tolerance(draws, 0.3, phi) == pytest.approx(
            tolerance_probability(inflated, 0.3, center=m), abs=1e-4
        )


def _product_forward(params, y):
    tau = np.asarray(params["u"]) * np.asarray(params["v"])
    shape = np.exp(-T_NS / tau[..., None])
    shape = shape / shape.sum(axis=-1, keepdims=True)
    return np.sum(y, axis=-1)[..., None] * shape


class TestMedianRepresentativeness:
    def test_unimodal_median_is_representative(self):
        lam = 1e4 * np.exp(-T_NS / 2.0) / np.exp(-T_NS / 2.0).sum()
        y = np.broadcast_to(lam, (3, K))
        rng = np.random.default_rng(14)
        samples = {
            "u": 2.0 + 0.05 * rng.standard_normal((3, 400)),
            "v": np.ones((3, 400)),
        }
        fat = FitAwareTolerance(_product_forward, {"u": 0.1}, n_params=2, t=T_NS)
        out = fat(samples, y)
        assert np.all(out["fit"]["A_M"] > 0.8)
        assert np.all(out["fit"]["A_M_tilde"] == 1.0)

    def test_bimodal_component_medians_are_unrepresentative(self):
        lam = 1e4 * np.exp(-T_NS / 1.0) / np.exp(-T_NS / 1.0).sum()
        y = np.broadcast_to(lam, (3, K))
        rng = np.random.default_rng(15)
        z = np.where(rng.random((3, 400)) < 0.5, -1.5, 1.5)
        z = z + 0.05 * rng.standard_normal(z.shape)
        samples = {"u": np.exp(z), "v": np.exp(-z)}
        fat = FitAwareTolerance(_product_forward, {"u": 0.1}, n_params=2, t=T_NS)
        out = fat(samples, y)
        assert np.all(out["fit"]["A_M"] < 0.05)
        assert np.all(out["T_gof"]["u"] < 0.1 * out["T_delta"]["u"] + 1e-12)
