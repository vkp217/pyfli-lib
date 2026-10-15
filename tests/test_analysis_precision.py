"""
Tests for pyfli.analysis.precision: Fisher information / CRLB, the lifetime CRLB of the
gated photon-counting model, fit-free lifetimes and the Monte Carlo precision study.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from pyfli.analysis import precision as pr
from pyfli.simulator.calibration.expected_decay import (
    DecayForwardModel,
    gated_bin_probabilities,
    wrapped_gate_exponential,
)

PERIOD = 12.5
T = 256


def gaussian_irf(n=T, center=20.0, width=1.5):
    irf = np.exp(-0.5 * ((np.arange(n) - center) / width) ** 2)
    return irf / irf.sum()


class TestFisherCRLB:
    def test_numerical_jacobian_of_linear_model(self):
        A = np.array([[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]])
        jac = pr.numerical_jacobian(
            lambda th: th @ A, np.array([[1.0, 2.0], [3.0, -1.0]])
        )
        assert jac.shape == (2, 2, 3)
        np.testing.assert_allclose(jac[0], A, rtol=1e-6)
        np.testing.assert_allclose(jac[1], A, rtol=1e-6)
        assert pr.numerical_jacobian(lambda th: th @ A, np.array([1.0, 2.0])).shape == (
            1,
            2,
            3,
        )

    def test_poisson_count_bound_is_sqrt_n(self):
        p = np.array([0.2, 0.3, 0.5, 0.0])
        n = np.array([[100.0], [10000.0]])
        info = pr.fisher_information(lambda th: th[:, :1] * p, n)
        np.testing.assert_allclose(info[:, 0, 0], 1 / n[:, 0], rtol=1e-6)
        np.testing.assert_allclose(pr.crlb_std(info, 0), np.sqrt(n[:, 0]), rtol=1e-6)
        np.testing.assert_allclose(
            pr.crlb_cv(info, n[:, 0], 0), 1 / np.sqrt(n[:, 0]), rtol=1e-6
        )

    def test_gaussian_information(self):
        jac = np.ones((1, 1, 5))
        np.testing.assert_allclose(
            pr.gaussian_fisher_information(2.0, jac), [[[5 / 4]]]
        )
        info = pr.fisher_information(
            lambda th: np.repeat(th, 5, axis=1),
            np.array([[1.0]]),
            noise="gaussian",
            sigma=2.0,
        )
        np.testing.assert_allclose(info, [[[5 / 4]]], rtol=1e-6)
        with pytest.raises(ValueError, match="sigma"):
            pr.fisher_information(lambda th: th, np.ones((1, 1)), noise="gaussian")
        with pytest.raises(ValueError, match="poisson"):
            pr.fisher_information(lambda th: th, np.ones((1, 1)), noise="laplace")

    def test_poisson_skips_empty_channels(self):
        info = pr.poisson_fisher_information(
            np.array([[0.0, 4.0]]), np.array([[[5.0, 2.0]]])
        )
        np.testing.assert_allclose(info, [[[1.0]]])

    def test_select_and_covariance(self):
        info = np.array([[[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]]])
        np.testing.assert_allclose(
            pr.select_parameters(info, [0, 2]), [[[4.0, 0.0], [0.0, 2.0]]]
        )
        np.testing.assert_allclose(
            pr.crlb_covariance(info)[0] @ info[0], np.eye(3), atol=1e-12
        )
        singular = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        assert np.all(np.isfinite(pr.crlb_covariance(singular)))

    def test_f_value_and_efficiency(self):
        assert pr.f_value(0.01, 10000) == pytest.approx(1.0)
        eff = pr.estimator_efficiency(np.array([0.02, 0.0]), np.array([0.01, 0.01]))
        assert eff[0] == pytest.approx(0.25) and np.isnan(eff[1])


class TestGatedModel:
    def test_wrapped_exponential_and_gated_probabilities(self):
        p = wrapped_gate_exponential(np.array([0.5, 3.0]), T, PERIOD)
        np.testing.assert_allclose(p.sum(-1), 1.0)
        g = gated_bin_probabilities(np.array([[0.5, 3.0]]), gaussian_irf(), PERIOD)
        assert g.shape == (1, 2, T) and np.allclose(g.sum(-1), 1.0) and np.all(g >= 0)
        delta = np.zeros(T)
        delta[0] = 1.0
        np.testing.assert_allclose(
            gated_bin_probabilities(2.0, delta, PERIOD),
            wrapped_gate_exponential(2.0, T, PERIOD),
            atol=1e-12,
        )

    def test_per_value_irf_matches_shared(self):
        irf = gaussian_irf()
        shared = gated_bin_probabilities(np.array([1.0, 2.0]), irf, PERIOD)
        per = gated_bin_probabilities(
            np.array([1.0, 2.0]), np.stack([irf, irf]), PERIOD
        )
        np.testing.assert_allclose(shared, per, atol=1e-14)

    def test_same_model_as_calibration_forward_model(self):
        irf = gaussian_irf()
        fm = DecayForwardModel(T, PERIOD, "discrete", jitter=False)
        mix = np.array([0.3, 0.7]) @ gated_bin_probabilities(
            np.array([0.4, 2.5]), irf, PERIOD
        )
        np.testing.assert_allclose(
            fm.shape(irf, [0.4, 2.5], [0.3, 0.7]), mix / mix.sum(), atol=1e-15
        )

    def test_gated_decay_model(self):
        mu = pr.gated_decay_model(
            np.array([[1000.0, 2.0, 0.5], [10.0, 1.0, 0.0]]), gaussian_irf(), PERIOD
        )
        assert mu.shape == (2, T)
        np.testing.assert_allclose(mu.sum(-1), [1000 + 0.5 * T, 10.0])
        assert pr.PARAMS[pr.TAU_INDEX] == "tau"


class TestCRB:
    def test_ideal_limit_f_is_one(self):
        n = 4096
        irf = np.zeros(n)
        irf[0] = 1.0
        f = pr.crb_f_value(1.0, irf, 200.0)
        assert f == pytest.approx(1.0, abs=0.02)

    def test_cv_scales_inverse_sqrt_n_without_background(self):
        cv = pr.crb_lifetime_cv(2.0, np.array([100.0, 10000.0]), gaussian_irf(), PERIOD)
        assert cv[0] / cv[1] == pytest.approx(10.0, rel=1e-4)
        f = pr.crb_f_value(2.0, gaussian_irf(), PERIOD)
        assert 1.0 < f < 2.0

    def test_background_and_free_offset_cost_precision(self):
        irf = gaussian_irf()
        clean = pr.crb_lifetime_cv(2.0, 1000.0, irf, PERIOD)
        noisy = pr.crb_lifetime_cv(2.0, 1000.0, irf, PERIOD, background_per_bin=0.5)
        free = pr.crb_lifetime_cv(
            2.0, 1000.0, irf, PERIOD, background_per_bin=0.5, fit_background=True
        )
        assert clean < noisy < free
        f_low = pr.crb_f_value(2.0, irf, PERIOD, n_photons=200, background_per_bin=0.5)
        f_high = pr.crb_f_value(
            2.0, irf, PERIOD, n_photons=20000, background_per_bin=0.5
        )
        assert f_low > f_high

    def test_invalid_and_broadcast(self):
        cv = pr.crb_lifetime_cv(
            np.array([[2.0, -1.0], [np.nan, 1.0]]),
            np.array([1000.0, 0.0]),
            gaussian_irf(),
            PERIOD,
        )
        assert (
            cv.shape == (2, 2)
            and np.isfinite(cv[0, 0])
            and np.isnan(cv[0, 1])
            and np.isnan(cv[1, 0])
        )
        assert np.isnan(cv[1, 1])

    def test_chunking_matches_single(self, monkeypatch):
        from pyfli.analysis.precision import crb_precision

        taus = np.linspace(0.5, 4.0, 9)
        full = pr.crb_lifetime_cv(taus, 500.0, gaussian_irf(), PERIOD)
        monkeypatch.setattr(crb_precision, "_CHUNK", 2)
        np.testing.assert_allclose(
            pr.crb_lifetime_cv(taus, 500.0, gaussian_irf(), PERIOD), full, rtol=1e-10
        )

    def test_cv_map_with_mask_and_irf_cube(self):
        tau_map = np.array([[1.0, 2.0], [0.0, 3.0]])
        n_map = np.full((2, 2), 1000.0)
        irf = gaussian_irf()
        cv = pr.crb_cv_map(tau_map, n_map, irf, PERIOD)
        assert np.isnan(cv[1, 0]) and np.all(np.isfinite(cv[[0, 0, 1], [0, 1, 1]]))
        cube = np.broadcast_to(irf, (2, 2, T))
        np.testing.assert_allclose(
            pr.crb_cv_map(tau_map, n_map, cube, PERIOD), cv, equal_nan=True
        )
        only = pr.crb_cv_map(
            tau_map, n_map, irf, PERIOD, mask=np.array([[True, False], [False, False]])
        )
        assert np.isfinite(only[0, 0]) and np.isnan(only[0, 1])

    def test_crb_for_cv_bins(self):
        tau_map = np.full((4, 4), 2.0)
        photons = np.repeat(np.array([100.0, 200.0, 400.0, 800.0])[:, None], 4, axis=1)
        clusters = np.zeros((4, 4), int)
        clusters[:, 2:] = 1
        cv_df = pd.DataFrame(
            {
                "cluster": ["a", "a", "b"],
                "bin_rank": [0, 1, 0],
                "bin_low": [100.0, 200.0, 100.0],
                "bin_high": [200.0, 800.0, 800.0],
                "bin_center": [150.0, 500.0, 450.0],
            }
        )
        out = pr.crb_for_cv_bins(
            cv_df,
            tau_map,
            photons,
            gaussian_irf(),
            PERIOD,
            cluster_mask=clusters,
            cluster_names={0: "a", 1: "b"},
        )
        assert list(out.columns) == [
            "cluster",
            "bin_rank",
            "bin_center",
            "tau_true",
            "n_signal",
            "cv_crb",
            "F_crb",
        ]
        assert len(out) == 3 and np.allclose(out["tau_true"], 2.0)
        row = out[(out.cluster == "a") & (out.bin_rank == 0)].iloc[0]
        assert row["n_signal"] == pytest.approx(150.0)
        expected = pr.crb_lifetime_cv(
            2.0, np.array([100.0, 200.0]), gaussian_irf(), PERIOD
        )
        assert row["cv_crb"] == pytest.approx(np.sqrt(np.mean(expected**2)), rel=1e-6)
        assert row["F_crb"] == pytest.approx(row["cv_crb"] * np.sqrt(150.0))
        empty = pd.DataFrame(
            {
                "cluster": [None],
                "bin_rank": [0],
                "bin_low": [1e6],
                "bin_high": [2e6],
                "bin_center": [1.5e6],
            }
        )
        assert pr.crb_for_cv_bins(empty, tau_map, photons, gaussian_irf(), PERIOD).empty


class TestFitFree:
    @pytest.fixture
    def model_decays(self):
        irf = gaussian_irf()
        taus = np.array([0.5, 1.5, 3.0, 6.0])
        return taus, 1e5 * gated_bin_probabilities(taus, irf, PERIOD), irf

    def test_gate_times(self):
        np.testing.assert_allclose(pr.gate_times(4, 10.0), [0, 2.5, 5.0, 7.5])

    def test_exact_estimators_recover_tau(self, model_decays):
        taus, decays, irf = model_decays
        np.testing.assert_allclose(
            pr.mean_arrival_lifetime(decays, irf, PERIOD), taus, rtol=1e-3
        )
        np.testing.assert_allclose(
            pr.phasor_lifetime(decays, irf, PERIOD, bin_corrected=True), taus, rtol=1e-3
        )
        x, phi = np.exp(-PERIOD / (T * taus)), 2 * np.pi / T
        binned_phase_tau = (
            x * np.sin(phi) / (1 - x * np.cos(phi)) / (2 * np.pi / PERIOD)
        )
        np.testing.assert_allclose(
            pr.phasor_lifetime(decays, irf, PERIOD), binned_phase_tau, rtol=1e-3
        )
        naive = pr.mean_arrival_time_naive(decays, irf, PERIOD)
        assert abs(naive[-1] - taus[-1]) > 0.5

    def test_background_subtraction(self, model_decays):
        taus, decays, irf = model_decays
        noisy = decays + 50.0
        biased = pr.mean_arrival_lifetime(noisy, irf, PERIOD)
        fixed = pr.mean_arrival_lifetime(noisy, irf, PERIOD, background_per_bin=50.0)
        assert np.abs(biased - taus).max() > 0.1
        np.testing.assert_allclose(fixed, taus, rtol=1e-3)
        np.testing.assert_allclose(
            pr.phasor_lifetime(noisy, irf, PERIOD, 50.0, bin_corrected=True),
            taus,
            rtol=1e-3,
        )

    def test_empty_and_bad_irf(self):
        irf = gaussian_irf()
        assert np.isnan(pr.mean_arrival_lifetime(np.zeros(T), irf, PERIOD))
        assert np.isnan(pr.mean_arrival_time_naive(np.zeros((1, T)), irf, PERIOD)[0])
        assert np.isnan(pr.phasor_lifetime(np.zeros((1, T)), irf, PERIOD)[0])
        with pytest.raises(ValueError, match="1-D IRF"):
            pr.mean_arrival_lifetime(np.ones(T), np.ones((2, T)), PERIOD)

    def test_maps(self, model_decays):
        taus, decays, irf = model_decays
        cube = np.zeros((2, 3, T))
        cube[0] = decays[:3]
        mask = cube.sum(-1) > 0
        maps = pr.fit_free_lifetime_maps(
            cube, np.broadcast_to(irf, cube.shape), PERIOD, mask=mask
        )
        assert set(maps) == {
            "tau_mean_arrival",
            "tau_mean_arrival_naive",
            "tau_phasor",
            "tau_phasor_bin_corrected",
        }
        np.testing.assert_allclose(maps["tau_mean_arrival"][0], taus[:3], rtol=1e-3)
        assert np.all(np.isnan(maps["tau_phasor"][1]))
        default = pr.fit_free_lifetime_maps(cube, irf, PERIOD)
        np.testing.assert_allclose(
            default["tau_phasor_bin_corrected"][0], taus[:3], rtol=1e-3
        )


class TestMonteCarlo:
    def test_simulate_fixed_lifetime(self):
        irf = gaussian_irf()
        decays, used = pr.simulate_fixed_lifetime_decays(irf, 2.0, 500.0, 300, seed=1)
        again, _ = pr.simulate_fixed_lifetime_decays(irf, 2.0, 500.0, 300, seed=1)
        assert decays.shape == (300, T) and used.sum() == pytest.approx(1.0)
        np.testing.assert_array_equal(decays, again)
        assert decays.sum(1).mean() == pytest.approx(500.0, rel=0.05)
        with pytest.raises(ValueError, match="minimum"):
            pr.simulate_fixed_lifetime_decays(irf, 2.0, 5.0, 3)

    def test_mean_arrival_reaches_bound(self):
        df = pr.run_monte_carlo(
            gaussian_irf(),
            2.0,
            [1000.0],
            n_repeats=600,
            methods=("mean_arrival", "phasor"),
            seed=0,
        )
        cols = {
            "method",
            "n_photons",
            "n_detected",
            "n_valid",
            "tau_true",
            "tau_mean",
            "tau_std",
            "bias",
            "rel_bias",
            "cv",
            "F",
            "cv_crb",
            "F_crb",
            "efficiency",
        }
        assert cols <= set(df.columns) and len(df) == 2
        row = df[df.method == "mean_arrival"].iloc[0]
        assert (
            abs(row["rel_bias"]) < 0.02
            and 0.7 < row["efficiency"] < 1.3
            and row["n_valid"] == 600
        )
        assert row["F"] == pytest.approx(row["cv"] * np.sqrt(1000.0))

    def test_fits_and_methods(self):
        irf = gaussian_irf()
        decays, used = pr.simulate_fixed_lifetime_decays(irf, 2.0, 2000.0, 6, seed=2)
        from pyfli.analyticalWorkflow import AnalyticalHelpers

        freq = AnalyticalHelpers(
            laser_period=PERIOD, gate_delay=PERIOD / T, num_gate=T
        ).freq_computation()
        tau = pr.estimate_lifetimes(decays, used, PERIOD, "MLE", freq=freq, n_jobs=1)
        assert tau.shape == (6,) and np.nanmedian(tau) == pytest.approx(2.0, rel=0.15)
        for method in pr.FIT_FREE_METHODS:
            assert pr.estimate_lifetimes(decays, used, PERIOD, method).shape == (6,)
        with pytest.raises(ValueError, match="Unknown method"):
            pr.estimate_lifetimes(decays, used, PERIOD, "guess")
        assert set(pr.FIT_METHODS) == {"NLSF", "MLE"}

    def test_plot(self):
        df = pr.run_monte_carlo(
            gaussian_irf(),
            1.5,
            [200.0, 2000.0],
            n_repeats=50,
            methods=("mean_arrival",),
            seed=3,
        )
        fig, (ax_cv, ax_f) = pr.plot_monte_carlo(df)
        assert (
            ax_cv.get_xscale() == "log"
            and len(ax_cv.lines) == 2
            and len(ax_f.lines) == 3
        )


def test_public_api():
    for name in pr.__all__:
        assert getattr(pr, name) is not None
