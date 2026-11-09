"""Unit tests for pyfli.bayes_utils.fit_aware_tolerance (API, shapes, forward model)."""

import numpy as np
import pytest

from pyfli.bayes_utils import (
    FitAwareTolerance,
    PosteriorDecayForward,
    tolerance_probability,
)
from pyfli.bayes_utils.fit_aware_tolerance import (
    acceptance_band,
    early_window_mask,
    gate,
    mean_lifetime_draws,
    median_representativeness,
    reduced_chi2,
    representativeness_penalty,
)
from pyfli.reconstruction import ParamToDecay

K = 128
FREQ = 80.0
H, W, S = 3, 4, 200


@pytest.fixture(scope="module")
def irf():
    t = np.arange(K) * 1000.0 / FREQ / K
    return np.exp(-0.5 * ((t - 1.0) / 0.1) ** 2)


@pytest.fixture(scope="module")
def forward(irf):
    return PosteriorDecayForward("bi-exponential", FREQ, irf=irf)


@pytest.fixture(scope="module")
def truth():
    return {
        "alpha1": np.full((H, W), 0.4),
        "tau1": np.full((H, W), 0.5),
        "tau2": np.full((H, W), 3.0),
    }


@pytest.fixture(scope="module")
def data(forward, truth):
    rng = np.random.default_rng(0)
    lam = forward(truth, np.full((H, W, K), 2e4 / K))
    y = rng.poisson(lam).astype(float)
    samples = {
        k: v[..., None] * (1 + 0.02 * rng.standard_normal((H, W, S)))
        for k, v in truth.items()
    }
    return samples, y


DELTA = {"tau1": 0.05, "tau2": 0.1, "alpha1": 0.05}


class TestHelpers:
    def test_tolerance_probability_counts_strictly_inside(self):
        s = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        assert tolerance_probability(s, 0.5, center=1.0) == pytest.approx(1 / 5)
        assert tolerance_probability(s, 0.51, center=1.0) == pytest.approx(3 / 5)

    def test_tolerance_probability_map_and_validation(self):
        s = np.random.default_rng(1).normal(size=(2, 3, 100))
        T = tolerance_probability(s, np.full((2, 3), 0.5))
        assert T.shape == (2, 3)
        assert np.all((T >= 0) & (T <= 1))
        with pytest.raises(ValueError, match="non-negative"):
            tolerance_probability(s, -1.0)

    def test_tolerance_axis(self):
        s = np.random.default_rng(2).normal(size=(5, 50))
        np.testing.assert_allclose(
            tolerance_probability(s.T, 0.3, axis=0), tolerance_probability(s, 0.3)
        )

    def test_mean_lifetime(self):
        assert mean_lifetime_draws(0.25, 0.4, 2.0) == pytest.approx(1.6)

    def test_reduced_chi2_validation_and_nan_dof(self):
        with pytest.raises(ValueError, match="statistic"):
            reduced_chi2(np.ones(5), np.ones(5), 1, statistic="neyman")
        chi2, nu = reduced_chi2(np.ones(5), np.ones(5), 5)
        assert np.isnan(chi2) and nu == 0

    def test_acceptance_band_and_gate_validation(self):
        lo, up = acceptance_band(np.array([0, 100]))
        assert np.isnan(lo[0]) and np.isnan(up[0])
        assert np.isnan(gate(np.nan, 0.8, 1.2))
        with pytest.raises(ValueError):
            acceptance_band(10, z0=-1)
        with pytest.raises(ValueError):
            gate(1.0, 0.8, 1.2, s=0)

    def test_median_representativeness_ties_count(self):
        a_m = median_representativeness(np.array([1.0, 1.0, 0.5, 2.0]), 1.0)
        assert a_m == pytest.approx(0.75)
        np.testing.assert_allclose(
            representativeness_penalty([0.1, 0.5, 0.9]), [0.2, 1.0, 1.0]
        )

    def test_early_window(self):
        t = np.arange(10, dtype=float)
        lam = np.zeros((2, 10))
        lam[0, 2] = lam[1, 5] = 1.0
        win = early_window_mask(t, lam, np.array([1.0, 0.5]), n_tau=3.0)
        assert win[0].sum() == 6 and win[1].sum() == 7
        masked = early_window_mask(t, lam, np.array([1.0, 0.5]), mask=t >= 1)
        assert masked[0].sum() == 5


class TestPosteriorDecayForward:
    def test_matches_param_to_decay_rescaled(self, forward, truth, data, irf):
        _, y = data
        lam = forward(truth, y)
        recon = ParamToDecay("bi-exponential", FREQ, irf=irf)
        params = {f"{k}_map": v for k, v in truth.items()}
        params["photon_count_map"] = np.ones((H, W))
        unit = recon.reconstruct_unit_amplitude(params)["convolved_map"]
        expected = recon.rescale_fit_to_measured_totals(unit, y)
        np.testing.assert_allclose(lam, expected, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(lam.sum(-1), y.sum(-1), rtol=1e-6)

    def test_draw_batch_axis_broadcasts(self, forward, data):
        samples, y = data
        batch = {k: np.moveaxis(v[..., :3], -1, 0) for k, v in samples.items()}
        lam = forward(batch, y)
        assert lam.shape == (3, H, W, K)
        single = forward({k: v[..., 1] for k, v in samples.items()}, y)
        np.testing.assert_allclose(lam[1], single, rtol=1e-6)

    def test_per_pixel_irf(self, irf, truth, data):
        _, y = data
        cube = np.broadcast_to(irf, (H, W, K))
        lam_cube = PosteriorDecayForward("bi-exponential", FREQ, irf=cube)(truth, y)
        lam_1d = PosteriorDecayForward("bi-exponential", FREQ, irf=irf)(truth, y)
        np.testing.assert_allclose(lam_cube, lam_1d, rtol=1e-6)

    def test_mono_without_irf_and_background(self):
        fwd = PosteriorDecayForward(
            "mono-exponential", FREQ, num_gates=K, background_key="b"
        )
        assert fwd.param_keys == ("tau", "b") and fwd.n_params == 3
        y = np.full(K, 10.0)
        lam = fwd({"tau": 2.0, "b": 1.5}, y)
        assert lam.shape == (K,)
        assert lam.sum() == pytest.approx(y.sum())
        assert lam.min() >= 1.5 - 1e-9

    def test_amplitude_mask(self, forward, truth, data):
        _, y = data
        window = np.arange(K) < 60
        fwd = PosteriorDecayForward(
            "bi-exponential", FREQ, irf=forward.irf, amplitude_mask=window
        )
        lam = fwd(truth, y)
        np.testing.assert_allclose(lam[..., window].sum(-1), y[..., window].sum(-1))

    def test_validation(self):
        with pytest.raises(ValueError, match="model_type"):
            PosteriorDecayForward("tri-exponential", FREQ, num_gates=K)
        with pytest.raises(ValueError, match="num_gates"):
            PosteriorDecayForward("mono-exponential", FREQ)
        fwd = PosteriorDecayForward("mono-exponential", FREQ, num_gates=K)
        with pytest.raises(ValueError, match="bins"):
            fwd({"tau": 1.0}, np.ones(K + 1))


class TestBackground:
    def test_fixed_value(self, irf, truth, data):
        _, y = data
        fwd = PosteriorDecayForward("bi-exponential", FREQ, irf=irf, background=0.7)
        lam = fwd(truth, y)
        assert fwd.n_params == 4
        assert lam.min() >= 0.7 - 1e-9
        np.testing.assert_allclose(lam.sum(-1), y.sum(-1), rtol=1e-6)
        np.testing.assert_allclose(lam[..., 0], 0.7, atol=1e-6)

    def test_map(self, irf, truth, data):
        _, y = data
        b = np.linspace(0.1, 2.0, H * W).reshape(H, W)
        fwd = PosteriorDecayForward("bi-exponential", FREQ, irf=irf, background=b)
        lam = fwd(truth, y)
        np.testing.assert_allclose(lam[..., 0], b, atol=1e-6)
        batch = {k: np.broadcast_to(v, (2, H, W)) for k, v in truth.items()}
        np.testing.assert_allclose(fwd(batch, y)[1], lam)

    def test_auto_pre_onset_bins(self, irf):
        fwd = PosteriorDecayForward("bi-exponential", FREQ, irf=irf, background="auto")
        onset = np.argmax(irf >= 0.01 * irf.max())
        assert fwd.background_bins.sum() == onset and fwd.background_bins[:onset].all()
        assert fwd.n_params == 5

    def test_auto_recovers_exact_background(self, irf):
        fwd = PosteriorDecayForward(
            "mono-exponential",
            FREQ,
            irf=irf,
            background="auto",
            background_bins=np.arange(K) < 6,
        )
        rng = np.random.default_rng(5)
        shape = rng.random((4, K))
        shape[:, :6] *= 0.01
        b_true, a_true = np.array([0.05, 0.3, 1.0, 7.0]), np.array([5.0, 50, 1, 0.2])
        y = b_true[:, None] + a_true[:, None] * shape
        np.testing.assert_allclose(
            fwd.auto_background(shape, y, (4,)), b_true, rtol=1e-5
        )
        assert np.all(fwd.auto_background(shape, np.zeros((4, K)), (4,)) == 0.0)

    def test_auto_satisfies_poisson_score_equations(self, irf, truth):
        rng = np.random.default_rng(6)
        fwd = PosteriorDecayForward("bi-exponential", FREQ, irf=irf, background="auto")
        lead = (H, W)
        shape = fwd.shape(truth, lead)
        y = rng.poisson(0.2 + 300 * shape / shape.sum(-1, keepdims=True)).astype(float)
        b = fwd.auto_background(shape, y, lead)
        lam = fwd(truth, y)
        assert np.all(b > 0)
        np.testing.assert_allclose(lam[..., 0], b, rtol=1e-6)
        np.testing.assert_allclose(np.sum(y / lam, axis=-1), K, rtol=1e-5)
        np.testing.assert_allclose(lam.sum(-1), y.sum(-1), rtol=1e-6)

    def test_auto_positive_when_counts_sit_where_shape_is_zero(self, irf, truth):
        fwd = PosteriorDecayForward("bi-exponential", FREQ, irf=irf, background="auto")
        y = fwd(truth, np.full((H, W, K), 2e4 / K))
        y = np.round(y)
        y[..., 0] = 1.0
        lam = fwd(truth, y)
        chi2, _ = reduced_chi2(y, lam, fwd.n_params)
        assert np.all(lam[..., 0] > 1e-3)
        assert np.all(chi2 < 10)

    def test_auto_exact_on_noise_free_data(self, irf, truth):
        fwd0 = PosteriorDecayForward("bi-exponential", FREQ, irf=irf, background=0.5)
        y = fwd0(truth, np.full((H, W, K), 2e4 / K))
        fwd = PosteriorDecayForward("bi-exponential", FREQ, irf=irf, background="auto")
        np.testing.assert_allclose(fwd(truth, y), y, rtol=1e-6, atol=1e-6)

    def test_auto_per_pixel_irf(self, irf, truth, data):
        _, y = data
        shifted = np.stack([np.roll(irf, 5 * i) for i in range(W)])
        cube = np.broadcast_to(shifted, (H, W, K))
        fwd = PosteriorDecayForward("bi-exponential", FREQ, irf=cube, background="auto")
        assert fwd.background_bins.shape == (H, W, K)
        assert np.all(np.diff(fwd.background_bins.sum(-1), axis=-1) == 5)
        assert fwd(truth, y).shape == (H, W, K)

    def test_auto_is_unbiased_for_dark_counts(self, irf, truth):
        rng = np.random.default_rng(3)
        fwd0 = PosteriorDecayForward("bi-exponential", FREQ, irf=irf)
        lam = fwd0(truth, np.full((H, W, K), 2e4 / K)) + 0.5
        y = rng.poisson(np.broadcast_to(lam, (500, H, W, K))).astype(float)
        fwd = PosteriorDecayForward("bi-exponential", FREQ, irf=irf, background="auto")
        b = fwd.background_map(truth, y, y.shape[:-1])
        assert b.mean() == pytest.approx(0.5, abs=0.02)

    def test_amplitude_never_negative(self, irf, truth):
        fwd = PosteriorDecayForward("bi-exponential", FREQ, irf=irf, background=5.0)
        lam = fwd(truth, np.full((H, W, K), 1.0))
        np.testing.assert_allclose(lam, 5.0)

    def test_validation(self, irf):
        with pytest.raises(ValueError, match="not both"):
            PosteriorDecayForward(
                "mono-exponential", FREQ, irf=irf, background=1.0, background_key="b"
            )
        with pytest.raises(ValueError, match="'auto'"):
            PosteriorDecayForward("mono-exponential", FREQ, irf=irf, background="mean")
        with pytest.raises(ValueError, match="needs an irf"):
            PosteriorDecayForward(
                "mono-exponential", FREQ, num_gates=K, background="auto"
            )
        with pytest.raises(ValueError, match="no bins"):
            PosteriorDecayForward(
                "mono-exponential",
                FREQ,
                irf=irf,
                background="auto",
                background_bins=np.zeros(K, bool),
            )
        with pytest.raises(ValueError, match="onset_fraction"):
            PosteriorDecayForward("mono-exponential", FREQ, irf=irf, onset_fraction=0)

    def test_dark_counts_need_background(self, irf, truth):
        rng = np.random.default_rng(4)
        fwd0 = PosteriorDecayForward("bi-exponential", FREQ, irf=irf)
        y = rng.poisson(fwd0(truth, np.full((H, W, K), 2e4 / K)) + 0.5).astype(float)
        samples = {
            k: v[..., None] * (1 + 0.02 * rng.standard_normal((H, W, S)))
            for k, v in truth.items()
        }
        none = FitAwareTolerance(fwd0, DELTA)(samples, y)
        auto = FitAwareTolerance(
            PosteriorDecayForward("bi-exponential", FREQ, irf=irf, background="auto"),
            DELTA,
        )(samples, y)
        fixed = FitAwareTolerance(
            PosteriorDecayForward("bi-exponential", FREQ, irf=irf, background=0.5),
            DELTA,
        )(samples, y)
        assert np.all(none["fit"]["chi2_nu"] > 100)
        assert np.all(none["T_gof"]["tau2"] == 0)
        for out in (auto, fixed):
            sd = np.sqrt(2 / out["fit"]["nu"])
            assert np.all(np.abs(out["fit"]["chi2_nu"] - 1) < 4 * sd)
            assert np.median(out["T_gof"]["tau2"] / out["T_delta"]["tau2"]) > 0.9
        assert np.all(auto["fit"]["nu"] == K - 5)
        assert np.all(fixed["fit"]["nu"] == K - 4)


class TestFitAwareTolerance:
    def test_outputs_shapes_and_ranges(self, forward, data):
        samples, y = data
        out = FitAwareTolerance(forward, DELTA)(samples, y)
        assert out["lam_median"].shape == (H, W, K)
        for group in FitAwareTolerance.GROUPS:
            assert set(out[group]) == set(DELTA)
            for v in out[group].values():
                assert v.shape == (H, W)
        for group in ("T_delta", "T_gof", "T_adj", "T_adj_gof", "A_chi", "A_R"):
            for v in out[group].values():
                assert np.all((v >= 0) & (v <= 1))
        assert np.all(out["fit"]["nu"] == K - forward.n_params)
        assert "chi2_nu_early" in out["fit"]

    def test_composition(self, forward, data):
        samples, y = data
        out = FitAwareTolerance(forward, DELTA)(samples, y)
        fit = out["fit"]
        for key in DELTA:
            np.testing.assert_allclose(
                out["T_gof"][key],
                out["T_delta"][key]
                * out["A_chi"][key]
                * out["A_R"][key]
                * fit["A_M_tilde"],
            )
        assert np.all(out["A_chi"]["tau1"] <= out["A_chi"]["tau2"])
        lo, up = acceptance_band(fit["nu"])
        np.testing.assert_allclose(out["A_chi"]["tau2"], gate(fit["chi2_nu"], lo, up))

    def test_adequate_fit_keeps_most_of_T_delta(self, forward, data):
        samples, y = data
        out = FitAwareTolerance(forward, DELTA)(samples, y)
        for key in DELTA:
            assert np.median(out["T_gof"][key] / out["T_delta"][key]) > 0.9

    def test_misfit_model_is_discounted(self, forward, data):
        samples, y = data
        wrong = dict(samples)
        wrong["tau2"] = samples["tau2"] * 1.5
        good = FitAwareTolerance(forward, DELTA)(samples, y)
        bad = FitAwareTolerance(forward, DELTA)(wrong, y)
        assert np.all(bad["fit"]["chi2_nu"] > 2)
        assert np.all(bad["T_gof"]["tau2"] < 0.01)
        assert np.all(bad["T_adj"]["tau2"] < good["T_adj"]["tau2"] + 1e-12)

    def test_draw_batching_and_subset(self, forward, data):
        samples, y = data
        a = FitAwareTolerance(forward, DELTA, draw_batch=1)(samples, y)
        b = FitAwareTolerance(forward, DELTA, draw_batch=64)(samples, y)
        np.testing.assert_allclose(a["fit"]["A_M"], b["fit"]["A_M"])
        sub = FitAwareTolerance(forward, DELTA, max_draws=50, seed=0)
        assert sub._draw_indices(S).size == 50
        out = sub(samples, y)
        assert np.all(
            np.isclose(out["fit"]["A_M"] * 50, np.round(out["fit"]["A_M"] * 50))
        )

    def test_fit_window_and_deviance(self, forward, data):
        samples, y = data
        mask = np.arange(K) < 100
        out = FitAwareTolerance(forward, DELTA, statistic="deviance")(
            samples, y, mask=mask
        )
        assert np.all(out["fit"]["nu"] == 100 - forward.n_params)

    def test_bool_mask(self, forward, data):
        samples, y = data
        keep = np.ones((H, W), bool)
        keep[0, 0] = False
        out = FitAwareTolerance(forward, DELTA)(samples, y, bool_mask=keep)
        assert np.isnan(out["T_gof"]["tau1"][0, 0])
        assert np.isnan(out["lam_median"][0, 0]).all()
        assert np.isfinite(out["T_gof"]["tau1"][keep]).all()
        with pytest.raises(ValueError, match="bool_mask"):
            FitAwareTolerance(forward, DELTA)(samples, y, bool_mask=keep.T)

    def test_derived_parameter(self, forward, data):
        samples, y = data
        samples = dict(samples)
        samples["tau_m"] = mean_lifetime_draws(
            samples["alpha1"], samples["tau1"], samples["tau2"]
        )
        out = FitAwareTolerance(forward, {"tau_m": 0.05})(samples, y)
        assert out["T_gof"]["tau_m"].shape == (H, W)
        assert "chi2_nu_early" not in out["fit"]

    def test_relative_tolerance(self, forward, data):
        samples, y = data
        out = FitAwareTolerance(forward, {"tau2": 0.02}, relative=True)(samples, y)
        np.testing.assert_allclose(
            out["T_delta"]["tau2"],
            tolerance_probability(samples["tau2"], 0.02, relative=True),
        )

    def test_validation(self, forward, data):
        samples, y = data
        with pytest.raises(ValueError, match="statistic"):
            FitAwareTolerance(forward, DELTA, statistic="cash")
        with pytest.raises(ValueError, match="delta"):
            FitAwareTolerance(forward, {})
        with pytest.raises(ValueError, match="n_params"):
            FitAwareTolerance(lambda p, y: y, DELTA)
        with pytest.raises(KeyError, match="delta keys"):
            FitAwareTolerance(forward, {"tau3": 0.1})(samples, y)
        with pytest.raises(ValueError, match="does not match"):
            FitAwareTolerance(forward, DELTA)(samples, y[:2])
