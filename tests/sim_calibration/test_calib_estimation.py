"""Tests for pixel selection, IRF and prior estimation, the simulator bridge, optimiser and coverage."""

from dataclasses import replace

import numpy as np
import pytest
from calib_helpers import (
    DCR,
    INACTIVE,
    N_BINS,
    PERIOD_NS,
    SAMPLE,
    SHAPE,
    TRUE_CONFIG,
    simulate_cube,
    true_irf,
)

from pyfli.simulator.calibration import coverage as cov
from pyfli.simulator.calibration import irf_estimation as ie
from pyfli.simulator.calibration import optimizer as opt
from pyfli.simulator.calibration import prior_estimation as pe
from pyfli.simulator.calibration import sim_bridge as sb
from pyfli.simulator.calibration.config import PixelSelectionCriteria
from pyfli.simulator.calibration.expected_decay import DecayForwardModel
from pyfli.simulator.calibration.pixel_selection import _otsu, select_signal_pixels
from pyfli.simulator.calibration.summary_stats import (
    StatsComparator,
    background_bins,
    compute_stats,
)


def fm_default(jitter=False):
    return DecayForwardModel(
        N_BINS, PERIOD_NS, "discrete", jitter=jitter, inactive_bins=INACTIVE
    )


def sample_decays(cube):
    return cube[SAMPLE].reshape(-1, N_BINS).astype(float)


@pytest.fixture(scope="module")
def context(calibration, synthetic_cube):
    exp = sample_decays(synthetic_cube)
    bg = background_bins(exp.sum(0))
    st = compute_stats(exp, bg)
    ctx = opt.ObjectiveContext(
        fm=fm_default(),
        detector_type="discrete",
        laser_freq_mhz=80.0,
        jitter=False,
        comparator=StatsComparator(st),
        bg_bins=bg,
        exp_counts=exp.sum(-1),
        exp_peaks=exp.max(-1),
        data_max=float(exp.max()),
        bit_depth=None,
        prior_inflation=1.0,
        n_sim=150,
        seed=0,
        inactive_bins=INACTIVE,
    )
    return ctx, calibration.state, st


class TestPixelSelection:
    def test_selects_sample_square(self, synthetic_cube):
        mask, report = select_signal_pixels(synthetic_cube)
        truth = np.zeros(SHAPE, bool)
        truth[SAMPLE] = True
        iou = (mask & truth).sum() / (mask | truth).sum()
        assert (
            iou > 0.85
            and report["mode"] == "auto"
            and report["n_selected"] == mask.sum()
        )
        assert set(report["rejected_by_test"]) >= {
            "signal_photons",
            "snr",
            "not_flat",
            "peak_position",
            "not_hot",
        }
        assert report["background_bins"][1] < 10

    def test_hot_pixel_rejected(self, synthetic_cube):
        cube = synthetic_cube.copy()
        cube[10, 10] *= 50
        mask, _ = select_signal_pixels(cube)
        assert not mask[10, 10]

    def test_adc_ceiling_rejects_saturated(self, synthetic_cube):
        cube = synthetic_cube.copy()
        cube[8, 8, 20] = 255
        mask, _ = select_signal_pixels(cube, adc_ceiling=255)
        assert not mask[8, 8]

    def test_user_mask_and_refinement(self, synthetic_cube):
        user = np.zeros(SHAPE, bool)
        user[:, :10] = True
        mask, report = select_signal_pixels(synthetic_cube, user_mask=user)
        assert report["mode"] == "user" and mask[0, 0] == bool(
            synthetic_cube[0, 0].sum() > 0
        )
        assert "agreement_user_vs_auto_iou" in report
        refined, report = select_signal_pixels(
            synthetic_cube, user, PixelSelectionCriteria(refine_user_mask=True)
        )
        assert (
            report["mode"] == "user+auto"
            and not refined[0, 0]
            and refined[SAMPLE].sum() > 0
        )

    def test_otsu_bimodal_and_flat(self):
        rng = np.random.default_rng(0)
        thr, eta = _otsu(np.r_[rng.normal(0, 1, 3000), rng.normal(6, 1, 3000)])
        assert 1.5 < thr < 4.5 and eta > 0.8
        assert _otsu(np.ones(10)) == (1.0, 0.0)
        assert _otsu(np.array([])) == (0.0, 0.0)


class TestIRFEstimation:
    def test_peak_position_subbin(self):
        y = np.exp(-0.5 * ((np.arange(N_BINS) - 20.3) / 1.5) ** 2)
        assert ie._peak_position(y[None, :])[0] == pytest.approx(20.3, abs=0.15)

    def test_experimental_template_1d_and_cube(self):
        irf = true_irf()
        template, a_half = ie.experimental_template(irf * 7, None)
        assert template.max() == pytest.approx(1.0) and a_half == 0.5
        cube = (
            np.stack([np.roll(irf, s) for s in (-2, 0, 2) for _ in range(30)]).reshape(
                9, 10, N_BINS
            )
            * 1000
        )
        template, a_half = ie.experimental_template(cube, np.ones((9, 10), bool))
        assert (
            abs(int(np.argmax(template)) - int(np.argmax(irf))) <= 1
            and 1.5 <= a_half <= 2.5
        )

    def test_fourier_shift_delays(self):
        y = np.exp(-0.5 * ((np.arange(N_BINS) - 20) / 2.0) ** 2)
        assert ie._peak_position(ie._fourier_shift(y, 3.0)[None, :])[
            0
        ] == pytest.approx(23.0, abs=0.2)
        assert np.all(ie._fourier_shift(y, 2.5) >= 0)

    def test_estimate_irf_simulated_recovers_truth(self, synthetic_cube):
        mask = np.zeros(SHAPE, bool)
        mask[SAMPLE] = True
        est = ie.estimate_irf(fm_default(), synthetic_cube, mask, irf=None, mode="auto")
        assert est.source == "simulated"
        assert est.params["mu_bin"] == pytest.approx(10, abs=1.0)
        assert est.params["sigma_ns"] == pytest.approx(0.08, rel=0.4)
        pf = est.pooled_fit
        assert pf["tau_short_ns"] < pf["tau_long_ns"] and 1.2 < pf["tau_long_ns"] < 3.0
        assert pf["dcr_per_bin_per_pixel"] >= 0

    def test_estimate_irf_experimental_and_auto(self, synthetic_cube):
        mask = np.zeros(SHAPE, bool)
        mask[SAMPLE] = True
        est = ie.estimate_irf(
            fm_default(), synthetic_cube, mask, irf=true_irf(), mode="experimental"
        )
        assert est.source == "experimental" and abs(est.a_center) < 2
        auto = ie.estimate_irf(
            fm_default(), synthetic_cube, mask, irf=true_irf(), mode="auto"
        )
        assert set(auto.pooled_fit["irf_choice"]) == {"experimental", "simulated"}
        with pytest.raises(ValueError):
            ie.estimate_irf(
                fm_default(), synthetic_cube, mask, irf=None, mode="experimental"
            )

    def test_theta_roundtrip_and_cube(self):
        fm = fm_default()
        est = ie.IRFEstimate(
            "simulated",
            fm.simulated_irf(10, 0.08, 0.05),
            0.0,
            0.5,
            1e-3,
            {"mu_bin": 10.0, "sigma_ns": 0.08, "tail_ns": 0.05},
        )
        theta = ie.irf_theta_from_estimate(est)
        assert ie.irf_theta_names("simulated") == [
            "mu_bin",
            "log_sigma_ns",
            "log_tail_ns",
        ]
        back = ie.apply_irf_theta(est, fm, theta)
        np.testing.assert_allclose(back.template, est.template, atol=1e-12)
        assert (
            est.irf_cube().shape == (1, 1, N_BINS)
            and est.a_range == (-0.5, 0.5)
            and est.b_range == (0, 1e-3)
        )
        lib = replace(
            est,
            library=np.stack([est.template] * 3),
            library_b_max=np.array([1e-3, 2e-3, 3e-3]),
        )
        assert lib.irf_cube().shape == (1, 3, N_BINS) and lib.b_ranges()[2] == (
            0.0,
            3e-3,
        )
        exp_est = ie.IRFEstimate("experimental", est.template, 2.0, 0.5, 1e-3)
        assert ie.irf_theta_names("experimental") == ["a_center"]
        moved = ie.apply_irf_theta(exp_est, fm, np.array([-1.0]))
        assert (
            moved.a_center == -1.0
            and moved.effective_irf().argmax() == est.template.argmax() - 1
        )


class TestPriorEstimation:
    def test_beta_prior_properties(self):
        p = pe.BetaPrior(2.0, 6.0, offset=1.0, scale=4.0)
        assert (
            p.mean_norm == pytest.approx(0.25)
            and p.mean == pytest.approx(2.0)
            and p.conc == 8
        )
        assert p.sd == pytest.approx(4 * np.sqrt(0.25 * 0.75 / 9))
        x = p.sample(20000, np.random.default_rng(0))
        assert (
            x.min() >= 1.0
            and x.max() <= 5.0
            and x.mean() == pytest.approx(2.0, abs=0.03)
        )
        q = pe.BetaPrior.from_mean_conc(0.999999, 1e6)
        assert q.a >= pe.MIN_SHAPE and q.b >= pe.MIN_SHAPE and q.conc <= pe.MAX_CONC + 1
        assert p.as_tuple() == (2.0, 6.0)

    def test_fit_beta_recovers_parameters(self):
        x = 0.5 + 3.0 * np.random.default_rng(1).beta(4, 9, 20000)
        p = pe.fit_beta(x, 0.5, 3.0)
        assert p.a == pytest.approx(4, rel=0.15) and p.b == pytest.approx(9, rel=0.15)
        wide = pe.fit_beta(x, 0.5, 3.0, inflation=4.0)
        assert wide.sd > 1.5 * p.sd and wide.mean == pytest.approx(p.mean, rel=0.05)
        fallback = pe.fit_beta(np.array([1.0, 2.0]), 0.0, 5.0)
        assert (fallback.a, fallback.b) == (1.5, 1.5)

    def test_quick_fits_recover_mono_lifetime(self):
        cfg = {
            **TRUE_CONFIG,
            "mono_fraction": 1.0,
            "tau2": (3000.0, 3000.0),
            "tau2_beta_range": (0.02, 1.99),
            "n_cycles": (200_000, 200_001),
            "photo_count": (50.0, 50.0),
        }
        decays = sample_decays(simulate_cube(seed=5, config=cfg))[:30]
        fm = fm_default()
        irf = true_irf()
        out = pe.quick_pixel_fits(
            fm, decays, irf / irf.sum(), np.full(len(decays), DCR), False, {"tau": 1.0}
        )
        assert np.nanmedian(out["tau"]) == pytest.approx(2.0, rel=0.08)

    def test_quick_fits_bi_return_all_keys(self, synthetic_cube):
        decays = sample_decays(synthetic_cube)[:10]
        irf = true_irf()
        init = {"tau2": 2.0, "efficiency": 0.6, "alpha1": 0.5}
        out = pe.quick_pixel_fits(
            fm_default(), decays, irf / irf.sum(), np.full(10, DCR), True, init
        )
        assert set(out) == {"tau1", "tau2", "alpha1", "efficiency", "tau"}
        ok = np.isfinite(out["tau2"])
        assert ok.sum() >= 8 and np.all(out["tau1"][ok] <= out["tau2"][ok])

    def test_photon_prior_discrete_covers_counts(self):
        counts = np.random.default_rng(2).uniform(300, 900, 2000)
        prior, cyc = pe.photon_prior_discrete(counts, N_BINS - len(INACTIVE), 0.0)
        assert (
            cyc[0] >= 1000
            and cyc[0] * 0.01 <= np.percentile(counts, 0.5)
            and cyc[1] * 0.01 >= counts.max() * 0.99
        )
        assert prior.offset == cyc[0] and prior.scale == cyc[1] - cyc[0]

    def test_photon_prior_continuous_and_bit_depth(self):
        peaks = np.random.default_rng(3).uniform(200, 400, 1000) * 0.25
        prior, bit = pe.photon_prior_continuous(peaks, data_max=110.0, bit=None)
        assert 8 <= bit <= 16 and 0 < prior.mean_norm < 1
        assert pe.infer_bit_depth(np.array([100.0]), 4000.0) == 12
        assert pe.infer_bit_depth(np.array([1.0]), 1.0) == 8

    def test_tau_bounds_and_lifetime_priors(self):
        rng = np.random.default_rng(4)
        tau2 = rng.uniform(1.5, 2.5, 500)
        offset, scale = pe.tau_bounds(tau2, PERIOD_NS)
        assert offset <= 1.5 and offset + scale >= 2.5
        values = {
            "tau2": tau2,
            "efficiency": rng.uniform(0.4, 0.8, 500),
            "alpha1": rng.uniform(0.3, 0.7, 500),
        }
        t2, eff, a1, moments = pe.lifetime_priors(values, True, PERIOD_NS, 1.0)
        assert t2.mean == pytest.approx(2.0, rel=0.05) and eff.mean == pytest.approx(
            0.6, rel=0.05
        )
        assert set(moments) == {"tau2", "efficiency", "alpha1"}
        t2m, effm, a1m, _ = pe.lifetime_priors({"tau": tau2}, False, PERIOD_NS, 1.0)
        assert effm is None and a1m is None and t2m.mean == pytest.approx(2.0, rel=0.05)

    def test_prior_set_summary(self, calibration):
        s = calibration.state.priors.summary()
        assert {"tau2", "efficiency", "alpha1", "photon", "photon_range", "bit"} <= set(
            s
        )
        assert s["tau2"]["mean"] == pytest.approx(calibration.state.priors.tau2.mean)


class TestSimBridge:
    def test_build_sim_config(self, calibration):
        cfg = sb.build_sim_config(
            calibration.state.priors, "discrete", 0.03, 80.0, jitter=False
        )
        assert (
            cfg["tau2_dist"] == "beta"
            and cfg["dcr"] == 0.03
            and cfg["seed"] is None
            and "n_cycles" in cfg
        )
        p = calibration.state.priors.tau2
        assert cfg["tau2_beta_range"] == (round(p.scale, 4), round(p.offset, 4))
        assert "n_cycles" not in sb.build_sim_config(
            calibration.state.priors, "continuous", 0.03, 80.0
        )

    def test_draw_pixels_reproducible(self):
        gen = sb.make_generator(
            true_irf()[None, None, :], dict(TRUE_CONFIG), (0, 0), (0, 0)
        )
        a = sb.stack_outputs(sb.draw_pixels(gen, 20, seed=7))
        b = sb.stack_outputs(sb.draw_pixels(gen, 20, seed=7))
        c = sb.stack_outputs(sb.draw_pixels(gen, 20, seed=8))
        np.testing.assert_array_equal(a["decay"], b["decay"])
        assert not np.array_equal(a["decay"], c["decay"]) and gen.config["seed"] is None
        assert set(sb.TRUTH_KEYS) <= set(a) and a["decay"].shape == (20, N_BINS)
        t2 = a["tau2"]
        assert np.all((t2 >= 1.5) & (t2 <= 2.5))

    def test_time_window_and_multi_irf(self):
        irf = true_irf()
        win = sb.make_generator(
            irf[None, None, :],
            dict(TRUE_CONFIG),
            (0, 0),
            (0, 0),
            inactive_bins=INACTIVE,
        )
        assert (
            isinstance(win, sb.TimeWindowedGenerator)
            and win.config is win.generator.config
        )
        out = sb.simulate_batch(
            irf[None, None, :],
            dict(TRUE_CONFIG),
            (0, 0),
            (0, 0),
            10,
            seed=1,
            inactive_bins=INACTIVE,
        )
        assert np.all(out["decay"][:, INACTIVE] == 0) and "irf" not in out
        lib = np.stack([irf, np.roll(irf, 20)])[None]
        multi = sb.make_generator(
            lib, dict(TRUE_CONFIG), (0, 0), (0, 0), irf_weights=[0.0, 1.0]
        )
        assert isinstance(multi, sb.MultiIRFSimGenerator)
        res = sb.stack_outputs(sb.draw_pixels(multi, 10, seed=2))
        assert np.all(res["irf_index"] == 1)

    def test_to_jsonable(self):
        out = sb.to_jsonable(
            {"a": np.float32(1.5), "b": (np.int64(2), [np.arange(2)]), "c": "x"}
        )
        assert out == {"a": 1.5, "b": [2, [[0, 1]]], "c": "x"}

    def test_saved_generator_roundtrip(self, calibration):
        gen = sb.load_calibrated_generator(calibration.files["config"])
        assert isinstance(gen, sb.TimeWindowedGenerator)
        assert np.all(gen.simulate_once()["decay"][INACTIVE] == 0)
        img = sb.simulate_image(calibration.files["config"], shape=(3, 4), seed=1)
        assert (
            img["decay"].shape == (3, 4, N_BINS)
            and img["tau2"].shape == (3, 4)
            and img["irf"].shape == (3, 4, N_BINS)
        )
        again = sb.simulate_image(calibration.files["config"], shape=(3, 4), seed=1)
        np.testing.assert_array_equal(img["decay"], again["decay"])


class TestOptimizer:
    def test_parameter_vector_roundtrip(self, context):
        ctx, state, _ = context
        names = [
            "mu_bin",
            "log_sigma_ns",
            "log_tail_ns",
            "log_a_half",
            "log_b_max",
            "log_dcr",
            "tau2_logit_m",
            "tau2_log_k",
            "efficiency_logit_m",
            "efficiency_log_k",
        ]
        pv = opt.ParameterVector(state, ctx, names)
        x0 = pv.initial()
        back = pv.state(x0)
        assert back.priors.tau2.mean == pytest.approx(state.priors.tau2.mean, rel=1e-6)
        assert back.dcr == pytest.approx(
            state.dcr
        ) and back.irf.a_half == pytest.approx(max(state.irf.a_half, 0.05))
        np.testing.assert_allclose(
            opt.ParameterVector(back, ctx, names).initial(), x0, atol=1e-6
        )

    def test_refresh_photon_prior(self, context):
        ctx, state, _ = context
        fresh = opt.refresh_photon_prior(state, ctx)
        assert fresh.priors.photon_range[0] >= 1000 and fresh.priors.bit == 16
        assert (
            opt.refresh_photon_prior(state, replace(ctx, fixed_photon_prior=True))
            is state
        )

    def test_evaluate_is_deterministic(self, context):
        ctx, state, _ = context
        d1, parts, sim, sim_stats = opt.evaluate(state, ctx, n=80, seed=3)
        d2, *_ = opt.evaluate(state, ctx, n=80, seed=3)
        assert d1 == d2 and set(parts) == {
            "log_counts",
            "edge",
            "bg_frac",
            "phasor1",
            "phasor2",
            "mean_decay",
        }
        assert sim["decay"].shape == (80, N_BINS) and np.all(
            sim["decay"][:, INACTIVE] == 0
        )

    def test_reference_penalty(self, context):
        ctx, state, _ = context
        assert opt._reference_penalty(state, ctx) == 0.0
        sd = max(0.3, 0.05 * state.priors.tau2.scale)
        ref = replace(
            ctx,
            reference_weight=1.0,
            reference_moments={"tau2": (state.priors.tau2.mean + sd, sd)},
        )
        assert opt._reference_penalty(state, ref) == pytest.approx(1.0, rel=1e-6)

    def test_reference_penalty_does_not_depend_on_support(self, context):
        ctx, state, _ = context
        ref = replace(
            ctx,
            reference_weight=1.0,
            reference_moments={"tau2": (state.priors.tau2.mean, 0.3)},
        )
        assert opt._reference_penalty(state, ref) == pytest.approx(0.0, abs=1e-12)
        moved = opt.recenter_tau2_support(state)
        assert moved.priors.tau2.offset != pytest.approx(state.priors.tau2.offset)
        assert opt._reference_penalty(moved, ref) == pytest.approx(0.0, abs=1e-4)

    def test_stage_parameter_names(self, context):
        ctx, state, _ = context
        nuis, prior = opt.stage_parameter_names(state, ctx, True, True, True, True)
        assert "log_dcr" in nuis and "log_photon_scale" not in nuis and len(prior) == 6
        nuis, prior = opt.stage_parameter_names(
            state, replace(ctx, detector_type="continuous"), False, True, False, False
        )
        assert nuis == ["log_dcr", "log_photon_scale"] and prior == []

    def test_optimize_stage_does_not_get_worse(self, context):
        ctx, state, _ = context
        start, *_ = opt.evaluate(state, ctx)
        best, hist = opt.optimize_stage(state, ctx, ["log_dcr"], 5, "test")
        assert len(hist) >= 1 and min(h["objective"] for h in hist) <= start + 1e-9
        assert opt.optimize_stage(state, ctx, [], 5, "none") == (state, [])

    def test_prior_bank(self, context):
        ctx, state, _ = context
        bank = opt.PriorBank(state, ctx, n=200, seed=1)
        w = bank.weights(state.priors)
        assert w.sum() == pytest.approx(1.0) and np.all(w >= 0)
        dist, parts, ess = bank.distance(state.priors)
        assert 1 <= ess <= 200 and np.isfinite(dist)

    def test_recenter_tau2_support(self, context):
        _, state, _ = context
        new = opt.recenter_tau2_support(state)
        assert new.priors.tau2.mean == pytest.approx(state.priors.tau2.mean, rel=0.02)
        assert new.priors.tau2.offset >= 0.05

    def test_prior_bounds(self):
        assert opt._prior_bounds(["tau2_logit_m", "tau2_log_k"]) == [
            (-6.0, 6.0),
            (0.0, float(np.log(pe.MAX_CONC))),
        ]

    def test_fit_priors_and_rounds(self, context):
        ctx, state, _ = context
        names = ["tau2_logit_m", "tau2_log_k"]
        new, hist = opt.fit_priors_on_bank(
            state, ctx, names, n_bank=200, max_evals=40, round_idx=0
        )
        assert hist and np.isfinite(new.priors.tau2.mean)
        assert opt.fit_priors_on_bank(state, ctx, [], 200, 40, 0) == (state, [])
        out, hist = opt.run_calibration_rounds(
            state, ctx, ["log_dcr"], names, 200, 1, 3, 30
        )
        assert {h["stage"] for h in hist} == {"priors_round1", "irf_noise_round1"}


class TestCoverage:
    def test_covered_pixels_self_and_shifted(self, context):
        _, _, st = context
        assert min(cov.coverage_fractions(st, st).values()) > 0.97
        far = {
            **st,
            "phasor1": st["phasor1"] + 5.0,
            "log_counts": st["log_counts"] + 10,
        }
        flags = cov.covered_pixels(far, st)
        assert flags["phasor1"].mean() == 0 and flags["log_counts"].mean() == 0

    def test_hdr_independent_of_sample_size(self):
        rng = np.random.default_rng(0)
        e = rng.normal(0, 1, (2000, 2))
        small = cov.hdr_covered(e, rng.normal(0, 1, (5000, 2))).mean()
        large = cov.hdr_covered(e, rng.normal(0, 1, (50000, 2))).mean()
        assert abs(small - large) < 0.01 and large > 0.99

    def test_tails_and_lifetime_side(self, context):
        _, _, st = context
        assert cov._tails(st, st, "edge") == pytest.approx((0.0, 0.0), abs=0.01)
        longer = {**st, "phasor1": st["phasor1"] * 0.5}
        short, long = cov._lifetime_side(longer, st)
        assert long > 0.5 and short == 0

    def test_beta_helpers(self):
        p = pe.BetaPrior(20.0, 40.0, 1.0, 2.0)
        flat = cov.flatten_beta(p, 0.5)
        assert flat.a == pytest.approx(10.5) and flat.offset == 1.0
        assert cov.flatten_beta(p, 0.0).as_tuple() == (1.0, 1.0)
        moved = cov.resupport_beta(p, 0.5, 4.0)
        assert moved.mean == pytest.approx(
            p.mean, rel=1e-6
        ) and moved.sd == pytest.approx(p.sd, rel=1e-3)
        assert cov.step_factor(0.5, 0.99) == 0.4 and cov.step_factor(0.989, 0.99) == 0.9

    def test_state_wideners(self, context):
        ctx, state, _ = context
        tiny = replace(state, priors=replace(state.priors, photon_range=(1000, 2000)))
        grown = cov.cover_photon_range(tiny, ctx.exp_counts, ctx.fm.n_active)
        assert grown.priors.photon_range[1] * 0.01 >= ctx.exp_counts.max()
        assert cov.cover_photon_range(grown, ctx.exp_counts, ctx.fm.n_active) is grown
        wider = cov.stretch_tau2_support(state, below=0.1, above=0.1)
        assert wider.priors.tau2.scale > state.priors.tau2.scale
        assert cov.stretch_tau2_support(state, 0.0, 0.0) is state
        flat = cov.flatten_lifetime_priors(state.priors, 0.5)
        assert (
            flat.tau2.conc < state.priors.tau2.conc
            and flat.photon is state.priors.photon
        )
        refit = cov.refit_photon_prior(tiny, ctx, ctx.exp_counts)
        assert refit.priors.photon_range[1] * 0.01 >= ctx.exp_counts.max()

    def test_widen_actions(self, context):
        ctx, state, st = context
        new, actions = cov.widen(
            state,
            ["edge", "bg_frac", "phasor1", "log_counts"],
            st,
            st,
            ctx,
            ctx.exp_counts,
            {"phasor1": 0.5, "phasor2": 0.9, "log_counts": 0.5},
            0.99,
            photon_refitted=True,
        )
        assert new.irf.a_half > state.irf.a_half and len(actions) == 4
        assert new.priors.tau2.conc < state.priors.tau2.conc

    def test_ensure_coverage_repairs_narrow_config(self, context):
        ctx, state, st = context

        def narrow(p):
            return pe.BetaPrior.from_mean_conc(p.mean_norm, 900, p.offset, p.scale)

        lo, hi = state.priors.photon_range
        bad = replace(
            state,
            irf=replace(state.irf, a_half=0.05),
            priors=replace(
                state.priors,
                tau2=narrow(state.priors.tau2),
                efficiency=narrow(state.priors.efficiency),
                photon=pe.BetaPrior(30, 30, lo, (hi - lo) / 4),
                photon_range=(lo, lo + (hi - lo) // 4),
            ),
        )
        before = cov.coverage_fractions(
            st, opt.evaluate(bad, replace(ctx, fixed_photon_prior=True), n=600)[3]
        )
        fixed, trace, after, *_ = cov.ensure_coverage(
            bad,
            replace(ctx, fixed_photon_prior=True),
            st,
            ctx.exp_counts,
            target=0.95,
            n_sim=600,
            max_iter=4,
        )
        assert min(after.values()) > min(before.values()) and len(trace) >= 2
        assert all("cover phasor1" in row and "widened" in row for row in trace)


class TestMathFixes:
    def test_noise_aware_distance(self):
        from pyfli.simulator.calibration.summary_stats import StatsComparator

        ratios = []
        for k in range(4):
            a = np.concatenate(
                [sample_decays(simulate_cube(seed=200 + 20 * k + j)) for j in range(12)]
            )
            b = np.concatenate(
                [sample_decays(simulate_cube(seed=500 + 20 * k + j)) for j in range(4)]
            )
            bg = background_bins(a.sum(0))
            comp = StatsComparator(compute_stats(a, bg))
            sb = compute_stats(b, bg)
            keep = comp.keep
            lr2 = (
                np.log(b.sum(0)[keep] / b.sum()) - np.log(a.sum(0)[keep] / a.sum())
            ) ** 2
            ratios.append(
                lr2.mean() / (sb["mean_decay_rel_var"][keep] + comp.var_pe).mean()
            )
        assert 0.6 < np.mean(ratios) < 1.2
        _, shifted = comp.distance(compute_stats(np.roll(b, 2, axis=-1), bg))
        assert shifted["mean_decay"] > 3.0

    def test_pooled_relative_variance(self):
        from pyfli.simulator.calibration.summary_stats import pooled_relative_variance

        rng = np.random.default_rng(0)
        d = rng.poisson(50, (4000, 16)).astype(float)
        np.testing.assert_allclose(
            pooled_relative_variance(d), 1 / (4000 * 50), rtol=0.1
        )
        w = rng.random(4000)
        w /= w.sum()
        m = w @ d
        direct = (w**2) @ (d - m) ** 2 / m**2
        np.testing.assert_allclose(pooled_relative_variance(d, w), direct, rtol=1e-12)
        bank = opt.PriorBank.__new__(opt.PriorBank)
        bank.sim = {"decay": d}
        iid = (w @ (d - m) ** 2) / len(w) / m**2
        np.testing.assert_allclose(bank._pooled_rel_var(w), iid, rtol=1e-9)
        spiky = np.zeros(4000)
        spiky[:5] = 0.2
        assert (
            bank._pooled_rel_var(spiky).mean()
            < pooled_relative_variance(d, spiky).mean()
        )
        assert np.all(pooled_relative_variance(np.zeros((3, 4))) == 0)

    def test_continuous_bit_depth_stays_fixed(self, context):
        ctx, state, _ = context
        peaks = np.random.default_rng(0).uniform(150, 260, 2000)
        cont = replace(
            ctx,
            detector_type="continuous",
            exp_peaks=peaks,
            data_max=300.0,
            bit_depth=11,
        )
        bits = {
            opt.refresh_photon_prior(replace(state, photon_scale=s), cont).priors.bit
            for s in (0.5, 1.0, 3.0)
        }
        assert bits == {11}
        free = replace(cont, bit_depth=None)
        bits_free = {
            opt.refresh_photon_prior(replace(state, photon_scale=s), free).priors.bit
            for s in (0.5, 3.0)
        }
        assert len(bits_free) == 2

    def test_active_photon_fraction_matches_simulator(self, context):
        ctx, state, _ = context
        f = opt.active_photon_fraction(state, ctx)
        sim = opt.evaluate(state, ctx, n=400, seed=5)[2]
        measured = (sim["decay"].sum(1) - ctx.fm.n_active * state.dcr).mean() / sim[
            "photon_count"
        ].mean()
        assert 0.9 < f <= 1.0 and f == pytest.approx(measured, abs=0.01)
        assert (
            opt.active_photon_fraction(state, replace(ctx, detector_type="continuous"))
            == 1.0
        )
        counts = np.random.default_rng(1).uniform(300, 900, 1000)
        p1, c1 = pe.photon_prior_discrete(counts, 58, 0.0)
        p2, c2 = pe.photon_prior_discrete(counts, 58, 0.0, active_fraction=0.8)
        assert c2[1] == pytest.approx(
            c1[1] / 0.8, rel=1e-3
        ) and p2.mean == pytest.approx(p1.mean / 0.8, rel=1e-3)

    def test_fit_beta_keeps_variance(self):
        x = np.random.default_rng(3).beta(4, 9, 200000)
        p = pe.fit_beta(x, 0.0, 1.0)
        assert p.sd == pytest.approx(x.std(), rel=0.01)

    def test_small_single_population_not_split(self):
        from pyfli.simulator.calibration import multicluster as mc

        shape, sample = (20, 20), (slice(3, 17), slice(3, 17))
        mask = np.zeros(shape, bool)
        mask[sample] = True
        for seed in (1, 3, 4):
            a = simulate_cube(
                seed=10 + seed,
                config={"mono_fraction": 1.0, "tau2_beta_range": (0.4, 3.0)},
                shape=shape,
                sample=sample,
            )
            b = simulate_cube(
                seed=20 + seed,
                config={"mono_fraction": 1.0, "tau2_beta_range": (0.4, 3.0)},
                shape=shape,
                sample=sample,
            )
            cube = a.copy()
            cube[:, 10:] = b[:, 10:]
            res = mc.detect_clusters(cube, mask, max_components=3, seed=0)
            assert res.n_clusters == 1
            assert np.array(res.report["dip_ratio_components"]).shape == (3, 3)

    def test_pooled_baseline_is_degenerate_with_dark_counts(self):
        fm = fm_default()
        tpl = true_irf() / true_irf().max()
        base = fm.shape(fm.with_baseline(tpl, 0.0), [0.5, 2.0], [0.4, 0.6])
        with_b = fm.shape(fm.with_baseline(tpl, 0.004), [0.5, 2.0], [0.4, 0.6])
        beta = np.linalg.lstsq((fm.flat() - base)[:, None], with_b - base, rcond=None)[
            0
        ][0]
        assert beta > 0.01
        np.testing.assert_allclose(
            with_b, (1 - beta) * base + beta * fm.flat(), atol=1e-12
        )

    def test_split_background_recovers_dark_counts_and_baseline(self):
        fm = fm_default()
        rng = np.random.default_rng(3)
        tpl = true_irf() / true_irf().max()
        b_true, dcr_true = 0.002, 0.1
        shape = fm.shape(fm.with_baseline(tpl, b_true), [0.5, 2.0], [0.4, 0.6])
        n_sig = rng.uniform(50, 1500, 80000)
        decays = rng.poisson(n_sig[:, None] * shape + dcr_true * fm.active)
        decays = decays.astype(float)
        est = ie.estimate_irf(
            fm,
            decays.reshape(200, 400, N_BINS),
            np.ones((200, 400), bool),
            irf=true_irf(),
            mode="experimental",
        )
        assert est.pooled_fit["irf_baseline_b"] == 0.0
        bg = background_bins(decays.sum(0))
        slope, intercept = np.polyfit(decays.sum(-1), decays[:, bg].mean(-1), 1)
        dcr = ie.split_background(est, fm, bg, slope, intercept)
        assert dcr == pytest.approx(dcr_true, rel=0.08)
        assert est.pooled_fit["irf_baseline_b"] == pytest.approx(b_true, rel=0.15)
        assert est.b_max == pytest.approx(2 * est.pooled_fit["irf_baseline_b"])
