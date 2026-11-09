"""
Tests for pyfli.phasor.harmonic_offset: offset invariant, mono estimator, bracket theorem,
noise model and chi-square test, modulation imbalance, unwrapping and correction.
"""

import numpy as np
import pytest

from pyfli.cm_funcs.forward_model import model_numpy
from pyfli.phasor.harmonic_offset import (
    HarmonicOffset,
    bracket_violations,
    calibrate_phasors,
    combine_offsets,
    correct_phasors,
    fit_offset_surface,
    h_shift_from_t0,
    harmonic_phasors,
    imbalance_offset,
    modulation_factor,
    mono_offset,
    offset_bracket,
    offset_covariance,
    phasor_noise,
    select_mono,
    t0_from_h_shift,
    unwrap_phases,
)
from pyfli.phasor.phasorS import PhasorAnalyzer
from pyfli.reconstruction import ParamToDecay

FREQ_HZ = 80e6
PERIOD = 12.5
N_BINS = 256
OMEGA = 2 * np.pi * FREQ_HZ * 1e-9
K = 3


def mono(tau):
    return lambda k: 1.0 / (1.0 - 1j * k * OMEGA * tau)


def mixture(taus, fractions):
    return lambda k: sum(
        a / (1.0 - 1j * k * OMEGA * t) for t, a in zip(taus, fractions)
    )


def trace(fhat, t0=0.0, background=0.0, mu=2.0, sigma=0.15):
    k = np.arange(N_BINS // 2 + 1)
    irf = np.exp(1j * k * OMEGA * mu - (k * OMEGA * sigma) ** 2 / 2)
    coeff = irf * fhat(k) * np.exp(1j * k * OMEGA * t0)
    x = np.clip(np.fft.irfft(np.conj(coeff), n=N_BINS), 0.0, None)
    x /= x.sum()
    return (1.0 - background) * x + background / N_BINS


@pytest.fixture
def analyzer():
    return PhasorAnalyzer(
        FREQ_HZ, np.arange(N_BINS) * PERIOD / N_BINS, n_harmonics=K, device="cpu"
    )


def calibrated(analyzer, decay, irf=None, n=K):
    irf = trace(lambda k: 1.0) if irf is None else irf
    return calibrate_phasors(
        harmonic_phasors(analyzer, decay, n), harmonic_phasors(analyzer, irf, n)
    )


class TestMeasurementModel:
    @pytest.mark.parametrize("t0", [-0.4, 0.0, 0.27, 0.9])
    def test_calibrated_phasor_is_shifted_fhat(self, analyzer, t0):
        fhat = mixture([0.4, 3.0], [0.3, 0.7])
        z = calibrated(analyzer, trace(fhat, t0))
        k = np.arange(1, K + 1)
        np.testing.assert_allclose(z, np.exp(1j * k * OMEGA * t0) * fhat(k), atol=1e-10)

    def test_background_imbalance_scales_by_c(self, analyzer):
        beta, beta_i = 0.2, 0.05
        z = calibrated(
            analyzer,
            trace(mono(2.0), 0.3, beta),
            trace(lambda k: 1.0, background=beta_i),
        )
        k = np.arange(1, K + 1)
        c = (1 - beta) / (1 - beta_i)
        np.testing.assert_allclose(
            z, c * np.exp(1j * k * OMEGA * 0.3) * mono(2.0)(k), atol=1e-10
        )

    def test_kernel_cancels(self, analyzer):
        irf = trace(lambda k: 1.0, mu=3.0, sigma=0.4)
        z = calibrated(analyzer, trace(mono(1.5), 0.2, mu=3.0, sigma=0.4), irf)
        np.testing.assert_allclose(
            np.abs(z), np.abs(mono(1.5)(np.arange(1, K + 1))), atol=1e-10
        )

    def test_reference_dye_calibration_matches_irf(self, analyzer):
        tau_ref, t0 = 4.0, 0.35
        decay = trace(mono(1.2), t0)[None, None]
        G, S = analyzer.create_phasor_cpu(decay)
        Gr, Sr = analyzer.calibrate_reference(G, S, trace(mono(tau_ref)), tau_ref)
        t0_k, tau_k = mono_offset(Gr + 1j * Sr, FREQ_HZ)
        np.testing.assert_allclose(t0_k, t0, atol=1e-9)
        np.testing.assert_allclose(tau_k, 1.2, atol=1e-9)


class TestMonoEstimator:
    @pytest.mark.parametrize("tau", [0.3, 1.0, 2.5, 8.0])
    @pytest.mark.parametrize("t0", [-0.5, 0.0, 0.6])
    def test_exact_and_harmonic_independent(self, analyzer, tau, t0):
        t0_k, tau_k = mono_offset(calibrated(analyzer, trace(mono(tau), t0)), FREQ_HZ)
        np.testing.assert_allclose(t0_k, t0, atol=1e-8)
        np.testing.assert_allclose(tau_k, tau, rtol=1e-8)

    def test_large_offset_unwraps_long_lifetime(self):
        k = np.arange(1, 7)[:, None]
        tau, t0 = 6.0, np.array([-2.5, 1.2, 1.4])
        z = np.exp(1j * k * OMEGA * t0) * mono(tau)(k)
        t0_k, _ = mono_offset(z, FREQ_HZ)
        np.testing.assert_allclose(t0_k, np.broadcast_to(t0, t0_k.shape), atol=1e-10)

    def test_naive_k_phi1_unwrap_fails_where_ours_does_not(self):
        k = np.arange(1, 5)
        phase = k * OMEGA * 0.5 + np.angle(mono(6.0)(k))
        principal = np.angle(np.exp(1j * phase))
        naive = principal + 2 * np.pi * np.round(
            (k * principal[0] - principal) / (2 * np.pi)
        )
        assert not np.allclose(naive, phase)
        z = np.exp(1j * k * OMEGA * 0.5) * mono(6.0)(k)
        np.testing.assert_allclose(unwrap_phases(z, FREQ_HZ), phase, atol=1e-12)

    def test_t0_guess_extends_range(self):
        k = np.arange(1, 4)
        t0 = 0.45 * PERIOD
        z = np.exp(1j * k * OMEGA * t0) * mono(5.0)(k)
        t0_k, _ = mono_offset(z, FREQ_HZ, t0_guess_ns=0.4 * PERIOD)
        np.testing.assert_allclose(t0_k, t0, atol=1e-10)


class TestBracketTheorem:
    def test_random_completely_monotone_decays(self):
        rng = np.random.default_rng(0)
        k = np.arange(1, 6)[:, None]
        n = 2000
        taus = rng.uniform(0.0, 15.0, (4, n))
        fr = rng.dirichlet(np.ones(4), n).T
        t0 = rng.uniform(-0.8, 0.8, n)
        fhat = sum(a / (1 - 1j * k * OMEGA * t) for t, a in zip(taus, fr))
        z = np.exp(1j * k * OMEGA * t0) * fhat
        t0_k, _ = mono_offset(z, FREQ_HZ)
        assert np.all(t0_k <= t0 + 1e-12)
        assert np.all(t0_k < t0 - 1e-6)
        lower, upper = offset_bracket(z, FREQ_HZ)
        assert np.all((lower <= t0 + 1e-12) & (t0 <= upper + 1e-12))

    def test_lifetime_distribution_and_scatter(self):
        k = np.arange(1, 4)
        taus = np.linspace(0.5, 4.0, 50)
        fhat = mixture(np.r_[0.0, taus], np.r_[0.2, np.full(50, 0.8 / 50)])(k)
        t0_k, _ = mono_offset(np.exp(1j * k * OMEGA * 0.3) * fhat, FREQ_HZ)
        assert np.all(t0_k < 0.3)

    def test_rise_component_violates_bound(self):
        k = np.arange(1, 4)
        t1, t2 = 0.5, 3.0
        fhat = (t2 * mono(t2)(k) - t1 * mono(t1)(k)) / (t2 - t1)
        t0_k, _ = mono_offset(np.exp(1j * k * OMEGA * 0.2) * fhat, FREQ_HZ)
        assert np.all(t0_k > 0.2)

    def test_bracket_violations(self):
        lower, upper = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
        out = bracket_violations(np.array([-0.1, 0.5, 1.2]), lower, upper)
        np.testing.assert_array_equal(out, [True, False, True])
        assert not bracket_violations(1.05, 0.0, 1.0, tol_ns=0.1)


class TestNoise:
    @pytest.fixture(scope="class")
    def mc(self):
        rng = np.random.default_rng(1)
        an = PhasorAnalyzer(
            FREQ_HZ, np.arange(N_BINS) * PERIOD / N_BINS, n_harmonics=K, device="cpu"
        )
        t0 = 0.3
        decay = rng.poisson(1e5 * trace(mono(2.5), t0), (4000, N_BINS)).astype(float)
        irf = trace(lambda k: 1.0)
        z_raw = harmonic_phasors(an, decay, 2 * K)
        z_irf = harmonic_phasors(an, irf, 2 * K)
        z = calibrate_phasors(z_raw[:K], z_irf)
        t0_k, tau_k = mono_offset(z, FREQ_HZ)
        noise = phasor_noise(z_raw, z_irf, decay.sum(-1), K)
        return t0, t0_k, offset_covariance(noise, FREQ_HZ), noise, tau_k

    def test_covariance_matches_monte_carlo(self, mc):
        _, t0_k, cov, _, _ = mc
        emp = np.cov(t0_k)
        pred = cov.mean(0)
        np.testing.assert_allclose(np.diag(emp), np.diag(pred), rtol=0.1)
        corr = lambda c: c / np.sqrt(np.outer(np.diag(c), np.diag(c)))  # noqa: E731
        np.testing.assert_allclose(corr(emp), corr(pred), atol=0.06)

    def test_joint_lifetime_covariance_matches_monte_carlo(self, mc):
        _, t0_k, _, noise, tau_k = mc
        joint = offset_covariance(noise, FREQ_HZ, include_lifetime=True)
        np.testing.assert_allclose(joint[..., :K, :K], mc[2])
        emp = np.cov(np.concatenate([t0_k, tau_k]))
        pred = joint.mean(0)
        np.testing.assert_allclose(np.diag(emp), np.diag(pred), rtol=0.1)
        corr = lambda c: c / np.sqrt(np.outer(np.diag(c), np.diag(c)))  # noqa: E731
        np.testing.assert_allclose(corr(emp), corr(pred), atol=0.06)

    def test_diagonal_matches_paper_eq9(self, mc):
        _, _, cov, noise, _ = mc
        m = noise["m"]
        kw = np.arange(1, K + 1) * OMEGA
        vx, vy = (
            np.diagonal(noise["XX"], 0, -2, -1),
            np.diagonal(noise["YY"], 0, -2, -1),
        )
        vxy = np.diagonal(noise["XY"], 0, -2, -1)
        eq9 = (vy / m**2 + vx / (1 - m**2) + 2 * vxy / (m * np.sqrt(1 - m**2))) / kw**2
        np.testing.assert_allclose(np.diagonal(cov, 0, -2, -1), eq9, rtol=1e-12)
        np.testing.assert_allclose(noise["XY"], np.swapaxes(noise["YX"], -1, -2))

    @pytest.mark.parametrize("full", [True, False])
    def test_chi2_acceptance_rate(self, mc, full):
        t0, t0_k, cov, _, _ = mc
        out = combine_offsets(t0_k, cov, full)
        assert out["dof"] == K - 1 and out["n_params"] == 1 and out["tau"] is None
        np.testing.assert_allclose(out["reduced_chi2"], out["chi2"] / (K - 1))
        t_hat = out["t0"]
        rate = select_mono(out["chi2"], out["dof"], 0.05).mean()
        assert abs(np.mean(t_hat) - t0) < 3 * np.std(t_hat) / np.sqrt(len(t_hat))
        if full:
            assert 0.93 < rate < 0.97
            np.testing.assert_allclose(
                np.var(t_hat), np.mean(out["t0_se"] ** 2), rtol=0.1
            )
            assert abs(np.mean(out["reduced_chi2"]) - 1) < 0.05

    def test_joint_chi2_has_2k_minus_2_dof(self, mc):
        t0, t0_k, _, noise, tau_k = mc
        joint = offset_covariance(noise, FREQ_HZ, include_lifetime=True)
        out = combine_offsets(t0_k, joint, tau_k=tau_k)
        assert out["dof"] == 2 * K - 2 and out["n_params"] == 2
        assert abs(np.mean(out["reduced_chi2"]) - 1) < 0.05
        assert 0.93 < select_mono(out["chi2"], out["dof"]).mean() < 0.97
        assert abs(np.mean(out["tau"]) - 2.5) < 0.1 * np.std(out["tau"])
        np.testing.assert_allclose(
            np.var(out["tau"]), np.mean(out["tau_se"] ** 2), rtol=0.1
        )
        with pytest.raises(ValueError):
            combine_offsets(t0_k, mc[2], tau_k=tau_k)

    def test_biexponential_rejected_at_high_counts(self, analyzer):
        rng = np.random.default_rng(2)
        decay = rng.poisson(
            1e6 * trace(mixture([0.3, 4.0], [0.5, 0.5]), 0.3), (200, N_BINS)
        ).astype(float)
        irf = trace(lambda k: 1.0)
        z_raw = harmonic_phasors(analyzer, decay, 2 * K)
        z_irf = harmonic_phasors(analyzer, irf, 2 * K)
        t0_k, _ = mono_offset(calibrate_phasors(z_raw[:K], z_irf), FREQ_HZ)
        cov = offset_covariance(phasor_noise(z_raw, z_irf, decay.sum(-1), K), FREQ_HZ)
        out = combine_offsets(t0_k, cov)
        assert select_mono(out["chi2"], out["dof"]).mean() < 0.05

    def test_needs_2k_harmonics(self, analyzer):
        z = harmonic_phasors(analyzer, trace(mono(1.0)), K)
        with pytest.raises(ValueError):
            phasor_noise(z, z, 1e4, K)

    def test_nonfinite_pixels_are_nan(self):
        t0_k = np.array([[0.1, np.nan], [0.1, 0.2]])
        cov = np.broadcast_to(np.eye(2) * 1e-4, (2, 2, 2)).copy()
        out = combine_offsets(t0_k, cov)
        assert np.isfinite(out["t0"][0]) and np.isnan(out["t0"][1])
        assert np.isnan(out["chi2"][1]) and np.isnan(out["t0_se"][1])


class TestImbalance:
    @pytest.mark.parametrize("c", [0.8, 1.0, 1.1])
    @pytest.mark.parametrize("n_harm", [2, 4])
    def test_closed_form_recovers_c_tau_t0(self, c, n_harm):
        k = np.arange(1, n_harm + 1)
        tau, t0 = 2.2, 0.35
        z = c * np.exp(1j * k * OMEGA * t0) * mono(tau)(k)
        out = imbalance_offset(z, FREQ_HZ)
        np.testing.assert_allclose(out["c"], c, rtol=1e-10)
        np.testing.assert_allclose(out["tau"], tau, rtol=1e-10)
        np.testing.assert_allclose(out["t0_k"], t0, atol=1e-10)
        np.testing.assert_allclose(out["t0"], t0, atol=1e-10)
        assert out["chi2"] < 1e-15

    def test_c_below_one_biases_mono_estimator_low(self):
        k = np.arange(1, 5)
        tau, t0, c = 2.0, 0.3, 0.85
        z = c * np.exp(1j * k * OMEGA * t0) * mono(tau)(k)
        t0_k, tau_k = mono_offset(z, FREQ_HZ)
        assert np.all(t0_k < t0) and np.all(np.diff(tau_k) < 0)
        expected = np.sqrt(tau**2 / c**2 + (c**-2 - 1) / (k * OMEGA) ** 2)
        np.testing.assert_allclose(tau_k, expected, rtol=1e-10)
        lower, upper = offset_bracket(z, FREQ_HZ)
        assert lower < t0 < upper

    def test_weighted_with_noise(self, analyzer):
        rng = np.random.default_rng(3)
        an = PhasorAnalyzer(
            FREQ_HZ, np.arange(N_BINS) * PERIOD / N_BINS, n_harmonics=3, device="cpu"
        )
        decay = rng.poisson(1e6 * trace(mono(2.0), 0.3, 0.15), (500, N_BINS))
        decay = decay.astype(float)
        z_raw = harmonic_phasors(an, decay, 6)
        z_irf = harmonic_phasors(an, trace(lambda k: 1.0), 6)
        noise = phasor_noise(z_raw, z_irf, decay.sum(-1), 3)
        out = imbalance_offset(calibrate_phasors(z_raw[:3], z_irf), FREQ_HZ, noise)
        assert abs(np.median(out["c"]) - 0.85) < 0.01
        assert abs(np.median(out["tau"]) - 2.0) < 0.05
        assert abs(np.median(out["t0"]) - 0.3) < 0.02


class TestCorrectionAndSurface:
    def test_correction_exact_for_any_decay(self, analyzer):
        fhat = mixture([0.2, 1.0, 5.0], [0.2, 0.3, 0.5])
        z = calibrated(analyzer, trace(fhat, 0.42))
        Gc, Sc = correct_phasors(z.real, z.imag, 0.42, FREQ_HZ)
        np.testing.assert_allclose(Gc + 1j * Sc, fhat(np.arange(1, K + 1)), atol=1e-10)

    def test_correction_broadcasts_map_over_analyzer_stack(self, analyzer):
        t0_map = np.array([[0.1, -0.2], [0.3, 0.0]])
        decay = np.stack([[trace(mono(2.0), t) for t in row] for row in t0_map])
        G, S = analyzer.create_phasor_cpu(decay)
        Gc, Sc = analyzer.calibrate(G, S, trace(lambda k: 1.0))
        Gc, Sc = correct_phasors(Gc, Sc, t0_map, FREQ_HZ)
        t0_k, tau_k = mono_offset(Gc + 1j * Sc, FREQ_HZ)
        np.testing.assert_allclose(t0_k, 0.0, atol=1e-9)
        np.testing.assert_allclose(tau_k, 2.0, rtol=1e-8)

    def test_surface_median_and_plane(self):
        yy, xx = np.meshgrid(
            np.linspace(-1, 1, 20), np.linspace(-1, 1, 30), indexing="ij"
        )
        plane = 0.3 + 0.05 * xx - 0.02 * yy
        accepted = np.ones(plane.shape, bool)
        contaminated = plane - 0.5 * (
            np.arange(plane.size).reshape(plane.shape) % 7 == 0
        )
        accepted_c = contaminated == plane
        fit, theta = fit_offset_surface(contaminated, accepted_c, order=1)
        np.testing.assert_allclose(fit, plane, atol=1e-12)
        med, _ = fit_offset_surface(np.full(plane.shape, 0.3), accepted)
        np.testing.assert_allclose(med, 0.3)
        with pytest.raises(ValueError):
            fit_offset_surface(plane, np.zeros(plane.shape, bool))


class TestHarmonicOffsetPipeline:
    def test_requires_two_harmonics(self):
        an = PhasorAnalyzer(FREQ_HZ, np.arange(N_BINS) * PERIOD / N_BINS, device="cpu")
        with pytest.raises(ValueError):
            HarmonicOffset(an)

    def test_end_to_end(self, analyzer):
        rng = np.random.default_rng(4)
        t0 = 0.37
        H, W = 12, 12
        biexp = np.zeros((H, W), bool)
        biexp[:, :4] = True
        taus = rng.uniform(1.0, 4.0, (H, W))
        clean = np.stack(
            [
                [
                    trace(
                        mixture([0.2, 5.0], [0.6, 0.4])
                        if biexp[i, j]
                        else mono(taus[i, j]),
                        t0,
                    )
                    for j in range(W)
                ]
                for i in range(H)
            ]
        )
        decay = rng.poisson(2e5 * clean).astype(float)
        irf = trace(lambda k: 1.0)
        ho = HarmonicOffset(analyzer)
        res = ho.estimate(decay, irf)
        assert res["accepted"][~biexp].mean() > 0.85
        assert res["accepted"][biexp].mean() < 0.1
        assert abs(res["theta"][0] - t0) < 0.01
        assert np.all(res["t0_k"][:, biexp] < t0)
        assert res["violations"].mean() < 0.05
        Gc, Sc = ho.correct(res["z"].real, res["z"].imag, res["t0_map"])
        np.testing.assert_allclose(
            np.angle((Gc + 1j * Sc)[0][~biexp]),
            np.arctan(OMEGA * taus[~biexp]),
            atol=0.03,
        )


class TestSolverConsistency:
    T_AXIS = np.arange(N_BINS) * PERIOD / N_BINS
    IRF = np.exp(-0.5 * ((np.arange(N_BINS) * PERIOD / N_BINS - 1.5) / 0.12) ** 2)

    def test_h_shift_conversion_round_trip(self):
        h = np.array([-0.3, 0.0, 0.25])
        np.testing.assert_allclose(
            h_shift_from_t0(h, self.T_AXIS) - h, PERIOD / N_BINS / 2
        )
        np.testing.assert_allclose(
            t0_from_h_shift(h_shift_from_t0(h, self.T_AXIS), self.T_AXIS), h
        )

    @pytest.mark.parametrize("h_shift", [-0.3, 0.0, 0.137, 0.5])
    @pytest.mark.parametrize("tau", [0.5, 1.0])
    def test_forward_model_h_shift_recovered(self, analyzer, h_shift, tau):
        d = model_numpy(
            self.T_AXIS, self.IRF, [1e5, tau, 0.0, h_shift], "mono-exponential"
        )
        z = calibrated(analyzer, d, self.IRF)
        t0_k, tau_k = mono_offset(z, FREQ_HZ)
        np.testing.assert_allclose(
            h_shift_from_t0(t0_k, self.T_AXIS), h_shift, atol=1e-3
        )
        np.testing.assert_allclose(tau_k, tau, rtol=1e-3)

    def test_forward_model_bi_exponential_is_lower_bound(self, analyzer):
        params = [1e5, 0.4, 2.5, 0.5, 0.0, 0.2]
        d = model_numpy(self.T_AXIS, self.IRF, params, "bi-exponential")
        z = calibrated(analyzer, d, self.IRF)
        lower, upper = offset_bracket(z, FREQ_HZ)
        h_low, h_up = h_shift_from_t0(np.array([lower, upper]), self.T_AXIS)
        assert h_low < 0.2 - 0.05 and 0.2 < h_up

    def test_param_to_decay_with_h_shift_map(self, analyzer):
        rng = np.random.default_rng(5)
        H, W, h_true = 6, 6, 0.21
        tau_map = rng.uniform(0.6, 1.4, (H, W))
        irf_cube = np.broadcast_to(self.IRF, (H, W, N_BINS)).copy()
        params = {
            "tau_map": tau_map,
            "photon_count_map": np.full((H, W), 2e5),
            "h_shift_map": np.full((H, W), h_true),
        }
        recon = ParamToDecay("mono-exponential", FREQ_HZ / 1e6, irf_cube, N_BINS)
        np.testing.assert_allclose(recon.t, self.T_AXIS)
        clean = recon.reconstruct(params, verbose=False)["fit_map"].astype(float)
        res = HarmonicOffset(analyzer).estimate(
            rng.poisson(clean).astype(float), self.IRF
        )
        assert res["accepted"].mean() > 0.8
        np.testing.assert_allclose(res["h_shift_map"], h_true, atol=0.01)
        tau_est = res["tau_k"][0]
        np.testing.assert_allclose(tau_est, tau_map, rtol=0.05)


class TestBackgroundImbalance:
    def test_modulation_factor_matches_eq3(self):
        beta, beta_i = 0.2, 0.05
        c = modulation_factor(1.0, beta / N_BINS, N_BINS, 1.0, beta_i / N_BINS)
        np.testing.assert_allclose(c, (1 - beta) / (1 - beta_i))
        np.testing.assert_allclose(modulation_factor(1e4, 0.0, N_BINS), 1.0)

    def test_estimate_with_v_shift_removes_bias(self, analyzer):
        rng = np.random.default_rng(6)
        H, W, t0 = 8, 8, 0.3
        S, v_shift, irf_v = 2e5, 150.0, 30.0
        taus = rng.uniform(1.0, 3.0, (H, W))
        clean = np.stack(
            [
                [S * trace(mono(taus[i, j]), t0) + v_shift for j in range(W)]
                for i in range(H)
            ]
        )
        irf = 1e5 * trace(lambda k: 1.0) + irf_v
        decay = rng.poisson(clean).astype(float)
        z = calibrate_phasors(
            harmonic_phasors(analyzer, decay), harmonic_phasors(analyzer, irf)
        )
        assert np.median(mono_offset(z, FREQ_HZ)[0][0]) < t0 - 0.01
        res = HarmonicOffset(analyzer).estimate(
            decay, irf, v_shift=np.full((H, W), v_shift), irf_v_shift=irf_v
        )
        beta = N_BINS * v_shift / decay.sum(-1)
        beta_i = N_BINS * irf_v / irf.sum()
        np.testing.assert_allclose(res["c"], (1 - beta) / (1 - beta_i))
        assert abs(res["theta"][0] - t0) < 0.01
        assert res["accepted"].mean() > 0.85
        np.testing.assert_allclose(res["tau_k"][0], taus, rtol=0.05)


class TestBiExponentialOffset:
    def _bi_z(self, t0, taus, a1, n_harm=4):
        k = np.arange(1, n_harm + 1)[:, None]
        t1, t2 = np.asarray(taus[0]), np.asarray(taus[1])
        fhat = a1 / (1 - 1j * k * OMEGA * t1) + (1 - a1) / (1 - 1j * k * OMEGA * t2)
        return np.exp(1j * k * OMEGA * np.asarray(t0)) * fhat

    def test_noise_free_recovers_offset_and_components(self):
        from pyfli.phasor.harmonic_offset import bi_exponential_offset

        t0 = np.array([0.1, 0.3, -0.2, 0.45])
        a1 = np.array([0.2, 0.5, 0.8, 0.35])
        taus = (np.array([0.4, 0.6, 0.3, 0.8]), np.array([2.5, 3.0, 1.8, 4.0]))
        z = self._bi_z(t0, taus, a1)
        eye = np.broadcast_to(np.eye(4) * 1e-8, (4, 4, 4))
        noise = {"XX": eye, "YY": eye, "XY": 0 * eye, "YX": 0 * eye}
        out = bi_exponential_offset(z, noise, FREQ_HZ, n_jobs=1)
        np.testing.assert_allclose(out["t0"], t0, atol=1e-5)
        np.testing.assert_allclose(out["tau1"], taus[0], rtol=1e-4)
        np.testing.assert_allclose(out["tau2"], taus[1], rtol=1e-4)
        np.testing.assert_allclose(out["alpha1"], a1, atol=1e-4)
        assert out["dof"] == 4 and out["n_params"] == 4

    def test_needs_three_harmonics(self):
        from pyfli.phasor.harmonic_offset import bi_exponential_offset

        z = self._bi_z([0.1], ([0.5], [2.0]), np.array([0.5]), n_harm=2)
        eye = np.broadcast_to(np.eye(4), (1, 4, 4))
        with pytest.raises(ValueError):
            bi_exponential_offset(
                z, {"XX": eye, "YY": eye, "XY": eye, "YX": eye}, FREQ_HZ
            )

    def test_estimate_on_noisy_bi_image_without_mono_pixels(self):
        rng = np.random.default_rng(5)
        an = PhasorAnalyzer(
            FREQ_HZ, np.arange(N_BINS) * PERIOD / N_BINS, n_harmonics=4, device="cpu"
        )
        H, W, t0 = 6, 8, 0.3
        a1 = rng.uniform(0.25, 0.75, (H, W))
        clean = np.stack(
            [
                [
                    2e5 * trace(mixture([0.6, 3.0], [a1[i, j], 1 - a1[i, j]]), t0)
                    for j in range(W)
                ]
                for i in range(H)
            ]
        )
        decay = rng.poisson(clean).astype(float)
        irf = trace(lambda k: 1.0)
        res = HarmonicOffset(an).estimate(decay, irf, model="bi-exponential", n_jobs=1)
        assert res["dof"] == 4 and res["cov"] is None
        assert res["accepted"].mean() > 0.8
        assert abs(res["theta"][0] - t0) < 0.01
        assert abs(np.median(res["tau2_hat"]) - 3.0) < 0.15
        assert abs(np.median(res["alpha1_hat"] - a1)) < 0.03

    def test_mono_model_rejects_bi_image(self):
        rng = np.random.default_rng(6)
        an = PhasorAnalyzer(
            FREQ_HZ, np.arange(N_BINS) * PERIOD / N_BINS, n_harmonics=4, device="cpu"
        )
        decay = rng.poisson(
            np.stack(
                [
                    [
                        2e5 * trace(mixture([0.6, 3.0], [0.5, 0.5]), 0.3)
                        for _ in range(4)
                    ]
                    for _ in range(4)
                ]
            )
        ).astype(float)
        irf = trace(lambda k: 1.0)
        with pytest.raises(ValueError):
            HarmonicOffset(an).estimate(decay, irf)
        with pytest.raises(ValueError):
            HarmonicOffset(an).estimate(decay, irf, model="tri-exponential")
