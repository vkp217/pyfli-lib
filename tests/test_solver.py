"""
Tests for BaseFLIFitter and FLICPUProcessor (CPU solver pipeline).

All tests use purely synthetic numpy arrays; no file I/O or GPU required.
Synthetic data is generated using the same model formula as the fitter so
parameter recovery assertions are meaningful.
"""

import numpy as np
import pytest

from pyfli.solver.base_fitter import BaseFLIFitter
from pyfli.solver.cpu_processor import FLICPUProcessor
from pyfli.solver.shared_metrics import expected_poisson_deviance

# ---------------------------------------------------------------------------
# Constants — 80 MHz system, 256 bins → T_acq = 12.5 ns, dt = 12.5/256 ns/bin
# ---------------------------------------------------------------------------
_N = 256
_DT = 12.5 / _N  # ≈ 0.04883 ns/bin (12.5 ns laser period / 256 bins)
_T = np.arange(_N) * _DT  # time axis
_FREQ = [80.0, 1000.0 / (_N * _DT)]  # [laser MHz, acq MHz] — both 80.0 MHz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_irf(center=0.3, sigma=0.1):
    irf = np.exp(-0.5 * ((_T - center) / sigma) ** 2)
    return irf / irf.sum()


def _make_mono(tau=2.0, S=5000, offset=10, irf=None, seed=42):
    """Synthetic mono-exponential photon-count decay."""
    if irf is None:
        irf = _gaussian_irf()
    decay_clean = (S / tau) * np.exp(-_T / tau)
    conv = np.convolve(decay_clean, irf, mode="full")[:_N] + offset
    rng = np.random.default_rng(seed)
    return rng.poisson(np.clip(conv, 0, None)).astype(float), irf


def _make_biexp(tau1=0.5, tau2=2.5, alpha1=0.6, S=5000, offset=10, irf=None, seed=42):
    """Synthetic bi-exponential photon-count decay.

    Uses the fitter's own model parameterisation:
        f(t) = S * (alpha1/tau1 * exp(-t/tau1) + (1-alpha1)/tau2 * exp(-t/tau2))
    so the true parameters match what the fitter should recover.
    """
    if irf is None:
        irf = _gaussian_irf()
    decay_clean = S * (
        alpha1 / tau1 * np.exp(-_T / tau1) + (1 - alpha1) / tau2 * np.exp(-_T / tau2)
    )
    conv = np.convolve(decay_clean, irf, mode="full")[:_N] + offset
    rng = np.random.default_rng(seed)
    return rng.poisson(np.clip(conv, 0, None)).astype(float), irf


# ---------------------------------------------------------------------------
# BaseFLIFitter — pixel-level unit tests
# ---------------------------------------------------------------------------


class TestBaseFLIFitterStructure:
    """Output structure and invariants, independent of parameter accuracy."""

    def test_mono_output_length(self):
        decay, irf = _make_mono()
        result = BaseFLIFitter(_FREQ, decay, irf).fit_with_estimator(
            model_type="mono-exponential"
        )
        popt, perr, r2, chi2, red_chi2, ssr, converged, rmse = result
        # [S, tau, offset, h_shift] — 4 parameters
        assert len(popt) == 4
        assert len(perr) == 4

    def test_bi_output_length(self):
        decay, irf = _make_biexp()
        result = BaseFLIFitter(_FREQ, decay, irf).fit_with_estimator(
            model_type="bi-exponential"
        )
        popt, perr, r2, chi2, red_chi2, ssr, converged, rmse = result
        # [S, a1, tau1, tau2, offset, h_shift] — 6 parameters
        assert len(popt) == 6
        assert len(perr) == 6

    def test_tau_ordering_enforced(self):
        """Post-processing must ensure tau1 <= tau2."""
        decay, irf = _make_biexp(tau1=0.5, tau2=2.5)
        popt, *_ = BaseFLIFitter(_FREQ, decay, irf).fit_with_estimator(
            model_type="bi-exponential"
        )
        assert popt[2] <= popt[3], "tau1 must be <= tau2 after post-processing"

    def test_alpha1_in_unit_interval(self):
        decay, irf = _make_biexp()
        popt, *_ = BaseFLIFitter(_FREQ, decay, irf).fit_with_estimator(
            model_type="bi-exponential"
        )
        assert 0.0 <= popt[1] <= 1.0

    def test_chi2_raw_greater_than_reduced(self):
        """red_chi2 is the deviance chi2 over its expectation minus the 4 params."""
        decay, irf = _make_mono()
        fitter = BaseFLIFitter(_FREQ, decay, irf)
        popt, _, _, chi2, red_chi2, _, _, _ = fitter.fit_with_estimator(
            model_type="mono-exponential"
        )
        model = fitter.model_fit(fitter.t, popt, model_type="mono-exponential")
        dof = np.sum(expected_poisson_deviance(model)) - 4
        assert abs(red_chi2 - chi2 / dof) < 1e-4, "red_chi2 must equal chi2 / dof"
        assert chi2 > red_chi2

    def test_r2_bounded(self):
        decay, irf = _make_mono(S=8000)
        _, _, r2, *_ = BaseFLIFitter(_FREQ, decay, irf).fit_with_estimator(
            model_type="mono-exponential"
        )
        assert -1.0 <= r2 <= 1.0

    def test_ssr_non_negative(self):
        decay, irf = _make_biexp()
        _, _, _, _, _, ssr, _, _ = BaseFLIFitter(_FREQ, decay, irf).fit_with_estimator(
            model_type="bi-exponential"
        )
        assert ssr >= 0.0

    def test_trust_region_estimator(self):
        decay, irf = _make_mono()
        result = BaseFLIFitter(_FREQ, decay, irf).fit_with_estimator(
            estimator_type="trust_region", model_type="mono-exponential"
        )
        assert len(result) == 8


class TestBaseFLIFitterRecovery:
    """Parameter recovery tests — tolerance is intentionally generous (±25 %)
    because we are fitting a single noisy pixel, not averaging over many."""

    def test_mono_tau_recovered(self):
        tau_true = 2.0
        decay, irf = _make_mono(tau=tau_true, S=8000, seed=0)
        popt, _, r2, _, red_chi2, _, converged, _ = BaseFLIFitter(
            _FREQ, decay, irf
        ).fit_with_estimator(model_type="mono-exponential")
        assert converged == 1
        assert abs(popt[1] - tau_true) / tau_true < 0.25
        assert r2 > 0.90

    def test_biexp_taus_recovered(self):
        tau1_true, tau2_true = 0.5, 2.5
        decay, irf = _make_biexp(tau1=tau1_true, tau2=tau2_true, S=10000, seed=1)
        popt, _, r2, _, red_chi2, _, converged, _ = BaseFLIFitter(
            _FREQ, decay, irf
        ).fit_with_estimator(model_type="bi-exponential")
        assert converged == 1
        assert abs(popt[2] - tau1_true) / tau1_true < 0.25
        assert abs(popt[3] - tau2_true) / tau2_true < 0.25
        assert r2 > 0.90

    def test_reduced_chi2_near_one_for_good_fit(self):
        """A well-fitted Poisson dataset should give reduced chi2 ≈ 1."""
        decay, irf = _make_mono(S=10000, seed=2)
        _, _, _, _, red_chi2, _, _, _ = BaseFLIFitter(
            _FREQ, decay, irf
        ).fit_with_estimator(model_type="mono-exponential")
        assert 0.5 < red_chi2 < 3.0, (
            f"reduced chi2 = {red_chi2:.3f} out of expected range"
        )

    def test_h_shift_near_zero_for_aligned_irf(self):
        """With a pre-aligned IRF the recovered h_shift should be close to 0."""
        decay, irf = _make_mono(S=8000, seed=3)
        popt, *_ = BaseFLIFitter(_FREQ, decay, irf).fit_with_estimator(
            model_type="mono-exponential"
        )
        # h_shift is now in ns; for aligned data the recovered delay should be small
        T_acq = 1000.0 / _FREQ[1]
        assert abs(popt[-1]) < T_acq / 4, (
            f"h_shift = {popt[-1]:.4f} ns, expected near 0"
        )


# ---------------------------------------------------------------------------
# FLICPUProcessor — image-level integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_biexp_image():
    """4×4 image with uniform bi-exponential decays and their IRF cube."""
    H, W = 4, 4
    irf = _gaussian_irf()
    tau1, tau2, alpha1, S, offset = 0.5, 2.5, 0.6, 3000, 5
    decay_clean = S * (
        alpha1 / tau1 * np.exp(-_T / tau1) + (1 - alpha1) / tau2 * np.exp(-_T / tau2)
    )
    conv = np.convolve(decay_clean, irf, mode="full")[:_N] + offset
    rng = np.random.default_rng(7)
    image = rng.poisson(np.clip(conv, 0, None).astype(float)).astype(float)
    image_cube = np.tile(image, (H, W, 1))
    irf_cube = np.tile(irf, (H, W, 1))
    return image_cube, irf_cube


@pytest.fixture(scope="module")
def biexp_result(small_biexp_image):
    image_cube, irf_cube = small_biexp_image
    proc = FLICPUProcessor(_FREQ, BaseFLIFitter)
    return proc.process_image(
        image_cube, irf_cube, model_type="bi-exponential", n_jobs=1
    )


@pytest.fixture(scope="module")
def mono_result(small_biexp_image):
    image_cube, irf_cube = small_biexp_image
    proc = FLICPUProcessor(_FREQ, BaseFLIFitter)
    return proc.process_image(
        image_cube, irf_cube, model_type="mono-exponential", n_jobs=1
    )


class TestCPUProcessorOutputKeys:
    """Verify map keys are correct — including h_shift_map."""

    def test_result_not_none(self, biexp_result):
        assert biexp_result is not None

    def test_chi2_map_key_present(self, biexp_result):
        assert "chi2_map" in biexp_result["results"]["maps"]

    def test_old_chi2_key_absent(self, biexp_result):
        assert "chi2_or_deviance_map" not in biexp_result["results"]["maps"]

    def test_reduced_chi2_map_present(self, biexp_result):
        assert "reduced_chi2_map" in biexp_result["results"]["maps"]

    def test_r2_map_present(self, biexp_result):
        assert "R2_map" in biexp_result["results"]["maps"]

    def test_rmse_map_present(self, biexp_result):
        assert "rmse_map" in biexp_result["results"]["maps"]

    def test_rmse_map_matches_residual_map(self, biexp_result):
        """rmse_map must equal sqrt(mean(residual_map**2, axis=-1)) per pixel."""
        maps = biexp_result["results"]["maps"]
        tr = biexp_result["results"]["TR_maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy pixels")
        expected = np.sqrt(np.mean(tr["residual_map"] ** 2, axis=-1))
        np.testing.assert_allclose(
            maps["rmse_map"][health], expected[health], rtol=1e-4
        )

    def test_pixel_health_map_present(self, biexp_result):
        assert "pixel_health_map" in biexp_result["results"]["maps"]

    def test_convergence_map_present(self, biexp_result):
        assert "convergence_map" in biexp_result["results"]["maps"]

    def test_biexp_param_keys(self, biexp_result):
        maps = biexp_result["results"]["maps"]
        for key in (
            "photon_count_map",
            "alpha1_map",
            "tau1_map",
            "tau2_map",
            "tau_mean_map",
            "v_shift_map",
            "h_shift_map",
        ):
            assert key in maps, f"Missing key: {key}"

    def test_mono_param_keys(self, mono_result):
        maps = mono_result["results"]["maps"]
        for key in (
            "photon_count_map",
            "tau_map",
            "v_shift_map",
            "chi2_map",
            "h_shift_map",
        ):
            assert key in maps, f"Missing key: {key}"

    def test_tr_maps_keys(self, biexp_result):
        tr = biexp_result["results"]["TR_maps"]
        assert "fit_map" in tr
        assert "residual_map" in tr


class TestCPUProcessorChi2Consistency:
    """chi2_map must be raw (not pre-divided by dof)."""

    def test_chi2_raw_greater_than_reduced(self, biexp_result):
        maps = biexp_result["results"]["maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy pixels")
        chi2_raw = maps["chi2_map"][health]
        chi2_red = maps["reduced_chi2_map"][health]
        assert np.all(chi2_raw >= chi2_red), (
            "chi2_map (raw) must be >= reduced_chi2_map for every healthy pixel"
        )

    def test_chi2_and_reduced_are_consistent(self, biexp_result):
        """reduced_chi2_map = chi2_map / (sum E[D](fit) - 6) for bi-exponential."""
        maps = biexp_result["results"]["maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy pixels")
        fit = biexp_result["results"]["TR_maps"]["fit_map"][health]
        dof = np.sum(expected_poisson_deviance(fit), axis=-1) - 6
        ratio = maps["chi2_map"][health] / maps["reduced_chi2_map"][health]
        np.testing.assert_allclose(ratio, dof, rtol=1e-3)


class TestCPUProcessorArrayShapes:
    def test_map_shapes_biexp(self, biexp_result, small_biexp_image):
        H, W, _ = small_biexp_image[0].shape
        maps = biexp_result["results"]["maps"]
        for key, arr in maps.items():
            assert arr.shape == (H, W), f"{key} has wrong shape {arr.shape}"

    def test_tr_map_shapes(self, biexp_result, small_biexp_image):
        H, W, T = small_biexp_image[0].shape
        tr = biexp_result["results"]["TR_maps"]
        assert tr["fit_map"].shape == (H, W, T)
        assert tr["residual_map"].shape == (H, W, T)

    def test_error_maps_shape(self, biexp_result, small_biexp_image):
        H, W, _ = small_biexp_image[0].shape
        e = biexp_result["results"]["error_maps"]
        assert e.shape == (H, W, 6)  # 6 params for bi-exponential


class TestCPUProcessorParamRecovery:
    """Median recovered parameters over healthy pixels should be close to truth."""

    def test_tau1_tau2_ordering(self, biexp_result):
        maps = biexp_result["results"]["maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy pixels")
        assert np.all(maps["tau1_map"][health] <= maps["tau2_map"][health])

    def test_tau2_within_25pct(self, biexp_result):
        maps = biexp_result["results"]["maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy pixels")
        tau2_med = float(np.median(maps["tau2_map"][health]))
        assert abs(tau2_med - 2.5) / 2.5 < 0.25, f"tau2 median = {tau2_med:.3f} ns"

    def test_r2_positive_for_healthy_pixels(self, biexp_result):
        maps = biexp_result["results"]["maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy pixels")
        assert np.all(biexp_result["results"]["maps"]["R2_map"][health] > 0)

    def test_fit_residuals_consistent(self, biexp_result, small_biexp_image):
        """fit_map + residual_map should reconstruct the original data."""
        image_cube, _ = small_biexp_image
        tr = biexp_result["results"]["TR_maps"]
        reconstructed = tr["fit_map"] + tr["residual_map"]
        np.testing.assert_allclose(reconstructed, image_cube, atol=1e-3)

    def test_h_shift_map_shape_and_range(self, biexp_result, small_biexp_image):
        """h_shift_map must have the correct shape and plausible values."""
        H, W, _ = small_biexp_image[0].shape
        maps = biexp_result["results"]["maps"]
        assert "h_shift_map" in maps
        assert maps["h_shift_map"].shape == (H, W)
        # For aligned test data the shift should be small (< 10 bins)
        health = maps["pixel_health_map"] > 0
        if health.any():
            assert np.all(np.abs(maps["h_shift_map"][health]) < 10.0)
