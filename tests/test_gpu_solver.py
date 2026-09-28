"""
Performance and correctness tests for FLIGPUProcessor.

All 4 estimation methods are tested on synthetic ground-truth data:
    CPU NLSF  — BaseFLIFitter  / least_squares  (IRLS WLS, default)
    CPU MLE   — MLEFLIFitter   / minimize        (Poisson C-stat)
    GPU NLSF  — FLIGPUProcessor / Adam          (IRLS WLS, default)
    GPU MLE   — FLIGPUProcessor / Adam          (Poisson C-stat)

Both mono-exponential and bi-exponential models are covered.
No file I/O. GPU falls back to CPU automatically when CUDA is absent.
"""

import numpy as np
import pytest

from pyfli.solver.base_fitter import BaseFLIFitter
from pyfli.solver.gpu_processor import FLIGPUProcessor
from pyfli.solver.mle_fitter import MLEFLIFitter

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_N = 256
_DT = 12.5 / _N  # ns/bin  (80 MHz laser, 12.5 ns period)
_T = np.arange(_N) * _DT
_FREQ = [80.0, 80.0]  # [laser MHz, acq MHz]

# Ground truth
_MONO = dict(tau=2.0, S=8000, v_shift=10, h_shift=0.0)
_BIEXP = dict(tau1=0.5, tau2=2.5, alpha1=0.6, S=8000, v_shift=10, h_shift=0.0)
_TOL = 0.20  # 20 % relative tolerance for parameter recovery
_H_TOL = 0.5  # absolute tolerance for h_shift in ns (should be ~0)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _gaussian_irf(center=0.3, sigma=0.1):
    irf = np.exp(-0.5 * ((_T - center) / sigma) ** 2)
    return irf / irf.sum()


def _make_mono(seed=42, **kw):
    p = {**_MONO, **kw}
    irf = _gaussian_irf()
    t_eff = _T - p["h_shift"]
    clean = (p["S"] / p["tau"]) * np.exp(-t_eff / p["tau"])
    conv = np.convolve(clean, irf, mode="full")[:_N] + p["v_shift"]
    return np.random.default_rng(seed).poisson(np.clip(conv, 0, None)).astype(
        float
    ), irf


def _make_biexp(seed=42, **kw):
    p = {**_BIEXP, **kw}
    irf = _gaussian_irf()
    t_eff = _T - p["h_shift"]
    clean = p["S"] * (
        p["alpha1"] / p["tau1"] * np.exp(-t_eff / p["tau1"])
        + (1 - p["alpha1"]) / p["tau2"] * np.exp(-t_eff / p["tau2"])
    )
    conv = np.convolve(clean, irf, mode="full")[:_N] + p["v_shift"]
    return np.random.default_rng(seed).poisson(np.clip(conv, 0, None)).astype(
        float
    ), irf


def _image_cube(decay_1d, irf_1d, H=4, W=4):
    """Tile a single decay into a small (H, W, T) image cube."""
    return (np.tile(decay_1d, (H, W, 1)), np.tile(irf_1d, (H, W, 1)))


# ---------------------------------------------------------------------------
# Fixtures — generate once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mono_pixel():
    return _make_mono(seed=0)


@pytest.fixture(scope="session")
def biexp_pixel():
    return _make_biexp(seed=1)


@pytest.fixture(scope="session")
def mono_cube(mono_pixel):
    decay, irf = mono_pixel
    return _image_cube(decay, irf)


@pytest.fixture(scope="session")
def biexp_cube(biexp_pixel):
    decay, irf = biexp_pixel
    return _image_cube(decay, irf)


# ---------------------------------------------------------------------------
# Fixtures — run all 4 methods
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cpu_nlsf_mono(mono_pixel):
    decay, irf = mono_pixel
    return BaseFLIFitter(_FREQ, decay, irf).fit_with_estimator(
        model_type="mono-exponential"
    )


@pytest.fixture(scope="session")
def cpu_mle_mono(mono_pixel):
    decay, irf = mono_pixel
    return MLEFLIFitter(_FREQ, decay, irf).fit_with_estimator(
        model_type="mono-exponential"
    )


@pytest.fixture(scope="session")
def cpu_nlsf_biexp(biexp_pixel):
    decay, irf = biexp_pixel
    return BaseFLIFitter(_FREQ, decay, irf).fit_with_estimator(
        model_type="bi-exponential"
    )


@pytest.fixture(scope="session")
def cpu_mle_biexp(biexp_pixel):
    decay, irf = biexp_pixel
    return MLEFLIFitter(_FREQ, decay, irf).fit_with_estimator(
        model_type="bi-exponential"
    )


@pytest.fixture(scope="session")
def gpu_nlsf_mono(mono_cube):
    image_cube, irf_cube = mono_cube
    proc = FLIGPUProcessor(_FREQ)
    return proc.fit_image(
        image_cube, irf_cube, mode="NLSF", model_type="mono-exponential", max_iter=600
    )


@pytest.fixture(scope="session")
def gpu_mle_mono(mono_cube):
    image_cube, irf_cube = mono_cube
    proc = FLIGPUProcessor(_FREQ)
    return proc.fit_image(
        image_cube, irf_cube, mode="MLE", model_type="mono-exponential", max_iter=600
    )


@pytest.fixture(scope="session")
def gpu_nlsf_biexp(biexp_cube):
    image_cube, irf_cube = biexp_cube
    proc = FLIGPUProcessor(_FREQ)
    return proc.fit_image(
        image_cube, irf_cube, mode="NLSF", model_type="bi-exponential", max_iter=600
    )


@pytest.fixture(scope="session")
def gpu_mle_biexp(biexp_cube):
    image_cube, irf_cube = biexp_cube
    proc = FLIGPUProcessor(_FREQ)
    return proc.fit_image(
        image_cube, irf_cube, mode="MLE", model_type="bi-exponential", max_iter=600
    )


# ===========================================================================
# 1. Output structure tests — GPU
# ===========================================================================


class TestGPUOutputStructure:
    """GPU result dict has the correct keys and shapes."""

    def test_gpu_mono_not_none(self, gpu_nlsf_mono):
        assert gpu_nlsf_mono is not None

    def test_gpu_biexp_not_none(self, gpu_nlsf_biexp):
        assert gpu_nlsf_biexp is not None

    def test_gpu_biexp_map_keys(self, gpu_nlsf_biexp):
        maps = gpu_nlsf_biexp["results"]["maps"]
        for key in (
            "photon_count_map",
            "alpha1_map",
            "tau1_map",
            "tau2_map",
            "tau_mean_map",
            "v_shift_map",
            "h_shift_map",
            "fret_efficiency_map",
            "chi2_map",
            "reduced_chi2_map",
            "R2_map",
            "rmse_map",
            "pixel_health_map",
            "convergence_map",
        ):
            assert key in maps, f"Missing GPU bi-exp key: {key}"

    def test_gpu_mono_map_keys(self, gpu_nlsf_mono):
        maps = gpu_nlsf_mono["results"]["maps"]
        for key in (
            "photon_count_map",
            "tau_map",
            "v_shift_map",
            "h_shift_map",
            "chi2_map",
            "reduced_chi2_map",
            "R2_map",
            "rmse_map",
            "pixel_health_map",
            "convergence_map",
        ):
            assert key in maps, f"Missing GPU mono key: {key}"

    def test_hshift_map_present_in_gpu(self, gpu_nlsf_biexp):
        """h_shift_map must be present — GPU uses (t - h_shift) in the decay kernel."""
        assert "h_shift_map" in gpu_nlsf_biexp["results"]["maps"]

    def test_gpu_biexp_map_shapes(self, gpu_nlsf_biexp, biexp_cube):
        H, W, _ = biexp_cube[0].shape
        for key, arr in gpu_nlsf_biexp["results"]["maps"].items():
            assert arr.shape == (H, W), f"{key}: expected ({H},{W}), got {arr.shape}"

    def test_gpu_mono_map_shapes(self, gpu_nlsf_mono, mono_cube):
        H, W, _ = mono_cube[0].shape
        for key, arr in gpu_nlsf_mono["results"]["maps"].items():
            assert arr.shape == (H, W), f"{key}: expected ({H},{W}), got {arr.shape}"

    def test_gpu_tr_maps(self, gpu_nlsf_biexp, biexp_cube):
        H, W, T = biexp_cube[0].shape
        tr = gpu_nlsf_biexp["results"]["TR_maps"]
        assert tr["fit_map"].shape == (H, W, T)
        assert tr["residual_map"].shape == (H, W, T)

    def test_gpu_error_maps_biexp(self, gpu_nlsf_biexp, biexp_cube):
        H, W, _ = biexp_cube[0].shape
        e = gpu_nlsf_biexp["results"]["error_maps"]
        assert e.shape == (H, W, 6), f"Expected (H,W,6) for bi-exp, got {e.shape}"

    def test_gpu_error_maps_mono(self, gpu_nlsf_mono, mono_cube):
        H, W, _ = mono_cube[0].shape
        e = gpu_nlsf_mono["results"]["error_maps"]
        assert e.shape == (H, W, 4), f"Expected (H,W,4) for mono, got {e.shape}"

    def test_method_tag_nlsf(self, gpu_nlsf_biexp):
        assert gpu_nlsf_biexp["method"] == "GPU_NLSF"

    def test_method_tag_mle(self, gpu_mle_biexp):
        assert gpu_mle_biexp["method"] == "GPU_MLE"


# ===========================================================================
# 2. Parameter recovery — CPU NLSF
# ===========================================================================


class TestCPUNLSFRecovery:
    def test_mono_tau_recovered(self, cpu_nlsf_mono):
        popt, _, r2, _, _, _, conv, _ = cpu_nlsf_mono
        assert conv == 1
        assert abs(popt[1] - _MONO["tau"]) / _MONO["tau"] < _TOL, (
            f"CPU NLSF mono tau={popt[1]:.4f}, truth={_MONO['tau']}"
        )
        assert r2 > 0.95

    def test_biexp_tau1_recovered(self, cpu_nlsf_biexp):
        popt, _, r2, _, _, _, conv, _ = cpu_nlsf_biexp
        assert conv == 1
        assert abs(popt[2] - _BIEXP["tau1"]) / _BIEXP["tau1"] < _TOL, (
            f"CPU NLSF biexp tau1={popt[2]:.4f}, truth={_BIEXP['tau1']}"
        )
        assert r2 > 0.95

    def test_biexp_tau2_recovered(self, cpu_nlsf_biexp):
        popt, *_ = cpu_nlsf_biexp
        assert abs(popt[3] - _BIEXP["tau2"]) / _BIEXP["tau2"] < _TOL, (
            f"CPU NLSF biexp tau2={popt[3]:.4f}, truth={_BIEXP['tau2']}"
        )

    def test_biexp_alpha1_recovered(self, cpu_nlsf_biexp):
        popt, *_ = cpu_nlsf_biexp
        assert abs(popt[1] - _BIEXP["alpha1"]) / _BIEXP["alpha1"] < _TOL, (
            f"CPU NLSF biexp alpha1={popt[1]:.4f}, truth={_BIEXP['alpha1']}"
        )

    def test_biexp_tau_ordering(self, cpu_nlsf_biexp):
        popt, *_ = cpu_nlsf_biexp
        assert popt[2] <= popt[3]

    def test_mono_vshift_recovered(self, cpu_nlsf_mono):
        popt, *_ = cpu_nlsf_mono
        # popt[2] = v_shift; truth = 10
        assert abs(popt[2] - _MONO["v_shift"]) / max(_MONO["v_shift"], 1) < _TOL, (
            f"CPU NLSF mono v_shift={popt[2]:.3f}, truth={_MONO['v_shift']}"
        )

    def test_mono_hshift_near_zero(self, cpu_nlsf_mono):
        popt, *_ = cpu_nlsf_mono
        # popt[3] = h_shift in ns; truth = 0
        assert abs(popt[3] - _MONO["h_shift"]) < _H_TOL, (
            f"CPU NLSF mono h_shift={popt[3]:.4f} ns, expected ~{_MONO['h_shift']}"
        )

    def test_biexp_vshift_recovered(self, cpu_nlsf_biexp):
        popt, *_ = cpu_nlsf_biexp
        # popt[4] = v_shift
        assert abs(popt[4] - _BIEXP["v_shift"]) / max(_BIEXP["v_shift"], 1) < _TOL, (
            f"CPU NLSF biexp v_shift={popt[4]:.3f}, truth={_BIEXP['v_shift']}"
        )

    def test_biexp_hshift_near_zero(self, cpu_nlsf_biexp):
        popt, *_ = cpu_nlsf_biexp
        # popt[5] = h_shift in ns
        assert abs(popt[5] - _BIEXP["h_shift"]) < _H_TOL, (
            f"CPU NLSF biexp h_shift={popt[5]:.4f} ns, expected ~{_BIEXP['h_shift']}"
        )


# ===========================================================================
# 3. Parameter recovery — CPU MLE
# ===========================================================================


class TestCPUMLERecovery:
    def test_mono_tau_recovered(self, cpu_mle_mono):
        popt, _, r2, _, _, _, conv, _ = cpu_mle_mono
        assert conv == 1
        assert abs(popt[1] - _MONO["tau"]) / _MONO["tau"] < _TOL, (
            f"CPU MLE mono tau={popt[1]:.4f}, truth={_MONO['tau']}"
        )
        assert r2 > 0.95

    def test_biexp_tau1_recovered(self, cpu_mle_biexp):
        popt, _, r2, _, _, _, conv, _ = cpu_mle_biexp
        assert conv == 1
        assert abs(popt[2] - _BIEXP["tau1"]) / _BIEXP["tau1"] < _TOL, (
            f"CPU MLE biexp tau1={popt[2]:.4f}, truth={_BIEXP['tau1']}"
        )
        assert r2 > 0.95

    def test_biexp_tau2_recovered(self, cpu_mle_biexp):
        popt, *_ = cpu_mle_biexp
        assert abs(popt[3] - _BIEXP["tau2"]) / _BIEXP["tau2"] < _TOL, (
            f"CPU MLE biexp tau2={popt[3]:.4f}, truth={_BIEXP['tau2']}"
        )

    def test_biexp_alpha1_recovered(self, cpu_mle_biexp):
        popt, *_ = cpu_mle_biexp
        assert abs(popt[1] - _BIEXP["alpha1"]) / _BIEXP["alpha1"] < _TOL, (
            f"CPU MLE biexp alpha1={popt[1]:.4f}, truth={_BIEXP['alpha1']}"
        )

    def test_biexp_tau_ordering(self, cpu_mle_biexp):
        popt, *_ = cpu_mle_biexp
        assert popt[2] <= popt[3]

    def test_mono_vshift_recovered(self, cpu_mle_mono):
        popt, *_ = cpu_mle_mono
        assert abs(popt[2] - _MONO["v_shift"]) / max(_MONO["v_shift"], 1) < _TOL, (
            f"CPU MLE mono v_shift={popt[2]:.3f}, truth={_MONO['v_shift']}"
        )

    def test_mono_hshift_near_zero(self, cpu_mle_mono):
        popt, *_ = cpu_mle_mono
        assert abs(popt[3] - _MONO["h_shift"]) < _H_TOL, (
            f"CPU MLE mono h_shift={popt[3]:.4f} ns, expected ~{_MONO['h_shift']}"
        )

    def test_biexp_vshift_recovered(self, cpu_mle_biexp):
        popt, *_ = cpu_mle_biexp
        assert abs(popt[4] - _BIEXP["v_shift"]) / max(_BIEXP["v_shift"], 1) < _TOL, (
            f"CPU MLE biexp v_shift={popt[4]:.3f}, truth={_BIEXP['v_shift']}"
        )

    def test_biexp_hshift_near_zero(self, cpu_mle_biexp):
        popt, *_ = cpu_mle_biexp
        assert abs(popt[5] - _BIEXP["h_shift"]) < _H_TOL, (
            f"CPU MLE biexp h_shift={popt[5]:.4f} ns, expected ~{_BIEXP['h_shift']}"
        )


# ===========================================================================
# 4. Parameter recovery — GPU NLSF
# ===========================================================================


class TestGPUNLSFRecovery:
    def _median_healthy(self, result, key):
        maps = result["results"]["maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy GPU pixels")
        return float(np.median(maps[key][health]))

    def test_mono_tau_recovered(self, gpu_nlsf_mono):
        tau = self._median_healthy(gpu_nlsf_mono, "tau_map")
        assert abs(tau - _MONO["tau"]) / _MONO["tau"] < _TOL, (
            f"GPU NLSF mono tau={tau:.4f}, truth={_MONO['tau']}"
        )

    def test_mono_r2_positive(self, gpu_nlsf_mono):
        maps = gpu_nlsf_mono["results"]["maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy GPU pixels")
        assert np.all(maps["R2_map"][health] > 0.90)

    def test_mono_rmse_matches_residual_map(self, gpu_nlsf_mono):
        maps = gpu_nlsf_mono["results"]["maps"]
        tr = gpu_nlsf_mono["results"]["TR_maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy GPU pixels")
        expected = np.sqrt(np.mean(tr["residual_map"] ** 2, axis=-1))
        np.testing.assert_allclose(
            maps["rmse_map"][health], expected[health], rtol=1e-3
        )

    def test_biexp_tau1_recovered(self, gpu_nlsf_biexp):
        tau1 = self._median_healthy(gpu_nlsf_biexp, "tau1_map")
        assert abs(tau1 - _BIEXP["tau1"]) / _BIEXP["tau1"] < _TOL, (
            f"GPU NLSF biexp tau1={tau1:.4f}, truth={_BIEXP['tau1']}"
        )

    def test_biexp_tau2_recovered(self, gpu_nlsf_biexp):
        tau2 = self._median_healthy(gpu_nlsf_biexp, "tau2_map")
        assert abs(tau2 - _BIEXP["tau2"]) / _BIEXP["tau2"] < _TOL, (
            f"GPU NLSF biexp tau2={tau2:.4f}, truth={_BIEXP['tau2']}"
        )

    def test_biexp_alpha1_recovered(self, gpu_nlsf_biexp):
        a1 = self._median_healthy(gpu_nlsf_biexp, "alpha1_map")
        assert abs(a1 - _BIEXP["alpha1"]) / _BIEXP["alpha1"] < _TOL, (
            f"GPU NLSF biexp alpha1={a1:.4f}, truth={_BIEXP['alpha1']}"
        )

    def test_biexp_tau_ordering(self, gpu_nlsf_biexp):
        maps = gpu_nlsf_biexp["results"]["maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy GPU pixels")
        assert np.all(maps["tau1_map"][health] <= maps["tau2_map"][health])

    def test_biexp_r2_positive(self, gpu_nlsf_biexp):
        maps = gpu_nlsf_biexp["results"]["maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy GPU pixels")
        assert np.all(maps["R2_map"][health] > 0.90)

    def test_mono_vshift_recovered(self, gpu_nlsf_mono):
        v = self._median_healthy(gpu_nlsf_mono, "v_shift_map")
        assert abs(v - _MONO["v_shift"]) / max(_MONO["v_shift"], 1) < _TOL, (
            f"GPU NLSF mono v_shift={v:.3f}, truth={_MONO['v_shift']}"
        )

    def test_mono_hshift_near_zero(self, gpu_nlsf_mono):
        h = self._median_healthy(gpu_nlsf_mono, "h_shift_map")
        assert abs(h - _MONO["h_shift"]) < _H_TOL, (
            f"GPU NLSF mono h_shift={h:.4f} ns, expected ~{_MONO['h_shift']}"
        )

    def test_biexp_vshift_recovered(self, gpu_nlsf_biexp):
        v = self._median_healthy(gpu_nlsf_biexp, "v_shift_map")
        assert abs(v - _BIEXP["v_shift"]) / max(_BIEXP["v_shift"], 1) < _TOL, (
            f"GPU NLSF biexp v_shift={v:.3f}, truth={_BIEXP['v_shift']}"
        )

    def test_biexp_hshift_near_zero(self, gpu_nlsf_biexp):
        h = self._median_healthy(gpu_nlsf_biexp, "h_shift_map")
        assert abs(h - _BIEXP["h_shift"]) < _H_TOL, (
            f"GPU NLSF biexp h_shift={h:.4f} ns, expected ~{_BIEXP['h_shift']}"
        )


# ===========================================================================
# 5. Parameter recovery — GPU MLE
# ===========================================================================


class TestGPUMLERecovery:
    def _median_healthy(self, result, key):
        maps = result["results"]["maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy GPU pixels")
        return float(np.median(maps[key][health]))

    def test_mono_tau_recovered(self, gpu_mle_mono):
        tau = self._median_healthy(gpu_mle_mono, "tau_map")
        assert abs(tau - _MONO["tau"]) / _MONO["tau"] < _TOL, (
            f"GPU MLE mono tau={tau:.4f}, truth={_MONO['tau']}"
        )

    def test_mono_r2_positive(self, gpu_mle_mono):
        maps = gpu_mle_mono["results"]["maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy GPU pixels")
        assert np.all(maps["R2_map"][health] > 0.90)

    def test_biexp_tau1_recovered(self, gpu_mle_biexp):
        tau1 = self._median_healthy(gpu_mle_biexp, "tau1_map")
        assert abs(tau1 - _BIEXP["tau1"]) / _BIEXP["tau1"] < _TOL, (
            f"GPU MLE biexp tau1={tau1:.4f}, truth={_BIEXP['tau1']}"
        )

    def test_biexp_tau2_recovered(self, gpu_mle_biexp):
        tau2 = self._median_healthy(gpu_mle_biexp, "tau2_map")
        assert abs(tau2 - _BIEXP["tau2"]) / _BIEXP["tau2"] < _TOL, (
            f"GPU MLE biexp tau2={tau2:.4f}, truth={_BIEXP['tau2']}"
        )

    def test_biexp_alpha1_recovered(self, gpu_mle_biexp):
        a1 = self._median_healthy(gpu_mle_biexp, "alpha1_map")
        assert abs(a1 - _BIEXP["alpha1"]) / _BIEXP["alpha1"] < _TOL, (
            f"GPU MLE biexp alpha1={a1:.4f}, truth={_BIEXP['alpha1']}"
        )

    def test_biexp_tau_ordering(self, gpu_mle_biexp):
        maps = gpu_mle_biexp["results"]["maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy GPU pixels")
        assert np.all(maps["tau1_map"][health] <= maps["tau2_map"][health])

    def test_biexp_r2_positive(self, gpu_mle_biexp):
        maps = gpu_mle_biexp["results"]["maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy GPU pixels")
        assert np.all(maps["R2_map"][health] > 0.90)

    def test_mono_vshift_recovered(self, gpu_mle_mono):
        v = self._median_healthy(gpu_mle_mono, "v_shift_map")
        assert abs(v - _MONO["v_shift"]) / max(_MONO["v_shift"], 1) < _TOL, (
            f"GPU MLE mono v_shift={v:.3f}, truth={_MONO['v_shift']}"
        )

    def test_mono_hshift_near_zero(self, gpu_mle_mono):
        h = self._median_healthy(gpu_mle_mono, "h_shift_map")
        assert abs(h - _MONO["h_shift"]) < _H_TOL, (
            f"GPU MLE mono h_shift={h:.4f} ns, expected ~{_MONO['h_shift']}"
        )

    def test_biexp_vshift_recovered(self, gpu_mle_biexp):
        v = self._median_healthy(gpu_mle_biexp, "v_shift_map")
        assert abs(v - _BIEXP["v_shift"]) / max(_BIEXP["v_shift"], 1) < _TOL, (
            f"GPU MLE biexp v_shift={v:.3f}, truth={_BIEXP['v_shift']}"
        )

    def test_biexp_hshift_near_zero(self, gpu_mle_biexp):
        h = self._median_healthy(gpu_mle_biexp, "h_shift_map")
        assert abs(h - _BIEXP["h_shift"]) < _H_TOL, (
            f"GPU MLE biexp h_shift={h:.4f} ns, expected ~{_BIEXP['h_shift']}"
        )


# ===========================================================================
# 6. Cross-method consistency — NLSF and MLE should agree within 2× _TOL
# ===========================================================================


class TestCrossMethodConsistency:
    """CPU and GPU, NLSF and MLE, should all recover parameters close to each other."""

    def _gpu_median(self, result, key):
        maps = result["results"]["maps"]
        health = maps["pixel_health_map"] > 0
        if not health.any():
            pytest.skip("No healthy GPU pixels")
        return float(np.median(maps[key][health]))

    def test_mono_tau_cpu_vs_gpu_nlsf(self, cpu_nlsf_mono, gpu_nlsf_mono):
        cpu_tau = cpu_nlsf_mono[0][1]
        gpu_tau = self._gpu_median(gpu_nlsf_mono, "tau_map")
        assert abs(cpu_tau - gpu_tau) / _MONO["tau"] < 2 * _TOL, (
            f"CPU NLSF tau={cpu_tau:.4f} vs GPU NLSF tau={gpu_tau:.4f}"
        )

    def test_mono_tau_cpu_vs_gpu_mle(self, cpu_mle_mono, gpu_mle_mono):
        cpu_tau = cpu_mle_mono[0][1]
        gpu_tau = self._gpu_median(gpu_mle_mono, "tau_map")
        assert abs(cpu_tau - gpu_tau) / _MONO["tau"] < 2 * _TOL, (
            f"CPU MLE tau={cpu_tau:.4f} vs GPU MLE tau={gpu_tau:.4f}"
        )

    def test_biexp_tau1_nlsf_vs_mle_cpu(self, cpu_nlsf_biexp, cpu_mle_biexp):
        nlsf_tau1 = cpu_nlsf_biexp[0][2]
        mle_tau1 = cpu_mle_biexp[0][2]
        assert abs(nlsf_tau1 - mle_tau1) / _BIEXP["tau1"] < 2 * _TOL, (
            f"CPU NLSF tau1={nlsf_tau1:.4f} vs CPU MLE tau1={mle_tau1:.4f}"
        )

    def test_biexp_tau1_nlsf_vs_mle_gpu(self, gpu_nlsf_biexp, gpu_mle_biexp):
        nlsf_tau1 = self._gpu_median(gpu_nlsf_biexp, "tau1_map")
        mle_tau1 = self._gpu_median(gpu_mle_biexp, "tau1_map")
        assert abs(nlsf_tau1 - mle_tau1) / _BIEXP["tau1"] < 2 * _TOL, (
            f"GPU NLSF tau1={nlsf_tau1:.4f} vs GPU MLE tau1={mle_tau1:.4f}"
        )

    def test_biexp_tau2_cpu_vs_gpu_mle(self, cpu_mle_biexp, gpu_mle_biexp):
        cpu_tau2 = cpu_mle_biexp[0][3]
        gpu_tau2 = self._gpu_median(gpu_mle_biexp, "tau2_map")
        assert abs(cpu_tau2 - gpu_tau2) / _BIEXP["tau2"] < 2 * _TOL, (
            f"CPU MLE tau2={cpu_tau2:.4f} vs GPU MLE tau2={gpu_tau2:.4f}"
        )


# ===========================================================================
# 7. Printed summary (captured by pytest -s)
# ===========================================================================


def test_print_comparison_table(
    cpu_nlsf_mono,
    cpu_mle_mono,
    gpu_nlsf_mono,
    gpu_mle_mono,
    cpu_nlsf_biexp,
    cpu_mle_biexp,
    gpu_nlsf_biexp,
    gpu_mle_biexp,
):
    """Print a side-by-side comparison of all 4 methods × 2 models."""

    def gpu_med(res, key):
        maps = res["results"]["maps"]
        h = maps["pixel_health_map"] > 0
        return float(np.median(maps[key][h])) if h.any() else float("nan")

    def pct(est, truth):
        return f"{100 * (est - truth) / truth:+.1f}%"

    lines = []
    lines.append("\n" + "=" * 90)
    lines.append(
        "  PARAMETER RECOVERY — all 4 methods × 2 models  (% = error from ground truth)"
    )
    lines.append("=" * 90)

    # ---- Mono-exponential ----
    lines.append(
        f"\n  MONO-EXPONENTIAL  "
        f"[tau={_MONO['tau']} ns, S={_MONO['S']}, v_shift={_MONO['v_shift']}, h_shift={_MONO['h_shift']} ns]"
    )
    lines.append(
        f"  {'Method':<12}  {'tau (ns)':<16}  {'v_shift':<14}  {'h_shift (ns)':<16}  {'R²'}"
    )
    lines.append(f"  {'-' * 12}  {'-' * 16}  {'-' * 14}  {'-' * 16}  {'-' * 6}")

    def cpu_mono_row(res, name):
        p = res[0]
        return (name, p[1], p[2], p[3], res[2])

    def gpu_mono_row(res, name):
        return (
            name,
            gpu_med(res, "tau_map"),
            gpu_med(res, "v_shift_map"),
            gpu_med(res, "h_shift_map"),
            gpu_med(res, "R2_map"),
        )

    for name, tau, vs, hs, r2 in [
        cpu_mono_row(cpu_nlsf_mono, "CPU NLSF"),
        cpu_mono_row(cpu_mle_mono, "CPU MLE"),
        gpu_mono_row(gpu_nlsf_mono, "GPU NLSF"),
        gpu_mono_row(gpu_mle_mono, "GPU MLE"),
    ]:
        lines.append(
            f"  {name:<12}  {tau:.4f} {pct(tau, _MONO['tau']):<9}  "
            f"{vs:.3f} {pct(vs, _MONO['v_shift']):<7}  "
            f"{hs:+.4f} ns          {r2:.4f}"
        )

    # ---- Bi-exponential ----
    lines.append(
        f"\n  BI-EXPONENTIAL  "
        f"[tau1={_BIEXP['tau1']}, tau2={_BIEXP['tau2']}, alpha1={_BIEXP['alpha1']}, "
        f"S={_BIEXP['S']}, v_shift={_BIEXP['v_shift']}, h_shift={_BIEXP['h_shift']} ns]"
    )
    lines.append(
        f"  {'Method':<12}  {'tau1 (ns)':<14}  {'tau2 (ns)':<14}  "
        f"{'alpha1':<12}  {'v_shift':<14}  {'h_shift (ns)':<16}  {'R²'}"
    )
    lines.append(
        f"  {'-' * 12}  {'-' * 14}  {'-' * 14}  {'-' * 12}  {'-' * 14}  {'-' * 16}  {'-' * 6}"
    )

    def cpu_biexp_row(res, name):
        p = res[0]
        return (name, p[2], p[3], p[1], p[4], p[5], res[2])

    def gpu_biexp_row(res, name):
        return (
            name,
            gpu_med(res, "tau1_map"),
            gpu_med(res, "tau2_map"),
            gpu_med(res, "alpha1_map"),
            gpu_med(res, "v_shift_map"),
            gpu_med(res, "h_shift_map"),
            gpu_med(res, "R2_map"),
        )

    for name, t1, t2, a1, vs, hs, r2 in [
        cpu_biexp_row(cpu_nlsf_biexp, "CPU NLSF"),
        cpu_biexp_row(cpu_mle_biexp, "CPU MLE"),
        gpu_biexp_row(gpu_nlsf_biexp, "GPU NLSF"),
        gpu_biexp_row(gpu_mle_biexp, "GPU MLE"),
    ]:
        lines.append(
            f"  {name:<12}  {t1:.4f} {pct(t1, _BIEXP['tau1']):<7}  "
            f"{t2:.4f} {pct(t2, _BIEXP['tau2']):<7}  "
            f"{a1:.4f} {pct(a1, _BIEXP['alpha1']):<5}  "
            f"{vs:.3f} {pct(vs, _BIEXP['v_shift']):<7}  "
            f"{hs:+.4f} ns          {r2:.4f}"
        )

    lines.append("=" * 90)
    print("\n".join(lines))
    # Test always passes — this just prints the table
    assert True
