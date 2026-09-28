"""
Tests for ParamToDecay (pyfli.reconstruction).

All tests use purely synthetic numpy arrays; no file I/O or GPU required.
Where relevant, outputs are cross-checked directly against
forward_model.model_numpy/decay_kernel, shared_metrics.compute_fli_stats, and
BaseFLIFitter's own time-axis construction, since the whole point of this
class is to stay in lockstep with pyfli.solver's conventions.
"""

import numpy as np
import pytest

from pyfli.reconstruction import ParamToDecay
from pyfli.solver.base_fitter import BaseFLIFitter
from pyfli.solver.forward_model import decay_kernel, model_numpy
from pyfli.solver.shared_metrics import compute_fli_stats

# ---------------------------------------------------------------------------
# Constants — 80 MHz system, 64 bins → T_acq = 12.5 ns
# ---------------------------------------------------------------------------
_N = 64
_FREQ = (80.0, 80.0)  # (laser MHz, acq MHz) -- for BaseFLIFitter only
_FREQ_ACQ = 80.0  # acquisition frequency (MHz) -- for ParamToDecay


def _gaussian_irf(n=_N, center=15, sigma=3.0):
    t = np.arange(n)
    irf = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    return irf


def _mono_params(H=4, W=4, S=1000.0, tau=2.0, v_shift=0.0, h_shift=0.0):
    return {
        "photon_count_map": np.full((H, W), S, dtype=np.float32),
        "tau_map": np.full((H, W), tau, dtype=np.float32),
        "v_shift_map": np.full((H, W), v_shift, dtype=np.float32),
        "h_shift_map": np.full((H, W), h_shift, dtype=np.float32),
    }


def _biexp_params(
    H=4, W=4, S=1000.0, alpha1=0.6, tau1=0.5, tau2=2.5, v_shift=0.0, h_shift=0.0
):
    return {
        "photon_count_map": np.full((H, W), S, dtype=np.float32),
        "alpha1_map": np.full((H, W), alpha1, dtype=np.float32),
        "tau1_map": np.full((H, W), tau1, dtype=np.float32),
        "tau2_map": np.full((H, W), tau2, dtype=np.float32),
        "v_shift_map": np.full((H, W), v_shift, dtype=np.float32),
        "h_shift_map": np.full((H, W), h_shift, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError):
            ParamToDecay("tri-exponential", _FREQ_ACQ, _gaussian_irf())

    def test_missing_num_gates_without_irf_raises(self):
        with pytest.raises(ValueError):
            ParamToDecay("mono-exponential", _FREQ_ACQ, irf=None)

    def test_num_gates_from_1d_irf(self):
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, _gaussian_irf(n=_N))
        assert recon.num_gates == _N

    def test_num_gates_from_3d_irf(self):
        irf_cube = np.tile(_gaussian_irf(), (3, 3, 1))
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf_cube)
        assert recon.num_gates == _N

    def test_num_gates_explicit_without_irf(self):
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf=None, num_gates=_N)
        assert recon.num_gates == _N
        assert recon.irf is None

    def test_time_axis_matches_base_fitter(self):
        """T_acq must match BaseFLIFitter's own t for the same acquisition freq."""
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf)
        dummy_decay = np.zeros(_N)
        bf = BaseFLIFitter(_FREQ, dummy_decay, irf)
        np.testing.assert_allclose(recon.t, bf.t)

    def test_t_acq_computed_from_freq(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", 80.0, irf)
        assert recon.T_acq == pytest.approx(1000.0 / 80.0)

    def test_required_keys_mono(self):
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, _gaussian_irf())
        assert recon.required_keys == ("tau_map",)

    def test_required_keys_biexp(self):
        recon = ParamToDecay("bi-exponential", _FREQ_ACQ, _gaussian_irf())
        assert recon.required_keys == ("alpha1_map", "tau1_map", "tau2_map")


# ---------------------------------------------------------------------------
# Parameter validation / defaults
# ---------------------------------------------------------------------------


class TestParamsAndDefaults:
    def test_missing_required_key_raises(self):
        recon = ParamToDecay("bi-exponential", _FREQ_ACQ, _gaussian_irf())
        with pytest.raises(KeyError):
            recon.reconstruct({"photon_count_map": np.ones((2, 2))}, verbose=False)

    def test_optional_keys_default_correctly(self):
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, _gaussian_irf())
        full = _mono_params(S=1.0, v_shift=0.0, h_shift=0.0)
        minimal = {"tau_map": full["tau_map"]}
        out_full = recon.reconstruct(full, verbose=False)["fit_map"]
        out_minimal = recon.reconstruct(minimal, verbose=False)["fit_map"]
        np.testing.assert_allclose(out_full, out_minimal)

    def test_photon_count_defaults_to_one_without_decay(self):
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, _gaussian_irf())
        minimal = {"tau_map": np.full((2, 2), 2.0)}
        with_s1 = _mono_params(H=2, W=2, S=1.0)
        out_minimal = recon.reconstruct(minimal, verbose=False)["fit_map"]
        out_s1 = recon.reconstruct(with_s1, verbose=False)["fit_map"]
        np.testing.assert_allclose(out_minimal, out_s1)

    def test_photon_count_derived_from_decay_matches_decay_total(self):
        """Auto-derived S must make fit_map's own total match decay's total
        per pixel -- NOT equal decay.sum(axis=-1) directly (the model
        kernel's own discrete sum at unit amplitude is not 1 in general, so
        naively equating S to the raw decay sum would over/under-scale the
        fit by that factor)."""
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf)
        H, W = 3, 3
        rng = np.random.default_rng(0)
        decay = rng.uniform(0, 500, (H, W, _N)).astype(np.float32)
        minimal = {"tau_map": np.full((H, W), 2.0, dtype=np.float32)}

        filled = recon._fill_photon_count_from_decay(minimal, decay)
        assert "photon_count_map" not in minimal, "must not mutate caller's dict"
        assert not np.allclose(filled["photon_count_map"], decay.sum(axis=-1)), (
            "S should NOT simply equal the raw decay sum"
        )

        out = recon.reconstruct(minimal, decay=decay, verbose=False)
        np.testing.assert_allclose(
            out["fit_map"].sum(axis=-1), decay.sum(axis=-1), rtol=1e-4
        )

    def test_photon_count_derived_matches_between_loop_and_vectorized(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("bi-exponential", _FREQ_ACQ, irf)
        H, W = 3, 3
        rng = np.random.default_rng(1)
        decay = rng.uniform(0, 500, (H, W, _N)).astype(np.float32)
        minimal = {
            "tau1_map": np.full((H, W), 0.5, dtype=np.float32),
            "tau2_map": np.full((H, W), 2.5, dtype=np.float32),
            "alpha1_map": np.full((H, W), 0.4, dtype=np.float32),
        }
        loop_out = recon.reconstruct(minimal, decay=decay, verbose=False)["fit_map"]
        vec_out = recon.reconstruct_vectorized(minimal, decay=decay, verbose=False)[
            "fit_map"
        ]
        np.testing.assert_allclose(loop_out, vec_out, atol=1e-3)
        np.testing.assert_allclose(loop_out.sum(axis=-1), decay.sum(axis=-1), rtol=1e-4)

    def test_explicit_photon_count_takes_priority_over_decay(self):
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, _gaussian_irf())
        H, W = 2, 2
        explicit = _mono_params(H=H, W=W, S=42.0)
        decay = np.full((H, W, _N), 999.0)
        out_with_decay = recon.reconstruct(explicit, decay=decay, verbose=False)
        out_without = recon.reconstruct(explicit, verbose=False)
        np.testing.assert_allclose(out_with_decay["fit_map"], out_without["fit_map"])


# ---------------------------------------------------------------------------
# Forward-model correctness (cross-checked against model_numpy / decay_kernel)
# ---------------------------------------------------------------------------


class TestForwardModelCorrectness:
    def test_mono_matches_model_numpy_directly(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf)
        params = _mono_params(H=1, W=1, S=500.0, tau=1.5, v_shift=2.0, h_shift=0.3)
        out = recon.reconstruct(params, verbose=False)["fit_map"][0, 0, :]
        direct = model_numpy(
            recon.t, irf, np.array([500.0, 1.5, 2.0, 0.3]), "mono-exponential"
        )
        np.testing.assert_allclose(out, direct)

    def test_biexp_matches_model_numpy_directly(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("bi-exponential", _FREQ_ACQ, irf)
        params = _biexp_params(
            H=1, W=1, S=800.0, alpha1=0.3, tau1=0.4, tau2=3.0, v_shift=1.0, h_shift=-0.2
        )
        out = recon.reconstruct(params, verbose=False)["fit_map"][0, 0, :]
        direct = model_numpy(
            recon.t, irf, np.array([800.0, 0.3, 0.4, 3.0, 1.0, -0.2]), "bi-exponential"
        )
        np.testing.assert_allclose(out, direct)

    def test_no_irf_matches_decay_kernel_directly(self):
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf=None, num_gates=_N)
        params = _mono_params(H=1, W=1, S=200.0, tau=2.0, v_shift=5.0, h_shift=0.0)
        out = recon.reconstruct(params, verbose=False)["fit_map"][0, 0, :]
        kernel, v_shift = decay_kernel(
            recon.t, (200.0, 2.0, 5.0), "mono-exponential", h_shift=0.0
        )
        np.testing.assert_allclose(out, kernel + v_shift, rtol=1e-5)

    def test_output_shape(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("bi-exponential", _FREQ_ACQ, irf)
        H, W = 5, 3
        out = recon.reconstruct(_biexp_params(H=H, W=W), verbose=False)["fit_map"]
        assert out.shape == (H, W, _N)

    def test_scaling_by_photon_count_is_linear(self):
        """Doubling S must double the (v_shift-free) fit."""
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf)
        p1 = _mono_params(H=1, W=1, S=100.0, tau=2.0)
        p2 = _mono_params(H=1, W=1, S=200.0, tau=2.0)
        f1 = recon.reconstruct(p1, verbose=False)["fit_map"]
        f2 = recon.reconstruct(p2, verbose=False)["fit_map"]
        np.testing.assert_allclose(f2, 2.0 * f1, rtol=1e-5)


# ---------------------------------------------------------------------------
# TR_maps / fit_stats_maps
# ---------------------------------------------------------------------------


class TestTRMapsAndStats:
    def test_no_decay_no_tr_maps(self):
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, _gaussian_irf())
        out = recon.reconstruct(_mono_params(), verbose=False)
        assert "TR_maps" not in out
        assert "fit_stats_maps" not in out

    def test_residual_map_is_decay_minus_fit(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf)
        params = _mono_params(H=2, W=2)
        fit = recon.reconstruct(params, verbose=False)["fit_map"]
        decay = fit + 3.0
        out = recon.reconstruct(params, decay=decay, verbose=False)
        np.testing.assert_allclose(
            out["TR_maps"]["residual_map"], decay - out["TR_maps"]["fit_map"]
        )

    def test_fit_stats_keys_match_solver_convention(self):
        """Keys must match FLICPUProcessor/FLIGPUProcessor's own map names."""
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf)
        params = _mono_params(H=2, W=2)
        fit = recon.reconstruct(params, verbose=False)["fit_map"]
        decay = fit + np.random.default_rng(1).normal(0, 0.5, fit.shape)
        out = recon.reconstruct(params, decay=decay, verbose=False)
        assert set(out["fit_stats_maps"].keys()) == {
            "R2_map",
            "chi2_map",
            "reduced_chi2_map",
            "pearson_chi2_map",
            "pearson_reduced_chi2_map",
            "rmse_map",
        }

    def test_fit_stats_match_compute_fli_stats_directly(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("bi-exponential", _FREQ_ACQ, irf)
        params = _biexp_params(H=1, W=1)
        fit = recon.reconstruct(params, verbose=False)["fit_map"]
        decay = fit + np.random.default_rng(2).normal(0, 1.0, fit.shape)
        out = recon.reconstruct(params, decay=decay, verbose=False)

        _ssr, chi_sq, red_chi_sq, r_sq, rmse = compute_fli_stats(
            fit[0, 0, :], decay[0, 0, :], n_params=6
        )
        assert out["fit_stats_maps"]["chi2_map"][0, 0] == pytest.approx(chi_sq)
        assert out["fit_stats_maps"]["reduced_chi2_map"][0, 0] == pytest.approx(
            red_chi_sq
        )
        assert out["fit_stats_maps"]["R2_map"][0, 0] == pytest.approx(r_sq)
        assert out["fit_stats_maps"]["rmse_map"][0, 0] == pytest.approx(rmse)


# ---------------------------------------------------------------------------
# bool_mask
# ---------------------------------------------------------------------------


class TestBoolMask:
    def test_masked_out_pixels_are_nan(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf)
        H, W = 3, 3
        params = _mono_params(H=H, W=W)
        mask = np.zeros((H, W), dtype=bool)
        mask[1, 1] = True
        out = recon.reconstruct(params, bool_mask=mask, verbose=False)
        fit = out["fit_map"]
        assert not np.isnan(fit[1, 1, :]).any()
        assert np.isnan(fit[0, 0, :]).all()

    def test_masked_out_stats_are_nan(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf)
        H, W = 3, 3
        params = _mono_params(H=H, W=W)
        decay = recon.reconstruct(params, verbose=False)["fit_map"] + 1.0
        mask = np.zeros((H, W), dtype=bool)
        mask[1, 1] = True
        out = recon.reconstruct(params, decay=decay, bool_mask=mask, verbose=False)
        assert np.isnan(out["fit_stats_maps"]["R2_map"][0, 0])
        assert not np.isnan(out["fit_stats_maps"]["R2_map"][1, 1])


# ---------------------------------------------------------------------------
# Single-pixel reconstruction
# ---------------------------------------------------------------------------


class TestSinglePixel:
    def test_scalar_params_give_1d_output(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf)
        out = recon.reconstruct(
            {
                "photon_count_map": 500.0,
                "tau_map": 2.0,
                "v_shift_map": 0.0,
                "h_shift_map": 0.0,
            },
            verbose=False,
        )
        assert out["fit_map"].shape == (_N,)

    def test_single_pixel_matches_1x1_image(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("bi-exponential", _FREQ_ACQ, irf)
        scalar_params = {
            "photon_count_map": 800.0,
            "alpha1_map": 0.4,
            "tau1_map": 0.6,
            "tau2_map": 2.8,
            "v_shift_map": 1.0,
            "h_shift_map": 0.0,
        }
        image_params = _biexp_params(
            H=1, W=1, S=800.0, alpha1=0.4, tau1=0.6, tau2=2.8, v_shift=1.0
        )
        single_out = recon.reconstruct(scalar_params, verbose=False)["fit_map"]
        image_out = recon.reconstruct(image_params, verbose=False)["fit_map"][0, 0, :]
        # image_params are float32 while the single-pixel path promotes to
        # float64 internally, so allow float32-level rounding noise.
        np.testing.assert_allclose(single_out, image_out, rtol=1e-5, atol=1e-4)

    def test_single_pixel_fit_stats_are_floats(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf)
        sp = {
            "photon_count_map": 500.0,
            "tau_map": 2.0,
            "v_shift_map": 0.0,
            "h_shift_map": 0.0,
        }
        fit = recon.reconstruct(sp, verbose=False)["fit_map"]
        decay = fit + np.random.default_rng(3).normal(0, 1.0, fit.shape)
        out = recon.reconstruct(sp, decay=decay, verbose=False)
        assert isinstance(out["fit_stats_maps"]["R2_map"], float)
        assert out["TR_maps"]["fit_map"].shape == (_N,)


# ---------------------------------------------------------------------------
# Vectorized parity
# ---------------------------------------------------------------------------


class TestVectorizedParity:
    def test_mono_vectorized_matches_loop(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf)
        params = _mono_params(H=4, W=4, S=750.0, tau=1.8, v_shift=1.0, h_shift=0.4)
        loop_out = recon.reconstruct(params, verbose=False)["fit_map"]
        vec_out = recon.reconstruct_vectorized(params, verbose=False)["fit_map"]
        np.testing.assert_allclose(loop_out, vec_out, atol=1e-4)

    def test_biexp_with_decay_and_mask_matches_loop(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("bi-exponential", _FREQ_ACQ, irf)
        H, W = 4, 4
        params = _biexp_params(H=H, W=W)
        rng = np.random.default_rng(5)
        decay = recon.reconstruct(params, verbose=False)["fit_map"] + rng.normal(
            0, 1.0, (H, W, _N)
        )
        mask = rng.uniform(size=(H, W)) > 0.4

        loop_out = recon.reconstruct(params, decay=decay, bool_mask=mask, verbose=False)
        vec_out = recon.reconstruct_vectorized(
            params, decay=decay, bool_mask=mask, verbose=False
        )
        np.testing.assert_allclose(
            loop_out["fit_map"], vec_out["fit_map"], atol=1e-4, equal_nan=True
        )
        for key in loop_out["fit_stats_maps"]:
            np.testing.assert_allclose(
                loop_out["fit_stats_maps"][key],
                vec_out["fit_stats_maps"][key],
                atol=1e-3,
                equal_nan=True,
            )

    def test_single_pixel_vectorized_matches_loop(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf)
        sp = {
            "photon_count_map": 500.0,
            "tau_map": 2.0,
            "v_shift_map": 0.0,
            "h_shift_map": 0.0,
        }
        loop_out = recon.reconstruct(sp, verbose=False)["fit_map"]
        vec_out = recon.reconstruct_vectorized(sp, verbose=False)["fit_map"]
        np.testing.assert_allclose(loop_out, vec_out, atol=1e-4)


# ---------------------------------------------------------------------------
# reconstruct_unit_amplitude / rescale_fit_to_measured_totals
# ---------------------------------------------------------------------------


class TestUnitAmplitudeAndRescale:
    def test_convolved_equals_kernel_without_irf(self):
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf=None, num_gates=_N)
        params = {"tau_map": np.full((2, 2), 2.0)}
        out = recon.reconstruct_unit_amplitude(params)
        np.testing.assert_array_equal(out["kernel_map"], out["convolved_map"])

    def test_rescale_matches_target_totals(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf)
        H, W = 3, 3
        unit_params = {
            "photon_count_map": np.ones((H, W), dtype=np.float32),
            "tau_map": np.full((H, W), 2.0, dtype=np.float32),
            "v_shift_map": np.zeros((H, W), dtype=np.float32),
            "h_shift_map": np.zeros((H, W), dtype=np.float32),
        }
        unit = recon.reconstruct_unit_amplitude(unit_params)
        target = np.random.default_rng(6).uniform(100, 5000, (H, W, _N))
        rescaled = recon.rescale_fit_to_measured_totals(unit["convolved_map"], target)
        np.testing.assert_allclose(
            rescaled.sum(axis=-1), target.sum(axis=-1), rtol=1e-4
        )

    def test_rescale_is_shape_preserving(self):
        irf = _gaussian_irf()
        recon = ParamToDecay("mono-exponential", _FREQ_ACQ, irf)
        H, W = 2, 2
        unit_params = {
            "photon_count_map": np.ones((H, W), dtype=np.float32),
            "tau_map": np.full((H, W), 2.0, dtype=np.float32),
            "v_shift_map": np.zeros((H, W), dtype=np.float32),
            "h_shift_map": np.zeros((H, W), dtype=np.float32),
        }
        unit = recon.reconstruct_unit_amplitude(unit_params)
        assert unit["kernel_map"].shape == (H, W, _N)
        assert unit["convolved_map"].shape == (H, W, _N)
