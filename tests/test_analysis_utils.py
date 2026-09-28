"""Tests for pyfli.reconstruction.DetailedRecon."""

import logging

import numpy as np
import pytest

from pyfli.reconstruction import DetailedRecon


@pytest.fixture
def irf_delta():
    """Near-delta IRF so convolution stays close to identity."""
    bins = 64
    irf = np.zeros(bins, dtype=np.float32)
    irf[0] = 1.0
    return irf


def make_reconstructor(irf, decay=None, **kwargs):
    return DetailedRecon(80.0, irf, binned_decay=decay, **kwargs)


# ---------------------------------------------------------------------------
# reconstruct()
# ---------------------------------------------------------------------------


def test_missing_required_params_raises(irf_delta):
    dr = make_reconstructor(irf_delta)
    with pytest.raises(KeyError):
        dr.reconstruct({}, "mono-exponential")


def test_rejects_unknown_model_type(irf_delta):
    H, W = 2, 2
    tau = np.full((H, W), 1.0, dtype=np.float32)
    dr = make_reconstructor(irf_delta)
    with pytest.raises(ValueError):
        dr.reconstruct({"tau_map": tau}, "triexponential")


def test_mono_branch_uses_tau_map_directly(irf_delta):
    H, W = 3, 3
    tau = np.full((H, W), 0.8, dtype=np.float32)
    decay = np.ones((H, W, irf_delta.shape[-1]), dtype=np.float32)

    dr = make_reconstructor(irf_delta, decay)
    out = dr.reconstruct({"tau_map": tau}, "mono-exponential")
    maps = out["results"]["maps"]
    tr_maps = out["results"]["TR_maps"]

    assert set(maps) >= {
        "photon_count_map",
        "tau_map",
        "R2_map",
        "chi2_map",
        "reduced_chi2_map",
        "rmse_map",
    }
    assert set(tr_maps) == {"fit_map", "residual_map", "sdf_map", "convolved_map"}
    np.testing.assert_allclose(maps["tau_map"], tau)
    np.testing.assert_allclose(
        maps["rmse_map"],
        np.sqrt(np.mean(tr_maps["residual_map"] ** 2, axis=-1)),
        rtol=1e-4,
    )


def test_log_summary_true_emits_multi_stat_line(irf_delta, caplog):
    H, W = 2, 2
    tau = np.full((H, W), 0.8, dtype=np.float32)
    decay = np.ones((H, W, irf_delta.shape[-1]), dtype=np.float32)
    dr = make_reconstructor(irf_delta, decay)

    with caplog.at_level(logging.INFO, logger="pyfli"):
        dr.reconstruct({"tau_map": tau}, "mono-exponential")

    msgs = [r.message for r in caplog.records]
    assert any(
        "mean reduced chi2" in m
        and "mean R2" in m
        and "mean RMSE" in m
        and "mean chi2" in m
        for m in msgs
    )


def test_log_summary_false_is_silent(irf_delta, caplog):
    H, W = 2, 2
    tau = np.full((H, W), 0.8, dtype=np.float32)
    decay = np.ones((H, W, irf_delta.shape[-1]), dtype=np.float32)
    dr = make_reconstructor(irf_delta, decay)

    with caplog.at_level(logging.INFO, logger="pyfli"):
        dr.reconstruct({"tau_map": tau}, "mono-exponential", log_summary=False)

    assert not any("reduced chi2" in r.message for r in caplog.records)


def test_biexponential_sdf_uses_per_tau_normalization(irf_delta):
    """Regression test for the 1/tau mixture-weight bug: sdf must match
    forward_model.decay_kernel's gate-integrated convention, where each component
    has unit area, a1*[exp(-t/tau1) - exp(-(t+dt)/tau1)] +
    (1-a1)*[exp(-t/tau2) - exp(-(t+dt)/tau2)], so a1 is the photon fraction of
    component 1 -- not the peak-height weighting a1*exp(...) + (1-a1)*exp(...)."""
    H, W, bins = 2, 2, irf_delta.shape[-1]
    tau1 = np.full((H, W), 0.4, dtype=np.float32)
    tau2 = np.full((H, W), 1.6, dtype=np.float32)
    alpha1 = np.full((H, W), 0.3, dtype=np.float32)

    dr = make_reconstructor(irf_delta)
    out = dr.reconstruct(
        {"tau1_map": tau1, "tau2_map": tau2, "alpha1_map": alpha1}, "bi-exponential"
    )
    sdf = out["results"]["TR_maps"]["sdf_map"]

    t = np.linspace(0, 1000.0 / 80.0, bins, endpoint=False)
    dt = t[1] - t[0]

    def gate(tau):
        return np.exp(-t / tau) - np.exp(-(t + dt) / tau)

    expected = 0.3 * gate(0.4) + 0.7 * gate(1.6)
    np.testing.assert_allclose(sdf[0, 0], expected, rtol=1e-4)


def test_photon_count_map_scales_convolved_fit_to_decay(irf_delta):
    """photon_count_map (S) must be exactly the factor that scales the
    unit-amplitude convolved_map up to fit_map: S * convolved_map == fit_map,
    and fit_map's total must match binned_decay's total at every pixel."""
    H, W, bins = 2, 2, irf_delta.shape[-1]
    tau = np.full((H, W), 1.0, dtype=np.float32)
    decay = np.full((H, W, bins), 2.0, dtype=np.float32)

    dr = make_reconstructor(irf_delta, decay)
    out = dr.reconstruct({"tau_map": tau}, "mono-exponential")
    maps = out["results"]["maps"]
    tr_maps = out["results"]["TR_maps"]

    S = maps["photon_count_map"]
    reconstructed = S[..., np.newaxis] * tr_maps["convolved_map"]
    np.testing.assert_allclose(reconstructed, tr_maps["fit_map"], rtol=1e-4)
    np.testing.assert_allclose(
        tr_maps["fit_map"].sum(axis=-1), decay.sum(axis=-1), rtol=1e-4
    )
    np.testing.assert_allclose(
        maps["rmse_map"],
        np.sqrt(np.mean(tr_maps["residual_map"] ** 2, axis=-1)),
        rtol=1e-4,
    )


def test_photon_count_map_scales_convolved_fit_to_decay_biexp(irf_delta):
    """Same invariant as the mono-exponential test, for the bi-exponential
    branch: S * convolved_map == fit_map, fit_map totals match decay totals."""
    H, W, bins = 2, 2, irf_delta.shape[-1]
    tau1 = np.full((H, W), 0.4, dtype=np.float32)
    tau2 = np.full((H, W), 1.6, dtype=np.float32)
    alpha1 = np.full((H, W), 0.3, dtype=np.float32)
    decay = np.full((H, W, bins), 3.0, dtype=np.float32)

    dr = make_reconstructor(irf_delta, decay)
    out = dr.reconstruct(
        {"tau1_map": tau1, "tau2_map": tau2, "alpha1_map": alpha1}, "bi-exponential"
    )
    maps = out["results"]["maps"]
    tr_maps = out["results"]["TR_maps"]

    S = maps["photon_count_map"]
    reconstructed = S[..., np.newaxis] * tr_maps["convolved_map"]
    np.testing.assert_allclose(reconstructed, tr_maps["fit_map"], rtol=1e-4)
    np.testing.assert_allclose(
        tr_maps["fit_map"].sum(axis=-1), decay.sum(axis=-1), rtol=1e-4
    )
    np.testing.assert_allclose(
        maps["rmse_map"],
        np.sqrt(np.mean(tr_maps["residual_map"] ** 2, axis=-1)),
        rtol=1e-4,
    )


def test_v_shift_map_defaults_to_zero_and_is_reported(irf_delta):
    """v_shift_map/h_shift_map are optional and default to 0.0 -- omitting
    them must reproduce the original (no-baseline) behavior exactly."""
    H, W, bins = 2, 2, irf_delta.shape[-1]
    tau = np.full((H, W), 1.0, dtype=np.float32)
    decay = np.full((H, W, bins), 2.0, dtype=np.float32)

    dr = make_reconstructor(irf_delta, decay)
    out = dr.reconstruct({"tau_map": tau}, "mono-exponential")
    maps = out["results"]["maps"]
    np.testing.assert_allclose(maps["v_shift_map"], np.zeros((H, W)))
    np.testing.assert_allclose(maps["h_shift_map"], np.zeros((H, W)))


def test_v_shift_map_baseline_is_additive_and_reported(irf_delta):
    """A nonzero v_shift_map must (a) be reported back unchanged, and (b)
    still leave fit_map's total matching decay's total (the baseline is
    subtracted before total-matching the peak, then added back)."""
    H, W, bins = 2, 2, irf_delta.shape[-1]
    tau = np.full((H, W), 1.0, dtype=np.float32)
    decay = np.full((H, W, bins), 5.0, dtype=np.float32)
    v_shift = np.full((H, W), 0.5, dtype=np.float32)

    dr = make_reconstructor(irf_delta, decay)
    out = dr.reconstruct({"tau_map": tau, "v_shift_map": v_shift}, "mono-exponential")
    maps = out["results"]["maps"]
    tr_maps = out["results"]["TR_maps"]

    np.testing.assert_allclose(maps["v_shift_map"], v_shift)
    np.testing.assert_allclose(
        tr_maps["fit_map"].sum(axis=-1), decay.sum(axis=-1), rtol=1e-4
    )
    # Every bin of the fit must be at least the baseline.
    assert np.all(tr_maps["fit_map"] >= v_shift[..., np.newaxis] - 1e-4)


def test_reconstruct_binned_decay_override(irf_delta):
    """Per-call binned_decay overrides the instance's default."""
    H, W, bins = 2, 2, irf_delta.shape[-1]
    tau = np.full((H, W), 1.0, dtype=np.float32)
    instance_decay = np.full((H, W, bins), 2.0, dtype=np.float32)
    override_decay = np.full((H, W, bins), 9.0, dtype=np.float32)

    dr = make_reconstructor(irf_delta, instance_decay)
    out = dr.reconstruct(
        {"tau_map": tau}, "mono-exponential", binned_decay=override_decay
    )
    np.testing.assert_allclose(
        out["results"]["TR_maps"]["fit_map"].sum(axis=-1),
        override_decay.sum(axis=-1),
        rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# collapse_to_mono()
# ---------------------------------------------------------------------------


def test_collapse_to_mono_picks_dominant_component_or_mean(irf_delta):
    """Per pixel: alpha1 above/below threshold -> the dominant component's
    tau; otherwise -> the amplitude-weighted mean (genuinely bi pixel);
    masked-out pixels are NaN in the returned reconstruction."""
    tau1_map = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    tau2_map = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    alpha1_map = np.array(
        [[0.99, 0.01], [0.5, 0.5]], dtype=np.float32
    )  # mono->tau1, mono->tau2, bi, bi (last one masked out)
    bool_mask = np.array([[True, True], [True, False]])

    dr = make_reconstructor(irf_delta)
    out = dr.collapse_to_mono(
        {"tau1_map": tau1_map, "tau2_map": tau2_map, "alpha1_map": alpha1_map},
        bool_mask,
        display=False,
    )
    tau_eff = out["results"]["maps"]["tau_map"]

    np.testing.assert_allclose(tau_eff[0, 0], 0.5, rtol=1e-4)  # dominant tau1
    np.testing.assert_allclose(tau_eff[0, 1], 2.0, rtol=1e-4)  # dominant tau2
    np.testing.assert_allclose(
        tau_eff[1, 0], 0.5 * 0.5 + 0.5 * 2.0, rtol=1e-4
    )  # bi -> amplitude-weighted mean
    assert np.isnan(tau_eff[1, 1])  # masked out


def test_collapse_to_mono_reconstructs_dominant_tau(irf_delta):
    H, W = 2, 2
    tau1_map = np.full((H, W), 0.4, dtype=np.float32)
    tau2_map = np.full((H, W), 1.6, dtype=np.float32)
    alpha1_map = np.full((H, W), 0.98, dtype=np.float32)  # dominant component-1
    bool_mask = np.ones((H, W), dtype=bool)

    dr = make_reconstructor(irf_delta)
    out = dr.collapse_to_mono(
        {"tau1_map": tau1_map, "tau2_map": tau2_map, "alpha1_map": alpha1_map},
        bool_mask,
        display=False,
    )
    np.testing.assert_allclose(out["results"]["maps"]["tau_map"], tau1_map, rtol=1e-4)


# ---------------------------------------------------------------------------
# split_mono_bi()
# ---------------------------------------------------------------------------


def test_split_mono_bi_partitions_pixels(irf_delta):
    """Mono-classified pixels are real (non-NaN) in the 'mono' result and NaN
    in the 'bi' result; bi-classified pixels are the reverse."""
    tau1_map = np.array([[0.5, 0.5]], dtype=np.float32)
    tau2_map = np.array([[2.0, 2.0]], dtype=np.float32)
    alpha1_map = np.array([[0.99, 0.5]], dtype=np.float32)  # pixel0 mono, pixel1 bi
    bool_mask = np.ones((1, 2), dtype=bool)

    dr = make_reconstructor(irf_delta)
    out = dr.split_mono_bi(
        {"tau1_map": tau1_map, "tau2_map": tau2_map, "alpha1_map": alpha1_map},
        bool_mask,
        display=False,
    )

    assert out["mono_mask"].tolist() == [[True, False]]
    assert out["bi_mask"].tolist() == [[False, True]]

    mono_tau = out["mono"]["results"]["maps"]["tau_map"]
    bi_alpha1 = out["bi"]["results"]["maps"]["alpha1_map"]

    assert not np.isnan(mono_tau[0, 0]) and np.isnan(mono_tau[0, 1])
    assert np.isnan(bi_alpha1[0, 0]) and not np.isnan(bi_alpha1[0, 1])
    np.testing.assert_allclose(mono_tau[0, 0], 0.5, rtol=1e-4)  # dominant tau1


def test_split_mono_bi_result_names(irf_delta):
    H, W = 1, 1
    tau1_map = np.full((H, W), 0.5, dtype=np.float32)
    tau2_map = np.full((H, W), 2.0, dtype=np.float32)
    alpha1_map = np.full((H, W), 0.99, dtype=np.float32)
    bool_mask = np.ones((H, W), dtype=bool)

    dr = make_reconstructor(irf_delta)
    out = dr.split_mono_bi(
        {"tau1_map": tau1_map, "tau2_map": tau2_map, "alpha1_map": alpha1_map},
        bool_mask,
        data_name="MyData",
        display=False,
    )
    assert out["mono"]["name"] == "MyData_mono"
    assert out["bi"]["name"] == "MyData_bi"
