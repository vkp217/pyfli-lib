"""
Tests for pyfli.bayes_utils.param_combinations.ParamSelector.

All tests use purely synthetic numpy arrays; no file I/O or GPU required.
"""

import numpy as np
import pytest

from pyfli.bayes_utils.param_combinations import ParamSelector
from pyfli.cm_funcs.forward_model import model_numpy

_H, _W, _T, _NUM_SAMPLES = 3, 3, 64, 5
_FREQ_ACQ = 80.0
_TAU1_TRUE, _TAU2_TRUE, _ALPHA1_TRUE, _S_TRUE = 1.0, 3.0, 0.6, 2000.0


def _gaussian_irf(n=_T, freq_acq=_FREQ_ACQ, center_ns=1.0, sigma_ns=0.2):
    t = np.linspace(0, 1000.0 / freq_acq, n, endpoint=False)
    irf = np.exp(-0.5 * ((t - center_ns) / sigma_ns) ** 2)
    return irf.astype(np.float32)


@pytest.fixture
def irf():
    return _gaussian_irf()


@pytest.fixture
def decay(irf, rng):
    t = np.linspace(0, 1000.0 / _FREQ_ACQ, _T, endpoint=False)
    clean = model_numpy(
        t,
        irf,
        [_S_TRUE, _ALPHA1_TRUE, _TAU1_TRUE, _TAU2_TRUE, 0.0, 0.0],
        "bi-exponential",
    )
    cube = np.stack(
        [
            rng.poisson(np.clip(clean, 0, None)).astype(np.float32)
            for _ in range(_H * _W)
        ]
    ).reshape(_H, _W, _T)
    return cube


@pytest.fixture
def output_combination(rng):
    tau1 = _TAU1_TRUE + 0.05 * rng.standard_normal((_H, _W, _NUM_SAMPLES))
    tau2 = _TAU2_TRUE + 0.15 * rng.standard_normal((_H, _W, _NUM_SAMPLES))
    alpha1 = np.clip(
        _ALPHA1_TRUE + 0.05 * rng.standard_normal((_H, _W, _NUM_SAMPLES)), 0.05, 0.95
    )
    return {
        "tau1": tau1.astype(np.float32),
        "tau2": tau2.astype(np.float32),
        "alpha1": alpha1.astype(np.float32),
    }


@pytest.fixture
def mono_output_combination(rng):
    tau = _TAU1_TRUE + 0.05 * rng.standard_normal((_H, _W, _NUM_SAMPLES))
    return {"tau": tau.astype(np.float32)}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_rejects_unknown_model_type(irf, decay):
    with pytest.raises(ValueError, match="model_type"):
        ParamSelector(_FREQ_ACQ, irf, decay, model_type="tri-exponential")


def test_rejects_unknown_backend(irf, decay):
    with pytest.raises(ValueError, match="backend"):
        ParamSelector(_FREQ_ACQ, irf, decay, backend="nope")


def test_bool_mask_defaults_to_none(irf, decay):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    assert selector.bool_mask is None


# ---------------------------------------------------------------------------
# evaluate_all_samples
# ---------------------------------------------------------------------------


def test_evaluate_all_samples_rejects_non_ndarray(irf, decay, output_combination):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    bad = dict(output_combination)
    bad["tau1"] = [1, 2, 3]  # not an ndarray
    with pytest.raises(TypeError):
        selector.evaluate_all_samples(bad)


def test_evaluate_all_samples_rejects_wrong_ndim(irf, decay, output_combination):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    bad = dict(output_combination)
    bad["tau1"] = bad["tau1"][..., 0]  # (H, W), not (H, W, NUM_SAMPLES)
    with pytest.raises(ValueError, match="expected 3D"):
        selector.evaluate_all_samples(bad)


def test_evaluate_all_samples_stack_shapes(irf, decay, output_combination):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    stacks = selector.evaluate_all_samples(output_combination)
    for key in ("chi2_stack", "reduced_chi2_stack", "rmse_stack", "r2_stack"):
        assert stacks[key].shape == (_H, _W, _NUM_SAMPLES)
    assert stacks["per_sample_results"] is None


def test_evaluate_all_samples_keep_per_sample_results(irf, decay, output_combination):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    stacks = selector.evaluate_all_samples(
        output_combination, keep_per_sample_results=True
    )
    assert len(stacks["per_sample_results"]) == _NUM_SAMPLES
    for result in stacks["per_sample_results"]:
        assert set(result["results"]) == {"maps", "error_maps", "TR_maps"}


def test_evaluate_all_samples_mono_exponential(irf, decay, mono_output_combination):
    selector = ParamSelector(_FREQ_ACQ, irf, decay, model_type="mono-exponential")
    stacks = selector.evaluate_all_samples(mono_output_combination)
    assert stacks["chi2_stack"].shape == (_H, _W, _NUM_SAMPLES)


# ---------------------------------------------------------------------------
# select_best_combination
# ---------------------------------------------------------------------------


def test_select_best_combination_rejects_unknown_metric(irf, decay, output_combination):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    stacks = selector.evaluate_all_samples(output_combination)
    with pytest.raises(ValueError, match="metric"):
        selector.select_best_combination(
            output_combination, stacks, metric="not_a_metric"
        )


@pytest.mark.parametrize(
    "metric,arg_extremum",
    [
        ("chi2", np.argmin),
        ("reduced_chi2", np.argmin),
        ("RMSE", np.argmin),
        ("R2", np.argmax),
    ],
)
def test_select_best_combination_picks_correct_extremum(
    irf, decay, output_combination, metric, arg_extremum
):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    stacks = selector.evaluate_all_samples(output_combination)
    selection = selector.select_best_combination(
        output_combination, stacks, metric=metric
    )

    stack_key = selector.METRICS[metric][0]
    expected_idx = arg_extremum(stacks[stack_key], axis=-1)
    np.testing.assert_array_equal(selection["best_sample_idx"], expected_idx)

    # best_params must be the per-pixel sample selected by best_sample_idx.
    for key, arr in output_combination.items():
        expected = np.take_along_axis(
            arr, expected_idx[..., np.newaxis], axis=-1
        ).squeeze(-1)
        np.testing.assert_array_equal(selection["best_params"][key], expected)


# ---------------------------------------------------------------------------
# compute_best_model_fit_result
# ---------------------------------------------------------------------------


def test_compute_best_model_fit_result_structure(irf, decay, output_combination):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    result = selector.compute_best_model_fit_result(
        output_combination, metric="reduced_chi2"
    )

    assert set(result) >= {"name", "method", "results", "sample_selection"}
    assert "best_sample_idx_map" in result["results"]["maps"]
    assert "reduced_chi2_selection_map" in result["results"]["maps"]
    assert result["results"]["maps"]["best_sample_idx_map"].shape == (_H, _W)
    assert set(result["sample_selection"]) == {
        "metric",
        "best_params",
        "best_sample_idx",
        "best_score",
        "stacks",
    }


def test_compute_best_model_fit_result_precomputed_stacks_matches_internal(
    irf, decay, output_combination
):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    stacks = selector.evaluate_all_samples(output_combination)

    via_internal = selector.compute_best_model_fit_result(
        output_combination, metric="reduced_chi2"
    )
    via_stacks = selector.compute_best_model_fit_result(
        output_combination, metric="reduced_chi2", stacks=stacks
    )

    for key in via_internal["results"]["maps"]:
        np.testing.assert_array_equal(
            via_internal["results"]["maps"][key],
            via_stacks["results"]["maps"][key],
        )
    np.testing.assert_array_equal(
        via_internal["results"]["TR_maps"]["fit_map"],
        via_stacks["results"]["TR_maps"]["fit_map"],
    )


def test_compute_best_model_fit_result_precomputed_stacks_skips_evaluation(
    irf, decay, output_combination, monkeypatch
):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    stacks = selector.evaluate_all_samples(output_combination)

    def _boom(*_args, **_kwargs):
        raise AssertionError("evaluate_all_samples should not be re-run")

    monkeypatch.setattr(selector, "evaluate_all_samples", _boom)
    result = selector.compute_best_model_fit_result(
        output_combination, metric="RMSE", stacks=stacks
    )
    assert result["sample_selection"]["stacks"] is stacks


def test_compute_aggregate_model_fit_result_best_forwards_stacks(
    irf, decay, output_combination, monkeypatch
):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    stacks = selector.evaluate_all_samples(output_combination)

    monkeypatch.setattr(
        selector,
        "evaluate_all_samples",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("stacks not forwarded through compute_aggregate")
        ),
    )
    result = selector.compute_aggregate_model_fit_result(
        output_combination, method="best", metric="RMSE", stacks=stacks
    )
    assert result["sample_selection"]["stacks"] is stacks


def test_compute_best_model_fit_result_no_mask_introduces_no_nans(
    irf, decay, output_combination
):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    result = selector.compute_best_model_fit_result(
        output_combination, metric="reduced_chi2"
    )
    for arr in result["results"]["maps"].values():
        assert not np.any(np.isnan(arr))


def test_compute_best_model_fit_result_constructor_bool_mask_nans_excluded_pixels(
    irf, decay, output_combination
):
    mask = np.ones((_H, _W), dtype=bool)
    mask[0, 0] = False
    selector = ParamSelector(_FREQ_ACQ, irf, decay, bool_mask=mask)
    result = selector.compute_best_model_fit_result(
        output_combination, metric="reduced_chi2"
    )

    maps = result["results"]["maps"]
    assert np.isnan(maps["pixel_health_map"][0, 0])
    assert np.isnan(maps["best_sample_idx_map"][0, 0])
    assert not np.isnan(maps["pixel_health_map"][1, 1])
    assert not np.isnan(maps["best_sample_idx_map"][1, 1])

    tr = result["results"]["TR_maps"]
    assert np.all(np.isnan(tr["fit_map"][0, 0]))
    assert not np.any(np.isnan(tr["fit_map"][1, 1]))

    # error_maps is (H, W, n_params) -- also masked.
    assert np.all(np.isnan(result["results"]["error_maps"][0, 0]))

    # sample_selection carries the raw (unmasked) diagnostics.
    assert not np.isnan(result["sample_selection"]["best_sample_idx"][0, 0])


def test_compute_best_model_fit_result_per_call_bool_mask_overrides_constructor(
    irf, decay, output_combination
):
    ctor_mask = np.ones((_H, _W), dtype=bool)
    ctor_mask[0, 0] = False
    call_mask = np.ones((_H, _W), dtype=bool)
    call_mask[2, 2] = False

    selector = ParamSelector(_FREQ_ACQ, irf, decay, bool_mask=ctor_mask)
    result = selector.compute_best_model_fit_result(
        output_combination, metric="reduced_chi2", bool_mask=call_mask
    )
    maps = result["results"]["maps"]
    assert not np.isnan(maps["pixel_health_map"][0, 0])  # ctor mask ignored
    assert np.isnan(maps["pixel_health_map"][2, 2])  # call mask honored


# ---------------------------------------------------------------------------
# compute_aggregate_model_fit_result
# ---------------------------------------------------------------------------


def test_compute_aggregate_model_fit_result_rejects_unknown_method(
    irf, decay, output_combination
):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    with pytest.raises(ValueError, match="method"):
        selector.compute_aggregate_model_fit_result(output_combination, method="mode")


def test_compute_aggregate_model_fit_result_best_matches_compute_best_model_fit_result(
    irf, decay, output_combination
):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    via_aggregate = selector.compute_aggregate_model_fit_result(
        output_combination, method="best", metric="RMSE", data_name="x"
    )
    via_direct = selector.compute_best_model_fit_result(
        output_combination, metric="RMSE", data_name="x"
    )
    np.testing.assert_array_equal(
        via_aggregate["results"]["maps"]["best_sample_idx_map"],
        via_direct["results"]["maps"]["best_sample_idx_map"],
    )


@pytest.mark.parametrize("method", ["mean", "median"])
def test_compute_aggregate_model_fit_result_reducer_matches_manual_reduction(
    irf, decay, output_combination, method
):
    selector = ParamSelector(_FREQ_ACQ, irf, decay)
    result = selector.compute_aggregate_model_fit_result(
        output_combination, method=method
    )

    reducer = {"mean": np.mean, "median": np.median}[method]
    expected_tau1 = reducer(output_combination["tau1"], axis=-1).astype(np.float32)
    np.testing.assert_allclose(
        result["results"]["maps"]["tau1_map"], expected_tau1, rtol=1e-5
    )
    assert "sample_selection" not in result


def test_compute_aggregate_model_fit_result_reducer_applies_bool_mask(
    irf, decay, output_combination
):
    mask = np.ones((_H, _W), dtype=bool)
    mask[1, 2] = False
    selector = ParamSelector(_FREQ_ACQ, irf, decay, bool_mask=mask)
    result = selector.compute_aggregate_model_fit_result(
        output_combination, method="mean"
    )
    assert np.isnan(result["results"]["maps"]["tau1_map"][1, 2])
    assert not np.isnan(result["results"]["maps"]["tau1_map"][0, 0])


# ---------------------------------------------------------------------------
# backend="reconstructor"
# ---------------------------------------------------------------------------


def test_reconstructor_backend_self_consistent(irf, decay, output_combination):
    selector = ParamSelector(_FREQ_ACQ, irf, decay, backend="reconstructor")
    result = selector.compute_best_model_fit_result(
        output_combination, metric="reduced_chi2"
    )

    assert set(result["results"]["maps"]) >= {
        "R2_map",
        "chi2_map",
        "reduced_chi2_map",
        "rmse_map",
        "best_sample_idx_map",
        "reduced_chi2_selection_map",
    }
    assert result["results"]["error_maps"] is None

    tr = result["results"]["TR_maps"]
    np.testing.assert_allclose(
        tr["fit_map"] + tr["residual_map"], decay, rtol=1e-4, atol=1e-2
    )


def test_reconstructor_backend_with_bool_mask_does_not_crash_on_none_error_maps(
    irf, decay, output_combination
):
    mask = np.ones((_H, _W), dtype=bool)
    mask[0, 0] = False
    selector = ParamSelector(
        _FREQ_ACQ, irf, decay, backend="reconstructor", bool_mask=mask
    )
    result = selector.compute_best_model_fit_result(
        output_combination, metric="reduced_chi2"
    )
    assert result["results"]["error_maps"] is None
    assert np.isnan(result["results"]["maps"]["chi2_map"][0, 0])
