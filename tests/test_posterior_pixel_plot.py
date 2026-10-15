"""Tests for pyfli.bayes_utils.posterior_pixel_plot."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyfli.bayes_utils.param_combinations import ParamSelector
from pyfli.bayes_utils.posterior_pixel_plot import (
    _MODEL_PARAM_KEYS,
    _reconstruct_sample_stack,
    _select_best_sample_idx,
    plot_pixel_posterior_fit,
)
from pyfli.solver.forward_model import model_numpy

_H, _W, _T, _NUM_SAMPLES = 3, 3, 64, 6
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
    """(H, W, T) decay cube; every pixel shares the same true bi-exponential
    params but gets independent Poisson noise, so pixels aren't identical."""
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
    """Posterior-sample bi-exponential params scattered around the truth --
    same (H, W, NUM_SAMPLES) / key convention BiPipeline.output_samples uses."""
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
# Cross-module compatibility contracts
# ---------------------------------------------------------------------------


def test_bipipeline_keys_match_posterior_plot_param_keys():
    """BiPipeline.MODEL_KEYS (what output_samples is keyed by) must exactly
    match posterior_pixel_plot's own registry, or output_samples from
    inference.py silently won't work as plot_pixel_posterior_fit's
    output_combination."""
    pytest.importorskip("keras")
    from pyfli.bayes_utils.inference import BiPipeline

    assert set(BiPipeline.MODEL_KEYS) == set(_MODEL_PARAM_KEYS)
    for model_type, keys in BiPipeline.MODEL_KEYS.items():
        assert set(keys) == set(_MODEL_PARAM_KEYS[model_type])


def test_output_samples_shaped_dict_is_accepted_directly(
    irf, decay, output_combination
):
    """A dict shaped exactly like BiPipeline.run_inference's `output_samples`
    return value (H, W, NUM_SAMPLES) per key) must work as-is."""
    fig, ax = plot_pixel_posterior_fit(
        output_combination, decay, irf, _FREQ_ACQ, pixel=(1, 1), center="median"
    )
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_model_type_raises(irf, decay, output_combination):
    with pytest.raises(ValueError, match="model_type"):
        plot_pixel_posterior_fit(
            output_combination,
            decay,
            irf,
            _FREQ_ACQ,
            pixel=(0, 0),
            model_type="tri-exponential",
        )


def test_unknown_center_raises(irf, decay, output_combination):
    with pytest.raises(ValueError, match="center"):
        plot_pixel_posterior_fit(
            output_combination, decay, irf, _FREQ_ACQ, pixel=(0, 0), center="mode"
        )


# ---------------------------------------------------------------------------
# Basic plotting behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("center", ["median", "mean", "best"])
def test_plot_smoke_bi_exponential(irf, decay, output_combination, center):
    fig, ax = plot_pixel_posterior_fit(
        output_combination, decay, irf, _FREQ_ACQ, pixel=(1, 1), center=center
    )
    assert fig is ax.figure
    # 2 CI bands + center curve + decay line = 4 legend entries by default.
    assert len(ax.get_legend().get_texts()) == 4
    lines = ax.get_lines()
    assert len(lines) == 2
    for line in lines:
        assert len(line.get_xdata()) == _T
    plt.close(fig)


def test_plot_smoke_mono_exponential(irf, decay, mono_output_combination):
    fig, ax = plot_pixel_posterior_fit(
        mono_output_combination,
        decay,
        irf,
        _FREQ_ACQ,
        pixel=(0, 0),
        model_type="mono-exponential",
        center="median",
    )
    assert fig is not None
    plt.close(fig)


def test_default_title_uses_pixel_coordinates(irf, decay, output_combination):
    fig, ax = plot_pixel_posterior_fit(
        output_combination, decay, irf, _FREQ_ACQ, pixel=(2, 1), center="mean"
    )
    assert ax.get_title() == "Pixel (2, 1)"
    plt.close(fig)


def test_custom_title(irf, decay, output_combination):
    fig, ax = plot_pixel_posterior_fit(
        output_combination, decay, irf, _FREQ_ACQ, pixel=(0, 0), title="Custom"
    )
    assert ax.get_title() == "Custom"
    plt.close(fig)


def test_accepts_existing_axes_and_does_not_create_a_second_figure(
    irf, decay, output_combination
):
    fig, ax = plt.subplots()
    n_before = len(plt.get_fignums())
    out_fig, out_ax = plot_pixel_posterior_fit(
        output_combination, decay, irf, _FREQ_ACQ, pixel=(0, 0), ax=ax
    )
    assert out_ax is ax
    assert out_fig is fig
    assert len(plt.get_fignums()) == n_before
    plt.close(fig)


def test_custom_ci_levels_change_band_count(irf, decay, output_combination):
    fig, ax = plot_pixel_posterior_fit(
        output_combination, decay, irf, _FREQ_ACQ, pixel=(0, 0), ci_levels=(50,)
    )
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels.count("50% credible interval") == 1
    assert not any("92%" in lbl or "68%" in lbl for lbl in labels)
    plt.close(fig)


@pytest.mark.parametrize("decay_alpha", [0.3, 1.0])
def test_decay_style_scatter_draws_points_with_alpha(
    irf, decay, output_combination, decay_alpha
):
    fig, ax = plot_pixel_posterior_fit(
        output_combination,
        decay,
        irf,
        _FREQ_ACQ,
        pixel=(1, 1),
        decay_style="scatter",
        decay_alpha=decay_alpha,
    )
    assert len(ax.get_lines()) == 1
    points = ax.collections[-1]
    np.testing.assert_allclose(points.get_offsets()[:, 0], np.arange(_T))
    np.testing.assert_allclose(points.get_offsets()[:, 1], decay[1, 1, :])
    assert points.get_alpha() == decay_alpha
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[-1] == "Decay (measured)" and len(labels) == 4
    plt.close(fig)


@pytest.mark.parametrize("decay_alpha", [0.25, 0.8])
def test_decay_style_line_default_uses_alpha(
    irf, decay, output_combination, decay_alpha
):
    fig, ax = plot_pixel_posterior_fit(
        output_combination, decay, irf, _FREQ_ACQ, pixel=(1, 1), decay_alpha=decay_alpha
    )
    decay_line = ax.get_lines()[1]
    np.testing.assert_allclose(decay_line.get_ydata(), decay[1, 1, :])
    assert decay_line.get_alpha() == decay_alpha
    plt.close(fig)


def test_unknown_decay_style_raises(irf, decay, output_combination):
    with pytest.raises(ValueError, match="decay_style"):
        plot_pixel_posterior_fit(
            output_combination, decay, irf, _FREQ_ACQ, pixel=(0, 0), decay_style="bars"
        )


# ---------------------------------------------------------------------------
# Numerical correctness / cross-module consistency
# ---------------------------------------------------------------------------


def test_reconstructed_samples_match_measured_photon_totals(
    irf, decay, output_combination
):
    """Every posterior sample's reconstructed curve must sum to the same
    total as the measured decay at that pixel -- the same
    rescale-to-measured-totals convention pyfli.reconstruction.compute_detailed_results
    uses (see ParamToDecay.rescale_fit_to_measured_totals)."""
    pixel = (1, 2)
    decay_px = decay[pixel].astype(np.float64)
    stack = _reconstruct_sample_stack(
        output_combination, pixel, irf, decay_px, _FREQ_ACQ, "bi-exponential"
    )
    assert stack.shape == (_NUM_SAMPLES, _T)
    np.testing.assert_allclose(stack.sum(axis=-1), decay_px.sum(), rtol=1e-4, atol=1e-2)


def test_credible_bands_are_properly_nested(irf, decay, output_combination):
    pixel = (0, 1)
    decay_px = decay[pixel].astype(np.float64)
    stack = _reconstruct_sample_stack(
        output_combination, pixel, irf, decay_px, _FREQ_ACQ, "bi-exponential"
    )
    lo_92, hi_92 = np.percentile(stack, 4, axis=0), np.percentile(stack, 96, axis=0)
    lo_68, hi_68 = np.percentile(stack, 16, axis=0), np.percentile(stack, 84, axis=0)
    assert np.all(lo_92 <= lo_68 + 1e-9)
    assert np.all(hi_68 <= hi_92 + 1e-9)


def test_median_center_lies_within_its_own_bands(irf, decay, output_combination):
    pixel = (1, 1)
    decay_px = decay[pixel].astype(np.float64)
    stack = _reconstruct_sample_stack(
        output_combination, pixel, irf, decay_px, _FREQ_ACQ, "bi-exponential"
    )
    median = np.median(stack, axis=0)
    lo_92, hi_92 = np.percentile(stack, 4, axis=0), np.percentile(stack, 96, axis=0)
    assert np.all(median >= lo_92 - 1e-6)
    assert np.all(median <= hi_92 + 1e-6)


def test_best_center_agrees_with_best_param_fit_selector(
    irf, decay, output_combination
):
    """center="best" must pick the same sample ParamSelector itself
    would pick, and its plotted curve must equal that sample's reconstructed
    curve -- posterior_pixel_plot's own ParamToDecay path
    and ParamSelector's compute_detailed_results path must agree."""
    pixel = (1, 1)
    metric = "reduced_chi2"

    expected_idx = _select_best_sample_idx(
        output_combination, pixel, irf, decay, _FREQ_ACQ, "bi-exponential", metric
    )

    # Cross-check independently via the public ParamSelector API too.
    selector = ParamSelector(_FREQ_ACQ, irf, decay, model_type="bi-exponential")
    stacks = selector.evaluate_all_samples(output_combination)
    selection = selector.select_best_combination(
        output_combination, stacks, metric=metric
    )
    assert expected_idx == int(selection["best_sample_idx"][pixel])

    decay_px = decay[pixel].astype(np.float64)
    stack = _reconstruct_sample_stack(
        output_combination, pixel, irf, decay_px, _FREQ_ACQ, "bi-exponential"
    )

    fig, ax = plot_pixel_posterior_fit(
        output_combination,
        decay,
        irf,
        _FREQ_ACQ,
        pixel=pixel,
        center="best",
        metric=metric,
    )
    center_line = ax.get_lines()[0]
    np.testing.assert_array_equal(center_line.get_ydata(), stack[expected_idx])
    plt.close(fig)


def test_shared_1d_irf_matches_equivalent_per_pixel_irf_cube(decay, output_combination):
    """A shared (T,) IRF and a (H, W, T) cube that broadcasts the same trace
    to every pixel must reconstruct identically at a given pixel."""
    shared_irf = _gaussian_irf()
    cube_irf = np.broadcast_to(shared_irf, (_H, _W, _T)).copy()
    pixel = (2, 0)
    decay_px = decay[pixel].astype(np.float64)

    stack_shared = _reconstruct_sample_stack(
        output_combination, pixel, shared_irf, decay_px, _FREQ_ACQ, "bi-exponential"
    )
    stack_cube = _reconstruct_sample_stack(
        output_combination, pixel, cube_irf, decay_px, _FREQ_ACQ, "bi-exponential"
    )
    np.testing.assert_allclose(stack_shared, stack_cube, rtol=1e-6, atol=1e-6)
