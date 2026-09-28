"""
Tests for the gate-integrated forward model and the NLSF / MLE weighting options.

Covers: the amplitude ``S`` being the photon count, ``h_shift`` acting as a true,
continuous onset shift (zero before onset, earlier curve for negative shifts), CPU/GPU
model agreement, the ``weighting`` options of ``BaseFLIFitter.least_squares_fit`` and
``MLEFLIFitter.fit_with_estimator`` (including removal of the Neyman low-count bias),
and ``max_iter`` reaching the MLE optimizer.
"""

import numpy as np
import pytest
import torch

import pyfli.solver.mle_fitter as mle_module
from pyfli.solver.base_fitter import NLSF_WEIGHTINGS, BaseFLIFitter
from pyfli.solver.forward_model import decay_kernel, model_numpy
from pyfli.solver.gpu_processor import FLIGPUProcessor
from pyfli.solver.mle_fitter import MLE_WEIGHTINGS, MLEFLIFitter

_N = 256
_PERIOD = 12.5
_DT = _PERIOD / _N
_T = np.arange(_N) * _DT
_FREQ = [80.0, 80.0]


def _irf(center=0.8, sigma=0.08):
    irf = np.exp(-0.5 * ((_T - center) / sigma) ** 2)
    return irf / irf.sum()


def _poisson_decays(n_photons, tau, n_repeats, seed=0, v_shift=0.0):
    mu = model_numpy(_T, _irf(), [n_photons, tau, v_shift, 0.0], "mono-exponential")
    rng = np.random.default_rng(seed)
    return rng.poisson(mu, size=(n_repeats, _N)).astype(float)


def test_amplitude_is_photon_count():
    S, tau = 5000.0, 1.5
    model = model_numpy(_T, _irf(), [S, tau, 0.0, 0.0], "mono-exponential")
    inside = 1.0 - np.exp(-_PERIOD / tau)
    assert model.sum() == pytest.approx(S * inside, rel=1e-3)


def test_kernel_is_zero_before_onset_and_sums_to_S():
    h = 3.3 * _DT
    kernel, _ = decay_kernel(_T, (1000.0, 2.0, 0.0), "mono-exponential", h_shift=h)
    assert np.all(kernel[:3] == 0.0)
    assert kernel[3] > 0.0
    assert kernel.sum() == pytest.approx(1000.0 * (1 - np.exp(-(_PERIOD - h) / 2.0)))


def test_h_shift_is_continuous_across_gate_boundaries():
    params = [4000.0, 1.0, 0.0]
    for boundary in (0.0, 5 * _DT, -3 * _DT):
        below = model_numpy(_T, _irf(), [*params, boundary - 1e-7], "mono-exponential")
        above = model_numpy(_T, _irf(), [*params, boundary + 1e-7], "mono-exponential")
        assert np.max(np.abs(above - below)) < 1e-2


@pytest.mark.parametrize("gates", [-4, 3])
def test_whole_gate_shift_moves_curve(gates):
    base = model_numpy(_T, _irf(), [4000.0, 1.0, 0.0, 0.0], "mono-exponential")
    shifted = model_numpy(
        _T, _irf(), [4000.0, 1.0, 0.0, gates * _DT], "mono-exponential"
    )
    lo, hi = 10, _N - 10
    np.testing.assert_allclose(
        shifted[lo:hi], base[lo - gates : hi - gates], rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("model_type", ["mono-exponential", "bi-exponential"])
def test_gpu_model_matches_numpy_model(model_type):
    proc = FLIGPUProcessor(_FREQ, device="cpu")
    if model_type == "mono-exponential":
        params = np.array([[5000.0, 1.3, 2.0, -0.37], [800.0, 0.6, 0.0, 0.21]])
    else:
        params = np.array(
            [[5000.0, 0.4, 0.5, 2.5, 1.0, -0.2], [900.0, 0.7, 0.3, 1.8, 0.0, 0.3]]
        )
    irf = np.tile(_irf(), (len(params), 1))
    gpu = (
        proc._model_kernel(
            torch.tensor(params, dtype=torch.float64),
            torch.tensor(_T, dtype=torch.float64),
            torch.tensor(irf, dtype=torch.float64),
            model_type,
        )
        .cpu()
        .numpy()
    )
    cpu = np.stack([model_numpy(_T, _irf(), p, model_type) for p in params])
    np.testing.assert_allclose(gpu, cpu, rtol=1e-5, atol=1e-6)


def test_nlsf_rejects_unknown_weighting():
    decay = _poisson_decays(2000, 1.0, 1)[0]
    fitter = BaseFLIFitter(_FREQ, decay, _irf())
    with pytest.raises(ValueError):
        fitter.fit_with_estimator("least_squares", "mono-exponential", weighting="bad")
    assert NLSF_WEIGHTINGS == ("irls", "none", "neyman")


def test_nlsf_use_weights_is_deprecated_alias():
    decay = _poisson_decays(2000, 1.0, 1)[0]
    neyman = BaseFLIFitter(_FREQ, decay, _irf()).fit_with_estimator(
        "least_squares", "mono-exponential", weighting="neyman"
    )[0]
    with pytest.warns(DeprecationWarning):
        legacy = BaseFLIFitter(_FREQ, decay, _irf()).fit_with_estimator(
            "least_squares", "mono-exponential", use_weights=True
        )[0]
    np.testing.assert_allclose(legacy, neyman)


def test_irls_matches_poisson_mle():
    decay = _poisson_decays(3000, 1.2, 1, seed=3, v_shift=0.5)[0]
    irls = BaseFLIFitter(_FREQ, decay, _irf()).fit_with_estimator(
        "least_squares", "mono-exponential"
    )[0]
    mle = MLEFLIFitter(_FREQ, decay, _irf()).fit_with_estimator(
        "poisson", model_type="mono-exponential"
    )[0]
    assert irls[1] == pytest.approx(mle[1], rel=5e-3)
    assert irls[0] == pytest.approx(mle[0], rel=5e-3)


def test_irls_removes_neyman_low_count_bias():
    tau = 1.0
    decays = _poisson_decays(400, tau, 60, seed=11)
    fits = {w: [] for w in NLSF_WEIGHTINGS}
    for d in decays:
        for w in NLSF_WEIGHTINGS:
            fits[w].append(
                BaseFLIFitter(_FREQ, d, _irf()).fit_with_estimator(
                    "least_squares", "mono-exponential", weighting=w
                )[0][1]
            )
    bias = {w: np.mean(v) / tau - 1.0 for w, v in fits.items()}
    assert bias["neyman"] < -0.05
    assert abs(bias["irls"]) < 0.03
    assert abs(bias["irls"]) < abs(bias["neyman"]) / 3


def test_fitted_amplitude_is_photon_count():
    n_photons = 20000
    decay = _poisson_decays(n_photons, 1.0, 1, seed=5)[0]
    popt = BaseFLIFitter(_FREQ, decay, _irf()).fit_with_estimator(
        "least_squares", "mono-exponential"
    )[0]
    assert popt[0] == pytest.approx(n_photons, rel=0.03)


@pytest.mark.parametrize("weighting", MLE_WEIGHTINGS)
def test_mle_weightings_run_and_recover_tau(weighting):
    decay = _poisson_decays(20000, 1.5, 1, seed=7)[0]
    popt = MLEFLIFitter(_FREQ, decay, _irf()).fit_with_estimator(
        model_type="mono-exponential", weighting=weighting
    )[0]
    assert popt[1] == pytest.approx(1.5, rel=0.05)


def test_mle_rejects_unknown_weighting():
    decay = _poisson_decays(2000, 1.0, 1)[0]
    with pytest.raises(ValueError):
        MLEFLIFitter(_FREQ, decay, _irf()).fit_with_estimator(
            model_type="mono-exponential", weighting="irls"
        )


def test_mle_passes_max_iter_to_optimizer(monkeypatch):
    seen = {}
    real_minimize = mle_module.minimize

    def spy(*args, **kwargs):
        seen.update(kwargs["options"])
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(mle_module, "minimize", spy)
    decay = _poisson_decays(2000, 1.0, 1)[0]
    MLEFLIFitter(_FREQ, decay, _irf()).fit_with_estimator(
        model_type="mono-exponential", max_iter=123
    )
    assert seen["maxiter"] == 123


@pytest.mark.parametrize("weighting", ["irls", "none", "neyman"])
def test_gpu_nlsf_weighting_options(weighting):
    decays = _poisson_decays(5000, 1.2, 4, seed=2)
    cube = decays.reshape(2, 2, _N)
    irf_cube = np.tile(_irf(), (2, 2, 1))
    out = FLIGPUProcessor(_FREQ, device="cpu").fit_image(
        cube,
        irf_cube,
        mode="NLSF",
        model_type="mono-exponential",
        max_iter=400,
        weighting=weighting,
    )
    tau = out["results"]["maps"]["tau_map"]
    assert np.all(np.isfinite(tau))
    assert np.median(tau) == pytest.approx(1.2, rel=0.15)


def test_gpu_rejects_unknown_weighting():
    cube = _poisson_decays(1000, 1.0, 1).reshape(1, 1, _N)
    with pytest.raises(ValueError):
        FLIGPUProcessor(_FREQ, device="cpu").fit_image(
            cube,
            np.tile(_irf(), (1, 1, 1)),
            mode="NLSF",
            model_type="mono-exponential",
            weighting="bad",
        )


def test_expected_poisson_deviance_matches_exact_values():
    from pyfli.solver.shared_metrics import expected_poisson_deviance

    mu = np.array([0.1, 1.0, 5.0, 100.0])
    np.testing.assert_allclose(
        expected_poisson_deviance(mu), [0.4741, 1.1468, 1.0467, 1.0017], atol=2e-3
    )
    assert expected_poisson_deviance(np.array([0.0]))[0] == 0.0


@pytest.mark.parametrize(
    ("truth", "irf_center"),
    [([3000.0, 0.8, 0.02, 0.0], 0.8), ([20000.0, 2.0, 5.0, 0.0], 1.5)],
)
def test_reduced_chi2_averages_one_at_any_count_level(truth, irf_center):
    from pyfli.solver.shared_metrics import compute_fli_stats, compute_pearson_stats

    irf = _irf(center=irf_center)
    mu = model_numpy(_T, irf, truth, "mono-exponential")
    rng = np.random.default_rng(4)
    reduced, pearson = [], []
    for _ in range(300):
        d = rng.poisson(mu).astype(float)
        reduced.append(compute_fli_stats(mu, d, 0)[2])
        pearson.append(compute_pearson_stats(mu, d, 0)[1])
    assert np.mean(reduced) == pytest.approx(1.0, abs=0.03)
    if np.sum(mu < 1) > 50:
        assert np.mean(pearson) < 0.9


def test_cpu_processor_reports_deviance_and_pearson_maps():
    from pyfli.solver import FLICPUProcessor
    from pyfli.solver.shared_metrics import reduced_poisson_deviance

    cube = _poisson_decays(4000, 1.1, 4, seed=8).reshape(2, 2, _N)
    irf_cube = np.tile(_irf(), (2, 2, 1))
    out = FLICPUProcessor(_FREQ, BaseFLIFitter).process_image(
        cube,
        irf_cube,
        mask=np.ones((2, 2), bool),
        model_type="mono-exponential",
        n_jobs=1,
    )
    maps = out["results"]["maps"]
    fit = out["results"]["TR_maps"]["fit_map"]
    deviance, reduced = reduced_poisson_deviance(fit, cube, 4)
    np.testing.assert_allclose(maps["chi2_map"], deviance, rtol=1e-3)
    np.testing.assert_allclose(maps["reduced_chi2_map"], reduced, rtol=1e-3)
    assert {"pearson_chi2_map", "pearson_reduced_chi2_map"} <= set(maps)


def test_gpu_processor_reports_pearson_maps():
    cube = _poisson_decays(4000, 1.1, 4, seed=9).reshape(2, 2, _N)
    out = FLIGPUProcessor(_FREQ, device="cpu").fit_image(
        cube,
        np.tile(_irf(), (2, 2, 1)),
        mode="MLE",
        model_type="mono-exponential",
        max_iter=200,
    )
    maps = out["results"]["maps"]
    assert {"pearson_chi2_map", "pearson_reduced_chi2_map"} <= set(maps)
    assert np.all(np.isfinite(maps["reduced_chi2_map"]))


@pytest.mark.parametrize("weighting", ["irls", "none"])
def test_nlsf_uncertainty_matches_scatter(weighting):
    decays = _poisson_decays(20000, 1.0, 40, seed=12)
    taus, errs = [], []
    for d in decays:
        popt, perr = BaseFLIFitter(_FREQ, d, _irf()).fit_with_estimator(
            "least_squares", "mono-exponential", weighting=weighting
        )[:2]
        taus.append(popt[1])
        errs.append(perr[1])
    assert np.mean(errs) == pytest.approx(np.std(taus, ddof=1), rel=0.35)


def test_binned_fitter_least_squares_runs_gpu_nlsf(monkeypatch):
    from pyfli.solver import BinnedFLIFitter

    seen = {}
    proc = FLIGPUProcessor(_FREQ, device="cpu")
    real_fit_image = proc.fit_image

    def spy(*args, **kwargs):
        out = real_fit_image(*args, **kwargs)
        seen["method"] = out["method"]
        return out

    monkeypatch.setattr(proc, "fit_image", spy)
    cube = _poisson_decays(3000, 1.0, 4, seed=3).reshape(2, 2, _N)
    BinnedFLIFitter(proc, bin_radius=0).fit(
        cube,
        np.tile(_irf(), (2, 2, 1)),
        estimator="least_squares",
        model_type="mono-exponential",
        max_iter=50,
    )
    assert "NLSF" in seen["method"]
