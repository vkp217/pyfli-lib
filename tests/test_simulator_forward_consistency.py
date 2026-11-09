"""
Simulator ground truth matches the solver forward model (model_numpy) for every
simulator family, sensor and pile-up mode; Bayes reconstructions honour h/v shifts.
"""

import numpy as np
import pytest

from pyfli.bayes_utils.param_combinations import ParamSelector
from pyfli.cm_funcs.forward_model import model_numpy
from pyfli.simulator.calibration.data_io import _amplitude_to_intensity
from pyfli.simulator.irf_sim.irf_generator import IRFGenerator
from pyfli.simulator.sim_workflow import SIMULATOR_TYPES

N, PERIOD = 128, 12.5
DT = PERIOD / N
IRF = IRFGenerator.gaussianExpIRF(mu=12, T=PERIOD, num_bins=N, sigma=0.12, tau=0.08)
CFG = dict(
    tau2=(2.5, 0.3),
    photo_count=(5.0, 2.0),
    bit=16,
    n_cycles=(400_000, 400_001),
    dcr=0.3,
    laser_feq=80,
    read_noise_on=False,
    round_on=False,
    clip_on=False,
    seed=1,
)


def _truth_model(maps, irf, shift_method):
    """Forward model evaluated at the simulator's reported truth maps."""
    t = np.arange(N) * DT
    if "tau_map" in maps:
        keys, model = ("tau_map",), "mono-exponential"
    else:
        keys, model = ("alpha1_map", "tau1_map", "tau2_map"), "bi-exponential"
    p = [maps["photon_count_map"], *(maps[k] for k in keys)]
    p += [maps["v_shift_map"], maps["h_shift_map"]]
    return model_numpy(t, irf, p, model, shift_method)


@pytest.mark.parametrize("family", ["separate", "combined"])
@pytest.mark.parametrize("sensor", ["discrete", "continuous"])
@pytest.mark.parametrize("mode", ["wrap", "truncate"])
@pytest.mark.parametrize("mono_fraction", [0.0, 1.0])
def test_fit_map_is_forward_model_at_truth(family, sensor, mode, mono_fraction):
    """fit_map equals model_numpy at the truth maps; circular jitter is whole bins."""
    np.random.seed(0)
    sim = SIMULATOR_TYPES[family][sensor](
        IRF, pileup_mode=mode, mono_fraction=mono_fraction, **CFG
    )
    for _ in range(4):
        out = sim()
        maps = out["results"]["maps"]
        fit = out["results"]["TR_maps"]["fit_map"]
        sm = sim.engine.shift_method
        np.testing.assert_allclose(
            _truth_model(maps, sim.engine.irf, sm), fit, rtol=1e-10, atol=1e-9
        )
        assert maps["h_shift_map"] == pytest.approx(maps["h_shift_jit_map"] * DT)
        if sm == "circular":
            assert float(maps["h_shift_jit_map"]).is_integer()
            signal = fit - maps["v_shift_map"]
            assert signal.sum() == pytest.approx(maps["photon_count_map"])


@pytest.mark.parametrize("family", ["separate", "combined"])
def test_amplitude_maps_match_photon_fraction(family):
    """A1_map/A2_map are the pre-exponential amplitudes of alpha1_map."""
    np.random.seed(0)
    sim = SIMULATOR_TYPES[family]["discrete"](IRF, mono_fraction=0.0, **CFG)
    maps = sim()["results"]["maps"]
    a1 = _amplitude_to_intensity(maps["A1_map"], maps["tau1_map"], maps["tau2_map"])
    assert a1 == pytest.approx(maps["alpha1_map"])
    assert maps["A1_map"] + maps["A2_map"] == pytest.approx(1.0)


@pytest.mark.parametrize("mode,jitter", [("wrap", 2.0), ("truncate", -1.4)])
def test_photon_histogram_mean_is_forward_model(mode, jitter):
    """Mean TCSPC histogram with a fixed jitter converges to the expected decay."""
    sim = SIMULATOR_TYPES["separate"]["discrete"](
        IRF, pileup_mode=mode, mono_fraction=0.0, dcr_on=False, **CFG
    )
    sim.engine.sample_jitter = lambda max_bins=2: jitter
    p = sim.engine.sample_bi_params()
    sim.engine.sample_params = lambda: p
    reps = 30
    total = sum(sim()["raw_data"]["decay"] for _ in range(reps)) / reps
    expected = sim.engine.expected_decay(p, 0.01 * 400_000, jitter)
    z = (total - expected) / np.sqrt(np.maximum(expected, 1e-9) / reps)
    assert 0.6 < np.mean(z[expected > 1] ** 2) < 1.4


def test_analytical_decay_convolves_to_expected():
    """Un-convolved analytical decay circularly convolved with the IRF is the expectation."""
    eng = SIMULATOR_TYPES["combined"]["discrete"](IRF, mono_fraction=0.0, **CFG).engine
    p = eng.sample_all_params()
    an = eng.get_analytical_decay(p)
    conv = np.real(np.fft.ifft(np.fft.fft(an) * np.fft.fft(eng.irf)))
    np.testing.assert_allclose(conv, eng.expected_decay(p), atol=1e-12)


@pytest.mark.parametrize("backend", ParamSelector.BACKENDS)
def test_param_selector_uses_shifts(backend):
    """True h/v shifts give reduced chi2 near 1; omitting them does not."""
    np.random.seed(3)
    sim = SIMULATOR_TYPES["separate"]["discrete"](
        IRF, pileup_mode="wrap", mono_fraction=0.0, **{**CFG, "seed": None}
    )
    outs = [sim() for _ in range(4)]
    decay = np.stack([o["raw_data"]["decay"] for o in outs]).reshape(2, 2, N)

    def get(key):
        return np.array([o["results"]["maps"][key] for o in outs]).reshape(2, 2)

    combo = {
        "tau1": get("tau1_map")[..., None],
        "tau2": get("tau2_map")[..., None],
        "alpha1": get("alpha1_map")[..., None],
    }
    kw = dict(backend=backend, shift_method="circular")
    irf = IRF / IRF.sum()
    with_shifts = ParamSelector(
        80.0, irf, decay, h_shift=get("h_shift_map"), v_shift=get("v_shift_map"), **kw
    ).evaluate_all_samples(combo, progress=False)
    per_sample = ParamSelector(80.0, irf, decay, **kw).evaluate_all_samples(
        {
            **combo,
            "h_shift": get("h_shift_map")[..., None],
            "v_shift": get("v_shift_map")[..., None],
        },
        progress=False,
    )
    no_shifts = ParamSelector(80.0, irf, decay, **kw).evaluate_all_samples(
        combo, progress=False
    )
    assert np.mean(with_shifts["reduced_chi2_stack"]) < 1.5
    np.testing.assert_allclose(
        per_sample["reduced_chi2_stack"], with_shifts["reduced_chi2_stack"]
    )
    assert np.mean(no_shifts["reduced_chi2_stack"]) > np.mean(
        with_shifts["reduced_chi2_stack"]
    )
