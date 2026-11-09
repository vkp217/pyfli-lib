"""
Forward-model options ``excitation`` and ``gate_width`` (pyfli.cm_funcs.forward_model):
exact mathematics, defaults unchanged, and consistency across solver, GPU, reconstruction,
Laguerre, simulator and phasor code.
"""

import itertools

import numpy as np
import pytest

from pyfli.cm_funcs.forward_model import (
    gate_fraction,
    gate_integrated_kernel,
    model_numpy,
    negative_lag_gates,
)
from pyfli.reconstruction.common_reconstruct import (
    gate_starts,
    periodic_bin_exponential,
    periodic_instrument_exponential,
)

PERIOD, N = 12.5, 128
DT = PERIOD / N
T_AXIS = gate_starts(PERIOD, N)
IRF = np.exp(-0.5 * ((T_AXIS - 2.0) / 0.12) ** 2) + 1e-6
TRUTH = [2e4, 0.35, 0.6, 3.0, 2.0, 0.37]


def _onset_average(a, b, tau, h, period):
    """Onset average of the bin-start fraction by Gauss-Legendre split at its kinks."""
    kinks = [h, h + DT, a, b] + (
        [a - PERIOD, b - PERIOD, a + PERIOD, b + PERIOD] if period else []
    )
    edges = np.unique(np.clip(kinks, h, h + DT))
    x, w = np.polynomial.legendre.leggauss(64)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        u = lo + (x + 1) / 2 * (hi - lo)
        vals = [gate_fraction(a, b, tau, ui, DT, period, "bin_start") for ui in u]
        total += np.sum(w / 2 * (hi - lo) * np.array(vals))
    return total / DT


@pytest.mark.parametrize("period", [None, PERIOD])
def test_uniform_is_exact_onset_average(period):
    """Uniform excitation equals the bin-start fraction averaged over the onset window."""
    rng = np.random.default_rng(0)
    for _ in range(60):
        tau = 10 ** rng.uniform(-2.5, 1.8)
        width = DT * rng.uniform(0.2, 9.0)
        h = rng.uniform(-3, 3) * DT
        a = rng.uniform(-6, N + 3) * DT
        got = float(
            gate_fraction(
                np.array(a), np.array(a + width), tau, h, DT, period, "uniform"
            )
        )
        assert got == pytest.approx(
            _onset_average(a, a + width, tau, h, period), abs=1e-12
        )


@pytest.mark.parametrize("excitation", ["bin_start", "uniform"])
@pytest.mark.parametrize("width", [None, 3 * DT])
def test_sum_rules(excitation, width):
    """Gates tiling a period (or all gates of a single decay) hold gate_width/dt photons."""
    ratio = 1.0 if width is None else width / DT
    for tau in (0.05, 0.7, 40.0):
        k = gate_integrated_kernel(
            T_AXIS,
            DT,
            (1.0, tau, 0.0),
            "mono-exponential",
            0.41 * DT,
            PERIOD,
            excitation,
            width,
        )[0]
        assert k.sum() == pytest.approx(ratio, abs=1e-12)
        lags = np.arange(-50, 40000) * DT
        k = gate_integrated_kernel(
            lags,
            DT,
            (1.0, tau, 0.0),
            "mono-exponential",
            0.41 * DT,
            None,
            excitation,
            width,
        )[0]
        assert k.sum() == pytest.approx(ratio, abs=1e-9)


def test_periodic_uniform_matches_instrument_kernel():
    """Periodic uniform kernel at onset 0 is periodic_instrument_exponential."""
    taus = np.array([0.05, 0.4, 2.5, 8.0])
    for tau, ref in zip(taus, periodic_instrument_exponential(taus, N, PERIOD)):
        k = gate_integrated_kernel(
            T_AXIS, DT, (1.0, tau, 0.0), "mono-exponential", 0.0, PERIOD, "uniform"
        )[0]
        np.testing.assert_allclose(k, ref, atol=1e-14)


def test_monte_carlo_photons():
    """Photons with uniform onset counted in overlapping gates follow the kernel."""
    rng = np.random.default_rng(3)
    n, tau, h, width = 1_000_000, 0.9, 0.3 * DT, 2.6 * DT
    t_ph = h + rng.random(n) * DT + rng.exponential(tau, n)
    starts = np.arange(-4, 60) * DT
    counts = np.array([np.sum((t_ph >= s) & (t_ph < s + width)) for s in starts])
    model = gate_fraction(starts, starts + width, tau, h, DT, None, "uniform") * n
    z = (counts - model) / np.sqrt(np.maximum(model, 1.0))
    assert np.mean(z**2) < 1.4


@pytest.mark.parametrize("period", [None, PERIOD])
def test_torch_values_and_gradients(period):
    """Torch evaluation equals NumPy and its autograd equals finite differences."""
    torch = pytest.importorskip("torch")
    a = torch.tensor(np.arange(-8, N) * DT, dtype=torch.float64)
    tau = torch.tensor([[0.7]], dtype=torch.float64, requires_grad=True)
    h = torch.tensor([[-0.4 * DT]], dtype=torch.float64, requires_grad=True)
    g = gate_fraction(a, a + 2.2 * DT, tau, h, DT, period, "uniform")
    ref = gate_fraction(
        a.numpy(), a.numpy() + 2.2 * DT, 0.7, -0.4 * DT, DT, period, "uniform"
    )
    np.testing.assert_allclose(g.detach().numpy()[0], ref, atol=1e-14)
    g.sum().backward()
    eps = 1e-6

    def total(tv, hv):
        return np.sum(
            gate_fraction(
                a.numpy(), a.numpy() + 2.2 * DT, tv, hv, DT, period, "uniform"
            )
        )

    assert tau.grad.item() == pytest.approx(
        (total(0.7 + eps, -0.4 * DT) - total(0.7 - eps, -0.4 * DT)) / (2 * eps),
        abs=1e-6,
    )
    assert h.grad.item() == pytest.approx(
        (total(0.7, -0.4 * DT + eps) - total(0.7, -0.4 * DT - eps)) / (2 * eps),
        abs=1e-6,
    )


def test_negative_lag_gates_and_validation():
    """Default lag count unchanged; wide gates add the lags they reach; bad options raise."""
    assert negative_lag_gates(-1.0, DT) == int(np.ceil(1.0 / DT))
    assert negative_lag_gates(0.5, DT) == 0
    assert negative_lag_gates(0.0, DT, 2.5 * DT) == 2
    assert negative_lag_gates(0.0, DT, DT) == negative_lag_gates(0.0, DT)
    with pytest.raises(ValueError):
        model_numpy(T_AXIS, IRF, TRUTH, "bi-exponential", excitation="gaussian")
    with pytest.raises(ValueError):
        model_numpy(T_AXIS, IRF, TRUTH, "bi-exponential", gate_width=-1.0)


@pytest.mark.parametrize("shift_method", ["zero_pad", "circular"])
@pytest.mark.parametrize(
    "excitation,width",
    [("uniform", None), ("uniform", 3.3 * DT), ("bin_start", 2.6 * DT)],
)
def test_gpu_and_reconstruction_match_model(shift_method, excitation, width):
    """GPU kernel and ParamToDecay (vectorized and loop) equal model_numpy."""
    torch = pytest.importorskip("torch")
    from pyfli.reconstruction import ParamToDecay
    from pyfli.solver import FLIGPUProcessor

    P = np.array(
        [[1e4, 0.4, 0.5, 2.8, 1.5, h] for h in (-1.7, -0.33 * DT, 0.41 * DT, 1.9)]
    )
    ref = np.stack(
        [
            model_numpy(
                T_AXIS, IRF, p, "bi-exponential", shift_method, excitation, width
            )
            for p in P
        ]
    )
    gpu = FLIGPUProcessor([80.0, 80.0])
    gpu.shift_method, gpu.excitation, gpu.gate_width = shift_method, excitation, width
    out = gpu._model_kernel(
        torch.tensor(P),
        torch.arange(N, dtype=torch.float64) * DT,
        torch.tensor(np.broadcast_to(IRF, (4, N)).copy()),
        "bi-exponential",
    ).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-9 * ref.max())
    keys = [
        "photon_count_map",
        "alpha1_map",
        "tau1_map",
        "tau2_map",
        "v_shift_map",
        "h_shift_map",
    ]
    maps = {k: P[:, i].reshape(1, 4) for i, k in enumerate(keys)}
    rec = ParamToDecay(
        "bi-exponential",
        80.0,
        irf=IRF,
        shift_method=shift_method,
        excitation=excitation,
        gate_width=width,
    )
    for fit in (
        rec.reconstruct_vectorized(maps, verbose=False),
        rec.reconstruct(maps, verbose=False),
    ):
        np.testing.assert_allclose(fit["fit_map"][0], ref, rtol=1e-6)


@pytest.mark.parametrize("shift_method", ["zero_pad", "circular"])
@pytest.mark.parametrize("width", [None, 2.6 * DT])
def test_mle_recovers_uniform_truth(shift_method, width):
    """MLE with the matching options recovers noise-free truth."""
    from pyfli.solver import MLEFLIFitter

    y = model_numpy(
        T_AXIS, IRF, TRUTH, "bi-exponential", shift_method, "uniform", width
    )
    fitter = MLEFLIFitter(
        [80.0, 80.0],
        y,
        IRF,
        shift_method=shift_method,
        excitation="uniform",
        gate_width=width,
    )
    np.testing.assert_allclose(
        fitter.fit_with_estimator("poisson", "bi-exponential")[0], TRUTH, rtol=1e-4
    )


@pytest.mark.parametrize("shift_method", ["zero_pad", "circular"])
@pytest.mark.parametrize(
    "excitation,width", [("uniform", None), ("bin_start", 2.5 * DT)]
)
@pytest.mark.parametrize("h", [0.0, 0.31 * DT])
def test_laguerre_recovers_truth(shift_method, excitation, width, h):
    """Laguerre with the matching options recovers noise-free lifetimes and fractions."""
    from pyfli.laguerre.laguerre_method import LaguerreFLI

    y = model_numpy(
        T_AXIS,
        IRF,
        [1e5, 0.5, 0.3, 4.0, 0.0, h],
        "bi-exponential",
        shift_method,
        excitation,
        width,
    )
    m = LaguerreFLI(
        2,
        n_laguerre=32,
        alpha=0.9,
        dt=DT,
        verbose=False,
        shift_method=shift_method,
        excitation=excitation,
        gate_width=width,
    )
    m.fit(y, IRF, h_shift=h or None)
    maps = m.get_parameters()["results"]["maps"]
    assert maps["alpha1_map"].item() == pytest.approx(0.5, rel=2e-3)
    assert maps["photon_count_map"].item() == pytest.approx(1e5, rel=2e-3)
    np.testing.assert_allclose(np.sort(m.taus_.ravel()), [0.3, 4.0], rtol=2e-3)


def test_simulator_options():
    """Uniform-excitation photons follow the expected decay; gated maps match the model."""
    from pyfli.simulator.sim_workflow import SIMULATOR_TYPES

    base = dict(
        tau2=(2.5, 0.3),
        photo_count=(5.0, 2.0),
        bit=16,
        n_cycles=(400_000, 400_001),
        dcr=0.3,
        laser_feq=80,
        read_noise_on=False,
        round_on=False,
        clip_on=False,
        mono_fraction=0.0,
    )
    sim = SIMULATOR_TYPES["separate"]["discrete"](
        IRF, dcr_on=False, excitation="uniform", **base
    )
    sim.engine.sample_jitter = lambda max_bins=2: 2.0
    p = sim.engine.sample_bi_params()
    sim.engine.sample_params = lambda: p
    mean = sum(sim()["raw_data"]["decay"] for _ in range(30)) / 30
    expected = sim.engine.expected_decay(p, 0.01 * 400_000, 2.0)
    z = (mean - expected) / np.sqrt(np.maximum(expected, 1e-9) / 30)
    assert 0.6 < np.mean(z[expected > 1] ** 2) < 1.4
    cont = SIMULATOR_TYPES["combined"]["continuous"](
        IRF, excitation="uniform", gate_width=2.4 * DT, **base
    )
    out = cont()
    m = out["results"]["maps"]
    params = [
        m["photon_count_map"],
        m["alpha1_map"],
        m["tau1_map"],
        m["tau2_map"],
        m["v_shift_map"],
        m["h_shift_map"],
    ]
    ref = model_numpy(
        T_AXIS,
        cont.engine.irf,
        params,
        "bi-exponential",
        "circular",
        "uniform",
        2.4 * DT,
    )
    np.testing.assert_allclose(out["results"]["TR_maps"]["fit_map"], ref, rtol=1e-10)
    with pytest.raises(ValueError):
        SIMULATOR_TYPES["separate"]["discrete"](IRF, gate_width=2 * DT, **base)()


def test_phasor_uses_forward_model():
    """Phasor kernels, reconstruction and gated loci agree with the forward model."""
    from pyfli.phasor.phasorS import PhasorAnalyzer
    from pyfli.phasor.phasorS.phasor_locus import MonoLocus

    pa = PhasorAnalyzer(frequency_hz=80e6, time_axis_ns=T_AXIS, n_harmonics=1)
    taus = np.array([0.2, 1.0, 3.0])
    np.testing.assert_allclose(
        pa.component_kernel(taus, "continuous"),
        periodic_instrument_exponential(taus, N, PERIOD),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        pa.component_kernel(taus, "bin_start"),
        periodic_bin_exponential(taus, N, PERIOD),
        atol=1e-15,
    )
    irf3 = np.broadcast_to(IRF, (1, 2, N)).copy()
    G, S = np.full((1, 2), 0.5), np.full((1, 2), 0.4)
    rec = pa.analyze_biexponential_and_reconstruct(
        G,
        S,
        irf3,
        0.5,
        3.0,
        plot=False,
        component_model="bin_start",
        h_shift=0.27,
        v_shift=2.0,
    )
    A1, A2 = pa.compute_fractions(
        G, S, 0.5, 3.0, plot_graph=False, component_model="bin_start"
    )
    total = A1[0, 0] + A2[0, 0]
    ref = model_numpy(
        T_AXIS,
        IRF,
        [total, A1[0, 0] / total, 0.5, 3.0, 2.0, 0.27],
        "bi-exponential",
        "circular",
    )
    np.testing.assert_allclose(rec[0, 0], ref, rtol=1e-6)
    loc = MonoLocus(80e6)
    for n_g, frac in ((16, 0.2), (8, 0.4)):
        k = np.stack(
            [
                gate_integrated_kernel(
                    gate_starts(PERIOD, n_g),
                    PERIOD / n_g,
                    (1.0, tau, 0.0),
                    "mono-exponential",
                    0.0,
                    PERIOD,
                    "bin_start",
                    frac * PERIOD,
                )[0]
                for tau in taus
            ]
        )
        z = np.sum(k * np.exp(2j * np.pi * np.arange(n_g) / n_g), -1) / k.sum(-1)
        g, s = loc._gated_n_gs(taus, frac, n_g)
        np.testing.assert_allclose(g + 1j * s, z, atol=1e-12)
