"""
Tests for the shared periodic-model equations (pyfli.reconstruction.common_reconstruct:
gate_starts, circular_convolve, periodic_gate_integral) and their use across the solver,
reconstruction, simulator calibration and phasor modules (shift_method="circular").
"""

import numpy as np
import pytest
import torch

from pyfli.analysis.utils import circular_convolution_fft
from pyfli.cm_funcs.forward_model import (
    _gate_integral,
    decay_kernel,
    model_numpy,
    validate_shift_method,
)
from pyfli.irf_deconvolution.fli_solver import cyclic_conv
from pyfli.phasor.harmonic_offset import (
    HarmonicOffset,
    calibrate_phasors,
    h_shift_from_t0,
    harmonic_phasors,
    mono_offset,
)
from pyfli.phasor.phasorS import PhasorAnalyzer
from pyfli.reconstruction import ParamToDecay
from pyfli.reconstruction.common_reconstruct import (
    circular_convolve,
    circular_convolve_torch,
    gate_starts,
    periodic_bin_exponential,
    periodic_gate_integral,
)
from pyfli.simulator.calibration.expected_decay import gated_bin_probabilities
from pyfli.solver import BaseFLIFitter
from pyfli.solver.gpu_processor import FLIGPUProcessor
from pyfli.solver.mle_fitter import MLEFLIFitter

N = 256
PERIOD = 12.5
DT = PERIOD / N
T = gate_starts(PERIOD, N)
IRF = np.exp(-0.5 * ((T - 1.5) / 0.12) ** 2)


class TestSharedEquations:
    def test_gate_starts_is_linspace_without_endpoint(self):
        np.testing.assert_allclose(T, np.linspace(0, PERIOD, N, endpoint=False))

    def test_circular_convolve_matches_definition(self):
        rng = np.random.default_rng(0)
        x, h = rng.random((2, 32)), rng.random(32)
        n = np.arange(32)
        brute = np.array(
            [[np.sum(xi * h[(k - n) % 32]) for k in range(32)] for xi in x]
        )
        np.testing.assert_allclose(circular_convolve(x, h), brute, atol=1e-12)
        np.testing.assert_allclose(
            circular_convolve_torch(torch.tensor(x), torch.tensor(h)).numpy(),
            brute,
            atol=1e-12,
        )
        with pytest.raises(ValueError):
            circular_convolve(x, h[:-1])

    def test_phasors_factorise_under_circular_convolution(self):
        rng = np.random.default_rng(1)
        x, h = rng.random(N), rng.random(N)
        an = PhasorAnalyzer(80e6, T, 3, device="cpu")
        z = harmonic_phasors(an, circular_convolve(x, h))
        np.testing.assert_allclose(
            z, harmonic_phasors(an, x) * harmonic_phasors(an, h), atol=1e-12
        )

    def test_existing_circular_convolutions_route_through_shared_definition(self):
        rng = np.random.default_rng(2)
        x, h = rng.random((2, 3, N)), rng.random((2, 3, N))
        ref = np.real(
            np.fft.ifft(np.fft.fft(x, axis=2) * np.fft.fft(h, axis=2), axis=2)
        )
        np.testing.assert_allclose(circular_convolution_fft(x, h), ref, atol=1e-12)
        np.testing.assert_allclose(cyclic_conv(h, x), ref, atol=1e-12)

    @pytest.mark.parametrize(
        "tau,h", [(0.7, 0.0), (3.0, 0.3), (8.0, -0.4), (2.0, 13.1)]
    )
    def test_periodic_gate_integral_is_folded_linear_kernel(self, tau, h):
        m = np.arange(-40 * N, 40 * N)
        lin = _gate_integral(m * DT, (m + 1) * DT, tau, h)
        folded = np.zeros(N)
        np.add.at(folded, m % N, lin)
        periodic = periodic_gate_integral(T, T + DT, tau, h, PERIOD)
        np.testing.assert_allclose(periodic, folded, atol=1e-14)
        np.testing.assert_allclose(periodic.sum(), 1.0)

    def test_periodic_gate_integral_reduces_to_bin_exponential(self):
        np.testing.assert_allclose(
            periodic_gate_integral(T, T + DT, 2.0, 0.0, PERIOD),
            periodic_bin_exponential(2.0, N, PERIOD),
            atol=1e-15,
        )

    def test_periodic_gate_integral_is_periodic_and_continuous_in_onset(self):
        a = periodic_gate_integral(T, T + DT, 1.5, 0.3, PERIOD)
        b = periodic_gate_integral(T, T + DT, 1.5, 0.3 + PERIOD, PERIOD)
        np.testing.assert_allclose(a, b, atol=1e-14)
        edge = 5 * DT
        lo = periodic_gate_integral(T, T + DT, 1.5, edge - 1e-10, PERIOD)
        hi = periodic_gate_integral(T, T + DT, 1.5, edge + 1e-10, PERIOD)
        np.testing.assert_allclose(lo, hi, atol=1e-8)

    def test_torch_periodic_gate_integral_matches_and_has_gradient(self):
        h = torch.tensor(0.21, dtype=torch.float64, requires_grad=True)
        tt = torch.tensor(T)
        tau = torch.tensor(2.0, dtype=torch.float64)
        out = periodic_gate_integral(tt, tt + DT, tau, h, PERIOD, expm1=torch.expm1)
        np.testing.assert_allclose(
            out.detach().numpy(), periodic_gate_integral(T, T + DT, 2.0, 0.21, PERIOD)
        )
        (out * tt).sum().backward()
        assert torch.isfinite(h.grad) and h.grad != 0


class TestCircularForwardModel:
    def test_validate_shift_method(self):
        assert validate_shift_method("circular", 12.5, 12.5) == "circular"
        with pytest.raises(ValueError):
            validate_shift_method("wrap")
        with pytest.raises(ValueError):
            validate_shift_method("circular", 12.5, 25.0)

    def test_matches_simulator_expected_decay(self):
        m = model_numpy(T, IRF, [1.0, 2.0, 0.0, 0.0], "mono-exponential", "circular")
        np.testing.assert_allclose(
            m, gated_bin_probabilities(2.0, IRF, PERIOD), atol=1e-15
        )

    @pytest.mark.parametrize(
        "params,model",
        [
            ([5e4, 1.2, 3.0, -0.25], "mono-exponential"),
            ([5e4, 0.3, 0.6, 6.0, 2.0, 0.4], "bi-exponential"),
        ],
    )
    def test_counts_background_and_time_origin(self, params, model):
        m = model_numpy(T, IRF, params, model, "circular")
        np.testing.assert_allclose(m.sum(), params[0] + N * params[-2], rtol=1e-12)
        np.testing.assert_allclose(
            model_numpy(T + 3.0, IRF, params, model, "circular"), m
        )

    def test_agrees_with_zero_pad_when_tails_vanish_and_differs_otherwise(self):
        short = [1e4, 0.5, 0.0, 0.2]
        np.testing.assert_allclose(
            model_numpy(T, IRF, short, "mono-exponential", "circular"),
            model_numpy(T, IRF, short, "mono-exponential"),
            atol=1e-6,
        )
        long = [1e4, 6.0, 0.0, 0.2]
        circ = model_numpy(T, IRF, long, "mono-exponential", "circular")
        lin = model_numpy(T, IRF, long, "mono-exponential")
        assert circ[0] > lin[0] + 10

    def test_decay_kernel_circular(self):
        k, v = decay_kernel(T, [1.0, 2.0, 0.5], "mono-exponential", 0.0, "circular")
        np.testing.assert_allclose(k, periodic_bin_exponential(2.0, N, PERIOD))
        assert v == 0.5

    def test_gpu_kernel_matches_numpy(self):
        gpu = FLIGPUProcessor([80.0, 80.0], device="cpu")
        gpu.shift_method = "circular"
        p = torch.tensor([[1e5, 0.4, 1.0, 4.0, 3.0, -0.2]], dtype=torch.float64)
        out = gpu._model_kernel(
            p, torch.tensor(T), torch.tensor(IRF[None]), "bi-exponential"
        )
        np.testing.assert_allclose(
            out.numpy()[0],
            model_numpy(T, IRF, p[0].numpy(), "bi-exponential", "circular"),
            rtol=1e-10,
        )

    def test_fitter_validates_and_recovers_parameters(self):
        with pytest.raises(ValueError):
            BaseFLIFitter([40.0, 80.0], IRF, IRF, shift_method="circular")
        truth = [2e5, 5.0, 2.0, 0.3]
        mu = model_numpy(T, IRF, truth, "mono-exponential", "circular")
        exact = BaseFLIFitter([80.0, 80.0], mu, IRF, shift_method="circular")
        popt = exact.fit_with_estimator("least_squares", "mono-exponential")[0]
        np.testing.assert_allclose(popt, truth, rtol=1e-3)
        zero_pad = BaseFLIFitter([80.0, 80.0], mu, IRF)
        biased = zero_pad.fit_with_estimator("least_squares", "mono-exponential")[0]
        assert abs(biased[1] - truth[1]) > 0.05
        y = np.random.default_rng(3).poisson(mu).astype(float)
        fitter = MLEFLIFitter([80.0, 80.0], y, IRF, shift_method="circular")
        popt = fitter.fit_with_estimator("poisson", model_type="mono-exponential")[0]
        assert len(popt) == 4
        np.testing.assert_allclose(popt[1], truth[1], rtol=0.03)
        np.testing.assert_allclose(popt[-1], truth[3], atol=0.01)

    def test_param_to_decay_circular_matches_model_numpy(self):
        rng = np.random.default_rng(4)
        tau = rng.uniform(0.5, 8.0, (3, 4))
        h = rng.uniform(-0.4, 0.6, (3, 4))
        params = {
            "tau_map": tau,
            "photon_count_map": np.full((3, 4), 1e4),
            "v_shift_map": np.full((3, 4), 1.5),
            "h_shift_map": h,
        }
        recon = ParamToDecay("mono-exponential", 80.0, IRF, shift_method="circular")
        fit = recon.reconstruct(params, verbose=False)["fit_map"]
        fast = recon.reconstruct_vectorized(params)["fit_map"]
        expected = np.array(
            [
                [
                    model_numpy(
                        T,
                        IRF,
                        [1e4, tau[i, j], 1.5, h[i, j]],
                        "mono-exponential",
                        "circular",
                    )
                    for j in range(4)
                ]
                for i in range(3)
            ]
        )
        np.testing.assert_allclose(fit, expected, rtol=1e-5)
        np.testing.assert_allclose(fast, expected, rtol=1e-5)


class TestPhasorConsistency:
    @pytest.mark.parametrize("h", [-0.3, 0.0, 0.137])
    @pytest.mark.parametrize("tau", [0.5, 4.0, 10.0])
    def test_harmonic_t0_is_h_shift_minus_half_gate(self, h, tau):
        an = PhasorAnalyzer(80e6, T, 3, device="cpu")
        d = model_numpy(T, IRF, [1e5, tau, 0.0, h], "mono-exponential", "circular")
        z = calibrate_phasors(harmonic_phasors(an, d), harmonic_phasors(an, IRF))
        t0_k, tau_k = mono_offset(z, 80e6)
        np.testing.assert_allclose(h_shift_from_t0(t0_k, T), h, atol=1e-3)
        np.testing.assert_allclose(tau_k, tau, rtol=1e-3)

    def test_harmonic_offset_rejects_endpoint_time_axis(self):
        bad = np.linspace(0, PERIOD, N)
        with pytest.raises(ValueError):
            HarmonicOffset(PhasorAnalyzer(80e6, bad, 3, device="cpu"))
        HarmonicOffset(PhasorAnalyzer(80e6, T + 1.0, 3, device="cpu"))


class TestProcessors:
    @pytest.fixture
    def cube(self):
        rng = np.random.default_rng(7)
        mu = model_numpy(T, IRF, [1e5, 3.0, 1.0, 0.2], "mono-exponential", "circular")
        return rng.poisson(np.broadcast_to(mu, (2, 2, N))).astype(
            float
        ), np.broadcast_to(IRF, (2, 2, N)).copy()

    def test_cpu_processor_records_and_validates(self, cube):
        from pyfli.solver import FLICPUProcessor

        decay, irf = cube
        proc = FLICPUProcessor([80.0, 80.0], BaseFLIFitter)
        out = proc.process_image(
            decay, irf, model_type="mono-exponential", n_jobs=1, shift_method="circular"
        )
        assert out["shift_method"] == "circular"
        np.testing.assert_allclose(out["results"]["maps"]["tau_map"], 3.0, rtol=0.05)
        with pytest.raises(ValueError):
            FLICPUProcessor([40.0, 80.0], BaseFLIFitter).process_image(
                decay, irf, n_jobs=1, shift_method="circular"
            )

    def test_gpu_processor_records_and_validates(self, cube):
        decay, irf = cube
        gpu = FLIGPUProcessor([80.0, 80.0], device="cpu")
        out = gpu.fit_image(
            decay,
            irf,
            model_type="mono-exponential",
            max_iter=50,
            shift_method="circular",
        )
        assert out["shift_method"] == "circular"
        with pytest.raises(ValueError):
            gpu.fit_image(decay, irf, shift_method="wrap")


class TestSharedWeightedLeastSquares:
    def test_fourier_shift_matches_irf_aligner_and_roll(self):
        from scipy.fft import fft, fftfreq, ifft

        from pyfli.reconstruction.common_reconstruct import fourier_shift

        rng = np.random.default_rng(8)
        x, s = rng.random((3, 4, 64)), rng.uniform(-3, 3, (3, 4))
        ref = np.real(
            ifft(
                fft(x, axis=2) * np.exp(-2j * np.pi * fftfreq(64) * s[..., None]),
                axis=2,
            )
        )
        np.testing.assert_allclose(fourier_shift(x, s), ref, atol=1e-12)
        np.testing.assert_allclose(
            fourier_shift(x[0, 0], 5), np.roll(x[0, 0], 5), atol=1e-12
        )

    def test_residual_variance_numpy_and_torch(self):
        from pyfli.reconstruction.common_reconstruct import residual_variance

        mu, d = np.array([0.2, 3.0]), np.array([0.0, 5.0])
        np.testing.assert_allclose(residual_variance(mu, d, "irls"), [1.0, 3.0])
        np.testing.assert_allclose(residual_variance(mu, d, "neyman"), [1.0, 5.0])
        np.testing.assert_allclose(residual_variance(mu, d, "none"), [1.0, 1.0])
        out = residual_variance(torch.tensor(mu), torch.tensor(d), "irls", 0.5)
        np.testing.assert_allclose(out.numpy(), [0.5, 3.0])
        with pytest.raises(ValueError):
            residual_variance(mu, d, "pearson")

    def test_weighted_linear_lstsq_matches_reference(self):
        from pyfli.reconstruction.common_reconstruct import weighted_linear_lstsq

        rng = np.random.default_rng(9)
        V, d, v = rng.random((64, 6)), rng.random((10, 64)), rng.random((10, 64)) + 0.5
        ref = np.array(
            [
                np.linalg.lstsq(
                    V / np.sqrt(v[i])[:, None], d[i] / np.sqrt(v[i]), rcond=None
                )[0]
                for i in range(10)
            ]
        )
        np.testing.assert_allclose(weighted_linear_lstsq(V, d, v), ref, atol=1e-10)
        np.testing.assert_allclose(
            weighted_linear_lstsq(np.broadcast_to(V, (10, 64, 6)), d, v),
            ref,
            atol=1e-10,
        )
        pen = np.arange(1, 7.0) ** 2
        lam = 0.1 * np.mean(np.diag(V.T @ V))
        reg = np.linalg.solve(V.T @ V + lam * np.diag(pen), V.T @ d.T).T
        np.testing.assert_allclose(
            weighted_linear_lstsq(V, d, None, pen, 0.1), reg, atol=1e-10
        )

    def test_irls_solves_poisson_likelihood_equations(self):
        from pyfli.reconstruction.common_reconstruct import irls, weighted_linear_lstsq

        rng = np.random.default_rng(10)
        V = np.column_stack([np.exp(-np.arange(64) / 10.0), np.ones(64)]) * [200.0, 2.0]
        d = rng.poisson(V @ [1.0, 1.0], (5, 64)).astype(float)

        def solve(_, variance):
            return weighted_linear_lstsq(V, d, variance), None

        c, _ = irls(
            solve, lambda c: c @ V.T, solve(None, None)[0], tol=1e-12, max_rounds=100
        )
        mu = c @ V.T
        np.testing.assert_allclose(((d - mu) / mu) @ V, 0.0, atol=1e-6)

    def test_irls_per_row_convergence_reproduces_single_rows(self):
        from pyfli.reconstruction.common_reconstruct import irls, weighted_linear_lstsq

        rng = np.random.default_rng(11)
        V = np.column_stack([np.exp(-np.arange(64) / 8.0), np.ones(64)]) * [100.0, 1.0]
        d = rng.poisson(V @ [1.0, 2.0], (6, 64)).astype(float)

        def run(rows):
            def solve(_, variance):
                return weighted_linear_lstsq(V, rows, variance), None

            def change(old, new):
                return np.max(np.abs(new - old), -1) / np.max(np.abs(old), -1)

            return irls(solve, lambda c: c @ V.T, solve(None, None)[0], change=change)[
                0
            ]

        batch = run(d)
        single = np.vstack([run(d[i : i + 1]) for i in range(6)])
        np.testing.assert_allclose(batch, single, rtol=1e-12)


def test_decay_histogram_matches_solver_models():
    from pyfli.reconstruction.common_reconstruct import decay_histogram

    params = [1.0, 0.4, 0.5, 3.0, 0.0, 0.0]
    for periodic, method in [(True, "circular"), (False, "zero_pad")]:
        hist = decay_histogram([0.5, 3.0], [0.4, 0.6], IRF, PERIOD, periodic)
        np.testing.assert_allclose(
            hist, model_numpy(T, IRF, params, "bi-exponential", method), atol=1e-14
        )
    np.testing.assert_allclose(
        decay_histogram([2.0], [1.0], IRF, PERIOD),
        gated_bin_probabilities(2.0, IRF, PERIOD),
        atol=1e-15,
    )
    np.testing.assert_allclose(decay_histogram([2.0], [1.0], IRF, PERIOD).sum(), 1.0)
