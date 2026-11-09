"""
Tests for NoiseEngine — the simulator noise model collection.
"""

import numpy as np
import pytest

from pyfli.simulator.noise_models import NoiseEngine


@pytest.fixture
def clean_signal():
    rng = np.random.default_rng(0)
    return rng.uniform(10.0, 100.0, size=(256,))


@pytest.fixture
def clean_2d():
    rng = np.random.default_rng(1)
    return rng.uniform(5.0, 50.0, size=(8, 64))


# ─────────────────────────────────────────────────────────────────────────────
# Poisson noise
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyPoisson:
    def test_shape_preserved(self, clean_signal):
        out = NoiseEngine.apply_poisson(clean_signal)
        assert out.shape == clean_signal.shape

    def test_output_non_negative(self, clean_signal):
        out = NoiseEngine.apply_poisson(clean_signal)
        assert np.all(out >= 0)

    def test_dtype_float64(self, clean_signal):
        out = NoiseEngine.apply_poisson(clean_signal)
        assert out.dtype == np.float64

    def test_2d_shape_preserved(self, clean_2d):
        out = NoiseEngine.apply_poisson(clean_2d)
        assert out.shape == clean_2d.shape

    def test_mean_close_to_lambda(self, rng):
        # E[Poisson(λ)] = λ; with many draws the sample mean converges
        signal = np.full(10_000, 20.0)
        out = NoiseEngine.apply_poisson(signal)
        assert abs(out.mean() - 20.0) < 1.0  # within 5 % of mean


# ─────────────────────────────────────────────────────────────────────────────
# Dark Count Rate (DCR)
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyDCR:
    def test_shape_preserved(self, clean_signal):
        out = NoiseEngine.apply_dcr(clean_signal, dcr_level=0.5)
        assert out.shape == clean_signal.shape

    def test_output_at_least_as_large_as_input_on_average(self):
        # DCR only adds, so the mean output ≥ mean input
        sig = np.full(5_000, 10.0)
        out = NoiseEngine.apply_dcr(sig, dcr_level=1.0)
        assert out.mean() >= sig.mean() - 0.5  # slight slack for randomness

    def test_zero_dcr_equals_input(self, clean_signal):
        # dcr_level=0 → Poisson(0) → always 0 → output == input
        out = NoiseEngine.apply_dcr(clean_signal, dcr_level=0)
        np.testing.assert_array_equal(out, clean_signal)

    def test_2d_shape_preserved(self, clean_2d):
        out = NoiseEngine.apply_dcr(clean_2d)
        assert out.shape == clean_2d.shape


# ─────────────────────────────────────────────────────────────────────────────
# Read noise (Gaussian)
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyReadNoise:
    def test_shape_preserved(self, clean_signal):
        out = NoiseEngine.apply_read_noise(clean_signal, sigma_read=1.5)
        assert out.shape == clean_signal.shape

    def test_noise_mean_near_zero(self):
        sig = np.zeros(20_000)
        out = NoiseEngine.apply_read_noise(sig, sigma_read=2.0)
        assert abs(out.mean()) < 0.1  # near-zero mean

    def test_noise_std_close_to_sigma(self):
        sig = np.zeros(20_000)
        sigma = 3.0
        out = NoiseEngine.apply_read_noise(sig, sigma_read=sigma)
        assert abs(out.std() - sigma) < 0.1

    def test_2d_shape_preserved(self, clean_2d):
        out = NoiseEngine.apply_read_noise(clean_2d)
        assert out.shape == clean_2d.shape


# ─────────────────────────────────────────────────────────────────────────────
# Timing jitter
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyJitter:
    def test_length_preserved(self, clean_signal):
        out = NoiseEngine.apply_jitter(clean_signal, max_shift=5)
        assert len(out) == len(clean_signal)

    def test_zero_max_shift_returns_original(self, clean_signal):
        out = NoiseEngine.apply_jitter(clean_signal, max_shift=0)
        np.testing.assert_array_equal(out, clean_signal)

    def test_output_has_zeros_at_boundary_for_positive_shift(self):
        # Force a known positive shift by seeding
        np.random.seed(0)
        sig = np.ones(20)
        # Run several times; at least one should shift and pad with zeros
        has_zero = False
        for _ in range(30):
            out = NoiseEngine.apply_jitter(sig, max_shift=3)
            if out[0] == 0:
                has_zero = True
                break
        assert has_zero, (
            "Expected at least one positive-shift result with leading zeros"
        )

    def test_total_energy_conserved_modulo_shift(self, clean_signal):
        # Jitter moves photons; the count of non-zero bins changes, but
        # the sum of the non-zero slice stays close (boundary bins are zero).
        out = NoiseEngine.apply_jitter(clean_signal, max_shift=2)
        # Sum should be within the range of the original signal (up to boundary loss)
        assert out.sum() <= clean_signal.sum() + 1e-9


def test_simulator_v_shift_map_reports_dark_counts():
    import numpy as np

    from pyfli.simulator import FLIModelImageGenerator

    t = np.arange(256) * 12.5 / 256
    irf = np.exp(-0.5 * ((t - 1.5) / 0.1) ** 2)
    common = dict(
        laser_feq=80.0, photo_count=(2, 5), mono_fraction=1.0, jitter=False, seed=1
    )
    for sensor, method, dcr_on in [
        ("discrete", "PHOTON_COUNTER", True),
        ("discrete", "PHOTON_COUNTER", False),
        ("continuous", "continuous", True),
    ]:
        cfg = dict(
            common,
            sensor_type=sensor,
            dcr_on=dcr_on,
            dcr=0.08,
            bit=12,
            n_cycles=(1_000_000, 1_000_000),
        )
        maps = FLIModelImageGenerator(
            irf_data=irf,
            roi_mask=np.ones((2, 2), int),
            image_shape=(2, 2),
            roi_params=[cfg],
            method=method,
            verbose=False,
        ).generate_image()["results"]["maps"]
        expected = (
            (0.08 * (12 / 8.0) if sensor == "continuous" else 0.08) if dcr_on else 0.0
        )
        np.testing.assert_allclose(maps["v_shift_map"], expected)


def test_simulator_fit_map_includes_background():
    import numpy as np

    from pyfli.simulator import FLIModelImageGenerator

    t = np.arange(256) * 12.5 / 256
    irf = np.exp(-0.5 * ((t - 1.5) / 0.1) ** 2)
    cfg = dict(
        sensor_type="discrete",
        laser_feq=80.0,
        photo_count=(2, 5),
        mono_fraction=1.0,
        jitter=False,
        qe_on=False,
        dcr_on=True,
        dcr=0.5,
        n_cycles=(1_000_000, 1_000_000),
        seed=2,
    )
    out = FLIModelImageGenerator(
        irf_data=irf,
        roi_mask=np.ones((8, 8), int),
        image_shape=(8, 8),
        roi_params=[cfg],
        method="PHOTON_COUNTER",
        verbose=False,
    ).generate_image()
    fit = out["results"]["TR_maps"]["fit_map"]
    residual = out["results"]["TR_maps"]["residual_map"]
    assert np.all(fit.min(-1) >= 0.5 - 1e-6)
    assert abs(residual[..., -40:].mean()) < 0.1
    np.testing.assert_allclose(
        fit.sum(-1).mean(), out["raw_data"]["decay"].sum(-1).mean(), rtol=0.02
    )


def _mean_and_fit(sim_cls, cfg, n=400):
    import numpy as np

    t = np.arange(256) * 12.5 / 256
    irf = np.exp(-0.5 * ((t - 1.5) / 0.1) ** 2)
    sim = sim_cls(irf, **cfg)
    outs = [sim() for _ in range(n)]
    obs = np.mean([o["raw_data"]["decay"] for o in outs], axis=0)
    fit = np.mean([o["results"]["TR_maps"]["fit_map"] for o in outs], axis=0)
    return obs, fit, n


def test_photon_simulator_fit_map_is_expectation():
    import numpy as np

    from pyfli.simulator.separate.main_factory_gen import PhotonCountSimulator

    for mode in ("wrap", "truncate"):
        cfg = dict(
            laser_feq=80.0,
            mono_fraction=1.0,
            tau2_dist="beta",
            tau2_beta_range=(1e-9, 4.0),
            jitter=False,
            qe_on=False,
            dcr_on=True,
            dcr=0.2,
            n_cycles=(1_000_000, 1_000_000),
            pileup_mode=mode,
            seed=3,
        )
        obs, fit, n = _mean_and_fit(PhotonCountSimulator, cfg)
        z = (obs - fit) / np.sqrt(fit / n)
        assert np.abs(z).max() < 5, (mode, np.abs(z).max())
        assert abs(z.mean()) < 0.3


def test_iccd_simulator_fit_map_is_expectation():
    import numpy as np

    from pyfli.simulator.separate.main_factory_gen import ContinuousSimulator

    cfg = dict(
        laser_feq=80.0,
        mono_fraction=1.0,
        tau2_dist="beta",
        tau2_beta_range=(1e-9, 2.5),
        photo_count=(50, 50),
        bit=12,
        jitter=False,
        qe_on=False,
        dcr_on=True,
        dcr=2.0,
        poisson=True,
        read_noise_on=True,
        round_on=True,
        clip_on=False,
        seed=4,
    )
    np.random.seed(4)
    obs, fit, n = _mean_and_fit(ContinuousSimulator, cfg, n=600)
    assert abs((obs - fit).mean()) < 0.2
    assert np.abs(obs - fit).max() < 0.05 * fit.max()
