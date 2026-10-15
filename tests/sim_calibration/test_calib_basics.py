"""Tests for calibration config, data loading, the expected-decay model and summary statistics."""

import json

import numpy as np
import pytest
from calib_helpers import INACTIVE, N_BINS, PERIOD_NS, SAMPLE, true_irf

from pyfli.simulator.calibration import config as cc
from pyfli.simulator.calibration import data_io
from pyfli.simulator.calibration import summary_stats as ss
from pyfli.simulator.calibration.expected_decay import DecayForwardModel
from pyfli.simulator.calibration.sim_bridge import (
    draw_pixels,
    make_generator,
    stack_outputs,
)


class TestConfig:
    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("bi", "bi_exponential"),
            ("Bi-Exponential", "bi_exponential"),
            ("mono exponential", "mono_exponential"),
            ("monoexponential", "mono_exponential"),
        ],
    )
    def test_model_aliases(self, alias, expected):
        assert cc.normalize_model_type(alias) == expected

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("TCSPC", "discrete"),
            ("iccd", "continuous"),
            ("continous", "continuous"),
            (" discrete ", "discrete"),
        ],
    )
    def test_detector_aliases(self, alias, expected):
        assert cc.normalize_detector_type(alias) == expected

    def test_invalid_names_raise(self):
        with pytest.raises(ValueError):
            cc.normalize_model_type("tri")
        with pytest.raises(ValueError):
            cc.normalize_detector_type("camera")
        with pytest.raises(ValueError):
            cc.CalibrationInputs("x.npy", "discrete", "bi", reference_source="unknown")

    def test_inputs_defaults(self):
        bi = cc.CalibrationInputs("x.npy", "tcspc", "bi", laser_freq_mhz=40.0)
        assert bi.detector_type == "discrete" and bi.is_bi and bi.mono_fraction == 0.0
        assert bi.laser_period_ns == pytest.approx(25.0)
        mono = cc.CalibrationInputs("x.npy", "iccd", "mono")
        assert not mono.is_bi and mono.mono_fraction == 1.0
        assert 0 < mono.target_coverage < 1 and mono.n_coverage_pixels > 0

    def test_explicit_mono_fraction_kept(self):
        assert (
            cc.CalibrationInputs(
                "x.npy", "discrete", "bi", mono_fraction=0.3
            ).mono_fraction
            == 0.3
        )


class TestDataIO:
    @pytest.mark.parametrize(
        "shape,expected",
        [
            ((7,), (1, 1, 7)),
            ((3, 7), (1, 3, 7)),
            ((2, 3, 7), (2, 3, 7)),
            ((2, 4, 5, 7), (4, 5, 7)),
            ((4, 5, 7, 2), (4, 5, 7)),
        ],
    )
    def test_as_cube(self, shape, expected):
        assert data_io._as_cube(np.ones(shape)).shape == expected

    def test_as_cube_rejects_5d(self):
        with pytest.raises(ValueError):
            data_io._as_cube(np.ones((1, 1, 1, 1, 1)))

    def test_load_with_irf_mask_and_binning(self, synthetic_dir, synthetic_cube):
        data = data_io.load_experimental_data(
            str(synthetic_dir / "exp_a.npy"),
            irf_path=str(synthetic_dir / "irf.npy"),
            mask_path=str(synthetic_dir / "mask_a.npy"),
        )
        assert data.decay.shape == synthetic_cube.shape and data.n_bins == N_BINS
        assert data.spatial_shape == synthetic_cube.shape[:2]
        assert data.irf.shape == (N_BINS,) and data.mask.sum() == 14 * 14
        binned = data_io.load_experimental_data(
            str(synthetic_dir / "exp_a.npy"), bin_radius=1
        )
        assert binned.decay.shape == synthetic_cube.shape
        assert binned.decay[10, 10].sum() == pytest.approx(
            synthetic_cube[9:12, 9:12].sum(), rel=1e-5
        )

    def test_irf_bin_mismatch_raises(self, synthetic_dir, tmp_path):
        np.save(tmp_path / "irf_short.npy", np.ones(N_BINS - 1))
        with pytest.raises(ValueError, match="time bins"):
            data_io.load_experimental_data(
                str(synthetic_dir / "exp_a.npy"),
                irf_path=str(tmp_path / "irf_short.npy"),
            )

    def test_mask_shape_error(self, synthetic_dir, tmp_path):
        np.save(tmp_path / "small.npy", np.ones((5, 5), bool))
        with pytest.raises(data_io.MaskShapeError):
            data_io.load_experimental_data(
                str(synthetic_dir / "exp_a.npy"), mask_path=str(tmp_path / "small.npy")
            )
        assert issubclass(data_io.MaskShapeError, ValueError)

    def test_3d_mask_is_collapsed(self, tmp_path):
        m = np.zeros((4, 4, 3), bool)
        m[1, 2, 0] = True
        np.save(tmp_path / "m.npy", m)
        out = data_io._load_mask(str(tmp_path / "m.npy"), (4, 4))
        assert out.dtype == bool and out.sum() == 1 and out[1, 2]

    def test_to_ns_and_amplitude_conversion(self):
        np.testing.assert_allclose(
            data_io._to_ns(np.array([500.0, 2500.0])), [0.5, 2.5]
        )
        np.testing.assert_allclose(data_io._to_ns(np.array([0.5, 2.5])), [0.5, 2.5])
        f = data_io._amplitude_to_intensity(
            np.array([0.5]), np.array([1.0]), np.array([3.0])
        )
        assert f[0] == pytest.approx(0.25)

    def test_canonical_swaps_and_converts(self):
        raw = {
            "tau1_map": np.array([[3.0, 0.5]]),
            "tau2_map": np.array([[1.0, 2.0]]),
            "A1_map": np.array([[1.0, 1.0]]),
            "A2_map": np.array([[1.0, 3.0]]),
        }
        out = data_io._canonical(raw, "amplitude")
        np.testing.assert_allclose(out["tau1"], [[1.0, 0.5]])
        np.testing.assert_allclose(out["tau2"], [[3.0, 2.0]])
        np.testing.assert_allclose(out["efficiency"], [[1 - 1 / 3, 0.75]])
        a_first = 0.5 * 3.0 / (0.5 * 3.0 + 0.5 * 1.0)
        assert out["alpha1"][0, 0] == pytest.approx(1 - a_first)
        assert np.all((out["alpha1"] >= 0) & (out["alpha1"] <= 1))

    def test_canonical_mono_only(self):
        out = data_io._canonical({"tau_map": np.array([[1500.0, -1.0]])}, "intensity")
        assert out["tau"][0, 0] == pytest.approx(1.5) and np.isnan(out["tau"][0, 1])

    def test_pyfli_reference_maps(self, tmp_path):
        maps = {
            "tau1_map": np.full((3, 3), 0.5),
            "tau2_map": np.full((3, 3), 2.5),
            "alpha1_map": np.full((3, 3), 0.4),
            "photon_count_map": np.full((3, 3), 100.0),
        }
        np.save(
            tmp_path / "CPU_MLE_poisson_bi.npy",
            {"results": {"maps": maps}},
            allow_pickle=True,
        )
        ref = data_io.load_reference_maps(str(tmp_path))
        assert (
            ref.source == "pyfli"
            and ref.aligned_with((3, 3))
            and not ref.aligned_with((4, 4))
        )
        np.testing.assert_allclose(ref.values("alpha1"), 0.4)
        np.testing.assert_allclose(ref.values("efficiency"), 0.8)
        mask = np.zeros((3, 3), bool)
        mask[0] = True
        assert len(ref.values("tau2", mask)) == 3 and len(ref.values("missing")) == 0

    def test_detect_source(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "tau1Map.txt").write_text("1")
        assert data_io._detect_source(str(tmp_path / "a")) == "alligator"
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "x.asc").write_text("1")
        assert data_io._detect_source(str(tmp_path / "b")) == "spcimage"
        (tmp_path / "c").mkdir()
        with pytest.raises(FileNotFoundError):
            data_io._detect_source(str(tmp_path / "c"))


class TestExpectedDecay:
    def test_shape_normalised_and_inactive_zero(self):
        fm = DecayForwardModel(
            N_BINS, PERIOD_NS, "discrete", jitter=True, inactive_bins=INACTIVE
        )
        irf = fm.with_baseline(fm.simulated_irf(10, 0.08, 0.05), 1e-3)
        y = fm.shape(irf, [0.5, 2.0], [0.4, 0.6])
        assert (
            y.sum() == pytest.approx(1.0)
            and np.all(y[INACTIVE] == 0)
            and np.all(y >= 0)
        )
        assert fm.n_active == N_BINS - len(INACTIVE)
        assert fm.flat().sum() == pytest.approx(1.0) and np.all(
            fm.flat()[INACTIVE] == 0
        )

    def test_simulated_irf_and_baseline(self):
        fm = DecayForwardModel(N_BINS, PERIOD_NS, "discrete")
        irf = fm.simulated_irf(12, 0.1, 0.05)
        assert irf.max() == pytest.approx(1.0) and abs(int(np.argmax(irf)) - 12) <= 1
        assert fm.with_baseline(irf, 0.01).sum() == pytest.approx(1.0)

    def test_components_mixture(self):
        fm = DecayForwardModel(N_BINS, PERIOD_NS, "discrete")
        single = fm.components([2.0], [1.0])
        assert single.sum() == pytest.approx(1.0)
        mix = fm.components([0.5, 2.0], [0.3, 0.7])
        np.testing.assert_allclose(
            mix, 0.3 * fm.components([0.5], [1.0]) + 0.7 * single
        )

    def test_jitter_average_conserves_interior_mass(self):
        fm = DecayForwardModel(N_BINS, PERIOD_NS, "discrete", jitter=True)
        y = np.zeros(N_BINS)
        y[30] = 1.0
        out = fm._jitter_average(y)
        assert out.sum() == pytest.approx(1.0) and np.count_nonzero(out) == 5

    @pytest.mark.parametrize("sensor", ["discrete", "continuous"])
    def test_matches_pyfli_simulator_mean(self, sensor):
        cfg = {
            "sensor_type": sensor,
            "tau2": (2000.0, 2000.0),
            "tau2_dist": "beta",
            "tau2_beta_range": (0.02, 1.99),
            "mono_fraction": 1.0,
            "photo_count": (50.0, 50.0),
            "n_cycles": (200_000, 200_001),
            "bit": 16,
            "dcr": 0.0,
            "dcr_on": False,
            "jitter": False,
            "laser_feq": 80.0,
            "pileup_mode": "wrap",
            "qe_on": False,
            "read_noise_on": False,
            "seed": None,
        }
        irf = true_irf()
        gen = make_generator(irf[None, None, :], cfg, (0, 0), (0, 0))
        mean = stack_outputs(draw_pixels(gen, 400, seed=3))["decay"].mean(0)
        fm = DecayForwardModel(N_BINS, PERIOD_NS, sensor, jitter=False)
        expected = fm.shape(irf / irf.sum(), [2.0], [1.0])
        rel = (
            np.abs(mean / mean.sum() - expected)[expected > 0.02 * expected.max()]
            / expected[expected > 0.02 * expected.max()]
        )
        assert np.median(rel) < 0.05


class TestSummaryStats:
    def test_inactive_bins(self):
        pooled = np.full(N_BINS, 500.0)
        pooled[INACTIVE] = 0
        np.testing.assert_array_equal(ss.inactive_bins(pooled), INACTIVE)
        assert ss.inactive_bins(np.r_[np.zeros(3), np.ones(61)]).size == 0

    def test_background_bins_before_edge(self):
        pooled = np.full(N_BINS, 5.0)
        pooled[30:] += 1000 * np.exp(-np.arange(N_BINS - 30) / 8.0)
        pooled[INACTIVE] = 0
        bg = ss.background_bins(pooled)
        assert len(bg) >= 4 and bg.max() < 30 and not np.isin(bg, INACTIVE).any()

    def test_background_bins_synthetic_pre_edge(self, synthetic_cube):
        bg = ss.background_bins(synthetic_cube[SAMPLE].reshape(-1, N_BINS).sum(0))
        assert len(bg) >= 4 and bg.max() < 10 and not np.isin(bg, INACTIVE).any()

    def test_background_bins_fallback_to_lowest(self):
        pooled = np.r_[np.zeros(3), 1000 * np.exp(-np.arange(N_BINS - 3) / 15.0)] + 2.0
        pooled[:3] = 0
        bg = ss.background_bins(pooled)
        active = np.setdiff1d(np.arange(N_BINS), np.arange(3))
        assert len(bg) >= 4 and not np.isin(bg, np.arange(3)).any()
        assert pooled[bg].max() <= np.sort(pooled[active])[len(bg) - 1] + 1e-9

    def test_phasor_matches_analytic_exponential(self):
        tau, dt = 2.0, PERIOD_NS / N_BINS
        k = np.arange(N_BINS)
        y = np.exp(-k * dt / tau) - np.exp(-(k + 1) * dt / tau)
        g, s = ss.phasor(y[None, :], 1)
        z = np.sum(y * np.exp(2j * np.pi * k / N_BINS)) / y.sum()
        assert g[0] == pytest.approx(z.real) and s[0] == pytest.approx(z.imag)
        g0, s0 = ss.phasor(np.zeros((1, N_BINS)), 2)
        assert g0[0] == 0 and s0[0] == 0

    def test_rising_edge_of_step(self):
        d = np.zeros((2, N_BINS))
        d[0, 20:] = 10.0
        d[1, 35:] = 10.0
        edge = ss.rising_edge(d)
        assert 18 < edge[0] < 21 and 33 < edge[1] < 36

    def test_compute_stats_keys(self, synthetic_cube):
        d = synthetic_cube[SAMPLE].reshape(-1, N_BINS).astype(float)
        st = ss.compute_stats(d, ss.background_bins(d.sum(0)))
        assert set(st) == {
            "log_counts",
            "phasor1",
            "phasor2",
            "edge",
            "bg_frac",
            "mean_decay",
            "mean_decay_rel_var",
            "counts",
        }
        assert st["phasor1"].shape == (len(d), 2) and st[
            "mean_decay"
        ].sum() == pytest.approx(1.0)
        assert st["mean_decay_rel_var"].shape == (N_BINS,)
        assert np.all(st["mean_decay_rel_var"] >= 0)

    def test_comparator_zero_for_self_and_grows_with_shift(self, synthetic_cube):
        d = synthetic_cube[SAMPLE].reshape(-1, N_BINS).astype(float)
        bg = ss.background_bins(d.sum(0))
        st = ss.compute_stats(d, bg)
        comp = ss.StatsComparator(st)
        same, parts = comp.distance(st)
        assert same == pytest.approx(0.0, abs=1e-6) and set(parts) == set(
            ss.STAT_WEIGHTS
        )
        shifted, _ = comp.distance(ss.compute_stats(np.roll(d, 4, axis=-1), bg))
        assert shifted > 1.0

    def test_prepared_sim_weights(self, synthetic_cube):
        d = synthetic_cube[SAMPLE].reshape(-1, N_BINS).astype(float)
        st = ss.compute_stats(d, ss.background_bins(d.sum(0)))
        prep = ss.PreparedSim(st)
        uniform = np.full(prep.n, 1.0 / prep.n)
        np.testing.assert_allclose(
            prep.quantiles("log_counts", uniform),
            prep.quantiles("log_counts", None),
            atol=0.05,
        )
        heavy = np.zeros(prep.n)
        heavy[np.argmax(st["log_counts"])] = 1.0
        assert np.allclose(prep.quantiles("log_counts", heavy), st["log_counts"].max())


def test_saved_json_is_plain(calibration):
    with open(calibration.files["config"]) as f:
        payload = json.load(f)
    assert payload["config"]["sensor_type"] == "discrete"


def test_public_api_and_removed_duplicates():
    import importlib

    import pyfli.simulator as sim
    from pyfli.simulator import calibration

    for name in calibration.__all__:
        assert getattr(calibration, name) is not None
    assert not hasattr(sim, "FLICalibrator") and hasattr(sim, "FLIValidator")
    for gone in ("pyfli.simulator.calibration_engine", "pyfli.simulator.sim_stat_test"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(gone)
