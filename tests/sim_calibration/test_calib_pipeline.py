"""End-to-end tests: calibrator, multi-cluster, pooled and sweep combination, folder mode and CLI."""

import importlib.util
import json
import logging
import os
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from calib_helpers import (
    INACTIVE,
    N_BINS,
    SAMPLE,
    SHAPE,
    TINY_BUDGET,
    copy_calibration,
    simulate_cube,
)

from pyfli.simulator.calibration import (
    CalibrationInputs,
    DetectorSimCalibrator,
    MaskShapeError,
    PooledCalibrator,
    calibrate_folder,
    combination_generator,
    combine_calibrations,
    find_data_files,
    find_session_configs,
    load_combined_setup,
    load_experimental_data,
    load_session_calibration,
    make_combined_generator,
    match_masks,
    verify_combined,
)
from pyfli.simulator.calibration import folder_calibration as fc
from pyfli.simulator.calibration import multicluster as mc
from pyfli.simulator.calibration import pooled_calibration as pc
from pyfli.simulator.calibration import run_calibration as rc
from pyfli.simulator.calibration import sweep_combination as sc
from pyfli.simulator.calibration.expected_decay import DecayForwardModel
from pyfli.simulator.calibration.prior_estimation import BetaPrior
from pyfli.simulator.calibration.sim_bridge import TimeWindowedGenerator


class TestCalibrator:
    def test_outputs_written(self, calibration):
        names = {
            "config",
            "irf",
            "mask",
            "history",
            "validation_sample",
            "plot_selection",
            "plot_validation",
            "plot_priors",
        }
        assert names <= set(calibration.files)
        assert all(os.path.exists(p) for p in calibration.files.values())
        assert isinstance(calibration.history, pd.DataFrame) and len(
            calibration.history
        )

    def test_payload_contents(self, calibration):
        with open(calibration.files["config"]) as f:
            p = json.load(f)
        assert {
            "inputs",
            "config",
            "sim_generator_kwargs",
            "irf",
            "priors",
            "priors_initial",
            "best_fit",
            "pixel_selection",
            "validation",
            "dcr_config",
        } <= set(p)
        assert p["sim_generator_kwargs"]["inactive_bins"] == INACTIVE.tolist()
        assert p["pixel_selection"]["inactive_bins"] == INACTIVE.tolist()
        v = p["validation"]
        assert {
            "distance_initial",
            "distance_best_fit",
            "distance_final",
            "coverage_best_fit",
            "coverage_final",
            "coverage_held_out",
            "coverage_trace",
            "fli_validator",
        } <= set(v)
        assert set(v["coverage_final"]) == {
            "phasor1",
            "phasor2",
            "log_counts",
            "edge",
            "bg_frac",
        }
        assert p["best_fit"]["config"]["tau2_dist"] == "beta"

    def test_recovers_ground_truth(self, calibration):
        priors, irf = calibration.state.priors, calibration.state.irf
        assert 1.4 < priors.tau2.mean < 2.6
        assert 0.35 < priors.efficiency.mean < 0.8 and 0.3 < priors.alpha1.mean < 0.7
        assert irf.source == "simulated" and irf.params["mu_bin"] == pytest.approx(
            10, abs=1.0
        )
        truth = np.zeros(SHAPE, bool)
        truth[SAMPLE] = True
        assert (calibration.mask & truth).sum() / (
            calibration.mask | truth
        ).sum() > 0.85

    def test_generator_and_default_output_dir(self, calibration, synthetic_dir):
        gen = calibration.make_generator()
        assert isinstance(gen, TimeWindowedGenerator) and np.all(
            gen.simulate_once()["decay"][INACTIVE] == 0
        )
        inp = CalibrationInputs(
            str(synthetic_dir / "exp_a.npy"), "discrete", "bi", bin_radius=2
        )
        expected = synthetic_dir / "exp_a_sim_calibration_bin2_discrete_bi_exponential"
        assert (
            DetectorSimCalibrator(inp).output_dir == str(expected) and expected.is_dir()
        )

    def test_mono_experimental_irf_user_mask_no_widening(self, synthetic_dir, tmp_path):
        cfg = {"mono_fraction": 1.0}
        np.save(tmp_path / "mono.npy", simulate_cube(seed=3, config=cfg))
        res = DetectorSimCalibrator(
            CalibrationInputs(
                data_path=str(tmp_path / "mono.npy"),
                detector_type="discrete",
                model_type="mono",
                irf_path=str(synthetic_dir / "irf.npy"),
                irf_mode="experimental",
                mask_path=str(synthetic_dir / "mask_a.npy"),
                target_coverage=0.0,
                output_dir=str(tmp_path / "out"),
                **TINY_BUDGET,
            )
        ).run()
        assert (
            res.state.irf.source == "experimental"
            and res.selection_report["mode"] == "user"
        )
        assert (
            res.state.priors.efficiency is None and res.config["mono_fraction"] == 1.0
        )
        assert (
            res.validation["coverage_trace"] == []
            and 1.4 < res.state.priors.tau2.mean < 2.6
        )

    def test_continuous_with_reference_maps(self, tmp_path):
        cfg = {
            "sensor_type": "continuous",
            "bit": 12,
            "photo_count": (30.0, 30.0),
            "dcr": 0.5,
            "mono_fraction": 1.0,
        }
        np.save(tmp_path / "cont.npy", simulate_cube(seed=4, config=cfg))
        tau = np.full(SHAPE, np.nan)
        tau[SAMPLE] = np.random.default_rng(0).uniform(1.6, 2.4, (14, 14))
        np.save(
            tmp_path / "CPU_MLE_poisson_mono.npy", {"tau_map": tau}, allow_pickle=True
        )
        res = DetectorSimCalibrator(
            CalibrationInputs(
                data_path=str(tmp_path / "cont.npy"),
                detector_type="continuous",
                model_type="mono",
                bit_depth=12,
                reference_path=str(tmp_path / "CPU_MLE_poisson_mono.npy"),
                output_dir=str(tmp_path / "out"),
                target_coverage=0.9,
                **TINY_BUDGET,
            )
        ).run()
        assert res.state.priors.source == "reference:pyfli" and res.config["bit"] == 12
        assert "n_cycles" not in res.config and res.state.dcr > 0


class TestMulticluster:
    @pytest.fixture(scope="class")
    def two_populations(self):
        shape, sample = (36, 36), (slice(3, 33), slice(3, 33))
        short = simulate_cube(
            seed=11,
            config={"mono_fraction": 1.0, "tau2_beta_range": (0.2, 0.4)},
            shape=shape,
            sample=sample,
        )
        long = simulate_cube(
            seed=21,
            config={"mono_fraction": 1.0, "tau2_beta_range": (0.4, 3.0)},
            shape=shape,
            sample=sample,
        )
        cube = short.copy()
        cube[:, 18:] = long[:, 18:]
        mask = np.zeros(shape, bool)
        mask[sample] = True
        return cube, mask, sample

    def test_detect_two_clusters(self, two_populations):
        cube, mask, sample = two_populations
        res = mc.detect_clusters(cube, mask, max_components=3, seed=0)
        assert res.n_clusters == 2 and sum(res.fractions) == pytest.approx(1.0)
        assert (
            np.all(res.labels[~mask] == 0)
            and res.mask(1).sum() + res.mask(2).sum() == mask.sum()
        )
        left, right = res.labels[sample][:, :10], res.labels[sample][:, -10:]
        assert (
            np.bincount(left.ravel()).argmax() == 1
            and np.bincount(right.ravel()).argmax() == 2
        )
        assert (
            res.report["n_clusters"] == 2
            and len(res.report["cluster_phasor_means"]) == 2
        )

    def test_min_fraction_merges_small_cluster(self, two_populations):
        cube, mask, _ = two_populations
        assert (
            mc.detect_clusters(
                cube, mask, max_components=3, min_fraction=0.6, seed=0
            ).n_clusters
            == 1
        )

    def test_single_population_merges(self, synthetic_cube):
        mask = np.zeros(SHAPE, bool)
        mask[SAMPLE] = True
        assert (
            mc.detect_clusters(
                synthetic_cube, mask, max_components=3, seed=0
            ).n_clusters
            == 1
        )

    def test_choose_tile(self):
        labels = np.zeros((12, 12), int)
        labels[2:10, 2:6] = 1
        labels[2:10, 6:10] = 2
        r, c = mc.choose_tile(labels, (4, 4), seed=0)
        tile = labels[r : r + 4, c : c + 4]
        assert (tile > 0).all() and set(np.unique(tile)) == {1, 2}

    def test_tile_helpers(self, calibration, tmp_path):
        cfg_path = calibration.files["config"]
        cube = mc.irf_tile(cfg_path, (3, 4), seed=1, peak=1e4)
        assert cube.shape == (3, 4, N_BINS) and cube.max() == pytest.approx(1e4)
        np.testing.assert_array_equal(
            cube, mc.irf_tile(cfg_path, (3, 4), seed=1, peak=1e4)
        )
        cfg = mc.roi_config(cfg_path, seed=5)
        assert isinstance(cfg["tau2"], tuple) and cfg["seed"] == 5
        base = CalibrationInputs("x.npy", "discrete", "bi")
        ci = mc.cluster_inputs(base, "m.npy", "irf.npy", str(tmp_path))
        assert (ci.mask_path, ci.irf_path, ci.irf_mode, ci.output_dir) == (
            "m.npy",
            "irf.npy",
            "experimental",
            str(tmp_path),
        )

    def test_simulate_multicluster_tile(self, calibration):
        labels = np.array([[1, 1, 2], [1, 2, 2], [0, 2, 2]])
        cfg = calibration.files["config"]
        out = mc.simulate_multicluster_tile(
            labels, {1: cfg, 2: cfg}, cfg, "discrete", seed=0
        )
        decay = np.asarray(out["raw_data"]["decay"])
        assert decay.shape == (3, 3, N_BINS) and np.all(decay[..., INACTIVE] == 0)
        assert decay[2, 0].sum() == 0 and decay[0, 0].sum() > 0


def make_second_experiment(
    calibration, tmp_path, name="exp_b", tau_shift=0.5, dcr=0.05
):
    path = copy_calibration(calibration.output_dir, str(tmp_path / name))
    with open(path) as f:
        p = json.load(f)
    p["priors"]["tau2"]["offset"] += tau_shift
    p["config"]["dcr"] = dcr
    p["config"]["efficiency"] = [5.0, 5.0]
    with open(path, "w") as f:
        json.dump(p, f)
    return path


class TestSweepCombination:
    @pytest.fixture
    def setup(self, calibration, tmp_path):
        a = copy_calibration(calibration.output_dir, str(tmp_path / "exp_a"))
        b = make_second_experiment(calibration, tmp_path)
        out = tmp_path / "combined"
        return (
            combine_calibrations({"exp_a": a, "exp_b": b}, str(out)),
            out,
            {"exp_a": a, "exp_b": b},
        )

    def test_structure(self, setup):
        s, out, _ = setup
        assert {"tau2", "dcr", "efficiency"} <= set(s["COMMON_SWEEP"])
        assert "A1_fraction" in s["COMMON_CONFIG"] and "jitter" in s["COMMON_CONFIG"]
        for k, v in s["COMMON_SWEEP"].items():
            assert len(v) == 2 and sum(s["COMMON_WEIGHTS"][k]) == pytest.approx(1.0)
        assert (
            s["DISCRETE_CONFIG"]["qe_on"] is False
            and len(s["DISCRETE_CONFIG"]["n_cycles"]) == 2
        )
        assert (
            s["N_COMBINATIONS"] == 2 ** len(s["COMMON_SWEEP"])
            and len(s["IRF_RUNS"]) == 2
        )
        scale, offset = s["COMMON_CONFIG"]["tau2_beta_range"]
        for path in s["EXPERIMENT_CONFIGS"].values():
            t2 = load_session_calibration(path).state.priors.tau2
            assert (
                offset <= t2.offset + 1e-4
                and offset + scale >= t2.offset + t2.scale - 1e-4
            )
        assert set(s["EXPERIMENT_SWEEP_VALUES"]) == {"exp_a", "exp_b"}
        assert (out / "combined_sim_setup.json").exists() and (
            out / "irfs" / "exp_a.txt"
        ).exists()

    def test_py_module_matches_json(self, setup):
        _, out, _ = setup
        spec = importlib.util.spec_from_file_location(
            "combined", out / "combined_sim_setup.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        loaded = load_combined_setup(str(out / "combined_sim_setup.json"))
        assert (
            mod.COMMON_SWEEP == loaded["COMMON_SWEEP"]
            and mod.COMMON_CONFIG == loaded["COMMON_CONFIG"]
        )
        assert mod.SENSOR_SETUP["discrete"][0] == loaded["DISCRETE_CONFIG"]
        assert isinstance(loaded["COMMON_SWEEP"]["tau2"][0], tuple) and isinstance(
            loaded["A_RANGE"], tuple
        )

    def test_resupport_keeps_moments(self, setup, calibration):
        s, _, paths = setup
        a = load_session_calibration(paths["exp_a"]).state.priors.tau2
        scale, offset = s["COMMON_CONFIG"]["tau2_beta_range"]
        shape = s["EXPERIMENT_SWEEP_VALUES"]["exp_a"]["tau2"]
        new = BetaPrior(shape[0], shape[1], offset, scale)
        assert new.mean == pytest.approx(a.mean, rel=1e-3) and new.sd == pytest.approx(
            a.sd, rel=2e-3
        )
        assert sc.cdf_error(a, a) == 0.0

    def test_sweep_entry_and_split(self):
        assert sc._sweep_entry([1, 2, 1], [0.2, 0.3, 0.5]) == ([1, 2], [0.7, 0.3])
        fixed, sweep, weights = sc._split({"a": [1, 1], "b": [1, 2]}, [0.5, 0.5])
        assert (
            fixed == {"a": 1}
            and sweep == {"b": [1, 2]}
            and weights == {"b": [0.5, 0.5]}
        )

    def test_generators(self, setup, monkeypatch):
        s, out, _ = setup
        path = str(out / "combined_sim_setup.json")
        ccg = combination_generator(load_combined_setup(path), "discrete")
        assert len(ccg) == s["N_COMBINATIONS"]
        monkeypatch.setattr(sc, "MAX_CARTESIAN", 1)
        with pytest.raises(ValueError):
            combination_generator(load_combined_setup(path), "discrete")
        gen = make_combined_generator(path, run=1, seed=0)
        assert isinstance(gen, TimeWindowedGenerator)
        o = gen.simulate_once()
        assert np.all(o["decay"][INACTIVE] == 0) and set(o["config_index"]) == set(
            s["COMMON_SWEEP"]
        )
        loose = make_combined_generator(path, run=0, seed=0, exact_ranges=False)
        assert isinstance(loose, sc.IndependentSweepGenerator)

    def test_independent_sweep_weights(self):
        gen = sc.IndependentSweepGenerator(
            np.ones((1, 1, 8)),
            {},
            {"k": [0, 1]},
            {"k": [0.9, 0.1]},
            (0, 0),
            (0, 0),
            seed=0,
        )
        idx = [int(gen.rng.choice(2, p=gen.probs[0])) for _ in range(2000)]
        assert np.mean(idx) == pytest.approx(0.1, abs=0.03)

    def test_pixel_weighting_and_errors(self, calibration, tmp_path):
        a = copy_calibration(calibration.output_dir, str(tmp_path / "a"))
        b = make_second_experiment(calibration, tmp_path, "b")
        with open(b) as f:
            p = json.load(f)
        p["pixel_selection"]["n_selected"] *= 3
        with open(b, "w") as f:
            json.dump(p, f)
        s = combine_calibrations(
            {"a": a, "b": b}, str(tmp_path / "c"), weighting="pixels"
        )
        assert s["EXPERIMENT_WEIGHTS"]["b"] == pytest.approx(0.75, abs=1e-3)
        p["inputs"]["detector_type"] = "continuous"
        with open(b, "w") as f:
            json.dump(p, f)
        with pytest.raises(ValueError, match="detector_type"):
            combine_calibrations({"a": a, "b": b}, str(tmp_path / "d"))
        with pytest.raises(ValueError):
            combine_calibrations({}, str(tmp_path / "e"))

    def test_verify_combined(self, setup):
        _, out, _ = setup
        table = verify_combined(
            str(out / "combined_sim_setup.json"), n_sim=400, n_exp=200
        )
        assert (
            set(table.index.get_level_values(1)) == {"own values", "combined"}
            and len(table) == 4
        )
        assert (out / "coverage_per_experiment.csv").exists()
        assert table.loc[("exp_a", "own values"), "edge"] > 0.8


class TestPooled:
    def test_helpers(self, calibration, tmp_path):
        sess = load_session_calibration(calibration.files["config"], name="a")
        assert sess.name == "a" and sess.n_bins == N_BINS
        assert sess.state.priors.tau2.mean == pytest.approx(
            calibration.state.priors.tau2.mean, rel=1e-3
        )
        lo, hi = pc.photon_range_photons(sess.state)
        assert hi > lo > 0
        rng = np.random.default_rng(0)
        samples = [rng.beta(20, 20, 3000), rng.beta(40, 10, 3000)]
        prior, info = pc.covering_beta(samples, np.ones(2), 0.0, 1.0)
        assert (
            min(info["covered_fraction_per_experiment"]) > 0.97
            and prior.a >= 1
            and prior.b >= 1
        )
        f = pc._background_fraction(
            0.0, np.array([0.0]), np.array([100.0]), 1.0, N_BINS
        )
        assert f[0] == 0.0
        copy_calibration(calibration.output_dir, str(tmp_path / "root" / "x"))
        assert set(find_session_configs(str(tmp_path / "root"))) == {"x"}

    def test_pooled_calibrator_runs(self, calibration, tmp_path):
        a = copy_calibration(calibration.output_dir, str(tmp_path / "a"))
        b = make_second_experiment(calibration, tmp_path, "b")
        res = PooledCalibrator(
            {"a": a, "b": b},
            str(tmp_path / "pooled"),
            n_per_exp=150,
            n_validation=600,
            max_flatten=1,
        ).run()
        assert (
            set(res.coverage.index) == {"a", "b"}
            and "cover phasor1" in res.coverage.columns
        )
        assert res.sim_generator_kwargs["inactive_bins"] == INACTIVE.tolist()
        assert os.path.exists(res.files["config"]) and res.state.irf.library.shape == (
            2,
            N_BINS,
        )
        assert res.config["tau2_dist"] == "beta"

    def test_combine_states_library(self, calibration):
        s = [
            load_session_calibration(calibration.files["config"], name=n)
            for n in ("a", "b")
        ]
        fm = DecayForwardModel(N_BINS, 12.5, "discrete")
        state, report = pc.combine_states(s, np.ones(2), fm, n_samples=2000)
        assert (
            state.irf.library.shape == (2, N_BINS) and len(state.irf.library_b_max) == 2
        )
        assert {"tau2", "efficiency", "alpha1", "photon", "background"} <= set(report)


class FakeCalibrator:
    source_dir: ClassVar[str | None] = None
    calls: ClassVar[list] = []

    def __init__(self, inputs):
        self.inputs = inputs

    def run(self):
        inp = self.inputs
        load_experimental_data(inp.data_path, mask_path=inp.mask_path)
        FakeCalibrator.calls.append((os.path.basename(inp.data_path), inp.mask_path))
        copy_calibration(
            self.source_dir,
            inp.output_dir,
            data_path=inp.data_path,
            mask_path=inp.mask_path,
            irf_path=inp.irf_path,
        )


class TestFolderCalibration:
    @pytest.fixture
    def folders(self, tmp_path, calibration, monkeypatch):
        data, masks = tmp_path / "data", tmp_path / "masks"
        data.mkdir()
        masks.mkdir()
        np.save(data / "exp_a.npy", simulate_cube(seed=0))
        np.save(data / "exp_b.npy", simulate_cube(seed=1))
        (data / "irf_notes.txt").write_text("not data")
        good = np.zeros(SHAPE, bool)
        good[SAMPLE] = True
        np.save(masks / "exp_a_mask.npy", good)
        np.save(masks / "exp_b.npy", np.ones((5, 5), bool))
        np.save(masks / "orphan.npy", good)
        FakeCalibrator.source_dir = calibration.output_dir
        FakeCalibrator.calls = []
        monkeypatch.setattr(fc, "DetectorSimCalibrator", FakeCalibrator)
        return data, masks

    def test_find_data_files(self, folders):
        data, _ = folders
        assert [os.path.basename(p) for p in find_data_files(str(data))] == [
            "exp_a.npy",
            "exp_b.npy",
        ]

    def test_match_masks(self, folders, tmp_path, caplog):
        data, masks = folders
        files = find_data_files(str(data))
        np.save(data / "exp_c.npy", np.zeros((2, 2, N_BINS)))
        files.append(str(data / "exp_c.npy"))
        with caplog.at_level(logging.WARNING, logger="pyfli"):
            m = match_masks(files, str(masks))
        assert (
            os.path.basename(m[files[0]]) == "exp_a_mask.npy"
            and os.path.basename(m[files[1]]) == "exp_b.npy"
        )
        assert files[2] not in m
        assert "exp_c" in caplog.text and "orphan.npy" in caplog.text
        assert match_masks(files, None) == {}
        with pytest.raises(NotADirectoryError):
            match_masks(files, str(data / "exp_a.npy"))

    def test_exact_name_preferred(self, tmp_path):
        (tmp_path / "m").mkdir()
        for n in ("s1.npy", "s1_mask.npy"):
            np.save(tmp_path / "m" / n, np.ones((2, 2)))
        m = match_masks([str(tmp_path / "s1.npy")], str(tmp_path / "m"))
        assert os.path.basename(m[str(tmp_path / "s1.npy")]) == "s1.npy"

    def test_folder_run_masks_reuse_and_combination(self, folders, tmp_path, caplog):
        data, masks = folders
        out = tmp_path / "out"
        with caplog.at_level(logging.WARNING, logger="pyfli"):
            res = calibrate_folder(
                str(data),
                "discrete",
                "bi",
                output_root=str(out),
                mask_folder=str(masks),
                verify=False,
            )
        assert set(res["per_experiment"]) == {"exp_a", "exp_b"} and res["failed"] == {}
        assert (
            res["masks"]["exp_a"].endswith("exp_a_mask.npy")
            and res["masks"]["exp_b"] == "automatic"
        )
        assert "does not match the data" in caplog.text
        assert (out / "exp_b" / fc.REJECTED_MASK_FILE).exists()
        assert sorted(FakeCalibrator.calls) == [
            ("exp_a.npy", str(masks / "exp_a_mask.npy")),
            ("exp_b.npy", None),
        ]
        assert os.path.exists(res["combined"]["py"]) and os.path.exists(
            res["combined"]["json"]
        )
        FakeCalibrator.calls = []
        again = calibrate_folder(
            str(data),
            "discrete",
            "bi",
            output_root=str(out),
            mask_folder=str(masks),
            verify=False,
        )
        assert FakeCalibrator.calls == [] and again["masks"] == res["masks"]
        os.remove(masks / "exp_a_mask.npy")
        calibrate_folder(
            str(data),
            "discrete",
            "bi",
            output_root=str(out),
            mask_folder=str(masks),
            verify=False,
        )
        assert FakeCalibrator.calls == [("exp_a.npy", None)]
        calibrate_folder(
            str(data),
            "discrete",
            "bi",
            output_root=str(out),
            mask_folder=str(masks),
            verify=False,
            force=True,
            combine=False,
        )
        assert sorted(FakeCalibrator.calls[1:]) == [
            ("exp_a.npy", None),
            ("exp_b.npy", None),
        ]

    def test_folder_verify_and_default_root(self, folders):
        data, _ = folders
        res = calibrate_folder(
            str(data), "discrete", "bi", n_verify_pixels=300, combine=True, verify=True
        )
        assert res["output_root"] == os.path.join(
            str(data), "sim_calibration_bin0_discrete_bi_exponential"
        )
        assert isinstance(res["coverage"], pd.DataFrame) and len(res["coverage"]) == 4

    def test_irf_rules(self, folders, synthetic_dir, tmp_path):
        data, _ = folders
        with pytest.raises(ValueError, match="ONE IRF file"):
            calibrate_folder(str(data), "discrete", "bi", irf_path=str(data))
        with pytest.raises(ValueError, match="ONE IRF file"):
            calibrate_folder(str(data), "discrete", "bi", irf_path=["a.txt", "b.txt"])
        with pytest.raises(FileNotFoundError):
            calibrate_folder(
                str(data), "discrete", "bi", irf_path=str(tmp_path / "missing.txt")
            )
        with pytest.raises(ValueError, match="mask_folder"):
            calibrate_folder(str(data), "discrete", "bi", mask_path="m.npy")
        res = calibrate_folder(
            str(data),
            "discrete",
            "bi",
            output_root=str(tmp_path / "o"),
            irf_path=str(synthetic_dir / "irf.npy"),
            verify=False,
        )
        assert res["irf_path"] == str(synthetic_dir / "irf.npy")
        assert all(
            json.load(open(p))["inputs"]["irf_path"] == res["irf_path"]
            for p in res["per_experiment"].values()
        )

    def test_failures_are_reported(self, folders, tmp_path, monkeypatch):
        data, _ = folders

        class Broken(FakeCalibrator):
            def run(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(fc, "DetectorSimCalibrator", Broken)
        res = calibrate_folder(
            str(data), "discrete", "bi", output_root=str(tmp_path / "o"), verify=False
        )
        assert set(res["failed"]) == {"exp_a", "exp_b"} and res["combined"] is None

    def test_empty_folder_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            calibrate_folder(str(tmp_path / "empty"), "discrete", "bi")

    def test_mask_shape_error_type(self, folders):
        data, masks = folders
        with pytest.raises(MaskShapeError):
            load_experimental_data(
                str(data / "exp_b.npy"), mask_path=str(masks / "exp_b.npy")
            )


class TestCLI:
    def test_parse_args_defaults(self):
        a = rc.parse_args(
            ["--data", "x.npy", "--detector", "discrete", "--model", "bi_exponential"]
        )
        assert (
            a.target_coverage == 0.99
            and a.weighting == "equal"
            and not a.force
            and a.jitter == "auto"
        )

    def test_single_file(self, monkeypatch, capsys, calibration):
        seen = {}

        class Fake:
            def __init__(self, inputs):
                seen["inputs"] = inputs

            def run(self):
                return calibration

        monkeypatch.setattr(rc, "DetectorSimCalibrator", Fake)
        rc.main(
            [
                "--data",
                "x.npy",
                "--detector",
                "discrete",
                "--model",
                "bi_exponential",
                "--mask",
                "m.npy",
                "--irf",
                "i.txt",
                "--jitter",
                "off",
                "--target-coverage",
                "0.9",
            ]
        )
        inp = seen["inputs"]
        assert (inp.mask_path, inp.irf_path, inp.jitter, inp.target_coverage) == (
            "m.npy",
            "i.txt",
            False,
            0.9,
        )
        assert "coverage of the experiment" in capsys.readouterr().out

    def test_folder_mode(self, monkeypatch, capsys, tmp_path):
        (tmp_path / "masks").mkdir()
        (tmp_path / "irf.txt").write_text("1\n")
        seen = {}

        def fake_folder(folder, detector, model, **kw):
            seen.update(kw, folder=folder)
            return {
                "per_experiment": {"e": "e.json"},
                "masks": {"e": "automatic"},
                "failed": {"f": "err"},
                "combined": {"py": "c.py"},
            }

        monkeypatch.setattr(rc, "calibrate_folder", fake_folder)
        rc.main(
            [
                "--data",
                str(tmp_path),
                "--detector",
                "discrete",
                "--model",
                "bi_exponential",
                "--mask",
                str(tmp_path / "masks"),
                "--irf",
                str(tmp_path / "irf.txt"),
                "--weighting",
                "pixels",
            ]
        )
        assert seen["mask_folder"] == str(tmp_path / "masks") and seen[
            "irf_path"
        ] == str(tmp_path / "irf.txt")
        assert seen["weighting"] == "pixels"
        out = capsys.readouterr().out
        assert "FAILED f" in out and "c.py" in out

    @pytest.mark.parametrize(
        "extra",
        [["--reference", "r.npy"], ["--mask", "m.npy"], ["--irf", "missing.txt"]],
    )
    def test_folder_mode_rejects(self, tmp_path, extra):
        with pytest.raises(SystemExit):
            rc.main(
                [
                    "--data",
                    str(tmp_path),
                    "--detector",
                    "discrete",
                    "--model",
                    "bi_exponential",
                    *extra,
                ]
            )
