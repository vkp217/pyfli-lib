"""Tests for pyfli.data_vnp.model_selector and pyfli.data_vnp.model_selector_plotter."""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyfli.data_vnp import ModelSelector, ModelSelectorPlotter

SHAPE = (20, 20)
N_GATES = 50
K_MONO, K_BI = 4, 6
DOF_BI = N_GATES - K_BI


def _regions():
    roi = np.zeros(SHAPE, bool)
    roi[2:18, 2:18] = True
    bi = np.zeros(SHAPE, bool)
    bi[2:8, 2:8] = True
    limit = np.zeros(SHAPE, bool)
    limit[2:5, 12:16] = True
    worse = np.zeros(SHAPE, bool)
    worse[15:18, 12:18] = True
    mono = roi & ~bi & ~limit & ~worse
    return roi, mono, bi, limit, worse


def _make_maps(seed=0):
    rng = np.random.default_rng(seed)
    roi, mono, bi, limit, worse = _regions()
    D_bi = np.full(SHAPE, float(DOF_BI))
    D_mono = D_bi + 0.5
    D_mono[bi | limit] += 100.0
    D_mono[worse] = D_bi[worse] - 3.0
    tau2 = np.full(SHAPE, 3.0)
    tau2[limit] = 12.5
    pc = rng.uniform(1e3, 5e3, SHAPE)
    maps_mono = {
        "chi2_map": D_mono,
        "photon_count_map": pc,
        "tau_map": rng.uniform(1.0, 3.0, SHAPE),
    }
    maps_bi = {
        "chi2_map": D_bi,
        "reduced_chi2_map": D_bi / DOF_BI,
        "photon_count_map": pc,
        "tau1_map": rng.uniform(0.3, 0.8, SHAPE),
        "tau2_map": tau2,
        "tau_mean_map": rng.uniform(1.0, 2.0, SHAPE),
        "alpha1_map": rng.uniform(0.0, 1.0, SHAPE),
        "fret_efficiency_map": rng.uniform(0.0, 1.0, SHAPE),
    }
    return maps_mono, maps_bi


def _save_fit(folder, prefix, maps_mono=None, maps_bi=None):
    for model, maps in (("mono", maps_mono), ("bi", maps_bi)):
        if maps is not None:
            np.save(
                os.path.join(folder, f"{prefix}_{model}-exponential.npy"),
                {"results": {"maps": maps}},
                allow_pickle=True,
            )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def maps():
    return _make_maps()


@pytest.fixture
def selector(maps):
    roi = _regions()[0]
    ms = ModelSelector(roi)
    ms.compare("MLE", *maps)
    return ms


@pytest.fixture
def fit_dir(tmp_path, maps):
    _save_fit(str(tmp_path), "CPU_MLE_poisson", *maps)
    return str(tmp_path)


class _Saver:
    def __init__(self):
        self.calls = []

    def save_plot(self, name, fig=None, close=True):
        self.calls.append((name, fig, close))


class TestBhFdr:
    def test_matches_scipy(self):
        if not hasattr(stats, "false_discovery_control"):
            pytest.skip("scipy.stats.false_discovery_control needs scipy >= 1.11")
        p = np.random.default_rng(1).uniform(0, 1, 200) ** 3
        np.testing.assert_allclose(
            ModelSelector.bh_fdr(p), stats.false_discovery_control(p, method="bh")
        )

    def test_known_values(self):
        q = ModelSelector.bh_fdr(np.array([0.01, 0.04, 0.03, 0.2]))
        np.testing.assert_allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.2])

    def test_monotone_in_p_and_bounded(self):
        p = np.random.default_rng(2).uniform(0, 1, 100)
        q = ModelSelector.bh_fdr(p)
        order = np.argsort(p)
        assert np.all(np.diff(q[order]) >= -1e-15)
        assert np.all((q >= p - 1e-15) & (q <= 1))


class TestModelSelector:
    def test_class_map(self, selector):
        roi, mono, bi, limit, worse = _regions()
        cls = selector.results["MLE"]["class_map"]
        assert np.all(cls[~roi] == 0)
        assert np.all(cls[mono] == 1)
        assert np.all(cls[bi] == 2)
        assert np.all(cls[limit] == 3)
        assert np.all(cls[worse] == 4)
        np.testing.assert_array_equal(selector.results["MLE"]["bi_justified"], bi)

    def test_n_points_recovered(self, selector):
        assert selector.results["MLE"]["n_points"] == N_GATES

    def test_f_test_and_aicc(self, selector, maps):
        roi = _regions()[0]
        res = selector.results["MLE"]
        dD = maps[0]["chi2_map"] - maps[1]["chi2_map"]
        dk = K_BI - K_MONO
        F = (dD / dk) / (maps[1]["chi2_map"] / DOF_BI)
        np.testing.assert_allclose(res["F_map"][roi], F[roi])
        np.testing.assert_allclose(res["p_map"][roi], stats.f.sf(F[roi], dk, DOF_BI))

        def aicc(D, k):
            return D + 2 * k + 2 * k * (k + 1) / (N_GATES - k - 1)

        expected = aicc(maps[0]["chi2_map"], K_MONO) - aicc(maps[1]["chi2_map"], K_BI)
        np.testing.assert_allclose(res["dAICc_map"][roi], expected[roi])
        np.testing.assert_allclose(res["dDev_map"][roi], dD[roi])

    def test_outside_roi_is_nan(self, selector):
        roi = _regions()[0]
        for key in ("F_map", "p_map", "q_map", "dAICc_map", "dDev_map"):
            assert np.all(np.isnan(selector.results["MLE"][key][~roi]))

    def test_invalid_pixels_excluded(self, maps):
        roi, _, bi, _, _ = _regions()
        maps_mono, maps_bi = maps
        maps_mono["chi2_map"][3, 3] = np.nan
        maps_bi["chi2_map"][4, 4] = 0.0
        res = ModelSelector(roi).compare("MLE", maps_mono, maps_bi)
        assert res["class_map"][3, 3] == 0
        assert res["class_map"][4, 4] == 0
        assert res["class_map"][5, 5] == 2

    def test_thresholds(self, maps):
        roi, _, bi, limit, _ = _regions()
        strict = ModelSelector(roi, daicc_min=1e6).compare("MLE", *maps)
        assert not np.any(strict["class_map"] == 2)
        assert np.all(strict["class_map"][bi | limit] == 1)
        high_bound = ModelSelector(roi, tau_bound=100.0).compare("MLE", *maps)
        assert np.all(high_bound["class_map"][limit] == 2)

    def test_summary(self, selector):
        df = selector.summary()
        assert isinstance(df, pd.DataFrame)
        assert list(df.index) == ["MLE"]
        row = df.loc["MLE"]
        roi, mono, bi, limit, worse = _regions()
        assert row["pixels"] == roi.sum()
        assert row["gates"] == N_GATES
        assert row["Bi justified"] == f"{bi.sum() / roi.sum():.1%}"
        assert row["Bi fit worse than mono"] == f"{worse.sum() / roi.sum():.1%}"

    def test_summary_empty_and_recompare_replaces_row(self, maps):
        ms = ModelSelector(_regions()[0])
        assert ms.summary().empty
        ms.compare("MLE", *maps)
        ms.compare("MLE", *maps)
        assert len(ms.summary()) == 1

    def test_compare_from_dir_skips_missing(self, fit_dir):
        ms = ModelSelector(_regions()[0])
        results = ms.compare_from_dir(fit_dir)
        assert list(results) == ["MLE"]

    def test_compare_from_dir_custom_fits(self, tmp_path, maps):
        _save_fit(str(tmp_path), "my_fit", *maps)
        ms = ModelSelector(_regions()[0])
        ms.compare_from_dir(str(tmp_path), fits={"custom": "my_fit"})
        assert list(ms.results) == ["custom"]

    def test_agreement(self, maps):
        roi, _, bi, _, _ = _regions()
        ms = ModelSelector(roi)
        ms.compare("MLE", *maps)
        assert ms.agreement() is None
        ms.compare("NLSF", *maps)
        assert ms.agreement() == pytest.approx(1.0)
        ms.results["NLSF"]["bi_justified"] = bi & (np.arange(SHAPE[1]) < 5)
        expected = (bi & (np.arange(SHAPE[1]) < 5)).sum() / bi.sum()
        assert ms.agreement() == pytest.approx(expected)

    def test_plot(self, selector):
        saver = _Saver()
        fig = selector.plot(saver=saver, name="ms_fig")
        assert fig is not None
        assert len([ax for ax in fig.axes if ax.get_label() != "<colorbar>"]) == 4
        assert saver.calls == [("ms_fig", fig, False)]

    def test_plot_empty_returns_none(self):
        assert ModelSelector(_regions()[0]).plot() is None


class TestPlotterSetup:
    def test_requires_mask_or_selector(self, fit_dir):
        with pytest.raises(ValueError, match="b_bool_mask"):
            ModelSelectorPlotter(None, fit_dir)

    def test_requires_save_dir_or_maps(self, selector):
        with pytest.raises(ValueError, match="save_dir"):
            ModelSelectorPlotter(selector)

    @pytest.mark.parametrize("kwargs", [{"model": "auto"}, {"fallback_model": "split"}])
    def test_invalid_model(self, selector, fit_dir, kwargs):
        with pytest.raises(ValueError):
            ModelSelectorPlotter(selector, fit_dir, **kwargs)

    def test_label_from_fit_prefix(self, selector, fit_dir):
        assert ModelSelectorPlotter(selector, fit_dir).label == "MLE"
        assert (
            ModelSelectorPlotter(selector, fit_dir, fit="CPU_NLSF_least_squares").label
            == "NLSF"
        )
        assert ModelSelectorPlotter(selector, fit_dir, fit="other").label == "other"

    def test_roi_defaults_to_selector(self, selector, fit_dir):
        p = ModelSelectorPlotter(selector, fit_dir)
        np.testing.assert_array_equal(p.roi, selector.roi)

    def test_load_maps(self, fit_dir, maps):
        p = ModelSelectorPlotter(None, fit_dir, b_bool_mask=_regions()[0])
        np.testing.assert_array_equal(
            p.load_maps("mono")["tau_map"], maps[0]["tau_map"]
        )

    def test_load_maps_missing(self, tmp_path):
        p = ModelSelectorPlotter(None, str(tmp_path), b_bool_mask=_regions()[0])
        with pytest.raises(FileNotFoundError):
            p.load_maps("bi")

    def test_class_masks(self, selector, fit_dir):
        _, mono, bi, _, worse = _regions()
        mono_px, bi_px = ModelSelectorPlotter(selector, fit_dir).class_masks()
        np.testing.assert_array_equal(mono_px, mono | worse)
        np.testing.assert_array_equal(bi_px, bi)

    def test_class_masks_without_selector(self, fit_dir):
        p = ModelSelectorPlotter(None, fit_dir, b_bool_mask=_regions()[0])
        with pytest.raises(RuntimeError):
            p.class_masks()


class TestResolveModel:
    def test_selector_split(self, selector, fit_dir):
        assert ModelSelectorPlotter(selector, fit_dir).resolve_model() == "split"

    def test_selector_mono_below_threshold(self, selector, fit_dir):
        p = ModelSelectorPlotter(selector, fit_dir, split_min_frac=0.9)
        assert p.resolve_model() == "mono"

    @pytest.mark.parametrize("model", ["mono", "bi", "split"])
    def test_explicit(self, selector, fit_dir, model):
        assert (
            ModelSelectorPlotter(selector, fit_dir, model=model).resolve_model()
            == model
        )

    @pytest.mark.parametrize("fallback", ["mono", "bi"])
    def test_fallback_without_selector(self, fit_dir, fallback):
        p = ModelSelectorPlotter(
            None, fit_dir, b_bool_mask=_regions()[0], fallback_model=fallback
        )
        assert p.resolve_model() == fallback

    def test_fallback_when_label_missing(self, selector, fit_dir):
        p = ModelSelectorPlotter(selector, fit_dir, fit="CPU_NLSF_least_squares")
        assert p.resolve_model() == "mono"

    def test_split_without_selector_raises(self, fit_dir):
        p = ModelSelectorPlotter(
            None, fit_dir, b_bool_mask=_regions()[0], model="split"
        )
        with pytest.raises(RuntimeError):
            p.resolve_model()


class TestPlotMaps:
    @pytest.mark.parametrize(
        "model, keys, files",
        [
            ("selector", ["mono", "bi"], ["mono_split", "bi_split"]),
            ("split", ["mono", "bi"], ["mono_split", "bi_split"]),
            ("mono", ["mono"], ["mono"]),
            ("bi", ["bi"], ["bi"]),
        ],
    )
    def test_modes_and_saved_files(self, selector, fit_dir, model, keys, files):
        figs = ModelSelectorPlotter(selector, fit_dir, model=model).plot_maps()
        assert list(figs) == keys
        for f in files:
            for ext in ("pdf", "svg", "tiff"):
                assert os.path.exists(
                    os.path.join(fit_dir, f"Fig_FLI_maps_CPU_MLE_poisson_{f}.{ext}")
                )

    def test_mono_needs_only_mono_fit(self, tmp_path, maps):
        _save_fit(str(tmp_path), "CPU_MLE_poisson", maps_mono=maps[0])
        p = ModelSelectorPlotter(
            None, str(tmp_path), b_bool_mask=_regions()[0], model="mono"
        )
        assert list(p.plot_maps()) == ["mono"]
        assert p.box_stats()["parameter"].tolist() == ["tau"]

    def test_bi_needs_only_bi_fit(self, tmp_path, maps):
        _save_fit(str(tmp_path), "CPU_MLE_poisson", maps_bi=maps[1])
        p = ModelSelectorPlotter(
            None, str(tmp_path), b_bool_mask=_regions()[0], model="bi"
        )
        assert list(p.plot_maps()) == ["bi"]

    def test_maps_without_save_dir(self, selector, maps, tmp_path):
        p = ModelSelectorPlotter(selector, maps={"mono": maps[0], "bi": maps[1]})
        assert list(p.plot_maps()) == ["mono", "bi"]
        fig, _ = p.plot_boxplots()
        assert fig is not None
        assert os.listdir(tmp_path) == []

    def _map_axes(self, fig):
        return [ax for ax in fig.axes if ax.images and ax.get_title()]

    def test_panel_counts(self, selector, fit_dir):
        figs = ModelSelectorPlotter(selector, fit_dir, tau_box=False).plot_maps()
        assert len(self._map_axes(figs["mono"])) == 3
        assert len(self._map_axes(figs["bi"])) == 6
        figs = ModelSelectorPlotter(
            selector, fit_dir, model="mono", tau_box=False
        ).plot_maps()
        assert len(self._map_axes(figs["mono"])) == 2

    def test_tau_box_panel(self, selector, fit_dir):
        fig = ModelSelectorPlotter(
            selector, fit_dir, model="mono", tau_box=True
        ).plot_maps()["mono"]
        titles = [ax.get_title() for ax in fig.axes]
        assert "Lifetime distribution" in titles

    def test_swap_xy_and_crop(self, selector, fit_dir):
        kw = dict(model="mono", tau_box=False, crop_to_roi=False)
        a = self._map_axes(
            ModelSelectorPlotter(selector, fit_dir, swap_xy=False, **kw).plot_maps()[
                "mono"
            ]
        )
        b = self._map_axes(
            ModelSelectorPlotter(selector, fit_dir, swap_xy=True, **kw).plot_maps()[
                "mono"
            ]
        )
        img_a = np.ma.filled(a[1].images[0].get_array(), np.nan)
        img_b = np.ma.filled(b[1].images[0].get_array(), np.nan)
        np.testing.assert_array_equal(img_a, img_b.T)
        cropped = self._map_axes(
            ModelSelectorPlotter(
                selector, fit_dir, model="mono", tau_box=False
            ).plot_maps()["mono"]
        )
        assert cropped[0].images[0].get_array().shape == (20, 20)

    def test_split_masks_other_class(self, selector, fit_dir):
        _, mono, bi, _, worse = _regions()
        figs = ModelSelectorPlotter(
            selector, fit_dir, swap_xy=False, crop_to_roi=False, tau_box=False
        ).plot_maps()
        tau_img = np.ma.filled(
            self._map_axes(figs["mono"])[2].images[0].get_array(), np.nan
        )
        assert np.all(np.isfinite(tau_img[mono | worse]))
        assert np.all(np.isnan(tau_img[bi]))
        all_counts = np.ma.filled(
            self._map_axes(figs["mono"])[0].images[0].get_array(), np.nan
        )
        assert np.all(np.isfinite(all_counts[bi]))

    def test_other_color_overlay(self, selector, fit_dir):
        figs = ModelSelectorPlotter(
            selector, fit_dir, other_color="grey", tau_box=False
        ).plot_maps()
        assert all(len(ax.images) == 2 for ax in self._map_axes(figs["bi"]))
        figs = ModelSelectorPlotter(selector, fit_dir, tau_box=False).plot_maps()
        assert all(len(ax.images) == 1 for ax in self._map_axes(figs["bi"]))

    def test_v_ranges_override(self, selector, fit_dir):
        fig = ModelSelectorPlotter(
            selector,
            fit_dir,
            model="mono",
            tau_box=False,
            v_ranges={"tau_map": (0.5, 1.5)},
        ).plot_maps()["mono"]
        assert self._map_axes(fig)[1].images[0].get_clim() == (0.5, 1.5)


class TestBoxPlots:
    def test_selector_stats(self, selector, fit_dir, maps):
        _, mono, bi, _, worse = _regions()
        df = ModelSelectorPlotter(selector, fit_dir).box_stats()
        assert df["panel"].tolist() == ["a", "b", "b", "b", "c", "c"]
        assert df["parameter"].tolist() == [
            "tau",
            "tau1",
            "tau2",
            "tau_mean",
            "FRET fraction",
            "FRET efficiency",
        ]
        n = dict(zip(df["parameter"], df["n"]))
        assert n["tau"] == (mono | worse).sum()
        assert n["tau1"] == bi.sum()
        row = df.set_index("parameter")
        assert row.loc["tau", "median"] == pytest.approx(
            np.median(maps[0]["tau_map"][mono | worse])
        )
        assert row.loc["FRET efficiency", "mean"] == pytest.approx(
            100 * maps[1]["fret_efficiency_map"][bi].mean()
        )
        assert row.loc["tau1", "unit"] == "ns"
        assert row.loc["FRET fraction", "unit"] == "%"

    @pytest.mark.parametrize(
        "model, panels",
        [("mono", ["a"]), ("bi", ["a", "a", "a", "b", "b"])],
    )
    def test_whole_roi_stats(self, selector, fit_dir, model, panels):
        roi = _regions()[0]
        df = ModelSelectorPlotter(selector, fit_dir, model=model).box_stats()
        assert df["panel"].tolist() == panels
        assert set(df["n"]) == {roi.sum()}
        assert df["class"].str.startswith("All ROI pixels").all()

    @pytest.mark.parametrize(
        "model, suffix, n_axes",
        [("selector", "", 3), ("mono", "_mono", 1), ("bi", "_bi", 2)],
    )
    def test_plot_boxplots_files(self, selector, fit_dir, model, suffix, n_axes):
        fig, df = ModelSelectorPlotter(selector, fit_dir, model=model).plot_boxplots(
            show_points=True
        )
        assert len(fig.axes) == n_axes
        name = f"Fig_FLI_boxplots_CPU_MLE_poisson{suffix}"
        saved = pd.read_csv(os.path.join(fit_dir, f"{name}_stats.csv"))
        assert len(saved) == len(df)
        for ext in ("pdf", "svg", "tiff"):
            assert os.path.exists(os.path.join(fit_dir, f"{name}.{ext}"))

    def test_height_scales_with_panels(self, selector, fit_dir):
        h = {
            m: ModelSelectorPlotter(selector, fit_dir, model=m)
            .plot_boxplots()[0]
            .get_size_inches()[1]
            for m in ("selector", "bi", "mono")
        }
        assert h["selector"] > h["bi"] > h["mono"]
        fig, _ = ModelSelectorPlotter(selector, fit_dir).plot_boxplots(height_mm=100)
        assert fig.get_size_inches()[1] == pytest.approx(100 / 25.4)
