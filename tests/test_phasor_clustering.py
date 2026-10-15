"""Tests for pyfli.phasor.phasorS.phasor_clustering (phasor features, labels, cluster summaries, plots)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from pyfli.phasor.phasorS import PhasorAnalyzer
from pyfli.phasor.phasorS import phasor_clustering as pf

FREQ_HZ = 80e6
OMEGA = 2 * np.pi * FREQ_HZ * 1e-9


def semicircle(tau):
    g = 1.0 / (1.0 + (OMEGA * tau) ** 2)
    return g, OMEGA * tau * g


@pytest.fixture
def two_population_phasors():
    rng = np.random.default_rng(0)
    tau = np.where(np.arange(20 * 30).reshape(20, 30) % 30 < 15, 1.0, 3.5)
    G, S = [], []
    for h in (1, 2):
        g, s = semicircle(h * tau)
        G.append(g + rng.normal(0, 0.01, g.shape))
        S.append(s + rng.normal(0, 0.01, s.shape))
    truth = (tau > 2).astype(int) + 1
    return np.stack(G), np.stack(S), tau, truth


class TestFeatures:
    def test_feature_matrix_cartesian_and_polar(self, two_population_phasors):
        G, S, _, _ = two_population_phasors
        X, pixels, names = pf.phasor_feature_matrix(G, S, harmonics=(1, 2))
        assert (
            X.shape == (600, 4)
            and pixels.shape == (600, 2)
            and names == ["G_h1", "S_h1", "G_h2", "S_h2"]
        )
        np.testing.assert_allclose(X[:, 0], G[0][pixels[:, 0], pixels[:, 1]])
        Xp, _, names_p = pf.phasor_feature_matrix(G, S, harmonics=(1,), polar=True)
        assert names_p == ["phase_h1", "modulation_h1"]
        np.testing.assert_allclose(Xp[:, 1], np.hypot(G[0], S[0]).ravel())

    def test_mask_nan_extra_and_single_harmonic(self, two_population_phasors):
        G, S, _, _ = two_population_phasors
        G = G.copy()
        G[0, 0, 0] = np.nan
        mask = np.zeros(G.shape[1:], bool)
        mask[:5] = True
        X, pixels, names = pf.phasor_feature_matrix(
            G, S, mask=mask, extra_maps={"log_i": np.ones(G.shape[1:])}
        )
        assert (
            len(X) == 5 * 30 - 1
            and names[-1] == "log_i"
            and not ((pixels == [0, 0]).all(1)).any()
        )
        X2, _, _ = pf.phasor_feature_matrix(G[1], S[1])
        assert X2.shape == (600, 2)

    def test_label_roundtrip(self):
        pixels = np.array([[0, 1], [1, 0], [2, 2]])
        img = pf.labels_to_image(np.array([5, 6, 7]), pixels, (3, 3), fill=0)
        assert img[0, 1] == 5 and img[2, 2] == 7 and img[0, 0] == 0
        np.testing.assert_array_equal(pf.image_to_labels(img, pixels), [5, 6, 7])
        names = pf.labels_to_image(np.array(["a", "b", "c"]), pixels, (3, 3))
        assert names.dtype == object and names[1, 0] == "b" and names[0, 0] == -1


class TestLabelAlignment:
    def test_permuted_labels_align_perfectly(self):
        truth = np.array([1, 1, 2, 2, 3, 3])
        pred = np.array([7, 7, 4, 4, 9, 9])
        aligned, mapping = pf.align_labels(pred, truth)
        np.testing.assert_array_equal(aligned, truth)
        assert (
            mapping == {4: 2, 7: 1, 9: 3} and pf.clustering_accuracy(pred, truth) == 1.0
        )

    def test_extra_cluster_keeps_own_id(self):
        aligned, mapping = pf.align_labels(
            np.array([0, 0, 1, 2]), np.array([5, 5, 6, 6])
        )
        assert mapping[0] == 5 and mapping[1] == 6 and mapping[2] == 2
        assert (
            pf.clustering_accuracy(np.array([0, 0, 1, 2]), np.array([5, 5, 6, 6]))
            == 0.75
        )


class TestSummaryAndSklearnPipeline:
    def test_cluster_summary_lifetimes(self, two_population_phasors):
        G, S, tau, truth = two_population_phasors
        g, s = semicircle(tau)
        phasor = PhasorAnalyzer(
            frequency_hz=FREQ_HZ, time_axis_ns=np.arange(256) * 12.5 / 256, device="cpu"
        )
        label_map = truth.copy()
        label_map[0, 0] = -1
        out = pf.cluster_summary(g, s, label_map, phasor, param_maps={"tau": tau})
        assert list(out.index) == [1, 2] and out.loc[1, "pixels"] == 299
        np.testing.assert_allclose(out["tau_phase_ns"], [1.0, 3.5], rtol=1e-6)
        np.testing.assert_allclose(out["tau_modulation_ns"], [1.0, 3.5], rtol=1e-6)
        np.testing.assert_allclose(out["mean_tau"], [1.0, 3.5])

    def test_pipeline_with_sklearn_separates_populations(self, two_population_phasors):
        G, S, _, truth = two_population_phasors
        X, pixels, _ = pf.phasor_feature_matrix(G, S, harmonics=(1, 2))
        scores = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
        labels = GaussianMixture(2, n_init=3, random_state=0).fit_predict(scores)
        y = pf.image_to_labels(truth, pixels)
        assert pf.clustering_accuracy(labels, y) > 0.99


class TestPlots:
    def test_labels_to_rgb_and_legend(self):
        label_map = np.array([[0, 1], [2, 1]])
        rgb, colors = pf.labels_to_rgb(label_map, background=0)
        assert rgb.shape == (2, 2, 3) and set(colors) == {1, 2}
        np.testing.assert_array_equal(rgb[0, 0], 0)
        np.testing.assert_allclose(rgb[0, 1], colors[1])
        fig, ax = plt.subplots()
        legend = pf.add_label_legend(ax, colors, title="cluster")
        assert [t.get_text() for t in legend.get_texts()] == ["1", "2"]
        plt.close(fig)

    def test_plot_scores_2d_1d_and_subsampling(self):
        rng = np.random.default_rng(1)
        scores = rng.normal(size=(500, 2))
        labels = np.repeat([0, 1], 250)
        ax = pf.plot_scores(scores, labels, max_points=100)
        assert (
            sum(len(c.get_offsets()) for c in ax.collections) == 100
            and ax.get_legend() is not None
        )
        ax1 = pf.plot_scores(scores[:, :1], labels)
        assert ax1.get_ylabel() == "pixels" and len(ax1.patches) > 0
        ax2 = pf.plot_scores(scores)
        assert ax2.get_legend() is None
        plt.close("all")

    def test_plot_explained_variance(self):
        ax = pf.plot_explained_variance(np.array([0.7, 0.2, 0.1]))
        assert len(ax.patches) == 3
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.7, 0.9, 1.0])
        plt.close("all")


def test_exports():
    from pyfli.phasor import phasorS

    for name in (
        "phasor_feature_matrix",
        "labels_to_image",
        "image_to_labels",
        "align_labels",
        "clustering_accuracy",
        "cluster_summary",
        "labels_to_rgb",
        "add_label_legend",
        "plot_scores",
        "plot_explained_variance",
        "discrete_phasor_lifetime",
    ):
        assert callable(getattr(phasorS, name))
