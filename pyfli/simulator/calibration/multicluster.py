"""Multi-cluster priors and images, following the pyfli multi-ROI example."""

import json
import os
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

from pyfli import logging

from ..irf_sim.irf_offset_gen import OffsetGen
from ..separate.sim_model_image_generator import FLIModelImageGenerator
from .config import CalibrationInputs
from .sim_bridge import TUPLE_KEYS
from .summary_stats import phasor


@dataclass
class ClusterResult:
    labels: np.ndarray  # (H, W) int: 0 outside the sample mask, 1..K clusters
    n_clusters: int
    fractions: list[float]
    report: dict = field(default_factory=dict)

    def mask(self, k: int) -> np.ndarray:
        return self.labels == k


def _ashman_d(gm: GaussianMixture, i: int, j: int) -> float:
    d = gm.means_[i] - gm.means_[j]
    u = d / max(np.linalg.norm(d), 1e-12)
    vi, vj = u @ gm.covariances_[i] @ u, u @ gm.covariances_[j] @ u
    return float(np.sqrt(2) * np.linalg.norm(d) / np.sqrt(vi + vj))


def _dip_ratio(
    z: np.ndarray,
    labels: np.ndarray,
    mean_i: np.ndarray,
    mean_j: np.ndarray,
    i: int,
    j: int,
) -> float:
    """
    Depth of the density valley between two components: the pixels assigned to
    either are projected onto the line joining the means, a Gaussian KDE (Scott
    bandwidth) of the projection is evaluated between the two projected means, and the
    minimum is divided by the smaller of the densities at the means. ~0 for two
    separate populations, close to 1 when there is no valley (one population).
    """
    d = mean_j - mean_i
    u = d / max(np.linalg.norm(d), 1e-12)
    proj = z[(labels == i) | (labels == j)] @ u
    if len(proj) < 5 or np.ptp(proj) == 0:
        return 1.0
    kde = gaussian_kde(proj)
    a, b = float(mean_i @ u), float(mean_j @ u)
    between = kde(np.linspace(min(a, b), max(a, b), 200)).min()
    return float(between / max(min(kde([a])[0], kde([b])[0]), 1e-300))


def _features(decays: np.ndarray, use_intensity: bool) -> np.ndarray:
    g, s = phasor(decays, 1)
    cols = [g, s] + ([np.log10(decays.sum(-1) + 1)] if use_intensity else [])
    return np.stack(cols, -1)


def detect_clusters(
    decay: np.ndarray,
    mask: np.ndarray,
    max_components: int = 4,
    separation: float = 2.0,
    min_fraction: float = 0.05,
    use_intensity: bool = False,
    n_fit: int = 20000,
    smooth: bool = True,
    seed: int = 0,
    max_dip_ratio: float = 0.5,
) -> ClusterResult:
    """
    Find well-separated pixel populations inside ``mask``; see the module docstring.

    Two mixture components stay separate only if Ashman's D is at least `separation`
    **and** the pixel density projected onto the line joining their means has a real
    dip between them (:func:`_dip_ratio` below `max_dip_ratio`). A mixture fitted to a
    few hundred pixels of one skewed population can reach D > 2 by chance, but its
    projected density has no dip (ratio ~0.7-1), while separate populations give ~0.
    """
    rng = np.random.default_rng(seed)
    coords = np.argwhere(mask)
    feats = np.concatenate(
        [
            _features(
                np.asarray(
                    decay[coords[i : i + 50000, 0], coords[i : i + 50000, 1]],
                    dtype=np.float64,
                ),
                use_intensity,
            )
            for i in range(0, len(coords), 50000)
        ]
    )
    mu, sd = feats.mean(0), feats.std(0) + 1e-12
    z = (feats - mu) / sd
    fit = z[rng.choice(len(z), min(n_fit, len(z)), replace=False)]
    gm = GaussianMixture(
        max_components, covariance_type="full", random_state=seed, n_init=3
    ).fit(fit)

    # merge components that are not clearly separated (union-find on Ashman's D)
    K = max_components
    D = np.array(
        [[_ashman_d(gm, i, j) if i != j else 0.0 for j in range(K)] for i in range(K)]
    )
    parent = list(range(K))

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    labels_fit = gm.predict(fit)
    dip = np.ones((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            if D[i, j] >= separation:
                dip[i, j] = dip[j, i] = _dip_ratio(
                    fit, labels_fit, gm.means_[i], gm.means_[j], i, j
                )
            if D[i, j] < separation or dip[i, j] >= max_dip_ratio:
                parent[find(j)] = find(i)
    comp_group = np.array([find(i) for i in range(K)])
    # tiny groups join the closest other group
    for _ in range(K):
        groups = np.unique(comp_group)
        w = {g: gm.weights_[comp_group == g].sum() for g in groups}
        small = [g for g in groups if w[g] < min_fraction]
        if not small or len(groups) == 1:
            break
        g = small[0]
        others = [h for h in groups if h != g]
        dist = {
            h: min(
                D[i, j]
                for i in np.flatnonzero(comp_group == g)
                for j in np.flatnonzero(comp_group == h)
            )
            for h in others
        }
        comp_group[comp_group == g] = min(dist, key=dist.get)

    groups = np.unique(comp_group)
    # order clusters by mean lifetime proxy (phasor G, high G = short lifetime) for stable labels
    resp = gm.predict_proba(z)
    group_prob = np.stack([resp[:, comp_group == g].sum(1) for g in groups], -1)
    pix_group = group_prob.argmax(1)
    g_mean = np.array([feats[pix_group == i, 0].mean() for i in range(len(groups))])
    order = np.argsort(-g_mean)  # cluster 1 = highest G (shortest lifetime)
    relabel = np.empty_like(order)
    relabel[order] = np.arange(len(order))
    labels = np.zeros(mask.shape, dtype=int)
    labels[coords[:, 0], coords[:, 1]] = relabel[pix_group] + 1

    n = len(groups)
    if smooth and n > 1:  # 3x3 majority filter inside the mask
        votes = np.stack(
            [
                ndimage.uniform_filter((labels == k).astype(float), 3)
                for k in range(1, n + 1)
            ],
            -1,
        )
        labels = np.where(mask, votes.argmax(-1) + 1, 0)

    fractions = [float(np.mean(labels[mask] == k)) for k in range(1, n + 1)]
    report = {
        "n_clusters": n,
        "fractions": fractions,
        "separation_threshold": separation,
        "ashman_d_components": D.round(2).tolist(),
        "dip_ratio_components": dip.round(3).tolist(),
        "component_group": comp_group.tolist(),
        "features": ["phasor G (h=1)", "phasor S (h=1)"]
        + (["log10 photons"] if use_intensity else []),
        "cluster_phasor_means": [
            [float(v) for v in feats[labels[coords[:, 0], coords[:, 1]] == k].mean(0)]
            for k in range(1, n + 1)
        ],
    }
    logging.info(
        f"Clusters: {n} (fractions {np.round(fractions, 3).tolist()}); max Ashman D {D.max():.2f}"
    )
    return ClusterResult(labels, n, fractions, report)


def cluster_inputs(
    base: CalibrationInputs,
    cluster_mask_path: str,
    whole_irf_path: str,
    output_dir: str,
) -> CalibrationInputs:
    """
    Inputs for calibrating one cluster: the experiment's inputs restricted to the cluster
    mask, with the IRF of the whole-experiment calibration (the IRF belongs to the detector,
    not to a cell population).
    """
    return replace(
        base,
        mask_path=cluster_mask_path,
        irf_path=whole_irf_path,
        irf_mode="experimental",
        output_dir=output_dir,
    )


# ------------------------------------------------------------------- image generation
def roi_config(json_path: str, seed: int | None = None) -> dict:
    """A calibrated config as a pyfli ``roi_params`` entry (tuples restored; optional fixed seed)."""
    with open(json_path) as f:
        cfg = dict(json.load(f)["config"])
    for k in TUPLE_KEYS:
        if isinstance(cfg.get(k), list):
            cfg[k] = tuple(cfg[k])
    cfg["seed"] = (
        seed  # FLIModelImageGenerator re-uses one simulator per ROI, so a seed is safe here
    )
    return cfg


def irf_tile(
    json_path: str, shape: tuple[int, int], seed: int = 0, peak: float = 1e4
) -> np.ndarray:
    """
    Per-pixel IRF cube ``(H, W, T)`` with the calibrated ``OffsetGen`` behaviour: each pixel
    gets ``roll(template, round(a)) + b``, a ~ U(a_range), b ~ U(b_range). It is scaled to
    ``peak`` because pyfli's ``irf_picker`` only accepts 3D IRFs whose peak exceeds 500
    (each pixel is sum-normalised by the generator anyway).
    """
    with open(json_path) as f:
        p = json.load(f)
    template = np.load(os.path.join(os.path.dirname(json_path), p["irf"]["file"]))[
        0, 0
    ].astype(float)
    a_lo, a_hi = p["sim_generator_kwargs"]["a_range"]
    b_lo, b_hi = p["sim_generator_kwargs"]["b_range"]
    base = np.broadcast_to(template, (*shape, len(template))).copy()
    state = np.random.get_state()
    np.random.seed(seed % (2**32))
    try:
        cube, _, _ = OffsetGen(
            base, a_range=(a_lo, a_hi), b_range=(b_lo, b_hi)
        ).sample_cube()
    finally:
        np.random.set_state(state)
    return cube * (peak / cube.max())


def choose_tile(
    labels: np.ndarray, shape: tuple[int, int] = (8, 8), seed: int = 0
) -> tuple[int, int]:
    """
    Top-left corner of a tile fully inside the sample mask that contains as many clusters
    as possible, with the most balanced composition (largest minority share).
    """
    H, W = shape
    win = sliding_window_view(labels, (H, W))
    inside = (win > 0).all(axis=(-1, -2))
    K = labels.max()
    counts = np.stack([(win == k).sum(axis=(-1, -2)) for k in range(1, K + 1)], -1)
    present = (counts > 0).sum(-1)
    minority = np.where(present > 1, np.where(counts > 0, counts, H * W).min(-1), 0)
    score = np.where(inside, present * 1000 + minority, -1)
    best = np.flatnonzero(score == score.max())
    r, c = np.unravel_index(np.random.default_rng(seed).choice(best), score.shape)
    return int(r), int(c)


def simulate_multicluster_tile(
    labels_tile: np.ndarray,
    cluster_configs: dict[int, str],
    irf_json: str,
    detector_type: str = "discrete",
    sample_mask_tile: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    """
    pyfli ``FLIModelImageGenerator`` image for a cluster label tile, as in the pyfli
    multi-ROI example: ``roi_params`` follows ``np.unique(roi_mask)``. Label 0 (outside the
    sample) is background with an empty config and is zeroed by the intensity / bool mask.
    """
    sample = labels_tile > 0 if sample_mask_tile is None else sample_mask_tile
    present = np.unique(labels_tile)
    roi_params = [
        {} if k == 0 else roi_config(cluster_configs[int(k)], seed=seed + int(k))
        for k in present
    ]
    np.random.seed(seed)
    gen = FLIModelImageGenerator(
        irf_data=irf_tile(irf_json, labels_tile.shape, seed=seed),
        intensity_image=sample.astype(float),
        roi_mask=labels_tile,
        roi_params=roi_params,
        image_shape=labels_tile.shape,
        method="PHOTON_COUNTER" if detector_type == "discrete" else "continuous",
        verbose=False,
        bool_mask=sample,
    )
    out = gen.generate_image()
    with open(irf_json) as f:
        inactive = json.load(f)["sim_generator_kwargs"].get("inactive_bins") or []
    if len(inactive):
        out["raw_data"]["decay"] = np.array(out["raw_data"]["decay"], dtype=float)
        out["raw_data"]["decay"][..., inactive] = 0.0
    out["roi_mask"] = labels_tile
    return out
