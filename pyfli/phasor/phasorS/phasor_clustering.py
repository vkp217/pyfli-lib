"""Convert phasor images to feature matrices and back for PCA, clustering and LDA."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _as_stack(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[None] if a.ndim == 2 else a


def phasor_feature_matrix(
    G: np.ndarray,
    S: np.ndarray,
    mask: np.ndarray | None = None,
    harmonics: Any = (1,),
    polar: bool = False,
    extra_maps: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    One feature row per pixel from multi-harmonic phasor coordinates.

    Parameters
    ----------
    G, S : np.ndarray
        Phasor coordinates, ``(n_harmonics, H, W)`` or ``(H, W)``.
    mask : np.ndarray | None
        Pixels to use (``(H, W)`` boolean); default all. Pixels with any non-finite
        feature are dropped as well.
    harmonics : iterable of int
        1-based harmonics to include.
    polar : bool
        Use phase ``atan2(S, G)`` and modulation ``sqrt(G^2 + S^2)`` instead of
        ``G`` / ``S``.
    extra_maps : dict[str, np.ndarray] | None
        Additional ``(H, W)`` maps appended as features (e.g. ``{"log_intensity":
        np.log1p(decay.sum(-1))}``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        ``(X, pixels, names)``: features ``(n_pixels, n_features)``, the ``(row,
        col)`` coordinates of every row, and the feature names.
    """
    G, S = _as_stack(G), _as_stack(S)
    h, w = G.shape[1:]
    mask = np.ones((h, w), bool) if mask is None else np.asarray(mask, dtype=bool)
    columns, names = [], []
    for k in harmonics:
        g, s = G[k - 1], S[k - 1]
        if polar:
            columns += [np.arctan2(s, g), np.hypot(g, s)]
            names += [f"phase_h{k}", f"modulation_h{k}"]
        else:
            columns += [g, s]
            names += [f"G_h{k}", f"S_h{k}"]
    for name, arr in (extra_maps or {}).items():
        columns.append(np.asarray(arr, dtype=float))
        names.append(name)
    stack = np.stack(columns, axis=-1)
    usable = mask & np.all(np.isfinite(stack), axis=-1)
    pixels = np.argwhere(usable)
    return stack[usable], pixels, names


def labels_to_image(
    labels: np.ndarray, pixels: np.ndarray, shape: tuple[int, int], fill: Any = -1
) -> np.ndarray:
    """Place per-pixel `labels` at their `pixels` coordinates in an image of `shape`."""
    labels = np.asarray(labels)
    out = np.full(
        shape,
        fill,
        dtype=labels.dtype if np.issubdtype(labels.dtype, np.number) else object,
    )
    out[pixels[:, 0], pixels[:, 1]] = labels
    return out


def image_to_labels(label_map: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Read a label (e.g. the ground-truth region id) for every pixel in `pixels`."""
    return np.asarray(label_map)[pixels[:, 0], pixels[:, 1]]


def align_labels(
    predicted: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, dict]:
    """
    Rename cluster ids so they best match `reference` labels (Hungarian assignment on
    the contingency table). Cluster ids are arbitrary; this makes them comparable with
    ground truth. Returns ``(aligned_labels, mapping)``; clusters without a partner
    keep their own id.
    """
    predicted, reference = np.asarray(predicted), np.asarray(reference)
    p_ids, t_ids = np.unique(predicted), np.unique(reference)
    table = np.array(
        [[np.sum((predicted == p) & (reference == t)) for t in t_ids] for p in p_ids]
    )
    rows, cols = linear_sum_assignment(-table)
    mapping = {p: p for p in p_ids}
    mapping.update({p_ids[r]: t_ids[c] for r, c in zip(rows, cols)})
    return np.array([mapping[p] for p in predicted]), mapping


def clustering_accuracy(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Fraction of pixels whose (aligned) cluster matches the reference label."""
    aligned, _ = align_labels(predicted, reference)
    return float(np.mean(aligned == np.asarray(reference)))


def cluster_summary(
    G1: np.ndarray,
    S1: np.ndarray,
    label_map: np.ndarray,
    phasor: Any,
    param_maps: dict[str, np.ndarray] | None = None,
    background: Any = -1,
    intensity: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Per-cluster summary of a first-harmonic phasor image: pixel count, mean ``G`` /
    ``S``, the phase and modulation lifetimes of the cluster-mean phasor -- computed
    with the given :class:`pyfli.phasor.phasorS.PhasorAnalyzer` (``compute_lifetime``
    / ``compute_modulation_lifetime``) -- and the mean of every map in `param_maps`
    (e.g. the simulator's ground-truth maps) inside the cluster. Pixels labelled
    `background` are ignored.

    By default every pixel counts equally. Pass the photon-count image as `intensity`
    to weight the mean phasor by photons instead: it is then exactly the phasor of the
    cluster's summed decay (phasors add with intensity weights).
    """
    label_map = np.asarray(label_map)
    rows = []
    for c in [c for c in np.unique(label_map) if c != background]:
        sel = (label_map == c) & np.isfinite(G1) & np.isfinite(S1)
        w = None if intensity is None else np.asarray(intensity, dtype=float)[sel]
        g = np.array([np.average(G1[sel], weights=w)])
        s = np.array([np.average(S1[sel], weights=w)])
        row = {
            "cluster": c,
            "pixels": int(sel.sum()),
            "G_mean": float(g[0]),
            "S_mean": float(s[0]),
            "tau_phase_ns": float(np.ravel(phasor.compute_lifetime(g, s))[0]),
            "tau_modulation_ns": float(
                np.ravel(phasor.compute_modulation_lifetime(g, s))[0]
            ),
        }
        for name, arr in (param_maps or {}).items():
            vals = np.asarray(arr, dtype=float)[sel]
            row[f"mean_{name}"] = float(np.nanmean(vals)) if vals.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("cluster")


def _cluster_colors(n: int) -> list:
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def labels_to_rgb(
    label_map: np.ndarray, background: Any = -1
) -> tuple[np.ndarray, dict]:
    """
    ``(H, W, 3)`` RGB image with one categorical color per label (background black),
    and the ``{label: color}`` legend. Pass the image as ``colors=`` to
    :meth:`pyfli.phasor.phasorS.PhasorAnalyzer.plot_phasor_diagram` to color the phasor
    points by cluster, or show it with ``imshow``.
    """
    label_map = np.asarray(label_map)
    classes = [c for c in np.unique(label_map) if c != background]
    colors = dict(zip(classes, (c[:3] for c in _cluster_colors(len(classes)))))
    rgb = np.zeros((*label_map.shape, 3))
    for c, color in colors.items():
        rgb[label_map == c] = color
    return rgb, colors


def add_label_legend(
    ax: Any, colors: dict, title: str | None = None, loc: str = "upper right"
) -> Any:
    """Legend of `colors` (from :func:`labels_to_rgb`) on `ax`, placed at `loc`."""
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", color=col, label=str(c))
        for c, col in colors.items()
    ]
    return ax.legend(handles=handles, title=title, frameon=False, loc=loc)


def plot_scores(
    scores: np.ndarray,
    labels: np.ndarray | None = None,
    ax: Any | None = None,
    axis_names: tuple[str, str] = ("component 1", "component 2"),
    title: str = "Scores",
    max_points: int = 20000,
    seed: Any = 0,
) -> Any:
    """
    Scatter of the first two columns of `scores` (PCA or LDA projections), colored by
    `labels` when given. A single-column projection is drawn as per-label histograms.
    """
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    scores = np.asarray(scores)
    labels = np.zeros(len(scores), int) if labels is None else np.asarray(labels)
    classes = np.unique(labels)
    idx = np.arange(len(scores))
    if len(idx) > max_points:
        idx = np.random.default_rng(seed).choice(idx, max_points, replace=False)
    for color, c in zip(_cluster_colors(len(classes)), classes):
        sel = idx[labels[idx] == c]
        if scores.shape[1] == 1:
            ax.hist(scores[sel, 0], bins=80, alpha=0.5, color=color, label=str(c))
        else:
            ax.scatter(
                scores[sel, 0],
                scores[sel, 1],
                s=2,
                alpha=0.4,
                color=color,
                label=str(c),
            )
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel("pixels" if scores.shape[1] == 1 else axis_names[1])
    ax.set_title(title, fontweight="bold")
    if len(classes) > 1:
        ax.legend(markerscale=4, frameon=False)
    return ax


def plot_explained_variance(
    ratio: np.ndarray, ax: Any | None = None, title: str = "Explained variance"
) -> Any:
    """Bar chart of per-component explained-variance ratio with the cumulative curve."""
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    ratio = np.asarray(ratio)
    x = np.arange(1, len(ratio) + 1)
    ax.bar(x, ratio, color="tab:blue", alpha=0.7, label="per component")
    ax.plot(x, np.cumsum(ratio), "o-", color="tab:orange", label="cumulative")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("component")
    ax.set_ylabel("explained variance ratio")
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False)
    return ax
