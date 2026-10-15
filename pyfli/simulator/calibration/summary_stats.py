"""
Model-free per-pixel summary statistics and distances between simulated and
experimental decays.
"""

from functools import lru_cache

import numpy as np
from scipy.ndimage import uniform_filter1d

from pyfli.phasor.phasorS.phasor_simple import PhasorAnalyzer

STAT_WEIGHTS = {
    "log_counts": 1.0,  # photon-count distribution
    "phasor1": 2.0,  # lifetime content (first harmonic)
    "phasor2": 1.0,  # second harmonic: IRF width / multi-exponential shape
    "edge": 1.0,  # rising-edge position: IRF position and its pixel-to-pixel spread
    "bg_frac": 1.0,  # pre-edge background relative to the mean level
    "mean_decay": 2.0,  # log-shape of the pooled decay
}


def inactive_bins(pooled: np.ndarray, min_counts_per_bin: float = 100.0) -> np.ndarray:
    """
    Time bins that never record a count (outside the TCSPC TAC / ADC acquisition window).
    Only reported when the pooled decay is bright enough (``min_counts_per_bin`` on
    average) for an all-zero bin to be a property of the detector rather than chance.
    The pyfli simulator fills every bin, so these bins are zeroed in the simulated decays.
    """
    pooled = np.asarray(pooled, dtype=float)
    if pooled.sum() < min_counts_per_bin * len(pooled):
        return np.array([], dtype=int)
    return np.flatnonzero(pooled == 0)


def background_bins(
    mean_decay: np.ndarray, guard: int = 2, min_bins: int = 4
) -> np.ndarray:
    """
    Bins before the rising edge of the pooled decay (pure background + previous-pulse
    tail). If the edge is too early for that, the lowest bins of the pooled decay are used.
    """
    T = len(mean_decay)
    sm = uniform_filter1d(mean_decay.astype(float), 3, mode="wrap")
    peak = int(np.argmax(sm))
    base = np.percentile(sm, 5)
    level = base + 0.1 * (sm[peak] - base)
    edge = peak
    while edge > 0 and sm[edge - 1] > level:
        edge -= 1
    alive = mean_decay > 0  # skip dead bins outside the TAC window
    pre = np.arange(0, max(edge - guard, 0))
    pre = pre[alive[pre]]
    if len(pre) >= min_bins:
        return pre
    order = np.argsort(np.where(alive, sm, np.inf))
    return np.sort(order[: max(min_bins, T // 16)])


@lru_cache(maxsize=16)
def _phasor_analyzer(n_bins: int, harmonic: int) -> PhasorAnalyzer:
    """CPU ``PhasorAnalyzer`` whose period is the full ``n_bins`` time axis (1 ns per bin)."""
    return PhasorAnalyzer(
        frequency_hz=1e9 / n_bins,
        time_axis_ns=np.arange(n_bins, dtype=float),
        n_harmonics=harmonic,
        device="cpu",
    )


def phasor(decays: np.ndarray, harmonic: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Raw (IRF-uncalibrated) phasor ``(G, S)`` of each decay at ``harmonic``, over the whole
    time window (one laser period), computed with pyfli's ``PhasorAnalyzer``.
    """
    G, S = _phasor_analyzer(decays.shape[-1], harmonic).create_phasor_cpu(decays)
    return G[harmonic - 1], S[harmonic - 1]


def rising_edge(decays: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """Sub-bin position where the (3-bin smoothed) decay first crosses ``frac`` of its rise."""
    sm = uniform_filter1d(decays.astype(float), 3, axis=-1, mode="wrap")
    peak = sm.argmax(-1)
    base = np.percentile(sm, 10, axis=-1)
    level = base + frac * (sm[np.arange(len(sm)), peak] - base)
    edge = np.full(len(sm), np.nan)
    for i in range(len(sm)):  # walk back from the peak; cheap for a few 1e4 pixels
        k = peak[i]
        while k > 0 and sm[i, k - 1] > level[i]:
            k -= 1
        if k > 0:
            lo, hi = sm[i, k - 1], sm[i, k]
            edge[i] = k - 1 + (level[i] - lo) / (hi - lo) if hi > lo else k
        else:
            edge[i] = 0.0
    return edge


def compute_stats(decays: np.ndarray, bg_bins: np.ndarray) -> dict[str, np.ndarray]:
    decays = np.asarray(decays, dtype=float)
    counts = decays.sum(-1)
    g1, s1 = phasor(decays, 1)
    g2, s2 = phasor(decays, 2)
    mean_level = np.where(counts > 0, counts / decays.shape[-1], 1)
    mean_decay = decays.sum(0)
    return {
        "mean_decay_rel_var": pooled_relative_variance(decays),
        "log_counts": np.log10(counts + 1),
        "phasor1": np.stack([g1, s1], -1),
        "phasor2": np.stack([g2, s2], -1),
        "edge": rising_edge(decays),
        "bg_frac": decays[:, bg_bins].mean(-1) / mean_level,
        "mean_decay": mean_decay / max(mean_decay.sum(), 1e-12),
        "counts": counts,
    }


def pooled_relative_variance(
    decays: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Relative variance, per time bin, of the pooled (summed or weighted-mean) decay of
    the pixels in `decays` ``(n, T)``: ``Var(m_k) / m_k^2`` for the weighted mean
    ``m_k = sum_i w_i d_ik`` (weights summing to 1, uniform by default), estimated
    from the pixels as ``sum_i w_i^2 (d_ik - m_k)^2``. It contains the photon noise and
    the pixel-to-pixel spread, and is the variance of ``log m_k`` to first order.
    Bins with no counts get 0.
    """
    decays = np.asarray(decays, dtype=float)
    n = len(decays)
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)
    m = w @ decays
    var = (w**2) @ (decays - m) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m > 0, var / m**2, 0.0)


def _scale(x: np.ndarray) -> float:
    q25, q75 = np.percentile(x, [25, 75])
    return max((q75 - q25) / 1.349, 1e-3 * (abs(np.median(x)) + 1e-6), 1e-6)


U_GRID = (
    np.arange(1000) + 0.5
) / 1000  # quantile levels for Wasserstein-1 = mean |Q_sim(u) - Q_exp(u)|
PROJ_ANGLES = np.linspace(
    0, np.pi, 12, endpoint=False
)  # sliced Wasserstein directions for the phasor clouds
_DIRS = np.stack([np.cos(PROJ_ANGLES), np.sin(PROJ_ANGLES)], -1)


def _stat_vectors(stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Flatten the statistics into 1D samples: scalar stats plus phasor projections."""
    vec = {k: stats[k] for k in ("log_counts", "edge", "bg_frac")}
    for h in ("phasor1", "phasor2"):
        proj = stats[h] @ _DIRS.T
        for i in range(len(PROJ_ANGLES)):
            vec[f"{h}_{i}"] = proj[:, i]
    return vec


class PreparedSim:
    """Simulated statistics sorted once, so any set of pixel weights is scored in ~1 ms."""

    def __init__(self, stats: dict[str, np.ndarray]) -> None:
        self.n = len(stats["log_counts"])
        self.sorted, self.order = {}, {}
        for k, v in _stat_vectors(stats).items():
            ok = np.flatnonzero(np.isfinite(v))
            o = ok[np.argsort(v[ok], kind="stable")]
            self.order[k], self.sorted[k] = o, v[o]

    def quantiles(self, key: str, weights: np.ndarray | None) -> np.ndarray:
        vals = self.sorted[key]
        if weights is None:
            idx = np.minimum((U_GRID * len(vals)).astype(int), len(vals) - 1)
            return vals[idx]
        cw = np.cumsum(weights[self.order[key]])
        idx = np.minimum(np.searchsorted(cw, U_GRID * cw[-1]), len(vals) - 1)
        return vals[idx]


class StatsComparator:
    """
    Scale-free distance between simulated and experimental statistics, as a weighted sum of:

    * ``log_counts``, ``edge``, ``bg_frac``: Wasserstein-1 / robust spread of the experiment;
    * ``phasor1``, ``phasor2``: sliced Wasserstein over 12 directions / phasor spread;
    * ``mean_decay``: RMS log-ratio of the pooled decay shape over bins above 1 % of the
      peak (lower bins are noise-dominated; ``bg_frac`` covers the baseline), in units of 5 %.
      The expected sampling noise of both pooled decays (:func:`pooled_relative_variance`)
      is subtracted from the mean squared log-ratio, so two samples of the same
      distribution score ~0 whatever the number of simulated pixels.

    Wasserstein-1 is evaluated as the mean absolute difference of the quantile functions
    on a fixed grid, which supports importance weights at negligible cost.
    """

    def __init__(
        self, exp_stats: dict[str, np.ndarray], weights: dict[str, float] | None = None
    ) -> None:
        self.weights = weights or STAT_WEIGHTS
        vec = _stat_vectors(exp_stats)
        self.q_exp = {
            k: np.quantile(v[np.isfinite(v)], U_GRID, method="inverted_cdf")
            for k, v in vec.items()
        }
        self.scale = {
            k: _scale(exp_stats[k][np.isfinite(exp_stats[k])])
            for k in ("log_counts", "edge", "bg_frac")
        }
        for h in ("phasor1", "phasor2"):
            self.scale[h] = max(_scale(exp_stats[h][:, 0]), _scale(exp_stats[h][:, 1]))
        pe = exp_stats["mean_decay"]
        self.keep = pe > 1e-2 * pe.max()
        self.log_pe = np.log(pe[self.keep])
        self.var_pe = np.asarray(
            exp_stats.get("mean_decay_rel_var", np.zeros_like(pe))
        )[self.keep]

    def distance(
        self,
        sim: PreparedSim | dict,
        mean_decay: np.ndarray | None = None,
        pixel_weights: np.ndarray | None = None,
        mean_decay_rel_var: np.ndarray | None = None,
    ) -> tuple[float, dict[str, float]]:
        if isinstance(sim, dict):
            mean_decay = sim["mean_decay"] if mean_decay is None else mean_decay
            if mean_decay_rel_var is None:
                mean_decay_rel_var = sim.get("mean_decay_rel_var")
            sim = PreparedSim(sim)

        def w1(k):
            return float(
                np.mean(np.abs(sim.quantiles(k, pixel_weights) - self.q_exp[k]))
            )

        parts = {k: w1(k) / self.scale[k] for k in ("log_counts", "edge", "bg_frac")}
        for h in ("phasor1", "phasor2"):
            parts[h] = (
                float(np.mean([w1(f"{h}_{i}") for i in range(len(PROJ_ANGLES))]))
                / self.scale[h]
            )
        md = mean_decay / mean_decay.sum()
        var_sim = (
            0.0
            if mean_decay_rel_var is None
            else np.asarray(mean_decay_rel_var)[self.keep]
        )
        sq = (np.log(md[self.keep] + 1e-12) - self.log_pe) ** 2 - var_sim - self.var_pe
        parts["mean_decay"] = float(np.sqrt(max(float(np.mean(sq)), 0.0))) / 0.05
        total = float(sum(self.weights.get(k, 0.0) * v for k, v in parts.items()))
        return total, parts
