"""Fit-aware tolerance probability T_GOF = T_delta * A_chi * A_R * A_M_tilde
for Bayesian FLI posteriors (trustworthiness note, Sections 3 and 7)."""

from typing import ClassVar

import numpy as np
from scipy.special import ndtr

from pyfli.cm_stats.chi_square import degrees_of_freedom, pearson_terms
from pyfli.cm_stats.deviance import deviance_dof, poisson_deviance_terms
from pyfli.reconstruction import ParamToDecay

_LAM_FLOOR = 1e-12
_SHAPE_EPS = 1e-12
_DRAW_BUDGET = 2**24

STATISTICS = ("pearson", "deviance")


def _expanded_delta(delta, center, axis, relative):
    center = np.asarray(center, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if np.any(delta < 0):
        raise ValueError("delta must be non-negative.")
    delta = np.broadcast_to(delta, center.shape)
    if relative:
        delta = delta * np.abs(center)
    return np.expand_dims(center, axis), np.expand_dims(delta, axis)


def tolerance_probability(samples, delta, center=None, relative=False, axis=-1):
    """T_delta = P(|theta - center| < delta | y); center defaults to the median (eq. 5)."""
    samples = np.asarray(samples, dtype=float)
    if center is None:
        center = np.median(samples, axis=axis)
    c, d = _expanded_delta(delta, center, axis, relative)
    return np.mean(np.abs(samples - c) < d, axis=axis)


def joint_tolerance_probability(samples, deltas, centers=None, relative=False, axis=-1):
    """Joint T_delta: probability all parameters in ``deltas`` are within tolerance."""
    centers = centers or {}
    inside = None
    for key, delta in deltas.items():
        s = np.asarray(samples[key], dtype=float)
        center = centers.get(key)
        if center is None:
            center = np.median(s, axis=axis)
        c, d = _expanded_delta(delta, center, axis, relative)
        hit = np.abs(s - c) < d
        inside = hit if inside is None else inside & hit
    if inside is None:
        raise ValueError("deltas must name at least one parameter.")
    return np.mean(inside, axis=axis)


def tolerance_standard_error(T, n_eff):
    """Monte-Carlo standard error sqrt(T (1 - T) / n_eff)."""
    T = np.asarray(T, dtype=float)
    return np.sqrt(T * (1.0 - T) / np.asarray(n_eff, dtype=float))


def gaussian_tolerance_probability(delta, sigma, phi=1.0):
    """Gaussian limit 2 Phi(delta / (sigma sqrt(phi))) - 1."""
    scale = np.asarray(sigma, dtype=float) * np.sqrt(np.asarray(phi, dtype=float))
    return 2.0 * ndtr(np.asarray(delta, dtype=float) / scale) - 1.0


def dispersion_adjusted_tolerance(
    samples, delta, phi, center=None, relative=False, axis=-1
):
    """T_adj: tolerance probability of draws inflated by sqrt(max(1, phi)) (eq. 32)."""
    samples = np.asarray(samples, dtype=float)
    if center is None:
        center = np.median(samples, axis=axis)
    center = np.asarray(center, dtype=float)
    phi = np.broadcast_to(np.asarray(phi, dtype=float), center.shape)
    phi_hat = np.where(np.isnan(phi), np.nan, np.fmax(1.0, phi))
    scaled = np.broadcast_to(np.asarray(delta, dtype=float), center.shape) / np.sqrt(
        phi_hat
    )
    T = tolerance_probability(
        samples, np.nan_to_num(scaled), center, relative=relative, axis=axis
    )
    return np.where(np.isnan(scaled), np.nan, T)


def mean_lifetime_draws(alpha1, tau1, tau2):
    """Per-draw mean lifetime tau_m = alpha1 tau1 + (1 - alpha1) tau2."""
    alpha1 = np.asarray(alpha1, dtype=float)
    return alpha1 * np.asarray(tau1, dtype=float) + (1.0 - alpha1) * np.asarray(
        tau2, dtype=float
    )


def _window_weights(shape, mask):
    if mask is None:
        return np.ones(shape)
    return np.broadcast_to(np.asarray(mask, dtype=bool), shape).astype(float)


def _dof(w, n_params):
    nu = degrees_of_freedom(np.sum(w, axis=-1), n_params, None)
    return nu, np.where(nu > 0, nu, np.nan)


def reduced_chi2(y, lam, n_params, mask=None, statistic="pearson"):
    """
    Return (chi2_nu, nu): Pearson chi2 / nu (eq. 20), or deviance D / (sum E[D_k] - p),
    the solver's reduced deviance (eq. 20's nu would bias it at low counts).
    """
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}, got {statistic!r}.")
    y = np.asarray(y, dtype=float)
    lam = np.fmax(np.asarray(lam, dtype=float), _LAM_FLOOR)
    shape = np.broadcast_shapes(y.shape, lam.shape)
    w = _window_weights(shape, mask)
    nu, nu_safe = _dof(w, n_params)
    if statistic == "pearson":
        terms = pearson_terms(lam, y, floor=_LAM_FLOOR)
    else:
        terms = poisson_deviance_terms(lam, y, floor=_LAM_FLOOR, nonpositive="nan")
        expected = deviance_dof(np.broadcast_to(lam, shape), n_params, weights=w)
        nu_safe = np.where(expected > 0, expected, np.nan)
    return np.sum(w * terms, axis=-1) / nu_safe, nu


def residual_noise_ratio(y, lam, n_params, mask=None):
    """Unweighted R2, its Poisson ceiling R2_ceil and rho_R = RSS / E[RSS] (eqs. 22-24)."""
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float)
    shape = np.broadcast_shapes(y.shape, lam.shape)
    y = np.broadcast_to(y, shape)
    w = _window_weights(shape, mask)
    k_win = np.sum(w, axis=-1)
    nu, nu_safe = _dof(w, n_params)
    with np.errstate(divide="ignore", invalid="ignore"):
        rss = np.sum(w * (y - lam) ** 2, axis=-1)
        e_rss = nu_safe / k_win * np.sum(w * lam, axis=-1)
        y_bar = np.sum(w * y, axis=-1, keepdims=True) / k_win[..., None]
        tss = np.sum(w * (y - y_bar) ** 2, axis=-1)
        tss_safe = np.where(tss > 0, tss, np.nan)
        e_rss_safe = np.where(e_rss > 0, e_rss, np.nan)
        return {
            "RSS": rss,
            "E_RSS": e_rss,
            "TSS": tss,
            "R2": 1.0 - rss / tss_safe,
            "R2_ceil": 1.0 - e_rss / tss_safe,
            "rho_R": rss / e_rss_safe,
            "nu": nu,
        }


def weighted_r2(y, lam, n_params, mask=None):
    """Return (R2_w, R2_w_ceil) with weights 1/lam (eq. 25)."""
    y = np.asarray(y, dtype=float)
    lam = np.fmax(np.asarray(lam, dtype=float), _LAM_FLOOR)
    shape = np.broadcast_shapes(y.shape, lam.shape)
    y = np.broadcast_to(y, shape)
    win = _window_weights(shape, mask)
    wk = win / lam
    nu, nu_safe = _dof(win, n_params)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_bar_w = np.sum(wk * y, axis=-1, keepdims=True) / np.sum(
            wk, axis=-1, keepdims=True
        )
        tss_w = np.sum(wk * (y - y_bar_w) ** 2, axis=-1)
        tss_w = np.where(tss_w > 0, tss_w, np.nan)
        return (
            1.0 - np.sum(wk * (y - lam) ** 2, axis=-1) / tss_w,
            1.0 - nu_safe / tss_w,
        )


def acceptance_band(nu, z0=2.0, eps=0.2):
    """Return (lower, upper) = (1 - z0 sqrt(2/nu), 1 + max(z0 sqrt(2/nu), eps)) (eq. 29)."""
    if z0 < 0 or eps < 0:
        raise ValueError("z0 and eps must be non-negative.")
    nu = np.asarray(nu, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        half = z0 * np.sqrt(2.0 / np.where(nu > 0, nu, np.nan))
    return 1.0 - half, 1.0 + np.maximum(half, eps)


def gate(x, lower, upper, s=0.2):
    """Tolerant Gaussian gate: 1 inside [lower, upper], half-Gaussian of width s outside (eq. 28)."""
    if s <= 0:
        raise ValueError("s must be positive.")
    x = np.asarray(x, dtype=float)
    excess = np.fmax(lower - x, 0.0) + np.fmax(x - upper, 0.0)
    out = np.exp(-(excess**2) / (2.0 * s**2))
    return np.where(np.isnan(x) | np.isnan(lower) | np.isnan(upper), np.nan, out)


def median_representativeness(chi2_draws, chi2_median, axis=-1):
    """A_M: fraction of draws with chi2 >= chi2 of the median curve (eq. 26)."""
    chi2_draws = np.asarray(chi2_draws, dtype=float)
    ref = np.expand_dims(np.asarray(chi2_median, dtype=float), axis)
    return np.mean(chi2_draws >= ref, axis=axis)


def representativeness_penalty(a_m):
    """A_M_tilde = min(1, 2 A_M) (eq. 27)."""
    a_m = np.asarray(a_m, dtype=float)
    return np.where(np.isnan(a_m), np.nan, np.fmin(1.0, 2.0 * a_m))


def early_window_mask(t, lam, tau1, n_tau=3.0, mask=None):
    """Boolean early window t_k <= t_peak + n_tau * tau1, t_peak = argmax of lam."""
    t = np.asarray(t, dtype=float)
    lam = np.asarray(lam, dtype=float)
    t_peak = t[np.argmax(np.nan_to_num(lam, nan=-np.inf), axis=-1)]
    limit = t_peak + n_tau * np.asarray(tau1, dtype=float)
    early = t <= limit[..., None]
    if mask is not None:
        early = early & np.broadcast_to(np.asarray(mask, dtype=bool), early.shape)
    return early


def combine_fit_aware_factors(
    T_delta,
    chi2_nu,
    rho_R,
    a_m,
    nu,
    z0=2.0,
    eps=0.2,
    s=0.2,
    chi2_nu_early=None,
    rho_R_early=None,
    nu_early=None,
):
    """Gate the fit statistics and return the factors and T_gof (eq. 31)."""
    lower, upper = acceptance_band(nu, z0, eps)
    a_chi = gate(chi2_nu, lower, upper, s)
    a_r = gate(rho_R, lower, upper, s)
    if chi2_nu_early is not None:
        lo_e, up_e = acceptance_band(nu_early, z0, eps)
        a_chi = np.fmin(a_chi, gate(chi2_nu_early, lo_e, up_e, s))
        a_r = np.fmin(a_r, gate(rho_R_early, lo_e, up_e, s))
    a_m_tilde = representativeness_penalty(a_m)
    return {
        "A_chi": a_chi,
        "A_R": a_r,
        "A_M_tilde": a_m_tilde,
        "lower": lower,
        "upper": upper,
        "T_gof": np.asarray(T_delta, dtype=float) * a_chi * a_r * a_m_tilde,
    }


class PosteriorDecayForward:
    """Forward model lambda(theta) via ParamToDecay, amplitude pinned to the measured total.

    ``background`` (v_shift) is None, a value, a map or ``"auto"`` (Poisson MLE, seeded from pre-onset bins);
    ``background_key`` takes it from the posterior instead. ``h_shift`` (onset, ns) is None, a value
    or a map; ``h_shift_key`` takes it from the posterior; ``excitation``/``gate_width`` are the
    forward-model options of :class:`~pyfli.reconstruction.ParamToDecay`. Called as
    ``forward(params, y)``.
    """

    PARAM_KEYS: ClassVar[dict[str, tuple[str, ...]]] = {
        "bi-exponential": ("alpha1", "tau1", "tau2"),
        "mono-exponential": ("tau",),
    }

    def __init__(
        self,
        model_type,
        freq,
        irf=None,
        num_gates=None,
        shift_method="zero_pad",
        background_key=None,
        amplitude_mask=None,
        background=None,
        background_bins=None,
        onset_fraction=0.01,
        h_shift=None,
        h_shift_key=None,
        excitation="bin_start",
        gate_width=None,
    ):
        if model_type not in self.PARAM_KEYS:
            raise ValueError(
                f"model_type must be one of {tuple(self.PARAM_KEYS)}, got {model_type!r}."
            )
        if background is not None and background_key is not None:
            raise ValueError("Give either background or background_key, not both.")
        if h_shift is not None and h_shift_key is not None:
            raise ValueError("Give either h_shift or h_shift_key, not both.")
        self.h_shift = None if h_shift is None else np.asarray(h_shift, dtype=float)
        self.h_shift_key = h_shift_key
        self.excitation = excitation
        self.gate_width = gate_width
        if isinstance(background, str) and background != "auto":
            raise ValueError(
                f"background must be None, a number, a map or 'auto', got {background!r}."
            )
        if not 0 < onset_fraction < 1:
            raise ValueError("onset_fraction must be in (0, 1).")
        self.model_type = model_type
        self.freq = float(freq)
        self.irf = None if irf is None else np.asarray(irf, dtype=float)
        if self.irf is not None:
            num_gates = self.irf.shape[-1]
        elif num_gates is None:
            raise ValueError("num_gates is required when irf is None.")
        self.num_gates = int(num_gates)
        self.shift_method = shift_method
        self.background_key = background_key
        self.amplitude_mask = (
            None if amplitude_mask is None else np.asarray(amplitude_mask, dtype=bool)
        )
        self.background = (
            background
            if background is None or isinstance(background, str)
            else np.asarray(background, dtype=float)
        )
        self.onset_fraction = onset_fraction
        self.background_bins = (
            None if background_bins is None else np.asarray(background_bins, dtype=bool)
        )
        if isinstance(self.background, str) and self.background_bins is None:
            if self.irf is None:
                raise ValueError(
                    "background='auto' needs an irf or explicit background_bins."
                )
            self.background_bins = self.pre_onset_bins()
        if self.background_bins is not None and not np.all(
            np.any(self.background_bins, axis=-1)
        ):
            raise ValueError("background_bins selects no bins for some pixel.")
        self.t = ParamToDecay(
            model_type, self.freq, num_gates=self.num_gates, shift_method=shift_method
        ).t

    def pre_onset_bins(self):
        """Bins before the IRF first reaches ``onset_fraction`` of its maximum."""
        irf = np.nan_to_num(self.irf)
        peak = np.max(irf, axis=-1, keepdims=True)
        onset = np.argmax(irf >= self.onset_fraction * peak, axis=-1)
        return np.arange(self.num_gates) < np.asarray(onset)[..., None]

    @property
    def param_keys(self):
        """Posterior keys used by the forward model."""
        keys = self.PARAM_KEYS[self.model_type]
        extra = (self.background_key, self.h_shift_key)
        return (*keys, *(k for k in extra if k is not None))

    @property
    def n_params(self):
        """Free parameters: lifetime keys + amplitude (+ posterior or auto background)."""
        auto = isinstance(self.background, str)
        return len(self.param_keys) + 1 + int(auto)

    def auto_background(self, shape, y, lead, max_iter=100, tol=1e-9):
        """Poisson MLE of b in lambda = b + A shape (profile Newton on [0, Y/N]), seeded from pre-onset bins."""
        k = self.num_gates
        y = np.broadcast_to(y, (*lead, k))
        pre = np.broadcast_to(self.background_bins, (*lead, k)).astype(float)
        w = np.broadcast_to(_window_weights((k,), self.amplitude_mask), (*lead, k))
        n_pre, s_pre, y_pre = (np.sum(pre * v, axis=-1) for v in (1.0, shape, y))
        n_tot, s_tot, y_tot = (np.sum(w * v, axis=-1) for v in (1.0, shape, y))
        valid = (s_tot > _SHAPE_EPS) & (y_tot > 0)
        s_safe = np.where(valid, s_tot, 1.0)
        c = 1.0 - n_tot[..., None] * shape / s_safe[..., None]
        d = y_tot[..., None] * shape / s_safe[..., None]
        wy = w * y

        def score(b):
            lam = b[..., None] * c + d
            inv = np.zeros_like(lam)
            np.divide(1.0, lam, out=inv, where=lam > 0)
            pos = (lam <= 0) & (wy > 0) & (c > 0)
            g = np.where(np.any(pos, axis=-1), np.inf, np.sum(wy * c * inv, axis=-1))
            return g, -np.sum(wy * (c * inv) ** 2, axis=-1)

        lo, hi = np.zeros(lead), np.where(valid, y_tot / n_tot, 0.0)
        det = n_pre * s_tot - n_tot * s_pre
        b = np.full(lead, np.nan)
        np.divide(
            y_pre * s_tot - y_tot * s_pre, det, out=b, where=np.abs(det) > _SHAPE_EPS
        )
        b = np.where((b > lo) & (b < hi), b, 0.5 * (lo + hi))
        for _ in range(max_iter):
            g, h = score(b)
            lo = np.where(g > 0, b, lo)
            hi = np.where(g > 0, hi, b)
            step = np.zeros(lead)
            np.divide(g, h, out=step, where=(h < 0) & np.isfinite(g))
            newton = b - step
            inside = np.isfinite(g) & (h < 0) & (newton > lo) & (newton < hi)
            b_new = np.where(inside, newton, 0.5 * (lo + hi))
            done = np.abs(b_new - b) <= tol * (1.0 + np.abs(b))
            b = b_new
            if np.all(done | ~valid):
                break
        return np.where(valid, b, 0.0)

    def background_map(self, params, y, lead, shape=None):
        """Background b broadcast to ``lead``: posterior key, fixed value/map, auto or zero."""
        if self.background_key is not None:
            b = np.asarray(params[self.background_key], dtype=float)
        elif isinstance(self.background, str):
            if shape is None:
                shape = self.shape(params, lead)
            b = self.auto_background(shape, y, lead)
        elif self.background is not None:
            b = self.background
        else:
            b = np.zeros(())
        return np.broadcast_to(b, lead)

    def shape(self, params, lead):
        """Unit-amplitude IRF-convolved decay of shape ``(*lead, K)``."""
        k = self.num_gates
        flat = {
            f"{key}_map": np.broadcast_to(
                np.asarray(params[key], dtype=float), lead
            ).reshape(1, -1)
            for key in self.PARAM_KEYS[self.model_type]
        }
        n = flat[next(iter(flat))].shape[1]
        flat["photon_count_map"] = np.ones((1, n))
        h = (
            params[self.h_shift_key]
            if self.h_shift_key is not None
            else (0.0 if self.h_shift is None else self.h_shift)
        )
        flat["h_shift_map"] = np.broadcast_to(np.asarray(h, dtype=float), lead).reshape(
            1, -1
        )
        irf = self.irf
        if irf is not None and irf.ndim > 1:
            irf = np.broadcast_to(irf, (*lead, k)).reshape(1, n, k)
        recon = ParamToDecay(
            self.model_type,
            self.freq,
            irf=irf,
            num_gates=k,
            shift_method=self.shift_method,
            excitation=self.excitation,
            gate_width=self.gate_width,
        )
        convolved = recon.reconstruct_unit_amplitude(flat)["convolved_map"]
        return np.fmax(np.asarray(convolved, dtype=float), 0.0).reshape((*lead, k))

    def __call__(self, params, y):
        y = np.asarray(y, dtype=float)
        if y.shape[-1] != self.num_gates:
            raise ValueError(
                f"y has {y.shape[-1]} bins but the forward model has {self.num_gates}."
            )
        lead = np.broadcast_shapes(
            y.shape[:-1], *(np.shape(params[key]) for key in self.param_keys)
        )
        shape = self.shape(params, lead)
        w = _window_weights((self.num_gates,), self.amplitude_mask)
        total = np.sum(np.broadcast_to(y, (*lead, self.num_gates)) * w, axis=-1)
        shape_sum = np.sum(shape * w, axis=-1)
        b = self.background_map(params, y, lead, shape)
        amp = np.zeros(lead)
        np.divide(
            total - b * np.sum(w), shape_sum, out=amp, where=shape_sum > _SHAPE_EPS
        )
        return b[..., None] + np.fmax(amp, 0.0)[..., None] * shape


class FitAwareTolerance:
    """Per-pixel T_GOF = T_delta * A_chi * A_R * A_M_tilde from posterior draws and the measured decay.

    ``delta`` maps parameter name to tolerance; ``forward(params, y)`` returns the model decay.
    """

    GROUPS = ("T_delta", "T_delta_se", "A_chi", "A_R", "T_gof", "T_adj", "T_adj_gof")

    def __init__(
        self,
        forward,
        delta,
        n_params=None,
        t=None,
        statistic="pearson",
        z0=2.0,
        eps=0.2,
        s=0.2,
        early_params=("tau1", "alpha1"),
        early_tau_key="tau1",
        n_tau_early=3.0,
        relative=False,
        draw_batch=None,
        max_draws=None,
        seed=None,
    ):
        if statistic not in STATISTICS:
            raise ValueError(
                f"statistic must be one of {STATISTICS}, got {statistic!r}."
            )
        if not delta:
            raise ValueError("delta must name at least one parameter.")
        if n_params is None:
            n_params = getattr(forward, "n_params", None)
        if n_params is None:
            raise ValueError("n_params is required when forward has no n_params.")
        if s <= 0:
            raise ValueError("s must be positive.")
        acceptance_band(1.0, z0, eps)
        self.forward = forward
        self.delta = dict(delta)
        self.n_params = int(n_params)
        self.t = getattr(forward, "t", None) if t is None else np.asarray(t, float)
        self.statistic = statistic
        self.z0 = z0
        self.eps = eps
        self.s = s
        self.early_params = tuple(early_params)
        self.early_tau_key = early_tau_key
        self.n_tau_early = n_tau_early
        self.relative = relative
        self.draw_batch = draw_batch
        self.max_draws = max_draws
        self.seed = seed

    def _draw_indices(self, n_samples):
        if self.max_draws is None or self.max_draws >= n_samples:
            return np.arange(n_samples)
        rng = np.random.default_rng(self.seed)
        return np.sort(rng.choice(n_samples, self.max_draws, replace=False))

    def median_representativeness(self, samples, y, chi2_median, mask=None):
        """A_M from draws streamed through the forward model in batches."""
        y = np.asarray(y, dtype=float)
        keys = list(samples)
        idx = self._draw_indices(np.shape(samples[keys[0]])[-1])
        batch = self.draw_batch or max(1, _DRAW_BUDGET // max(1, y.size))
        count = np.zeros(np.shape(chi2_median))
        for start in range(0, idx.size, batch):
            sel = idx[start : start + batch]
            params = {
                k: np.moveaxis(np.asarray(samples[k])[..., sel], -1, 0) for k in keys
            }
            lam = self.forward(params, y)
            chi2, _ = reduced_chi2(y, lam, self.n_params, mask, self.statistic)
            count += np.sum(chi2 >= chi2_median, axis=0)
        return np.where(np.isnan(chi2_median), np.nan, count / idx.size)

    def __call__(self, samples, y, mask=None, bool_mask=None):
        """Return a dict of medians, fit statistics and per-parameter T_gof maps."""
        missing = [k for k in self.delta if k not in samples]
        if missing:
            raise KeyError(f"samples is missing delta keys: {missing}")
        samples = {k: np.asarray(v, dtype=float) for k, v in samples.items()}
        y = np.asarray(y, dtype=float)
        map_shape = y.shape[:-1]
        for key, v in samples.items():
            if v.shape[:-1] != map_shape:
                raise ValueError(
                    f"samples[{key!r}] shape {v.shape} does not match y map shape {map_shape}."
                )
        median = {k: np.median(v, axis=-1) for k, v in samples.items()}
        n_samples = next(iter(samples.values())).shape[-1]

        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.asarray(self.forward(median, y), dtype=float)
            chi2_nu, nu = reduced_chi2(y, lam, self.n_params, mask, self.statistic)
            r2 = residual_noise_ratio(y, lam, self.n_params, mask)
            fit = {
                "chi2_nu": chi2_nu,
                "nu": nu,
                "rho_R": r2["rho_R"],
                "R2": r2["R2"],
                "R2_ceil": r2["R2_ceil"],
            }
            early = None
            if (
                self.t is not None
                and self.early_tau_key in median
                and any(p in self.delta for p in self.early_params)
            ):
                win = early_window_mask(
                    self.t, lam, median[self.early_tau_key], self.n_tau_early, mask
                )
                chi2_e, nu_e = reduced_chi2(y, lam, self.n_params, win, self.statistic)
                rho_e = residual_noise_ratio(y, lam, self.n_params, win)["rho_R"]
                early = {
                    "chi2_nu_early": chi2_e,
                    "rho_R_early": rho_e,
                    "nu_early": nu_e,
                }
                fit.update(early)

            a_m = self.median_representativeness(samples, y, chi2_nu, mask)
            gates = (chi2_nu, r2["rho_R"], a_m, nu, self.z0, self.eps, self.s)
            base = combine_fit_aware_factors(1.0, *gates)
            with_early = (
                base
                if early is None
                else combine_fit_aware_factors(1.0, *gates, **early)
            )
            fit.update(
                lower=base["lower"],
                upper=base["upper"],
                A_M=a_m,
                A_M_tilde=base["A_M_tilde"],
            )

            out = {group: {} for group in self.GROUPS}
            for key, delta in self.delta.items():
                factors = with_early if key in self.early_params else base
                T = tolerance_probability(
                    samples[key], delta, median[key], relative=self.relative
                )
                T_adj = dispersion_adjusted_tolerance(
                    samples[key], delta, chi2_nu, median[key], relative=self.relative
                )
                out["T_delta"][key] = T
                out["T_delta_se"][key] = tolerance_standard_error(T, n_samples)
                out["A_chi"][key] = factors["A_chi"]
                out["A_R"][key] = factors["A_R"]
                out["T_gof"][key] = T * factors["T_gof"]
                out["T_adj"][key] = T_adj
                out["T_adj_gof"][key] = T_adj * factors["A_R"] * factors["A_M_tilde"]

        if bool_mask is not None:
            keep = np.asarray(bool_mask, dtype=bool)
            if keep.shape != map_shape:
                raise ValueError(
                    f"bool_mask shape {keep.shape} does not match map shape {map_shape}."
                )
            lam = np.where(keep[..., None], lam, np.nan)
            fit = {k: np.where(keep, v, np.nan) for k, v in fit.items()}
            median = {k: np.where(keep, v, np.nan) for k, v in median.items()}
            out = {
                group: {k: np.where(keep, v, np.nan) for k, v in vals.items()}
                for group, vals in out.items()
            }

        return {"median": median, "lam_median": lam, "fit": fit, **out}
