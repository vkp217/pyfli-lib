"""Per-pixel posterior uncertainty (MAD), credible-interval and goodness-of-fit
trustworthiness maps."""

import numpy as np
from scipy.stats import median_abs_deviation

from pyfli.cm_stats.chi_square import degrees_of_freedom
from pyfli.cm_stats.deviance import (
    poisson_deviance_moments,
    poisson_deviance_terms,
)

_TRUST_EPS = 1e-12


def unc(samples, axis=-1, scale=1.0):
    """
    Median absolute deviation (MAD) of posterior samples.

    Parameters
    ----------
    samples : np.ndarray
        Posterior draws, e.g. ``(H, W, NUM_SAMPLES)`` from
        ``BiPipeline.output_samples[key]``.
    axis : int
        Sample axis.
    scale : float or str
        Passed to :func:`scipy.stats.median_abs_deviation`. ``1.0`` gives the raw MAD
        (same as ``BiPipeline`` uncertainties); ``"normal"`` gives a
        standard-deviation-comparable value.

    Returns
    -------
    np.ndarray
        MAD map with the sample axis removed.
    """
    return median_abs_deviation(np.asarray(samples, float), axis=axis, scale=scale)


def trustworthiness_CrI(samples, median_map=None, ci=90.0, axis=-1):
    """
    Trustworthiness from the relative credible-interval width:
    ``1 / (1 + (q_high - q_low) / |median|)``, in ``(0, 1]``.

    Parameters
    ----------
    samples : np.ndarray
        Posterior draws, e.g. ``(H, W, NUM_SAMPLES)``.
    median_map : np.ndarray | None
        Posterior median map; computed from ``samples`` when ``None``.
    ci : float
        Central credible interval in percent (90 uses the 5th and 95th percentiles).
    axis : int
        Sample axis.

    Returns
    -------
    np.ndarray
        Trustworthiness map. ``NaN`` where the median is ~0 (relative width undefined,
        including all-zero masked-out pixels).
    """
    if not 0 < ci < 100:
        raise ValueError(f"ci must be in (0, 100), got {ci}.")
    samples = np.asarray(samples, float)
    if median_map is None:
        median_map = np.median(samples, axis=axis)
    median_map = np.asarray(median_map, float)
    map_shape = tuple(np.delete(samples.shape, axis % samples.ndim))
    if median_map.shape != map_shape:
        raise ValueError(
            f"median_map shape {median_map.shape} does not match samples shape "
            f"{samples.shape} without axis {axis}."
        )
    tail = (100.0 - ci) / 2.0
    q_low, q_high = np.percentile(samples, [tail, 100.0 - tail], axis=axis)
    denom = np.abs(median_map)
    defined = denom > _TRUST_EPS
    rciw = np.divide(
        q_high - q_low, denom, out=np.full(denom.shape, np.nan), where=defined
    )
    return 1.0 / (1.0 + rciw)


class trustworthiness_gof:
    """
    Goodness-of-fit score ``G`` in ``(0, 1]`` for FLIM / TCSPC decay fits.

    The fit's Poisson deviance ``D`` is standardized against its exact expected mean
    and variance under Poisson noise (so dim pixels are not rewarded for
    ``chi2_nu < 1``), giving ``z``, which is mapped to a score:

    - ``mode="one-sided"``: ``G = exp(-max(0, z)^2 / (2 z_plus^2))`` penalizes only
      underfitting.
    - ``mode="two-sided"``: additionally multiplies by
      ``exp(-max(0, -z)^2 / (2 z_minus^2))`` to penalize implausibly good fits.

    The time-bin axis is ``axis`` (last by default), so ``y`` and ``mu`` can be
    ``(K,)``, ``(n_pixels, K)`` or ``(H, W, K)``.

    Parameters
    ----------
    n_params : int
        Number of fitted parameters, subtracted from the expected deviance.
    mode : str
        ``"one-sided"`` or ``"two-sided"``.
    z_plus : float
        Width of the penalty for ``z > 0`` (underfitting).
    z_minus : float
        Width of the penalty for ``z < 0`` (overfitting), two-sided mode only.
    expectation : str
        ``"exact"``: ``z = (D - (sum E_k - n_params)) / sqrt(sum V_k)``.
        ``"asymptotic"``: ``z = (D - nu) / sqrt(2 nu)``.
    axis : int
        Time-bin axis.
    """

    MODES = ("one-sided", "two-sided")
    EXPECTATIONS = ("exact", "asymptotic")

    def __init__(
        self,
        n_params,
        mode="one-sided",
        z_plus=3.0,
        z_minus=5.0,
        expectation="exact",
        axis=-1,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}.")
        if expectation not in self.EXPECTATIONS:
            raise ValueError(
                f"expectation must be one of {self.EXPECTATIONS}, got {expectation!r}."
            )
        if z_plus <= 0 or z_minus <= 0:
            raise ValueError("z_plus and z_minus must be positive.")
        self.n_params = n_params
        self.mode = mode
        self.z_plus = z_plus
        self.z_minus = z_minus
        self.expectation = expectation
        self.axis = axis

    @staticmethod
    def deviance_terms(y, mu):
        """Per-bin Poisson deviance ``2 [y ln(y/mu) - (y - mu)]`` (``2 mu`` when ``y = 0``)."""
        return poisson_deviance_terms(mu, y, nonpositive="nan")

    @staticmethod
    def expected_moments(mu):
        """Exact per-bin ``(E[d], Var[d])`` for ``y ~ Poisson(mu)`` (:func:`~pyfli.cm_stats.deviance.poisson_deviance_moments`)."""
        return poisson_deviance_moments(mu)

    def zscore(self, y, mu, mask=None):
        """
        Standardized deviance per pixel.

        Parameters
        ----------
        y : np.ndarray
            Measured counts.
        mu : np.ndarray
            Fitted model counts, same shape as ``y``.
        mask : np.ndarray | None
            Boolean array broadcastable to ``y`` selecting the fit window.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(z, D, nu)`` with the time-bin axis removed.
        """
        d = self.deviance_terms(y, mu)
        w = (
            np.ones_like(d)
            if mask is None
            else np.broadcast_to(mask, d.shape).astype(float)
        )
        D = np.sum(w * d, axis=self.axis)
        nu = degrees_of_freedom(np.sum(w, axis=self.axis), self.n_params, None)
        if self.expectation == "exact":
            E, V = self.expected_moments(np.broadcast_to(mu, d.shape))
            mean = np.sum(w * E, axis=self.axis) - self.n_params
            var = np.sum(w * V, axis=self.axis)
        else:
            mean, var = nu, 2.0 * nu
        z = (D - mean) / np.sqrt(np.maximum(var, 1e-12))
        return z, D, nu

    def score(self, z):
        """Map standardized deviance ``z`` to the fit score ``G`` in ``(0, 1]``."""
        z = np.asarray(z, dtype=float)
        penalty = np.maximum(0.0, z) ** 2 / (2.0 * self.z_plus**2)
        if self.mode == "two-sided":
            penalty = penalty + np.maximum(0.0, -z) ** 2 / (2.0 * self.z_minus**2)
        return np.exp(-penalty)

    def __call__(self, y, mu, mask=None):
        """
        Goodness of fit for every pixel.

        Parameters
        ----------
        y : np.ndarray
            Measured counts.
        mu : np.ndarray
            Fitted model counts, same shape as ``y``.
        mask : np.ndarray | None
            Boolean array broadcastable to ``y`` selecting the fit window.

        Returns
        -------
        dict
            ``D``, ``nu``, ``chi2_nu``, ``z`` and ``G`` maps.
        """
        z, D, nu = self.zscore(y, mu, mask=mask)
        return {
            "D": D,
            "nu": nu,
            "chi2_nu": D / np.maximum(nu, 1),
            "z": z,
            "G": self.score(z),
        }
