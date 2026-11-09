"""
pyfli.cm_stats definitions: each variant reproduces the formula it replaced.
"""

import numpy as np
import pytest
from scipy.special import xlogy
from scipy.stats import f

from pyfli.cm_stats import (
    MAD_TO_SIGMA,
    degrees_of_freedom,
    median_absolute_deviation,
    nested_f_test,
    neyman_chi_square,
    pearson_chi_square,
    poisson_deviance,
    poisson_deviance_terms,
    r_squared,
    rmse,
    sum_squared_residuals,
)

RNG = np.random.default_rng(0)
MU = RNG.uniform(0.05, 50.0, (3, 40))
D = RNG.poisson(MU).astype(float)
D[0, :5] = 0.0
NEG = D - 2.0


def test_deviance_modes():
    """zero/nan/floor modes match their original closed forms."""
    mu = np.clip(MU, 1e-12, None)
    pos = NEG > 0
    zero = 2.0 * (
        mu - NEG + np.where(pos, NEG * np.log(np.where(pos, NEG, 1.0) / mu), 0.0)
    )
    assert np.array_equal(poisson_deviance_terms(MU, NEG), zero)
    nan_mode = 2.0 * (xlogy(NEG, NEG / mu) - (NEG - mu))
    assert np.array_equal(
        poisson_deviance_terms(MU, NEG, nonpositive="nan"), nan_mode, equal_nan=True
    )
    m9 = np.clip(MU, 1e-9, None)
    floor = 2.0 * np.sum(m9 - NEG + NEG * np.log(np.clip(NEG, 1e-9, None) / m9))
    got = poisson_deviance(MU, NEG, axis=None, floor=1e-9, nonpositive="floor")
    assert got == floor
    with pytest.raises(ValueError):
        poisson_deviance_terms(MU, D, nonpositive="bad")


def test_chi_square_and_dof():
    """Pearson/Neyman floors and the degrees-of-freedom floor."""
    assert np.array_equal(
        pearson_chi_square(MU, D), np.sum((D - MU) ** 2 / np.clip(MU, 1.0, None), -1)
    )
    assert np.array_equal(
        neyman_chi_square(MU, D), np.sum((D - MU) ** 2 / np.clip(D, 1.0, None), -1)
    )
    assert degrees_of_freedom(10, 6) == 4 and degrees_of_freedom(5, 6) == 1
    assert degrees_of_freedom(5, 6, None) == -1
    assert type(degrees_of_freedom(10, 6)) is int
    np.testing.assert_array_equal(degrees_of_freedom(np.array([3, 9]), 6), [1, 3])


@pytest.mark.parametrize(
    "empty,threshold,dtype",
    [(0.0, 0.0, None), (1.0, 1e-8, None), (0.0, 1e-8, np.float32)],
)
def test_r_squared_variants(empty, threshold, dtype):
    """R2 matches 1 - SS_res/SS_tot and returns `empty_value` on flat rows."""
    data = np.vstack([D, np.full((1, 40), 3.0)])
    res = data - np.vstack([MU, np.full((1, 40), 3.0)])
    r2 = r_squared(res, data, empty_value=empty, threshold=threshold, dtype=dtype)
    ss_tot = np.sum((data - data.mean(-1, keepdims=True)) ** 2, -1)
    ref = 1.0 - np.sum(res**2, -1) / ss_tot
    np.testing.assert_allclose(r2[:-1], ref[:-1], rtol=1e-6)
    assert r2[-1] == empty


def test_residual_stats_numpy_and_torch():
    """SSR/RMSE/R2 agree between NumPy and torch inputs."""
    torch = pytest.importorskip("torch")
    res = D - MU
    assert sum_squared_residuals(res, axis=None) == np.sum(res**2)
    np.testing.assert_array_equal(rmse(res), np.sqrt(np.mean(res**2, -1)))
    tr = torch.as_tensor(res)
    np.testing.assert_allclose(rmse(tr, axis=1).numpy(), rmse(res, axis=1))
    np.testing.assert_allclose(
        r_squared(tr, torch.as_tensor(D), axis=1).numpy(), r_squared(res, D, axis=1)
    )


def test_nested_f_test_and_mad():
    """F statistic, sf/cdf p-values and MAD helpers."""
    stat, p = nested_f_test(120.0, 90.0, 2, 58)
    assert stat == pytest.approx((30.0 / 2) / (90.0 / 58))
    assert p == f.sf(stat, 2, 58)
    assert nested_f_test(120.0, 90.0, 2, 58, p_value="cdf")[1] == 1.0 - f.cdf(
        stat, 2, 58
    )
    center, mad = median_absolute_deviation(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert (center, mad) == (3.0, 1.0) and MAD_TO_SIGMA == 1.4826


def test_deviance_moments_match_exact_sums():
    """Unified E[d]/Var[d] match direct Poisson sums from 1e-6 to 1e5."""
    from scipy.special import gammaln

    from pyfli.cm_stats import poisson_deviance_moments

    for m in (1e-6, 0.1, 1.0, 7.3, 999.0, 1001.0, 1e5):
        hw = 14 * np.sqrt(m) + 30
        y = np.arange(max(0, int(m - hw)), int(np.ceil(m + hw)) + 1, dtype=float)
        p = np.exp(xlogy(y, m) - m - gammaln(y + 1))
        d = 2 * (xlogy(y, y / m) - (y - m))
        e = np.sum(p * d)
        E, V = poisson_deviance_moments(np.array(m))
        assert E == pytest.approx(e, rel=1e-5)
        assert V == pytest.approx(np.sum(p * (d - e) ** 2), rel=1e-5)


def test_fit_aware_deviance_is_normalised_by_expectation():
    """fit_aware reduced deviance averages ~1 for the true model at low counts."""
    from pyfli.bayes_utils.fit_aware_tolerance import reduced_chi2
    from pyfli.cm_stats import reduced_poisson_deviance

    lam = np.full(200, 0.2)
    y = np.random.default_rng(2).poisson(lam, (3000, 200)).astype(float)
    chi, nu = reduced_chi2(y, lam, 0, statistic="deviance")
    assert chi.mean() == pytest.approx(1.0, abs=0.01)
    np.testing.assert_allclose(chi, reduced_poisson_deviance(lam, y, 0)[1])
    assert np.all(nu == 200)


def test_flat_decay_r2_is_zero_everywhere():
    """DetailedRecon and Laguerre report R2 = 0 on a flat decay, like the solver."""
    from pyfli.cm_funcs.forward_model import model_numpy
    from pyfli.laguerre.laguerre_method import LaguerreFLI
    from pyfli.reconstruction import DetailedRecon
    from pyfli.reconstruction.common_reconstruct import gate_starts

    t = gate_starts(12.5, 64)
    irf = np.exp(-0.5 * ((t - 1.5) / 0.15) ** 2)
    decay = np.broadcast_to(
        model_numpy(t, irf, [5e3, 0.4, 0.5, 2.8, 1.0, 0.0], "bi-exponential"),
        (2, 2, 64),
    ).copy()
    decay[1, 1] = 4.0
    params = {"alpha1_map": np.full((2, 2), 0.4), "tau1_map": np.full((2, 2), 0.5)}
    params["tau2_map"] = np.full((2, 2), 2.8)
    det = DetailedRecon(80.0, irf, binned_decay=decay)
    r2 = det.reconstruct(params, "bi-exponential", log_summary=False)
    assert r2["results"]["maps"]["R2_map"][1, 1] == 0.0
    lag = LaguerreFLI(2, n_laguerre=16, alpha=0.9, dt=12.5 / 64, verbose=False)
    maps = lag.fit(decay, irf).get_parameters()["results"]["maps"]
    assert maps["R2_map"][1, 1] == 0.0


def test_model_selector_uses_fitted_gates():
    """ModelSelector takes n from n_points / fit_indices instead of the chi2 ratio."""
    from pyfli.data_vnp import ModelSelector

    roi = np.ones((2, 2), bool)
    mono = {"chi2_map": np.full((2, 2), 80.0)}
    bi = {
        "chi2_map": np.full((2, 2), 60.0),
        "reduced_chi2_map": np.full((2, 2), 60.0 / 30.0),
        "tau1_map": np.full((2, 2), 0.5),
        "tau2_map": np.full((2, 2), 3.0),
    }
    res = ModelSelector(roi, n_points=56).compare("x", mono, bi)
    assert res["n_points"] == 56
    assert res["F_map"][0, 0] == pytest.approx((20.0 / 2) / (60.0 / 50))
    assert ModelSelector(roi).compare("y", mono, bi)["n_points"] == 36
    assert ModelSelector.fitted_gates({"fit_indices": (5, 61)}) == 56
    fit_only = {"results": {"TR_maps": {"fit_map": np.zeros((2, 2, 64))}}}
    assert ModelSelector.fitted_gates(fit_only) == 64
