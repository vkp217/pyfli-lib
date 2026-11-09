"""
Tests for LaguerreFLI — the Laguerre Expansion Technique (LET) fitter.

All tests use purely synthetic numpy arrays; no file I/O is required.
"""

from math import comb

import numpy as np
import pytest

from pyfli import LaguerreFLI

# ─────────────────────────────────────────────────────────────────────────────
# Initialisation validation
# ─────────────────────────────────────────────────────────────────────────────


class TestInit:
    def test_defaults_accepted(self):
        m = LaguerreFLI()
        assert m.n_components == 2
        assert 0.0 < m.alpha < 1.0

    def test_n_components_below_1_raises(self):
        with pytest.raises(ValueError, match="n_components"):
            LaguerreFLI(n_components=0)

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            LaguerreFLI(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            LaguerreFLI(alpha=1.0)
        with pytest.raises(ValueError, match="alpha"):
            LaguerreFLI(alpha=1.5)

    def test_dt_non_positive_raises(self):
        with pytest.raises(ValueError, match="dt"):
            LaguerreFLI(dt=0.0)
        with pytest.raises(ValueError, match="dt"):
            LaguerreFLI(dt=-1.0)

    def test_n_laguerre_below_n_components_raises(self):
        with pytest.raises(ValueError, match="n_laguerre"):
            LaguerreFLI(n_components=3, n_laguerre=2)

    def test_n_laguerre_defaults_to_max_4_2n(self):
        m = LaguerreFLI(n_components=1)
        assert m.n_laguerre >= 4

    def test_repr_contains_key_params(self):
        m = LaguerreFLI(n_components=2, alpha=0.9, dt=0.05)
        r = repr(m)
        assert "n_components=2" in r
        assert "alpha=0.900" in r

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            LaguerreFLI().predict()

    def test_get_parameters_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            LaguerreFLI().get_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# Static helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestBasis:
    def test_basis_shape(self):
        T, L = 64, 5
        B = LaguerreFLI._discrete_laguerre_basis(T, 0.85, L)
        assert B.shape == (L, T)

    def test_basis_first_row_decays(self):
        T = 64
        B = LaguerreFLI._discrete_laguerre_basis(T, 0.85, 4)
        # b[0] = sqrt(1-alpha) * alpha^(n/2) — strictly decreasing for alpha<1
        assert B[0, 0] > B[0, 1] > B[0, -1]

    def test_basis_finite(self):
        B = LaguerreFLI._discrete_laguerre_basis(128, 0.7, 6)
        assert np.all(np.isfinite(B))


class TestConvolveWithIRF:
    def test_output_shape(self, gaussian_irf, n_bins):
        L = 4
        B = LaguerreFLI._discrete_laguerre_basis(n_bins, 0.85, L)
        V = LaguerreFLI._convolve_with_irf(B, gaussian_irf)
        assert V.shape == (n_bins, L)

    def test_irf_area_normalisation(self, n_bins):
        """IRF is area-normalised internally; scaling the input changes nothing."""
        irf = np.ones(n_bins, dtype=float)
        B = LaguerreFLI._discrete_laguerre_basis(n_bins, 0.85, 4)
        V1 = LaguerreFLI._convolve_with_irf(B, irf)
        V2 = LaguerreFLI._convolve_with_irf(B, irf * 10.0)
        np.testing.assert_allclose(V1, V2)

    def test_output_finite(self, gaussian_irf, n_bins):
        B = LaguerreFLI._discrete_laguerre_basis(n_bins, 0.85, 4)
        V = LaguerreFLI._convolve_with_irf(B, gaussian_irf)
        assert np.all(np.isfinite(V))


# ─────────────────────────────────────────────────────────────────────────────
# Fit on 1-D (single pixel) input
# ─────────────────────────────────────────────────────────────────────────────


class TestFit1D:
    @pytest.fixture
    def fitted_mono(self, mono_decay_1d, gaussian_irf, dt):
        m = LaguerreFLI(n_components=1, n_laguerre=6, alpha=0.85, dt=dt)
        return m.fit(mono_decay_1d, gaussian_irf)

    def test_fit_returns_self(self, mono_decay_1d, gaussian_irf, dt):
        m = LaguerreFLI(n_components=1, n_laguerre=6, alpha=0.85, dt=dt)
        result = m.fit(mono_decay_1d, gaussian_irf)
        assert result is m

    def test_attributes_populated(self, fitted_mono):
        assert fitted_mono.coeffs_ is not None
        assert fitted_mono.taus_ is not None
        assert fitted_mono.amplitudes_ is not None
        assert fitted_mono.fractions_ is not None
        assert fitted_mono.tau_mean_ is not None
        assert fitted_mono.reconstructed_ is not None

    def test_coeffs_shape_1d(self, fitted_mono, n_bins):
        # 1-D input → treated as (1,1,T)
        assert fitted_mono.coeffs_.shape[0] == 1
        assert fitted_mono.coeffs_.shape[1] == 1

    def test_reconstructed_shape_1d(self, fitted_mono, n_bins):
        assert fitted_mono.reconstructed_.shape == (1, 1, n_bins)

    def test_taus_positive(self, fitted_mono):
        assert np.all(fitted_mono.taus_ > 0)

    def test_fractions_sum_to_one(self, fitted_mono):
        np.testing.assert_allclose(fitted_mono.fractions_.sum(axis=-1), 1.0, atol=1e-6)

    def test_tau_mean_positive(self, fitted_mono):
        assert np.all(fitted_mono.tau_mean_ >= 0)

    def test_predict_returns_reconstructed(self, fitted_mono):
        pred = fitted_mono.predict()
        np.testing.assert_array_equal(pred, fitted_mono.reconstructed_)


# ─────────────────────────────────────────────────────────────────────────────
# Fit on 3-D image cube
# ─────────────────────────────────────────────────────────────────────────────


class TestFit3D:
    @pytest.fixture
    def fitted_bi(self, decay_cube, gaussian_irf, dt):
        m = LaguerreFLI(n_components=2, n_laguerre=6, alpha=0.85, dt=dt)
        return m.fit(decay_cube, gaussian_irf)

    def test_coeffs_shape_3d(self, fitted_bi, decay_cube):
        X, Y, _ = decay_cube.shape
        assert fitted_bi.coeffs_.shape[:2] == (X, Y)

    def test_taus_map_shape(self, fitted_bi, decay_cube):
        X, Y, _ = decay_cube.shape
        assert fitted_bi.taus_.shape == (X, Y, 2)

    def test_amplitudes_shape(self, fitted_bi, decay_cube):
        X, Y, _ = decay_cube.shape
        assert fitted_bi.amplitudes_.shape == (X, Y, 2)

    def test_tau_mean_shape(self, fitted_bi, decay_cube):
        X, Y, _ = decay_cube.shape
        assert fitted_bi.tau_mean_.shape == (X, Y)

    def test_taus_ascending_per_pixel(self, fitted_bi):
        # taus_ are sorted ascending within each pixel
        assert np.all(fitted_bi.taus_[..., 0] <= fitted_bi.taus_[..., 1] + 1e-12)

    def test_fractions_sum_to_one(self, fitted_bi):
        np.testing.assert_allclose(fitted_bi.fractions_.sum(axis=-1), 1.0, atol=1e-6)

    def test_fit_curve_shape(self, fitted_bi, decay_cube):
        assert fitted_bi.fit_curve_.shape == decay_cube.shape

    def test_residual_curve_shape(self, fitted_bi, decay_cube):
        assert fitted_bi.residual_curve_.shape == decay_cube.shape

    def test_residuals_finite(self, fitted_bi):
        assert np.all(np.isfinite(fitted_bi.residuals_))


# ─────────────────────────────────────────────────────────────────────────────
# get_parameters dict structure
# ─────────────────────────────────────────────────────────────────────────────


class TestGetParameters:
    @pytest.fixture
    def params(self, bi_decay_1d, gaussian_irf, dt):
        m = LaguerreFLI(n_components=2, n_laguerre=6, alpha=0.85, dt=dt)
        m.fit(bi_decay_1d, gaussian_irf)
        return m.get_parameters("TestDataset")

    def test_top_level_keys(self, params):
        assert set(params.keys()) == {"name", "method", "results"}

    def test_name_field(self, params):
        assert params["name"] == "TestDataset"

    def test_method_contains_exp(self, params):
        assert "exp" in params["method"]

    def test_results_keys(self, params):
        assert "maps" in params["results"]
        assert "error_maps" in params["results"]
        assert "TR_maps" in params["results"]

    def test_maps_contain_tau_and_alpha(self, params):
        maps = params["results"]["maps"]
        assert "tau1_map" in maps
        assert "tau2_map" in maps
        assert "alpha1_map" in maps
        assert "alpha2_map" in maps

    def test_maps_contain_standard_keys(self, params):
        maps = params["results"]["maps"]
        assert "photon_count_map" in maps
        assert "tau_mean_map" in maps
        assert "chi2_map" in maps
        assert "pixel_health_map" in maps

    def test_tr_maps_keys(self, params):
        tr = params["results"]["TR_maps"]
        assert "fit_map" in tr
        assert "residual_map" in tr


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────


class TestFitValidation:
    def test_2d_decay_raises(self, gaussian_irf, dt):
        bad_decay = np.ones((4, 64))
        with pytest.raises(ValueError):
            LaguerreFLI(dt=dt).fit(bad_decay, gaussian_irf[:64])

    def test_irf_length_mismatch_raises(self, mono_decay_1d, dt):
        bad_irf = np.ones(10)
        with pytest.raises(ValueError):
            LaguerreFLI(dt=dt).fit(mono_decay_1d, bad_irf)

    def test_per_pixel_irf_shape_mismatch_raises(self, decay_cube, dt):
        bad_irf = np.ones((3, 3, decay_cube.shape[-1]))  # wrong X,Y
        with pytest.raises(ValueError):
            LaguerreFLI(dt=dt).fit(decay_cube, bad_irf)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-alpha optimisation
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoAlpha:
    def test_auto_alpha_updates_alpha(self, mono_decay_1d, gaussian_irf, dt):
        m = LaguerreFLI(n_components=1, alpha=0.5, dt=dt, auto_alpha=True)
        m.fit(mono_decay_1d, gaussian_irf)
        # alpha should have been updated by the optimiser
        assert 0.0 < m.alpha < 1.0

    def test_auto_alpha_fit_completes(self, bi_decay_1d, gaussian_irf, dt):
        m = LaguerreFLI(n_components=2, dt=dt, auto_alpha=True)
        m.fit(bi_decay_1d, gaussian_irf)
        assert m.taus_ is not None


# ─────────────────────────────────────────────────────────────────────────────
# Lifetime accuracy on noise-free synthetic data
# ─────────────────────────────────────────────────────────────────────────────


class TestLifetimeAccuracy:
    def test_mono_tau_ballpark(self, mono_decay_1d, gaussian_irf, dt):
        """Recovered tau should be within 50% of the ground truth (2 ns)."""
        m = LaguerreFLI(n_components=1, n_laguerre=8, alpha=0.85, dt=dt)
        m.fit(mono_decay_1d, gaussian_irf)
        tau_recovered = float(m.taus_.mean())
        assert 1.0 < tau_recovered < 4.0, (
            f"Mono tau out of range: {tau_recovered:.3f} ns"
        )

    def test_mean_lifetime_positive(self, bi_decay_1d, gaussian_irf, dt):
        m = LaguerreFLI(n_components=2, n_laguerre=6, alpha=0.85, dt=dt)
        m.fit(bi_decay_1d, gaussian_irf)
        assert float(m.tau_mean_.mean()) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Regression guards
# ─────────────────────────────────────────────────────────────────────────────
# These tests pin down the properties that a *correct* implementation must
# satisfy, so that future edits cannot silently reintroduce historical bugs:
#   * basis orthonormality and closed-form identity
#   * spatially-invariant 3-D IRF == 1-D path
#   * lifetime recovery (1-D and pixel-variant IRF)
#   * per-pixel IRF grouping matches naive loop exactly
#   * output contract used by downstream plotting/saving code
# ─────────────────────────────────────────────────────────────────────────────


def _closed_form_laguerre(T, alpha, L):
    """
    Canonical discrete orthonormal Laguerre functions (Marmarelis, 1993):

        b_j(n) = alpha^((n-j)/2) * sqrt(1-alpha)
                 * sum_{k=0}^{j} (-1)^k C(n,k) C(j,k) alpha^(j-k) (1-alpha)^k

    This is the unambiguous definition the recurrence must reproduce.
    """
    b = np.zeros((L, T), dtype=np.float64)
    for j in range(L):
        for n in range(T):
            s = 0.0
            for k in range(j + 1):
                s += (
                    ((-1) ** k)
                    * comb(n, k)
                    * comb(j, k)
                    * alpha ** (j - k)
                    * (1 - alpha) ** k
                )
            b[j, n] = alpha ** ((n - j) / 2.0) * np.sqrt(1.0 - alpha) * s
    return b


def _reg_gaussian_irf(T, dt, center_ns=0.6, width_ns=0.15):
    n = np.arange(T)
    irf = np.exp(-0.5 * ((n * dt - center_ns) / width_ns) ** 2)
    return irf / irf.sum()


def _measured_cube(comps, irf, dt, scale, shape=(4, 4), seed=0, noise=False):
    """Build (X, Y, T) measured cube as IRF (*) sum_i a_i exp(-t/tau_i)."""
    T = irf.shape[-1]
    n = np.arange(T)
    h = sum(a * np.exp(-n * dt / tau) for a, tau in comps)
    yc = np.convolve(irf, h, mode="full")[:T] * scale
    cube = np.tile(yc, (*shape, 1))
    if noise:
        cube = np.random.default_rng(seed).poisson(cube).astype(float)
    return cube


def _measured_cube_varying_irf(comps, irfs_2d, labels, dt, scale, seed=0, noise=False):
    """Build (decay_cube, irf_cube) where IRF varies per pixel via `labels`."""
    X, Y = labels.shape
    T = irfs_2d.shape[1]
    n = np.arange(T)
    rng = np.random.default_rng(seed)
    f1 = rng.uniform(0.4, 0.6, (X, Y))
    decay = np.zeros((X, Y, T))
    irf_cube = np.zeros((X, Y, T))
    for i in range(X):
        for j in range(Y):
            g = irfs_2d[labels[i, j]]
            h = f1[i, j] * np.exp(-n * dt / comps[0][1]) + (1 - f1[i, j]) * np.exp(
                -n * dt / comps[1][1]
            )
            irf_cube[i, j] = g
            decay[i, j] = np.convolve(g, h, mode="full")[:T] * scale
    if noise:
        decay = rng.poisson(decay).astype(float)
    return decay, irf_cube


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7, 0.85, 0.95])
def test_orthonormality(alpha):
    """B @ B.T must equal the identity on an adequately long window."""
    L, T = 10, 4096
    B = LaguerreFLI._discrete_laguerre_basis(T, alpha, L)
    G = B @ B.T
    assert np.max(np.abs(G - np.eye(L))) < 1e-9, "basis is not orthonormal"


@pytest.mark.parametrize("alpha,L", [(0.5, 4), (0.5, 8), (0.85, 4), (0.85, 8)])
def test_closed_form_match(alpha, L):
    """Recurrence output must equal the Marmarelis closed-form definition."""
    T = 1000
    B = LaguerreFLI._discrete_laguerre_basis(T, alpha, L)
    B_ref = _closed_form_laguerre(T, alpha, L)
    assert np.max(np.abs(B - B_ref)) < 1e-10, "basis is orthonormal but not Laguerre"


def test_irf_collapse_equivalence():
    """A tiled (constant-across-pixels) 3-D IRF must give identical taus to 1-D."""
    T = 256
    dt = 12.5 / T
    irf = _reg_gaussian_irf(T, dt)
    y = _measured_cube(
        [(0.6, 0.5), (0.4, 1.5)], irf, dt, scale=3000, shape=(4, 4), seed=1, noise=True
    )

    kw = dict(
        n_components=2, n_laguerre=8, dt=dt, auto_alpha=True, laser_period_ns=12.5
    )
    taus_1d = LaguerreFLI(**kw).fit(y, irf).taus_
    taus_3d = LaguerreFLI(**kw).fit(y, np.tile(irf, (4, 4, 1))).taus_

    assert np.array_equal(taus_1d, taus_3d), (
        "invariant-IRF fast path diverges from the 1-D path"
    )


@pytest.mark.parametrize("true_taus", [(0.5, 1.5), (0.4, 2.5)])
def test_biexp_recovery(true_taus):
    """Well-separated bi-exponential lifetimes are recovered within ~10%."""
    T = 256
    dt = 12.5 / T
    irf = _reg_gaussian_irf(T, dt)
    comps = [(0.55, true_taus[0]), (0.45, true_taus[1])]
    y = _measured_cube(comps, irf, dt, scale=5000, shape=(6, 6), seed=0, noise=False)

    m = LaguerreFLI(
        n_components=2,
        n_laguerre=8,
        dt=dt,
        auto_alpha=True,
        laser_period_ns=12.5,
        taus_init=np.array(true_taus),
    ).fit(y, irf)
    got = np.sort(m.taus_, axis=-1).mean(axis=(0, 1))
    assert np.allclose(got, sorted(true_taus), rtol=0.10, atol=0.10), (
        f"recovered {got}, expected {sorted(true_taus)}"
    )


def test_monoexp_autoalpha_not_degenerate():
    """
    auto_alpha must select a physical pole and recover a clean mono-exponential.
    Guards the degenerate-pole failure mode where a near-zero alpha makes the
    convolved fit flat while driving the deconvolved decay strongly negative.
    """
    T = 128
    dt = 12.5 / T
    irf = _reg_gaussian_irf(T, dt, center_ns=0.3, width_ns=0.10)
    y = _measured_cube(
        [(1.0, 0.8)], irf, dt, scale=5000, shape=(1, 1), seed=0, noise=False
    )

    m = LaguerreFLI(
        n_components=1, n_laguerre=6, dt=dt, auto_alpha=True, laser_period_ns=12.5
    ).fit(y, irf)
    tau = float(m.taus_.mean())
    recon = m.reconstructed_[0, 0]
    neg_frac = float((recon < -0.02 * np.abs(recon).max()).mean())

    assert abs(tau - 0.8) < 0.08, f"mono-exp recovered tau={tau:.3f}, expected ~0.8"
    assert 0.4 < m.alpha < 0.98, f"auto_alpha selected an extreme pole: {m.alpha:.3f}"
    assert neg_frac < 0.10, (
        f"deconvolution went strongly negative (neg_frac={neg_frac:.2f})"
    )


def test_pixel_variant_irf_recovery():
    """Lifetimes are recovered when the IRF genuinely varies across pixels."""
    X = Y = 8
    T = 256
    dt = 12.5 / T
    n = np.arange(T)
    true = (0.5, 1.6)
    centers = np.linspace(0.4, 0.9, X)
    widths = np.linspace(0.10, 0.22, Y)
    irfs, labels = [], np.zeros((X, Y), dtype=int)
    k = 0
    for i in range(X):
        for j in range(Y):
            g = np.exp(-0.5 * ((n * dt - centers[i]) / widths[j]) ** 2)
            irfs.append(g / g.sum())
            labels[i, j] = k
            k += 1
    irfs = np.array(irfs)

    y, irf_cube = _measured_cube_varying_irf(
        [(0.5, true[0]), (0.5, true[1])],
        irfs,
        labels,
        dt,
        scale=4000,
        seed=0,
        noise=True,
    )

    m = LaguerreFLI(
        n_components=2,
        n_laguerre=8,
        dt=dt,
        auto_alpha=True,
        laser_period_ns=12.5,
        taus_init=np.array(true),
    ).fit(y, irf_cube)
    got = np.sort(m.taus_, axis=-1).mean(axis=(0, 1))
    assert np.allclose(got, sorted(true), rtol=0.12, atol=0.12), (
        f"recovered {got}, expected {sorted(true)}"
    )


def test_pixel_variant_irf_beats_single_irf():
    """
    Using the true per-pixel IRF must fit better than collapsing to one
    averaged IRF when the IRF really varies.
    """
    X = Y = 8
    T = 256
    dt = 12.5 / T
    n = np.arange(T)
    centers = np.linspace(0.4, 0.9, X)
    irfs = np.array(
        [
            (lambda g: g / g.sum())(np.exp(-0.5 * ((n * dt - c) / 0.15) ** 2))
            for c in centers
        ]
    )
    labels = np.repeat(np.arange(X), Y).reshape(X, Y)

    y, irf_cube = _measured_cube_varying_irf(
        [(0.5, 0.5), (0.5, 1.6)], irfs, labels, dt, scale=4000, seed=1, noise=True
    )

    kw = dict(
        n_components=2,
        n_laguerre=8,
        dt=dt,
        auto_alpha=True,
        laser_period_ns=12.5,
        taus_init=np.array([0.5, 1.6]),
    )
    m_var = LaguerreFLI(**kw).fit(y, irf_cube)
    m_avg = LaguerreFLI(**kw).fit(y, irf_cube.reshape(-1, T).mean(0))

    sse_var = float((m_var.residual_curve_**2).sum())
    sse_avg = float((m_avg.residual_curve_**2).sum())
    assert sse_var < 0.5 * sse_avg, (
        f"per-pixel IRF did not improve the fit (var={sse_var:.2e}, avg={sse_avg:.2e})"
    )


def test_unique_irf_grouping_matches_naive_loop():
    """
    The unique-IRF grouping optimisation must give identical results to a
    naive per-pixel loop (which it replaces for efficiency).
    """
    X = Y = 6
    T = 128
    dt = 12.5 / T
    n = np.arange(T)
    n_unique = 3
    centers = np.linspace(0.4, 0.8, n_unique)
    irfs = np.array(
        [
            (lambda g: g / g.sum())(np.exp(-0.5 * ((n * dt - c) / 0.15) ** 2))
            for c in centers
        ]
    )
    rng = np.random.default_rng(2)
    labels = rng.integers(0, n_unique, (X, Y))

    y, irf_cube = _measured_cube_varying_irf(
        [(0.5, 0.6), (0.5, 1.4)], irfs, labels, dt, scale=4000, seed=2, noise=True
    )

    kw = dict(
        n_components=2,
        n_laguerre=6,
        dt=dt,
        auto_alpha=False,
        alpha=0.88,
        laser_period_ns=12.5,
        taus_init=np.array([0.6, 1.4]),
    )
    m = LaguerreFLI(**kw).fit(y, irf_cube)
    assert m.n_unique_irf_ == n_unique, (
        f"expected {n_unique} unique IRFs, grouped into {m.n_unique_irf_}"
    )

    inst = LaguerreFLI(**kw)
    basis = inst._discrete_laguerre_basis(T, 0.88, 6)
    naive = np.zeros((X, Y, T))
    for i in range(X):
        for j in range(Y):
            Vp = inst._convolve_with_irf(basis, irf_cube[i, j])
            c = inst._solve_coefficients(Vp, y[i, j][:, None]).ravel()
            naive[i, j] = basis.T @ c
    assert np.allclose(m.reconstructed_, naive, atol=1e-9), (
        "grouped reconstruction differs from naive per-pixel reconstruction"
    )


def test_output_contract():
    """get_parameters() must expose the keys downstream plotting/saving code relies on."""
    T = 128
    dt = 12.5 / T
    irf = _reg_gaussian_irf(T, dt)
    y = _measured_cube(
        [(0.6, 0.5), (0.4, 1.4)], irf, dt, scale=3000, shape=(3, 3), seed=0, noise=True
    )

    out = (
        LaguerreFLI(n_components=2, n_laguerre=6, dt=dt, laser_period_ns=12.5)
        .fit(y, irf)
        .get_parameters("regtest")
    )

    maps = out["results"]["maps"]
    tr = out["results"]["TR_maps"]
    for key in (
        "tau1_map",
        "tau2_map",
        "alpha1_map",
        "alpha2_map",
        "tau_mean_map",
        "R2_map",
        "photon_count_map",
    ):
        assert key in maps, f"missing map: {key}"
    for key in ("fit_map", "residual_map", "sdf_map"):
        assert key in tr, f"missing TR map: {key}"
    assert maps["tau1_map"].shape == (3, 3)
    assert tr["fit_map"].shape == (3, 3, T)
    assert np.isfinite(maps["tau_mean_map"]).all()


# ─────────────────────────────────────────────────────────────────────────────
# Consistency with the solver forward model, parameter counts and statistics
# ─────────────────────────────────────────────────────────────────────────────


class TestSolverConsistency:
    N_GATES = 256
    PERIOD = 12.5
    DT = PERIOD / N_GATES

    @pytest.fixture
    def axis(self):
        from pyfli.reconstruction.common_reconstruct import gate_starts

        t = gate_starts(self.PERIOD, self.N_GATES)
        return t, np.exp(-0.5 * ((t - 1.0) / 0.1) ** 2)

    def _fit(self, decay, irf, n_components, **kwargs):
        m = LaguerreFLI(
            n_components, n_laguerre=16, alpha=0.9, dt=self.DT, verbose=False, **kwargs
        )
        return m.fit(decay, irf), m.fit(decay, irf).get_parameters()

    @pytest.mark.parametrize(
        "shift_method,background",
        [("zero_pad", 0.0), ("circular", 0.0), ("zero_pad", 3.0)],
    )
    def test_bi_parameters_follow_solver_definitions(
        self, axis, shift_method, background
    ):
        from pyfli.cm_funcs.forward_model import model_numpy
        from pyfli.reconstruction import ParamToDecay

        t, irf = axis
        truth = [1e5, 0.5, 0.3, 4.0, background, 0.0]
        decay = model_numpy(t, irf, truth, "bi-exponential", shift_method)
        m, out = self._fit(
            decay, irf, 2, shift_method=shift_method, fit_background=background > 0
        )
        maps = out["results"]["maps"]
        np.testing.assert_allclose(maps["alpha1_map"], 0.5, atol=1e-3)
        np.testing.assert_allclose(maps["photon_count_map"], 1e5, rtol=1e-4)
        np.testing.assert_allclose(
            maps["tau_mean_map"], 0.5 * 0.3 + 0.5 * 4.0, rtol=1e-3
        )
        np.testing.assert_allclose(maps["v_shift_map"], background, atol=1e-2)
        np.testing.assert_allclose(m.taus_.ravel(), [0.3, 4.0], rtol=1e-3)
        keys = ["tau1_map", "tau2_map", "alpha1_map", "photon_count_map", "v_shift_map"]
        recon = ParamToDecay("bi-exponential", 80.0, irf, shift_method=shift_method)
        fit = recon.reconstruct({k: maps[k] for k in keys}, verbose=False)["fit_map"]
        np.testing.assert_allclose(
            out["results"]["TR_maps"]["fit_map"], fit, rtol=1e-5, atol=1e-3
        )

    def test_mono_parameters_follow_solver_definitions(self, axis):
        from pyfli.cm_funcs.forward_model import model_numpy

        t, irf = axis
        decay = model_numpy(t, irf, [5e4, 2.0, 0.0, 0.0], "mono-exponential")
        m, out = self._fit(decay, irf, 1)
        maps = out["results"]["maps"]
        np.testing.assert_allclose(maps["tau_map"], 2.0, rtol=1e-4)
        np.testing.assert_allclose(maps["tau_mean_map"], 2.0, rtol=1e-4)
        np.testing.assert_allclose(maps["photon_count_map"], 5e4, rtol=1e-4)
        np.testing.assert_allclose(maps["alpha_map"], 1.0)

    @pytest.mark.parametrize(
        "n_components,background", [(1, False), (2, False), (2, True)]
    )
    def test_n_params_and_statistics(self, axis, n_components, background):
        from pyfli.cm_funcs.forward_model import model_numpy
        from pyfli.cm_stats.shared_metrics import (
            pearson_chi_square,
            reduced_poisson_deviance,
        )

        t, irf = axis
        rng = np.random.default_rng(0)
        mu = model_numpy(
            t, irf, [2e4, 0.5, 0.4, 3.0, 2.0 * background, 0.0], "bi-exponential"
        )
        decay = rng.poisson(np.broadcast_to(mu, (3, 3, self.N_GATES))).astype(float)
        m, out = self._fit(decay, irf, n_components, fit_background=background)
        maps, tr = out["results"]["maps"], out["results"]["TR_maps"]
        n_params = 2 * n_components + background
        _, red = reduced_poisson_deviance(tr["fit_map"], decay, n_params)
        np.testing.assert_allclose(maps["reduced_chi2_map"], red, rtol=1e-4)
        _, lag_red = reduced_poisson_deviance(
            tr["laguerre_fit_map"], decay, 16 + background
        )
        np.testing.assert_allclose(
            maps["laguerre_reduced_chi2_map"], lag_red, rtol=1e-4
        )
        pearson = pearson_chi_square(tr["fit_map"], decay)
        np.testing.assert_allclose(maps["pearson_chi2_map"], pearson, rtol=1e-4)
        np.testing.assert_allclose(
            maps["pearson_reduced_chi2_map"],
            pearson / (self.N_GATES - n_params),
            rtol=1e-4,
        )
        np.testing.assert_allclose(tr["residual_map"], decay - tr["fit_map"], atol=1e-2)
        assert out["results"]["error_maps"].shape[-1] == 2 * n_components + 2

    def test_reduced_chi2_is_one_for_correct_model(self, axis):
        from pyfli.cm_funcs.forward_model import model_numpy

        t, irf = axis
        rng = np.random.default_rng(1)
        mu = model_numpy(t, irf, [2e4, 2.0, 0.0, 0.0], "mono-exponential")
        decay = rng.poisson(np.broadcast_to(mu, (15, 15, self.N_GATES))).astype(float)
        _, out = self._fit(decay, irf, 1)
        assert abs(out["results"]["maps"]["reduced_chi2_map"].mean() - 1.0) < 0.05

    def test_circular_background_fitted_in_exponential_stage(self, axis):
        from pyfli.cm_funcs.forward_model import model_numpy
        from pyfli.reconstruction import ParamToDecay

        t, irf = axis
        truth = [1e5, 0.5, 0.3, 4.0, 3.0, 0.0]
        decay = model_numpy(t, irf, truth, "bi-exponential", "circular")
        m, out = self._fit(decay, irf, 2, shift_method="circular", fit_background=True)
        maps = out["results"]["maps"]
        np.testing.assert_allclose(maps["v_shift_map"], 3.0, atol=0.05)
        np.testing.assert_allclose(maps["alpha1_map"], 0.5, atol=2e-3)
        np.testing.assert_allclose(maps["photon_count_map"], 1e5, rtol=2e-3)
        np.testing.assert_allclose(m.taus_.ravel(), [0.3, 4.0], rtol=2e-3)
        keys = ["tau1_map", "tau2_map", "alpha1_map", "photon_count_map", "v_shift_map"]
        recon = ParamToDecay("bi-exponential", 80.0, irf, shift_method="circular")
        fit = recon.reconstruct({k: maps[k] for k in keys}, verbose=False)["fit_map"]
        np.testing.assert_allclose(
            out["results"]["TR_maps"]["fit_map"], fit, rtol=1e-4, atol=1e-2
        )
        mu = model_numpy(
            t, irf, [1e5, 0.5, 0.3, 2.0, 3.0, 0.0], "bi-exponential", "circular"
        )
        noisy = np.random.default_rng(9).poisson(
            np.broadcast_to(mu, (6, 6, self.N_GATES))
        )
        m, out = self._fit(
            noisy.astype(float), irf, 2, shift_method="circular", fit_background=True
        )
        noisy_maps = out["results"]["maps"]
        assert abs(np.median(noisy_maps["v_shift_map"]) - 3.0) < 1.5
        assert abs(np.median(noisy_maps["tau2_map"]) - 2.0) < 0.06
        assert abs(np.median(noisy_maps["reduced_chi2_map"]) - 1.0) < 0.1

    def test_invalid_combinations_raise(self, axis):
        t, irf = axis
        with pytest.raises(ValueError):
            LaguerreFLI(shift_method="wrap")
        with pytest.raises(ValueError):
            LaguerreFLI(dt=self.DT, laser_period_ns=25.0, shift_method="circular").fit(
                irf, irf
            )


class TestWeightingAndOffset:
    N_GATES = 256
    PERIOD = 12.5
    DT = PERIOD / N_GATES

    @pytest.fixture
    def axis(self):
        from pyfli.reconstruction.common_reconstruct import gate_starts

        t = gate_starts(self.PERIOD, self.N_GATES)
        return t, np.exp(-0.5 * ((t - 1.5) / 0.1) ** 2) + 1e-3

    def _model(self, **kwargs):
        return LaguerreFLI(
            2, n_laguerre=12, alpha=0.9, dt=self.DT, verbose=False, **kwargs
        )

    def test_irls_calibrates_reduced_chi2_with_background(self, axis):
        from pyfli.cm_funcs.forward_model import model_numpy

        t, irf = axis
        mu = model_numpy(t, irf, [2e4, 0.5, 0.4, 3.0, 2.0, 0.0], "bi-exponential")
        rng = np.random.default_rng(0)
        decay = rng.poisson(np.broadcast_to(mu, (12, 12, self.N_GATES))).astype(float)
        maps = (
            self._model(fit_background=True)
            .fit(decay, irf)
            .get_parameters()["results"]["maps"]
        )
        assert abs(maps["reduced_chi2_map"].mean() - 1.0) < 0.05
        assert abs(maps["laguerre_reduced_chi2_map"].mean() - 1.0) < 0.05
        assert abs(np.median(maps["v_shift_map"]) - 2.0) < 0.2

    def test_weighting_validation_and_unweighted_option(self, axis):
        t, irf = axis
        with pytest.raises(ValueError):
            self._model(weighting="pearson")
        assert self._model(weighting="none").fit(irf, irf).taus_ is not None

    @pytest.mark.parametrize("shift_method", ["zero_pad", "circular"])
    @pytest.mark.parametrize("h", [0.0, 8 * 12.5 / 256, 0.4, -0.3, 0.0115])
    def test_irf_shift_is_solver_h_shift(self, axis, shift_method, h):
        from pyfli.cm_funcs.forward_model import model_numpy
        from pyfli.reconstruction import ParamToDecay

        t, irf = axis
        decay = model_numpy(
            t, irf, [5e4, 0.5, 0.4, 3.0, 0.0, h], "bi-exponential", shift_method
        )
        m = self._model(shift_method=shift_method).fit(
            decay, irf, irf_shift=h / self.DT
        )
        out = m.get_parameters()
        maps = out["results"]["maps"]
        np.testing.assert_allclose(m.taus_.ravel(), [0.4, 3.0], rtol=1e-3)
        np.testing.assert_allclose(maps["alpha1_map"], 0.5, atol=1e-3)
        np.testing.assert_allclose(maps["photon_count_map"], 5e4, rtol=1e-3)
        np.testing.assert_allclose(maps["h_shift_map"], h, atol=1e-12)
        keys = ["tau1_map", "tau2_map", "alpha1_map", "photon_count_map", "h_shift_map"]
        recon = ParamToDecay("bi-exponential", 80.0, irf, shift_method=shift_method)
        fit = recon.reconstruct({k: maps[k] for k in keys}, verbose=False)["fit_map"]
        np.testing.assert_allclose(
            out["results"]["TR_maps"]["fit_map"], fit, rtol=1e-4, atol=1e-2
        )

    def test_irf_aligner_shift_map_with_manual_correction(self, axis):
        import warnings

        from pyfli.cm_funcs.forward_model import model_numpy
        from pyfli.data_cc.irfAligner import IRFAligner

        t, irf = axis
        h = 0.4
        mu = model_numpy(t, irf, [5e4, 0.5, 0.4, 3.0, 0.0, h], "bi-exponential")
        rng = np.random.default_rng(3)
        decay = rng.poisson(np.broadcast_to(mu, (8, 8, self.N_GATES))).astype(float)
        irf_cube = np.broadcast_to(irf * 1e4, decay.shape).copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = IRFAligner(
                decay, irf_cube, laser_period=self.PERIOD
            ).estimate_shift_debiased()
            shift = IRFAligner(
                decay, irf_cube, laser_period=self.PERIOD
            ).estimate_shift_debiased(
                manual_correction=float(np.median(raw) - h / self.DT)
            )
        m = self._model().fit(decay, irf, irf_shift=shift)
        maps = m.get_parameters()["results"]["maps"]
        np.testing.assert_allclose(maps["h_shift_map"], shift * self.DT)
        np.testing.assert_allclose(
            np.median(m.taus_.reshape(-1, 2), 0), [0.4, 3.0], rtol=0.03
        )
        assert abs(maps["reduced_chi2_map"].mean() - 1.0) < 0.1
        n_params = 2 * 2 + 1
        from pyfli.cm_stats.shared_metrics import reduced_poisson_deviance

        _, red = reduced_poisson_deviance(
            m.get_parameters()["results"]["TR_maps"]["fit_map"], decay, n_params
        )
        np.testing.assert_allclose(maps["reduced_chi2_map"], red, rtol=1e-4)

    def test_irf_shift_validation(self, axis):
        t, irf = axis
        with pytest.raises(ValueError):
            self._model().fit(
                np.ones((2, 2, self.N_GATES)), irf, irf_shift=np.zeros((3, 3))
            )


class TestHShiftInput:
    N_GATES = 256
    PERIOD = 12.5
    DT = PERIOD / N_GATES

    @pytest.fixture
    def axis(self):
        from pyfli.reconstruction.common_reconstruct import gate_starts

        t = gate_starts(self.PERIOD, self.N_GATES)
        return t, np.exp(-0.5 * ((t - 1.5) / 0.1) ** 2) + 1e-3

    def _model(self, **kwargs):
        return LaguerreFLI(
            2, n_laguerre=12, alpha=0.9, dt=self.DT, verbose=False, **kwargs
        )

    def test_h_shift_equals_irf_shift(self, axis):
        from pyfli.cm_funcs.forward_model import model_numpy

        t, irf = axis
        h = np.full((2, 2), 0.37)
        decay = np.broadcast_to(
            model_numpy(t, irf, [5e4, 0.5, 0.4, 3.0, 0.0, 0.37], "bi-exponential"),
            (2, 2, self.N_GATES),
        ).copy()
        a = self._model().fit(decay, irf, h_shift=h)
        b = self._model().fit(decay, irf, irf_shift=h / self.DT)
        np.testing.assert_allclose(a.taus_, b.taus_)
        np.testing.assert_allclose(
            a.get_parameters()["results"]["maps"]["h_shift_map"], h, atol=1e-12
        )
        with pytest.raises(ValueError):
            self._model().fit(decay, irf, h_shift=h, irf_shift=h / self.DT)

    def test_h_shift_map_from_harmonic_offset(self, axis):
        from pyfli.cm_funcs.forward_model import model_numpy
        from pyfli.phasor.harmonic_offset import HarmonicOffset
        from pyfli.phasor.phasorS import PhasorAnalyzer

        t, irf = axis
        h, X = 0.3, 6
        rng = np.random.default_rng(4)
        mono = model_numpy(t, irf, [2e5, 2.0, 0.0, h], "mono-exponential", "circular")
        bi = model_numpy(
            t, irf, [2e5, 0.5, 0.4, 3.0, 0.0, h], "bi-exponential", "circular"
        )
        clean = np.empty((X, X, self.N_GATES))
        clean[:, :2], clean[:, 2:] = mono, bi
        decay = rng.poisson(clean).astype(float)
        analyzer = PhasorAnalyzer(80e6, t, n_harmonics=3, device="cpu")
        res = HarmonicOffset(analyzer).estimate(decay, irf)
        assert abs(np.median(res["h_shift_map"]) - h) < 0.01
        m = self._model(shift_method="circular").fit(
            decay[:, 2:], irf, h_shift=res["h_shift_map"][:, 2:]
        )
        np.testing.assert_allclose(
            np.median(m.taus_.reshape(-1, 2), 0), [0.4, 3.0], rtol=0.05
        )
