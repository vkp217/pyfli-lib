"""
Tests for the acquisition-geometry loci (pyfli.phasor.phasorS.MonoLocus) and the
locus helpers of pyfli.phasor.phasorS.phasor_locus_tools.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyfli.phasor.phasorS import (
    AcquisitionConfig,
    AcquisitionMode,
    MonoLocus,
    discrete_locus_circle,
    lifetime_from_locus,
    modulation_lifetime,
    phase_lifetime,
    phase_lifetime_gated,
    plot_discrete_n_sweep,
)

FREQ_HZ = 80e6
PERIOD_NS = 12.5
TAU = np.array([0.5, 1.0, 2.5, 3.5, 6.0])


def _numerical_phasor(tau, weight, t, harmonic):
    omega = 2 * np.pi * harmonic / PERIOD_NS
    g = (weight * np.cos(omega * t)).sum(1) / weight.sum(1)
    s = (weight * np.sin(omega * t)).sum(1) / weight.sum(1)
    return g, s


@pytest.mark.parametrize("t_rec_frac", [0.3, 0.5, 0.6432, 0.9])
@pytest.mark.parametrize("harmonic", [1, 2])
def test_truncated_locus_matches_numerical_integration(t_rec_frac, harmonic):
    """Truncated locus equals the integral of e^{-t/tau} e^{i w t} over [0, T_rec]."""
    n = 200_000
    t = (np.arange(n) + 0.5) * t_rec_frac * PERIOD_NS / n
    decay = np.exp(-t[None, :] / TAU[:, None])
    g_ref, s_ref = _numerical_phasor(TAU, decay, t, harmonic)

    g, s, _, _ = MonoLocus(FREQ_HZ).truncated_locus(
        t_rec_frac, harmonic=harmonic, tau_ns=TAU, draw=False
    )
    np.testing.assert_allclose(g, g_ref, atol=1e-6)
    np.testing.assert_allclose(s, s_ref, atol=1e-6)


def test_full_window_truncated_equals_continuous():
    locus = MonoLocus(FREQ_HZ)
    g_t, s_t, _, _ = locus.truncated_locus(1.0, tau_ns=TAU, draw=False)
    g_c, s_c, _, _ = locus.continuous_locus(tau_ns=TAU, draw=False)
    np.testing.assert_allclose(g_t, g_c, atol=1e-12)
    np.testing.assert_allclose(s_t, s_c, atol=1e-12)


def test_continuous_locus_on_universal_semicircle():
    g, s, _, _ = MonoLocus(FREQ_HZ).continuous_locus(tau_ns=TAU, draw=False)
    np.testing.assert_allclose((g - 0.5) ** 2 + s**2, 0.25, atol=1e-12)


def test_acquisition_config_validation():
    with pytest.raises(ValueError):
        AcquisitionConfig(frequency_hz=-1.0)
    with pytest.raises(ValueError):
        AcquisitionConfig(harmonic=0)
    with pytest.raises(ValueError):
        AcquisitionConfig(t_rec_frac=1.5)
    with pytest.raises(ValueError):
        AcquisitionConfig(mode="not_a_mode")
    assert AcquisitionConfig(mode="truncated").mode is AcquisitionMode.TRUNCATED


@pytest.mark.parametrize(
    ("cfg", "method", "args"),
    [
        (AcquisitionConfig(mode="continuous", harmonic=2), "continuous_locus", ()),
        (AcquisitionConfig(mode="discrete", n_bins=16), "discrete_locus", (16,)),
        (
            AcquisitionConfig(mode="gated_single", gate_width_frac=0.4),
            "gated_single_locus",
            (0.4,),
        ),
        (
            AcquisitionConfig(mode="gated_n", gate_width_frac=0.3, n_gates=4),
            "gated_n_locus",
            (0.3, 4),
        ),
        (
            AcquisitionConfig(mode="truncated", t_rec_frac=0.64),
            "truncated_locus",
            (0.64,),
        ),
        (AcquisitionConfig(mode="offset", t0_frac=0.1), "offset_locus", (0.1,)),
    ],
)
def test_acquisition_config_locus_dispatches_to_monolocus(cfg, method, args):
    g, s, tau, ax = cfg.locus(tau_ns=TAU)
    g_ref, s_ref, _, _ = getattr(MonoLocus(FREQ_HZ), method)(
        *args, harmonic=cfg.harmonic, tau_ns=TAU, draw=False
    )
    assert ax is None
    np.testing.assert_allclose(g, g_ref)
    np.testing.assert_allclose(s, s_ref)
    assert cfg.mode.value in cfg.describe()


@pytest.mark.parametrize("harmonic", [1, 2, 3])
def test_phase_and_modulation_lifetime_invert_continuous_locus(harmonic):
    g, s, _, _ = MonoLocus(FREQ_HZ).continuous_locus(harmonic, tau_ns=TAU, draw=False)
    np.testing.assert_allclose(phase_lifetime(g, s, FREQ_HZ, harmonic), TAU, rtol=1e-10)
    np.testing.assert_allclose(
        modulation_lifetime(g, s, FREQ_HZ, harmonic), TAU, rtol=1e-8
    )


def test_lifetime_from_locus_recovers_lifetime_and_respects_mask():
    cfg = AcquisitionConfig(mode="truncated", t_rec_frac=0.6)
    grid = np.linspace(0.01, 10.0, 20_000)
    lg, ls, lt, _ = cfg.locus(tau_ns=grid)
    g, s, _, _ = cfg.locus(tau_ns=TAU)
    mask = np.array([True, True, False, True, True])

    tau = lifetime_from_locus(g, s, lg, ls, lt, mask=mask)
    np.testing.assert_allclose(tau[mask], TAU[mask], atol=5e-3)
    assert np.isnan(tau[~mask]).all()


def test_phase_lifetime_gated_inverts_gated_locus():
    g, s, _, _ = MonoLocus(FREQ_HZ).gated_single_locus(0.4, tau_ns=TAU, draw=False)
    tau = phase_lifetime_gated(g.reshape(1, -1), s.reshape(1, -1), FREQ_HZ, 0.4)
    assert tau.shape == (1, TAU.size)
    np.testing.assert_allclose(tau[0], TAU, rtol=1e-6)
    assert np.isnan(phase_lifetime_gated([np.nan], [0.1], FREQ_HZ, 0.4)).all()


@pytest.mark.parametrize(("n_bins", "harmonic"), [(4, 1), (8, 1), (16, 2), (64, 1)])
def test_discrete_locus_lies_on_analytic_circle(n_bins, harmonic):
    gc, sc, r = discrete_locus_circle(n_bins, harmonic)
    g, s, _, _ = MonoLocus(FREQ_HZ).discrete_locus(
        n_bins, harmonic, tau_ns=TAU, draw=False
    )
    np.testing.assert_allclose(np.hypot(g - gc, s - sc), r, atol=1e-10)


def test_discrete_locus_circle_degenerate_and_limit():
    assert discrete_locus_circle(2, 1) == (0.5, 0.0, 0.5)
    gc, sc, r = discrete_locus_circle(100_000)
    assert gc == 0.5
    assert abs(sc) < 1e-4
    assert abs(r - 0.5) < 1e-8


def test_plot_discrete_n_sweep_draws_one_line_per_n():
    ax = plot_discrete_n_sweep(FREQ_HZ, n_values=(4, 16, 64))
    labels = [line.get_label() for line in ax.get_lines()]
    assert {"N = 4", "N = 16", "N = 64"} <= set(labels)
    plt.close(ax.figure)
