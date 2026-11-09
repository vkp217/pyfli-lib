import numpy as np


def periodic_bin_exponential(tau, n_bins, period):
    """
    Fraction of the photons of a periodically excited mono-exponential decay (lifetime
    ``tau``, any shape ``S``) that falls in each of ``n_bins`` equal bins covering one
    repetition ``period`` (same time unit as ``tau``). In steady state the tails of all
    earlier pulses add up to the same exponential shape, so bin ``k`` receives
    ``exp(-k dt / tau) (1 - exp(-dt / tau)) / (1 - exp(-period / tau))`` with
    ``dt = period / n_bins``. Returns ``S + (n_bins,)``, summing to 1 along the last axis.
    """
    tau = np.asarray(tau, dtype=float)[..., None]
    dt = period / n_bins
    m = np.arange(n_bins)
    return -np.expm1(-dt / tau) * np.exp(-m * dt / tau) / -np.expm1(-n_bins * dt / tau)


def periodic_instrument_exponential(tau, n_bins, period):
    """
    Per-bin photon distribution of a periodically excited mono-exponential decay as a
    real time-resolved instrument records it: excitation (IRF) photons arrive anywhere
    inside their bin, not at its start, so this is the decay kernel that, circularly
    convolved with the binned IRF, gives the binned measurement when the IRF density is
    uniform within each bin. With ``x = dt / tau`` and ``dt = period / n_bins``:

    * bin 0: ``1 - (1 - e^-x) / x`` (decays starting and ending in the excitation bin)
      plus the wrapped tails of earlier pulses;
    * bin ``m >= 1``: ``e^(-m x) (1 - e^-x)(e^x - 1) / x``, divided by
      ``1 - exp(-period / tau)`` for the periodic steady state.

    Its discrete Fourier transform at the laser frequency equals the continuous
    phasor ``1 / (1 - i omega tau)`` up to ~1e-4, so IRF-calibrated phasors of such data
    lie on the universal semicircle. :func:`periodic_bin_exponential` is the kernel when
    every excitation is at a bin start (pyfli's ``PhotonCountSimulator``). Returns
    ``S + (n_bins,)``, summing to 1 along the last axis.
    """
    tau = np.asarray(tau, dtype=float)[..., None]
    dt = period / n_bins
    x = dt / tau
    m = np.arange(n_bins)
    inner = -np.expm1(-x) * np.expm1(x) / x
    steady = 1.0 / -np.expm1(-n_bins * x)
    out = np.exp(-m * x) * inner * steady
    first = 1.0 - (-np.expm1(-x)) / x + np.exp(-n_bins * x) * inner * steady
    out[..., 0] = first[..., 0]
    return out


def gate_starts(period, n_gates):
    """
    Start times ``m * period / n_gates``, ``m = 0..n_gates-1``, of `n_gates` equal gates
    covering ``[0, period)``: the time axis of the fitters, reconstruction, simulators
    and phasors (``linspace(0, period, n_gates, endpoint=False)``).
    """
    return np.arange(n_gates) * (period / n_gates)


def circular_convolve(x, h, axis=-1):
    """
    Circular convolution ``y[n] = sum_m x[m] h[(n - m) mod N]`` along `axis` (inputs
    broadcast, both of length ``N`` there): the discrete periodic measurement model
    ``F = I (*) f`` over one repetition period, under which phasors factorise,
    ``z_k(F) = z_k(I) z_k(f)``.
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    n = x.shape[axis]
    if h.shape[axis] != n:
        raise ValueError(f"length mismatch along axis {axis}: {n} vs {h.shape[axis]}")
    return np.fft.irfft(
        np.fft.rfft(x, axis=axis) * np.fft.rfft(h, axis=axis), n=n, axis=axis
    )


def circular_convolve_torch(x, h, dim=-1):
    """Torch form of :func:`circular_convolve` (differentiable)."""
    import torch

    n = x.shape[dim]
    if h.shape[dim] != n:
        raise ValueError(f"length mismatch along dim {dim}: {n} vs {h.shape[dim]}")
    return torch.fft.irfft(
        torch.fft.rfft(x, dim=dim) * torch.fft.rfft(h, dim=dim), n=n, dim=dim
    )


def _periodic_exponential_cdf(x, tau, h_shift, period, expm1):
    u = x - h_shift
    r = u % period
    return (u - r) / period + expm1(-r / tau) / expm1(-period / tau)


def periodic_gate_integral(a, b, tau, h_shift, period, expm1=np.expm1):
    """
    Fraction of the photons of a periodically excited exponential (lifetime `tau`, onset
    `h_shift` after each pulse, repetition `period`) that falls in ``[a, b)``, in steady
    state. With ``u = x - h_shift`` the cumulative count since ``t = h_shift`` is
    ``Psi(x) = floor(u / T) + (1 - e^{-(u mod T) / tau}) / (1 - e^{-T / tau})``, continuous
    in `x` and `h_shift`, and the gate holds ``Psi(b) - Psi(a)``; gates tiling one period
    sum to 1. For ``h_shift = 0`` on :func:`gate_starts` gates this is
    :func:`periodic_bin_exponential`. Pass ``expm1=torch.expm1`` for tensors.
    """
    return _periodic_exponential_cdf(
        b, tau, h_shift, period, expm1
    ) - _periodic_exponential_cdf(a, tau, h_shift, period, expm1)


def fourier_shift(x, shift):
    """
    Delay `x` by `shift` samples (fractional; scalar or broadcast against
    ``x.shape[:-1]``) along the last axis with the phase ramp ``e^{-2 pi i k shift / N}``:
    ``x[n] -> x[n - shift]``
    circularly, exact for integer shifts and for band-limited periodic signals. A
    positive shift delays, as :meth:`pyfli.data_cc.irfAligner.IRFAligner.estimate_shift`
    reports it; delaying the IRF by ``s`` gates is the solver's onset
    ``h_shift = s * dt``.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    shift = np.asarray(shift, dtype=float)[..., None]
    phase = np.exp(-2j * np.pi * np.fft.rfftfreq(n) * shift)
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * phase, n=n, axis=-1)


NLSF_WEIGHTINGS = ("irls", "none", "neyman")


def residual_variance(model, data, weighting, variance_floor=1.0):
    """
    Per-gate variance ``v_k`` of weighted least squares ``sum_k (d_k - mu_k)^2 / v_k``
    (NumPy arrays or torch tensors): ``"irls"`` ``max(mu_k, variance_floor)`` from the
    model held fixed (Poisson variance; see :func:`irls`), ``"neyman"`` ``max(d_k, 1)``
    from the data, ``"none"`` 1.
    """
    if weighting == "irls":
        return model.clip(min=variance_floor)
    if weighting == "neyman":
        return data.clip(min=1.0)
    if weighting == "none":
        return data * 0 + 1
    raise ValueError(f"weighting must be one of {NLSF_WEIGHTINGS}, got {weighting!r}")


def _relative_change(old, new):
    return np.max(np.abs(new - old) / np.maximum(np.abs(old), 1e-12))


def irls(solve, predict, x0, variance_floor=1.0, max_rounds=20, tol=1e-6, change=None):
    """
    Iteratively reweighted least squares for Poisson counts: each round solves the
    weighted problem with the variance frozen at ``max(mu, variance_floor)`` of the
    previous round, ``x_{r+1} = solve(x_r, residual_variance(predict(x_r)))``, until
    ``change(x_r, x_{r+1}) < tol`` (default: largest element-wise relative change) or
    `max_rounds`. A `change` returning one value per row of ``x`` (pixels) freezes
    each row at its own convergence, so a batch reproduces per-pixel runs exactly. At
    convergence ``sum_k (d_k - mu_k) / mu_k * dmu_k/dx = 0``, the
    Poisson likelihood equations.

    Parameters
    ----------
    solve : callable
        ``solve(x, variance) -> (x_new, extra)``.
    predict : callable
        ``predict(x) -> mu``, the model counts on the fitted gates.
    x0 : array_like
        Starting parameters.

    Returns
    -------
    tuple
        ``(x, extra)`` of the last round.
    """
    change = _relative_change if change is None else change
    x, extra = np.asarray(x0, dtype=float), None
    active = np.ones(np.shape(change(x, x)), dtype=bool)
    for _ in range(max_rounds):
        variance = residual_variance(
            np.asarray(predict(x)), None, "irls", variance_floor
        )
        x_new, extra = solve(x, variance)
        delta = change(x, x_new)
        if active.ndim:
            x = np.where(active[:, None], x_new, x)
        else:
            x = x_new
        active = active & ~(delta < tol)
        if not active.any():
            break
    return x, extra


def weighted_linear_lstsq(V, d, variance=None, penalty=None, reg_strength=0.0):
    """
    Per-pixel solution of ``min_c sum_t (d_t - (V c)_t)^2 / v_t + lam sum_k p_k c_k^2``
    with ``lam = reg_strength * mean_k sum_t V_tk^2 / v_t``, by QR of the
    sqrt-weighted, penalty-augmented design.

    Parameters
    ----------
    V : np.ndarray
        Design ``(T, K)`` shared by all pixels, or ``(P, T, K)`` per pixel.
    d : np.ndarray
        Data ``(P, T)``.
    variance : np.ndarray | None
        ``(P, T)`` per-gate variances (:func:`residual_variance`); ``None`` is 1.
    penalty : np.ndarray | None
        ``(K,)`` penalty weights ``p_k`` for `reg_strength`.

    Returns
    -------
    np.ndarray
        ``(P, K)`` coefficients.
    """
    d = np.asarray(d, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.ndim == 2 and variance is None and reg_strength == 0.0:
        return np.linalg.lstsq(V, d.T, rcond=None)[0].T
    P, T = d.shape
    K = V.shape[-1]
    sw = np.ones((P, T)) if variance is None else 1.0 / np.sqrt(variance)
    A = sw[..., None] * np.broadcast_to(V, (P, T, K))
    y = sw * d
    if reg_strength > 0.0:
        p = np.ones(K) if penalty is None else np.asarray(penalty, dtype=float)
        lam = reg_strength * np.mean(np.sum(A**2, axis=1), axis=-1)
        aug = np.sqrt(lam)[:, None, None] * np.diag(np.sqrt(p))
        A = np.concatenate([A, aug], axis=1)
        y = np.concatenate([y, np.zeros((P, K))], axis=1)
    q, r = np.linalg.qr(A)
    try:
        return np.linalg.solve(r, np.einsum("ptk,pt->pk", q, y)[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.einsum("pkt,pt->pk", np.linalg.pinv(A), y)


def decay_histogram(taus, fractions, irf, period, periodic=True):
    """
    Expected fraction of the emitted photons in each of the ``n`` gates tiling one
    `period` (:func:`gate_starts`), for a decay whose photons come from exponential
    components `taus` with photon fractions `fractions`, each excitation at the start of an
    IRF gate (``irf`` normalised here). ``periodic=True``: steady state of the pulse train
    (photons wrapped over the period), :func:`periodic_gate_integral` circularly convolved
    with the IRF; sums to 1, as pyfli's photon simulator with ``pileup_mode="wrap"`` and the
    solver's ``shift_method="circular"``. ``periodic=False``: a single pulse, photons past
    the period lost, gate-integrated exponentials linearly convolved and truncated, as
    ``pileup_mode="truncate"`` and ``shift_method="zero_pad"``.
    """
    irf = np.asarray(irf, dtype=float)
    irf = irf / irf.sum()
    n = irf.size
    dt = period / n
    t = gate_starts(period, n)
    kernel = np.zeros(n)
    for tau, fraction in zip(np.atleast_1d(taus), np.atleast_1d(fractions)):
        if periodic:
            kernel += fraction * periodic_gate_integral(
                t, t + dt, float(tau), 0.0, period
            )
        else:
            kernel += fraction * -np.diff(np.exp(-np.append(t, period) / float(tau)))
    if periodic:
        return circular_convolve(kernel, irf)
    return np.convolve(kernel, irf)[:n]


def mono_reconstruction(t, tau, S=1.0):
    return (S / tau) * np.exp(-t / tau)


def bi_reconstruction(t, tau1, tau2, A1, A2):
    return (A1 / tau1) * np.exp(-t / tau1) + (A2 / tau2) * np.exp(-t / tau2)


def mono_reconstruction_torch(t, tau, S=1.0):
    return (S / tau) * (-t / tau).exp()


def bi_reconstruction_torch(t, tau1, tau2, A1, A2):
    return (A1 / tau1) * (-t / tau1).exp() + (A2 / tau2) * (-t / tau2).exp()
