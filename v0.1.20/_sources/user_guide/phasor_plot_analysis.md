---
myst:
  html_meta:
    description: "Step-by-step guide to phasor plot analysis of FLIM data in Python with PyFLI: compute phasor coordinates from TCSPC or SPAD or ICCD decays, calibrate with the IRF, map lifetimes and separate two-component mixtures."
    keywords: "phasor plot analysis, FLIM phasor, phasor approach fluorescence lifetime, Python phasor FLIM, TCSPC phasor, IRF calibration, PyFLI"
---

# How to perform phasor plot analysis in Python with PyFLI

The **phasor approach** turns every pixel of a fluorescence lifetime imaging (FLIM) dataset into a single point on a 2-D plot, without fitting a decay model. Pixels with similar lifetimes cluster together, mono-exponential decays
fall on the *universal semicircle*, and mixtures of two lifetime species lie on the straight line joining their two pure-species points.
This makes the phasor plot a fast, fit-free way to explore FLIM data from TCSPC systems, SPAD arrays or gated cameras.

This guide shows the complete phasor workflow in PyFLI with
{py:class}`~pyfli.phasor.phasorS.phasor_simple.PhasorAnalyzer`:

1. compute phasor coordinates from a decay cube,
2. calibrate them with the instrument response function (IRF),
3. convert phasors to a per-pixel lifetime map,
4. plot the phasor diagram and its harmonics, and
5. resolve two-component mixtures into fractional contributions.

## What the phasor coordinates are

For a decay {math}`I(t)` recorded over one laser period at angular modulation
frequency {math}`\omega = 2\pi f`, the phasor coordinates of harmonic {math}`n` are

```{math}
G_n = \frac{\int I(t)\cos(n\omega t)\,dt}{\int I(t)\,dt}, \qquad
S_n = \frac{\int I(t)\sin(n\omega t)\,dt}{\int I(t)\,dt}.
```

A mono-exponential decay with lifetime {math}`\tau` maps to
{math}`G = 1/(1+\omega^2\tau^2)` and {math}`S = \omega\tau/(1+\omega^2\tau^2)`, a point on the semicircle. Its *phase lifetime* is {math}`\tau_\phi = S / (\omega G)`.

The measured decay is the true decay convolved with the IRF, so raw phasors are rotated and shrunk. Dividing by the IRF's own phasor (IRF calibration) removes that distortion and puts the data back on the correct scale.

## Step 1: get a decay cube and an IRF

PyFLI's phasor tools work on a decay cube of shape `(H, W, T)` (image height, width, time bins) and an IRF cube of the same shape. With measured data, load both with {py:class}`~pyfli.io.data_operations.DataOperations` (see {doc}`../quickstart`):

```python
from pyfli import DataOperations

loader = DataOperations(data_path="experiment.sdt", irf_path="irf.txt")
decay = loader.load_data()
irf = loader.load_irf()
```

To follow along without any files, simulate a small mono-exponential TCSPC
image with a Gaussian IRF. Because the simulator records the true lifetime of
every pixel, you can check the phasor result against ground truth:


## Step 1: compute and calibrate the phasors

Create a {py:class}`~pyfli.phasor.phasorS.phasor_simple.PhasorAnalyzer` with the laser
repetition frequency and the time axis of the decay, then compute the phasor
coordinates. `G` and `S` are stacked over harmonics, with shape
`(n_harmonics, H, W)`:

```python
from pyfli.phasor.phasorS import PhasorAnalyzer

phasor = PhasorAnalyzer(
    frequency_hz=80e6,
    time_axis_ns=np.linspace(0, T_NS, N_BINS),
    n_harmonics=3,
)
G, S = phasor.create_phasor_cpu(decay)
Gc, Sc = phasor.calibrate_pixelwise(G, S, irf)
```

`create_phasor_gpu` is a drop-in replacement for `create_phasor_cpu` when a
CUDA GPU is available, which helps for large SPAD or wide-field datasets.
`calibrate_pixelwise` calibrates each pixel against its own IRF, which matters
for SPAD arrays and other detectors whose IRF shifts across the field of view.

## Step 2: map lifetimes

The phase lifetime of the first harmonic gives a lifetime image in
nanoseconds:

```python
tau_phasor = phasor.compute_lifetime(Gc[0], Sc[0])

```
For a pure mono-exponential decay,
{math}`\tau_\phi = \tau_m`, so a large gap between the two is a quick sign of multi-exponential behavior.

## Step 3: plot the phasor diagram

```python
fig = phasor.plot_phasor_diagram(Gc[0], Sc[0], half_circle=True)
```

Pass `mask=` to restrict the plot to a region of interest, and
`hexbin_color=` to choose the density colormap. Two more views are useful when exploring an image:

- `phasor.plot_overlay_subplots(decay, Gc[0], Sc[0], mask=mask)` maps each
  pixel's phasor position back onto the image as a color overlay.
- `phasor.plot_phasor_harmonics(Gc, Sc, harmonics=(1, 2, 3), mask=mask)` shows
  higher harmonics, which spread out components that overlap at the first
  harmonic.

## Step 4: separate two lifetime species

If a sample contains two species with known lifetimes, every pixel's phasor lies on the line between their two pure-species points. Its position along that line gives the fractional contribution of each species:

```python
frac_short, frac_long = phasor.compute_fractions(Gc[0], Sc[0], 0.5, 2.0)
```

`compute_fractions` also draws the phasor plot with the two reference points and the connecting line (pass `plot_graph=False` to skip it). For a full bi-exponential reconstruction of each pixel's decay from its phasor, see `analyze_biexponential_and_reconstruct` in the bi-exponential example below.

## Full worked examples

The example notebooks run this workflow on a simulated "FLIM" letters image
with known lifetimes, including intensity-weighted phasor overlays and fit
comparisons:

- {doc}`../examples/Phasor_analysis_mono`: mono-exponential phasor analysis
- {doc}`../examples/Phasor_analysis_bi`: bi-exponential phasor analysis and
  decay reconstruction

To compare phasor lifetimes with fitted ones, the same data can be fitted with non-linear least squares or maximum likelihood estimation (MLE); see
{doc}`../examples/Monoexponential_fitting`.
