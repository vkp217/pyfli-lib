---
myst:
  html_meta:
    description: "Generate labeled fluorescence lifetime imaging (FLIM) training data in Python with the PyFLI simulator: realistic IRFs, TCSPC and SPAD detector noise, randomized acquisition conditions, and deep-learning lifetime inference."
    keywords: "deep learning FLIM, FLIM simulation, FLIM training data, simulated TCSPC data, SPAD noise model, lifetime estimation neural network, BayesFlow FLIM, PyFLI"
---

# Training deep-learning FLIM models on simulated data

Deep-learning lifetime estimators can be much faster than pixel-wise fitting and more robust at low photon counts, but they need large training sets with known lifetimes. Real FLIM measurements rarely come with ground truth, so the
standard approach is to **train on simulated decays** and then apply the model to measured data.

For this to work, the simulated decays must cover the distribution of the experimental ones: same instrument response function (IRF), same time binning, same photon budget and the same detector noise.
Otherwise the network learns the simulator rather than the physics.

The PyFLI simulator ({py:mod}`pyfli.simulator`) is built for this. This guide covers:

1. choosing an IRF that matches your instrument,
2. configuring the decay model and detector noise,
3. generating a labeled training set,
4. randomizing acquisition conditions so the model generalizes, and
5. running a trained model over a lifetime image.

## Step 1: choose an IRF

The IRF shapes every decay, so a model trained with the wrong IRF will be biased on real data. Use your instrument's measured IRF cube where possible or use simulated IRF:

```python
from pyfli import DataOperations

irf_cube = DataOperations(irf_path="irf.txt").load_irf()
```

The simulator samples from a 3-D IRF cube of shape `(H, W, T)`. For a quick start, or to study a hypothetical system, generate a synthetic IRF instead. Here it is a 100 ps Gaussian on a 12.5 ns, 256-bin time axis (80 MHz laser):

```python
from pyfli.simulator import IRFGenerator

irf_cube = IRFGenerator.gaussianIRF(mu=40, T=12.5, num_bins=256, sigma=0.1, H=1, W=1)
```

## Step 2: configure the decay model and the detector

A simulator configuration is a plain dictionary. The most important entries are:

- `sensor_type`: `"discrete"` simulates photon counting (TCSPC, SPAD);
  `"continuous"` simulates intensity-scaled, gated or ICCD-type detectors.
- `laser_feq`: laser repetition rate in MHz.
- `mono_fraction`: share of mono-exponential decays in the sampled data (`1.0`
  mono-exponential only, `0.0` bi-exponential only).
- `photo_count`: shape of the photon-budget distribution, which controls the
  range of signal levels the model sees.
- Noise toggles: `jitter` (timing jitter), `dcr_on` with `dcr` (dark count
  rate), `qe_on` (quantum efficiency), `clip_on` (counter or ADC saturation),
  plus `poisson` and `read_noise_on` for large-format continuous detectors.

```python
base_config = {
    "sensor_type": "discrete",
    "laser_feq": 80,
    "mono_fraction": 0.0,
    "photo_count": (2, 5),
    "jitter": True,
    "dcr_on": True,
    "dcr": 0.08,
    "qe_on": True,
    "clip_on": True,
}
```

The {doc}`../examples/single_decay_sim` and {doc}`../examples/whole_image_sim`
examples walk through the full set of options, including per-region lifetime
distributions.

## Step 3: generate a labeled training set

{py:class}`~pyfli.simulator.sim_workflow.SimGenerator` draws one decay at a time, randomly shifting the IRF (time-of-flight effect) in time (`a_range`, in time bins) and adding a baseline offset (`b_range`) so the model learns to tolerate the IRF drift seen in real acquisitions.
Set both to `(0, 0)` to disable this augmentation.
{py:func}`~pyfli.simulator.sim_workflow.make_simulator` stacks many draws into arrays:

```python
from pyfli.simulator import SimGenerator, make_simulator

gen = SimGenerator(irf_cube, base_config, a_range=(-20, 20), b_range=(0, 10))
train = make_simulator(gen.simulate_once, 50_000)
```

`train` is a dictionary of NumPy arrays with one row per simulated decay:

| Key | Content |
| --- | --- |
| `decay` | noisy decay histogram, shape `(N, T)` (network input) |
| `irf` | the (shifted, normalized) IRF used for that decay, shape `(N, T)` |
| `tau1`, `tau2`, `alpha1` | bi-exponential ground truth (lifetimes in ns, amplitude fraction) |
| `tau` | mean lifetime; equals the single lifetime for mono-exponential decays |
| `photon_count` | photon budget of the decay |
| `h_shift`, `v_shift_bgp` | applied IRF time shift and baseline offset |

These arrays can be fed directly to PyTorch, TensorFlow/Keras or JAX training code.

## Step 4: randomize acquisition conditions

A model trained under one fixed noise setting often fails when the dark-count rate or jitter differs on the day of the experiment.
Sweep over the conditions you expect and sample from all of them during training.
Use {py:class}`~pyfli.data_cc.config_combinations.ConfigCombinationGenerator` with
{py:class}`~pyfli.simulator.sim_workflow.WeightedConfigSimGenerator`:

```python
from pyfli.data_cc.config_combinations import ConfigCombinationGenerator
from pyfli.simulator import WeightedConfigSimGenerator

combos = ConfigCombinationGenerator(
    base_config,
    sweep={"jitter": [True, False], "dcr": [0.02, 0.08, 0.2]},
)
configs, probs = combos.combination_table()

mixed = WeightedConfigSimGenerator(irf_cube, configs, probs, seed=0)
train = make_simulator(mixed.simulate_once, 50_000)
```

Pass `weights=` to `ConfigCombinationGenerator` to over-sample the conditions
you care about most.

## Step 5: run a trained model on a lifetime image

Run a model (a saved Keras approximator conditioned on `decay` and `irf`, the same keys the simulator produces) over a full decay image.

It needs the TensorFlow extra (`pip install pyfli-lib[tf]`):


## Benchmark against analytical methods

Simulated data has known ground truth, so it is also the right way to check a trained model. Simulate a held-out test image and compare the network against non-linear least squares, maximum likelihood estimation (MLE) and {doc}`phasor analysis <phasor_plot_analysis>` on the same pixels.

See
{doc}`../examples/Monoexponential_fitting_CPU_GPU` and
{doc}`../examples/Biexponential_fitting_CPU_GPU` for the analytical baselines,
and the [PyFLI manuscript](https://arxiv.org/abs/2609.11994) for a full benchmark.
