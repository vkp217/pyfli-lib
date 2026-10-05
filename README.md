<p align="center">
  <img src="pyfli/img/PyFLI_logo.png" alt="PyFLI logo: open-source FLIM analysis software in Python" width="300"/>
</p>

# PyFLI (`pyfli-lib`): A Unified Open-Source Python Library for FLIM Analysis and Fluorescence Lifetime Imaging


[![Website](https://img.shields.io/badge/website-pyfli.org-blue.svg)](https://pyfli.org)
[![PyPI version](https://img.shields.io/pypi/v/pyfli-lib.svg)](https://pypi.org/project/pyfli-lib/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/vkp217/pyfli-lib/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/vkp217/pyfli-lib/actions/workflows/tests.yml)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/vkp217/pyfli-lib/issues)



**PyFLI** is an open-source Python library for **Fluorescence Lifetime Imaging (FLI/FLIM)** analysis. It loads TCSPC, SPAD and ICCD data from different manufacturers through one interface, for both microscopy (FLIM) and macroscopic imaging (MFLI), and provides:

* **Traditional analysis:** NLSF and MLE lifetime fitting, RLD, Laguerre deconvolution and phasor plot analysis, on CPU or GPU.
* **Advanced analysis:** FLIM Deep-learning inference, and compressed-sensing reconstruction for single-pixel hyperspectral FLI.
* **Advanced statistical analysis:**
  * **Precision limits:** CRLB-derived bounds for fitted parameters, computed from the Poisson Fisher information matrix (Cramér-Rao lower bound, CRLB).
  * **Photon economy:** coefficient of variation of lifetime estimates as a function of photon count.
  * **Error analysis:** per-pixel error against simulated ground truth, and goodness-of-fit with Poisson deviance.
  * **Method comparison:** results from every fitting method binned and compared across factors such as photon count, plus mono- vs. bi-exponential model classification.
  * **Distribution tests:** classical and multivariate tests that compare simulated and experimental data.
  * **Cross-software benchmarking:** import results processed in other software and compare them with PyFLI.
* **A standardized platform for FLIM deep learning:** a complete pipeline for training, running and benchmarking FLIM deep-learning models. It includes a detector-aware simulator that generates labeled training data with known ground truth, and a common benchmark that compares networks against analytical methods.

---

## Supported Data Acquisition Methods

The platform provides native support for several high-end imaging systems:

1. **ICCD:** Intensified Charge-Coupled Device cameras for fast-gated, wide-field imaging.
2. **SPAD:** High-speed SPAD (Single-Photon Avalanche Diode) architectures for high-resolution photon counting.
3. **TCSPC:** Standardized processing for Time-Correlated Single Photon Counting microscopy data.

## Data Processing & Analysis

PyFLI (`pyfli-lib`) implements standard analytical methods to extract lifetime information:

* **Non-linear Least Squares Fitting (NLSF):** Robust mathematical approach for exponential decay modeling.
* **Phasor Plot Analysis:** Graphical, model-free transformation of fluorescence decay into a 2D polar plot for easy species separation.
* **Maximum Likelihood Estimation (MLE):** Statistical estimator optimized for low-photon regimes.
* **Rapid Lifetime Determination (RLD):** Computationally efficient method for real-time applications and high-frame-rate data.
* **Laguerre Method:** Model-free IRF deconvolution followed by multi-exponential lifetime extraction on a per-pixel basis.
* **Bayesian Inference:** Probabilistic parameter estimation with uncertainty quantification for lifetime fitting.

---

## Installation

Install the stable version directly from PyPI:

```bash
pip install pyfli-lib
```

For users requiring GPU-based processing, install the optional tensor/AI dependencies:

```bash
pip install "pyfli-lib[gpu]"
```

## Quick Start

Even though the package is installed as `pyfli-lib`, you import it as `pyfli` in your scripts:

```python
from pyfli import DataOperations

loader = DataOperations(
    data_path="experimental_data.sdt",
    irf_path="instrument_data.txt",
    bg_path="background_data.tif",
    mask_path="background_data.png",
)
decay_data = loader.load_data()
irf_data = loader.load_irf()
```

## Documentation

Full documentation, examples and API reference are at [pyfli.org](https://pyfli.org). Guides:

* [How to perform phasor plot analysis in Python with PyFLI](https://pyfli.org/main/user_guide/phasor_plot_analysis.html)
* [Training deep-learning FLIM models on simulated data](https://pyfli.org/main/user_guide/deep_learning_flim_simulation.html)
* [Example notebooks: NLSF/MLE fitting, phasor analysis, FLIM simulation and SPAD data I/O](https://pyfli.org/main/examples.html)

## Citation

If you use PyFLI in your research, please cite this package:

> Pandey V., Erbas I., Barroso M., Radev S., Intes X. *PyFLI: A Python Library for Simulation, Parameter Estimation, and Benchmarking in Fluorescence Lifetime Imaging.*
> https://arxiv.org/abs/2609.11994

```bibtex
@misc{pandey2026pyflipythonlibrarysimulation,
      title={PyFLI: A Python Library for Simulation, Parameter Estimation, and Benchmarking in Fluorescence Lifetime Imaging},
      author={Vikas Pandey and Ismail Erbas and Margarida Barroso and Stefan Radev and Xavier Intes},
      year={2026},
      eprint={2609.11994},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2609.11994},
}
```

---

## Repository & Issues

The source code is hosted on GitHub. Please report any bugs or feature requests via the issues tracker.
* **Website:** [PyFLI, the open-source FLIM analysis framework (pyfli.org)](https://pyfli.org)
* **GitHub:** [https://github.com/vkp217/pyfli-lib](https://github.com/vkp217/pyfli-lib)
* **Contact:** For any queries, reach out at [pyfli4lifetime@gmail.com](mailto:pyfli4lifetime@gmail.com) or [support@pyfli.org](mailto:support@pyfli.org)
