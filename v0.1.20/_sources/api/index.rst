API Reference
=============

Reference for every public module in the ``pyfli`` package, organized by subpackage:

- :mod:`pyfli.io` — Data loader, file readers, saving helpers, and
  processed-data loaders.
- :mod:`pyfli.solver` — Least-squares, maximum-likelihood, CPU/GPU, binned, and
  global FLI fitting routines.
- :mod:`pyfli.reconstruction` — Rebuilds modeled decay cubes, generate fit statistics
  maps from fitted parameter maps.
- :mod:`pyfli.phasor` — Phasor-domain lifetime analysis: a compact CPU/GPU
  phasor analyzer, acquisition-geometry loci and locus-based lifetime helpers
  (:mod:`~pyfli.phasor.phasorS`).
- :mod:`pyfli.analyticalWorkflow` — Analytical FLI reconstruction helpers.
- :mod:`pyfli.laguerre` — Laguerre-basis deconvolution and fitting method.
- :mod:`pyfli.simulator` — Synthetic FLI/FLIM data generation, hardware noise
  modeling, calibration, and validation tools.
- :mod:`pyfli.analysis` — Post-processing, diagnostics, statistical
  comparison, and result-loading utilities for fitted FLI/FLIM datasets.
- :mod:`pyfli.bayes_utils` — Bayesian posterior-sampling,
  inference over decay images.
- :mod:`pyfli.irf_deconvolution` — Detector-aware IRF deconvolution and joint
  FLI fitting utilities.
- :mod:`pyfli.sp_analysis` — Single-pixel camera basis generation,
  acquisition simulation, and reconstruction solvers.
- :mod:`pyfli.data_cc` — Array preprocessing: normalization, masking, ROI
  extraction, and IRF alignment.
- :mod:`pyfli.data_vnp` — Visualization, normalization, plotting, and
  mono-/bi-exponential comparison tools.
- :mod:`pyfli.data_text` — Text display helpers for interactive fitting
  workflows.
- :mod:`pyfli.roi_maker` — Interactive ROI creation and threshold-mask
  tooling.
- :mod:`pyfli.log_save` — Logging setup and small logging convenience helpers.

The full listing below is generated automatically and includes every module.

.. autosummary::
   :toctree: generated
   :template: module
   :recursive:

   pyfli
