"""
Provide Bayesian-inference tooling for PyFLI.

This module belongs to :mod:`pyfli.bayes_utils` and covers the direct-inference
(BayesFlow/Keras) side of FLI decay fitting, downstream of :mod:`pyfli.solver`
and :mod:`pyfli.reconstruction`: running a trained posterior-sampling model
over a decay image (:class:`BiPipeline`), selecting/aggregating per-pixel
posterior-sample parameter combinations against measured decay
(:class:`ParamSelector`), and visualizing a single pixel's posterior
predictive fit (:func:`plot_pixel_posterior_fit`).
"""

from .param_combinations import ParamSelector
from .posterior_pixel_plot import plot_pixel_posterior_fit
from .unc_trustworthiness import trustworthiness_CrI, trustworthiness_gof, unc

# BiPipeline needs Keras (the "tf" extra) -- keep the rest of this package
# importable without it, matching pyfli.analysis's fbi_analysis fallback.
try:
    from .inference import BiPipeline

    _KERAS_AVAILABLE = True
except ImportError:
    _KERAS_AVAILABLE = False

    class BiPipeline:  # type: ignore[no-redef]
        """Placeholder raised in place of :class:`BiPipeline` when Keras
        (the ``tf`` extra, ``pip install pyfli-lib[tf]``) isn't installed."""

        def __init__(self, *_args, **_kwargs) -> None:
            raise ImportError(
                "BiPipeline requires the 'tf' extra (Keras/TensorFlow). "
                "Install it with: pip install pyfli-lib[tf]"
            )


__all__ = [
    "BiPipeline",
    "ParamSelector",
    "plot_pixel_posterior_fit",
    "trustworthiness_CrI",
    "trustworthiness_gof",
    "unc",
]
