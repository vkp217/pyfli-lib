"""Top-level public API for PyFLI."""
# ruff: noqa: E402, F401, I001
# NOTE: import order below is load-bearing, not arbitrary — `from .log_save import
# logging` must run first so `pyfli.logging` exists before any submodule imported
# afterward (e.g. sp_analysis.simulator.reconstructor) can do `from pyfli import
# logging`. Do not let an isort/ruff auto-fix alphabetize this block.

from importlib.metadata import PackageNotFoundError, version as _version

# Version comes from the installed distribution metadata (built from pyproject.toml).
try:
    __version__ = _version("pyfli-lib")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

del _version, PackageNotFoundError


def setup() -> None:
    """
    Configure PyFLI logging and package defaults.

    Returns
    -------
    None
        No object is returned; package logging is configured for later imports.
    """
    from .log_save.logging import configure_logging

    configure_logging()


setup()
del setup

from .log_save import logging
from . import phasor, sp_analysis
from .laguerre.laguerre_method import LaguerreFLI
from .data_cc.norm import Normalization
from .io.data_operations import DataOperations
from .io.detector import Detector
from .io.data_saving import DataSaver
from .io.save_direction import SaveLoadDirector
from .io.flim_decay_cube import collapse_to_xyt, load_flim_data, plot_xyt
from .simulator.combined.main_factory import MacroSimulator, TCSPCSimulator
from .simulator.separate.sim_model_image_generator import FLIModelImageGenerator
from .sp_analysis import BasisPatterns, MeasurementSimulator, Reconstructor
