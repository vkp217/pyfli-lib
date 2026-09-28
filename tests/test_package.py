"""Smoke tests for the curated top-level pyfli API."""

import pyfli
import pyfli.io


def test_version_matches_distribution_metadata():
    from importlib.metadata import version

    assert pyfli.__version__ == version("pyfli-lib")


def test_no_all_export():
    assert not hasattr(pyfli, "__all__")


def test_top_level_core_symbols_are_public():
    expected = [
        "DataOperations",
        "Detector",
        "LaguerreFLI",
        "Normalization",
        "DataSaver",
        "SaveLoadDirector",
        "MacroSimulator",
        "TCSPCSimulator",
        "BasisPatterns",
        "MeasurementSimulator",
        "Reconstructor",
    ]
    for name in expected:
        assert callable(getattr(pyfli, name))


def test_subpackages_available():
    assert pyfli.phasor is not None
    assert pyfli.sp_analysis is not None


def test_logging_module_is_configured():
    assert pyfli.logging.logger.name == "pyfli"
    assert callable(pyfli.logging.info)


def test_laguerre_instantiation():
    model = pyfli.LaguerreFLI(n_components=2, alpha=0.85, dt=0.05)
    assert model.n_components == 2


def test_flim_decay_cube_helpers_are_public():
    for name in ["load_flim_data", "collapse_to_xyt", "plot_xyt"]:
        assert callable(getattr(pyfli, name))


def test_io_subpackage_symbols_are_public():
    expected = [
        "Detector",
        "DataOperations",
        "DataSaver",
        "SaveLoadDirector",
        "AlliGprocessedImport",
        "BHprocessedImport",
        "DatasetPlotter",
        "PyFliprocessedImport",
        "DataIOUtils",
        "load_flim_data",
        "collapse_to_xyt",
        "plot_xyt",
    ]
    for name in expected:
        assert callable(getattr(pyfli.io, name))
