from __future__ import annotations

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# docsrc/ lives at the repo root, next to the ``pyfli`` package itself, so
# add the repo root to sys.path for autodoc/autosummary to import ``pyfli``.
sys.path.insert(0, os.path.abspath(".."))

try:
    from sphinx_polyversion import load

    # Imported for its side effect only: this registers GitRef with the
    # package's JSON decoder, so load()["current"] below deserializes to a
    # GitRef (with .name) instead of a plain dict.
    from sphinx_polyversion.git import GitRef  # noqa: F401

    USE_POLYVERSION = True
    current = load(globals())["current"].name
except ImportError:
    USE_POLYVERSION = False
    current = "local"

import pyfli

# -- Project information ------------------------------------------------------
project = "PyFli"
author = "Vikas Pandey"
copyright = f"2025-{datetime.now().year}, PyFli developer and maintainer: {author}"
release = current if USE_POLYVERSION and current != "local" else pyfli.__version__
version = release

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # examples/ex_helper/ holds plain .py helper scripts the example notebooks
    # import at runtime (for local reproduction) — not build-time doc sources.
    "examples/ex_helper",
]

source_suffix = {
    ".rst": "restructuredtext",
}

# Modules that are heavy, optional, or hardware/GUI dependent. They are
# mocked so the docs can be built in environments where these extras
# (GPU/CUDA, TensorFlow, Qt platform plugins, etc.) are not installed.
autodoc_mock_imports = [
    "torch",
    "tensorflow",
    "keras",
    "bayesflow",
    "PySide6",
    "cv2",
    "nvidia",
]

# -- Autodoc / Autosummary -----------------------------------------------------
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True

add_module_names = False
python_use_unqualified_type_names = True

# -- Napoleon (NumPy / Google style docstrings) --------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True
napoleon_use_ivar = False
napoleon_preprocess_types = True

# -- Intersphinx ----------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}

# -- MyST (Markdown) -----------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -- MyST-NB (Jupyter notebooks) ------------------------------------------------
# Notebooks under docsrc/examples/ ship with their outputs already saved, and
# may depend on acquisition hardware/data not available in CI, so render the
# saved outputs as-is rather than re-executing on every docs build.
nb_execution_mode = "off"

# -- HTML output ----------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "PyFli"
html_baseurl = "https://pyfli.org/"
html_show_sourcelink = True
html_sourcelink_suffix = ""

html_theme_options = {
    "logo": {
        "text": "PyFli",
        "image_light": "../pyfli/img/PyFLI_logo_light.png",
        "image_dark": "../pyfli/img/PyFLI_logo_dark.png",
    },
    "github_url": "https://github.com/vkp217/pyfli-pkg",
    "navbar_end": ["theme-switcher", "navbar-icon-links", "version-switcher"],
    "switcher": {
        "json_url": "https://pyfli.org/versions.json",
        "version_match": current,
    },
    "check_switcher": False,
    "navbar_align": "left",
    "navigation_with_keys": True,
    "show_prev_next": False,
    "show_toc_level": 2,
    "collapse_navigation": True,
    # "sourcelink" (the "Show Source" link) is dropped just for the index
    # page, where it only ever offered to download index.rst; every other
    # page keeps it, since examples.md relies on it to let readers download
    # and re-run the notebooks.
    "secondary_sidebar_items": {
        "**": ["page-toc", "sourcelink"],
        "index": ["page-toc"],
    },
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}

html_context = {
    "github_user": "vkp217",
    "github_repo": "pyfli-pkg",
    "github_version": current,
    "doc_path": "docsrc",
}

html_logo = "../pyfli/img/PyFLI_logo.png"
html_favicon = "../pyfli/img/PyFLI_logo.png"

html_css_files = ["custom.css"]
html_sidebars = {
    "index": [],
    "install": [],
    "quickstart": [],
    "changelog": [],
    "citation": [],
    "contributing": [],
    "faq": [],
}

# -- Options for autosummary/autodoc member ordering ---------------------------
suppress_warnings = ["autosummary"]

# -- Hide proprietary members --------------------------------------------------
# pyfli.analysis.__init__ defines fallback stubs for the FBI model workflow so
# that importing these names still works when the private fbi_analysis module
# (excluded from the public repo, see .gitignore) isn't installed. The stubs
# are real, public, module-level functions, so autodoc would otherwise
# document them here even though the underlying implementation is never
# published.
_HIDDEN_MEMBERS = {
    "load_fbi_model",
    "run_fbi_inference",
    "compute_fbi_results",
    "plot_fbi_maps",
}


def _skip_proprietary_members(app, what, name, obj, skip, options):
    if name in _HIDDEN_MEMBERS:
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", _skip_proprietary_members)
