from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

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
    _poly_data = load(globals())
    current = _poly_data["current"].name
    latest = _poly_data["latest"].name
except ImportError:
    USE_POLYVERSION = False
    current = "local"
    latest = "main"

import pyfli

# -- Project information ------------------------------------------------------
project = "PyFLI"
author = "Vikas Pandey"
copyright = f"2025-{datetime.now().year}, PyFLI developer and maintainer: {author}"
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
html_title = "PyFLI"
SITE_URL = "https://pyfli.org"
html_baseurl = f"{SITE_URL}/{latest}/"
html_show_sourcelink = True
html_sourcelink_suffix = ""

html_theme_options = {
    "logo": {
        "text": "PyFLI",
        "image_light": "../pyfli/img/PyFLI_logo_light.png",
        "image_dark": "../pyfli/img/PyFLI_logo_dark.png",
    },
    "github_url": "https://github.com/vkp217/pyfli-lib",
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
    "github_repo": "pyfli-lib",
    "github_version": current,
    "doc_path": "docsrc",
    "seo_home_url": html_baseurl,
    "seo_home_title": (
        "PyFLI | Open-Source FLIM Analysis & Fluorescence Lifetime Imaging in Python"
    ),
    "seo_description": (
        "PyFLI is an open-source Python library for Fluorescence Lifetime Imaging "
        "Microscopy (FLIM) analysis: TCSPC and SPAD data processing, phasor plot "
        "analysis, NLSF/MLE lifetime fitting with IRF deconvolution, and FLIM "
        "data simulation for deep-learning model training."
    ),
    "seo_image_url": f"{html_baseurl}_static/pyfli-social-card.png",
    "seo_pypi_url": "https://pypi.org/project/pyfli-lib/",
    "seo_repo_url": "https://github.com/vkp217/pyfli-lib",
    "seo_paper_url": "https://arxiv.org/abs/2609.11994",
}
html_context["seo_jsonld"] = json.dumps(
    {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": "PyFLI",
        "alternateName": "pyfli-lib",
        "url": html_context["seo_home_url"],
        "description": html_context["seo_description"],
        "image": html_context["seo_image_url"],
        "applicationCategory": "DeveloperApplication",
        "applicationSubCategory": "Fluorescence Lifetime Imaging (FLIM) analysis",
        "operatingSystem": "Windows, Linux, macOS",
        "programmingLanguage": "Python",
        "softwareVersion": pyfli.__version__,
        "downloadUrl": html_context["seo_pypi_url"],
        "installUrl": html_context["seo_pypi_url"],
        "license": "https://www.gnu.org/licenses/agpl-3.0.html",
        "isAccessibleForFree": True,
        "keywords": (
            "FLIM analysis, fluorescence lifetime imaging software, "
            "fluorescence lifetime imaging microscopy, TCSPC, SPAD, "
            "phasor plot analysis, maximum likelihood estimation, "
            "instrument response function, IRF deconvolution, FLIM simulation, "
            "deep learning FLIM, Python"
        ),
        "author": {"@type": "Person", "name": author},
        "sameAs": [html_context["seo_repo_url"], html_context["seo_pypi_url"]],
        "citation": {
            "@type": "ScholarlyArticle",
            "name": (
                "PyFLI: A Python Library for Simulation, Parameter Estimation, "
                "and Benchmarking in Fluorescence Lifetime Imaging"
            ),
            "url": html_context["seo_paper_url"],
        },
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
    },
    indent=2,
)

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


def _write_sitemap(app, exception):
    """Write ``sitemap.xml`` listing every page under its canonical URL."""
    if exception is not None or app.builder.format != "html":
        return
    base = app.config.html_baseurl
    urls = []
    for docname in sorted(app.env.found_docs):
        if docname == app.config.root_doc:
            urls.append(base)
        else:
            urls.append(base + app.builder.get_target_uri(docname))
    entries = "\n".join(f"  <url><loc>{xml_escape(url)}</loc></url>" for url in urls)
    sitemap = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{entries}\n"
        "</urlset>\n"
    )
    (Path(app.outdir) / "sitemap.xml").write_text(sitemap, encoding="utf-8")


def setup(app):
    app.connect("autodoc-skip-member", _skip_proprietary_members)
    app.connect("build-finished", _write_sitemap)
