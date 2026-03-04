"""Sphinx configuration for Transformation Portal API documentation."""

import os
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Project information
project = "Transformation Portal"
copyright = "2026, Transformation Portal Contributors"  # pylint: disable=redefined-builtin
author = "Transformation Portal Contributors"
release = "2.0.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Napoleon settings (Google/NumPy docstring support)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autosummary
autosummary_generate = True

# Templates
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = []
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable", None),
}

if os.environ.get("SPHINX_OFFLINE", "").strip() == "1":
    # Local/offline docs builds cannot fetch
    # inventory URLs; disable intersphinx
    # to keep `-W` checks actionable.
    extensions = [ext for ext in extensions if ext != "sphinx.ext.intersphinx"]
    intersphinx_mapping = {}

# Suppress warnings for missing references in external packages
nitpicky = False

# Suppress autodoc warnings for missing imports (optional dependencies)
autodoc_mock_imports = [
    "skimage",
    "lpips",
    "coremltools",
    "torch",
    "torchvision",
    "transformers",
    "diffusers",
]
