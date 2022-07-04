# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "GluonTS"
copyright = "2022, Amazon"
author = "Amazon"

nitpicky = True

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "nbsphinx",
    # "IPython.sphinxext.ipython_console_highlighting",
    # "IPython.sphinxext.ipython_directive",
    "myst_parser",
    "mdinclude",
]


autosummary_generate = True

autodoc_preserve_defaults = True
autodoc_type_aliases = {"DataEntry": "gluonts.dataset.DataEntry"}

python_use_unqualified_type_names = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# are we running locally?
if not "CI" in os.environ:
    html_theme_options = {
        "announcement": "<strong>Warning:</strong> Locally build docs.",
        "light_css_variables": {
            "color-announcement-background": "var(--color-background-secondary)",
            "color-announcement-text": "#119431",
            "color-brand-primary": "#119431",
            "color-brand-content": "#119431",
        },
        "dark_css_variables": {
            "color-announcement-background": "var(--color-background-secondary)",
            "color-announcement-text": "#76D652",
            "color-brand-primary": "#76D652",
            "color-brand-content": "#76D652",
        },
    }
elif os.environ.get("GITHUB_REF_NAME") == "dev":
    html_theme_options = {
        "announcement": "<strong>Warning:</strong> You are looking at the development docs.",
        "light_css_variables": {
            "color-announcement-background": "var(--color-background-secondary)",
            "color-announcement-text": "#db6a00",
            "color-brand-primary": "#ff6f00",
            "color-brand-content": "#ff6f00",
        },
        "dark_css_variables": {
            "color-announcement-background": "var(--color-background-secondary)",
            "color-announcement-text": "#db6a00",
            "color-brand-primary": "#ff6f00",
            "color-brand-content": "#ff6f00",
        },
    }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["custom.css"]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/gluon-logo.svg"

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/gluon.ico"

# Enable Markdown
source_suffix = [".rst", ".md"]
