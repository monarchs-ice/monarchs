# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.append(os.path.abspath("../.."))

project = "MONARCHS"
copyright = "2024, Sammie Buzzard, Jon Elsey and Alex Robel"
author = "Sammie Buzzard, Jon Elsey and Alex Robel"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
]

extensions.append("autoapi.extension")

autoapi_dirs = ["../../src/monarchs"]
autoapi_ignore = ["*venv*", "*.run*", "*data*", "*conf.py*", "tests"]
# autosummary_generate = True # Turn on sphinx.ext.autosummary
# autoapi_keep_files = True
autoapi_member_order = "groupwise"
autoapi_template_dir = "_autoapi_templates"
autoapi_own_page_level = "function"
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
autoapi_template_dir = (
    "./_templates/autoapi"  # exclude_patterns = ['_build', '_templates']
)
autoapi_python_class_content = "both"
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
