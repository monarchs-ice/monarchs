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
sys.path.insert(0, os.path.abspath("../../src"))


from pathlib import Path

from monarchs.config import to_markdown as _settings_markdown
from monarchs.variables import to_markdown as _variables_markdown
from monarchs.met_data import to_markdown as _met_markdown

# write text from the config files
pwd = Path(__file__).parent
(pwd / "variables.md").write_text(_variables_markdown())
(pwd / "settings.md").write_text(_settings_markdown())
(pwd / "met.md").write_text(_met_markdown())

project = "MONARCHS"
copyright = "2024, Sammie Buzzard, Jon Elsey and Alex Robel"
author = "Sammie Buzzard, Jon Elsey and Alex Robel"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
]

extensions.append("autoapi.extension")
extensions.append("myst_parser")  # lets us use Markdown rather than reST

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
autoapi_template_dir = (  # exclude_patterns = ['_build', '_templates']
    "./source/_templates/autoapi"
)
autoapi_python_class_content = "both"
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
