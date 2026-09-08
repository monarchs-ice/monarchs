"""
MONARCHS grid variable schema.

* ``catalogue.py``    - Catalogue of variables in the model.
* ``definitions.py``  - Definitions of a Variable, Dimension, and other stuff needed
                        to properly initialise variables.
* ``initial_value_functions.py`` - Functions describing initial values of variables
                                that depend on other variables.
* ``build.py``        - Take the catalogue, and turn it into the stuff the model needs
* ``docs.py``         - renders the catalogue as a Markdown reference.
"""

from monarchs.variables.build import (
    build_dtype,
    make_grid,
    validate_catalogue,
    variable_metadata,
)
from monarchs.variables.catalogue import CATALOGUE
from monarchs.variables.definitions import Dim, Variable
from monarchs.variables.docs import to_markdown

__all__ = [
    "build_dtype",
    "make_grid",
    "variable_metadata",
    "validate_catalogue",
    "to_markdown",
    "CATALOGUE",
    "Variable",
    "Dim",
]
