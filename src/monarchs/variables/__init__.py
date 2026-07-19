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

from monarchs.variables import build
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


def build_dtype(vert_grid, vert_grid_lake, vert_grid_lid, n_directions=8):
    """Structured-array dtype for the model grid (drop-in for get_spec)."""
    return build.build_dtype(
        CATALOGUE, vert_grid, vert_grid_lake, vert_grid_lid, n_directions
    )


def make_grid(
    num_rows, num_cols, vert_grid, vert_grid_lake, vert_grid_lid, inputs=None
):
    """
    Build the model grid. This explicitly passes CATALOGUE automatically, so
    use this public-facing import rather than importing build.make_grid directly
    """
    return build.make_grid(
        CATALOGUE, num_rows, num_cols, vert_grid, vert_grid_lake, vert_grid_lid, inputs
    )


def variable_metadata():
    """{name: {attr: value}} netCDF metadata for output-eligible variables."""
    return build.variable_metadata(CATALOGUE)


def validate_catalogue():
    """Sanity-check the catalogue (duplicate names, bad dims)."""
    return build.validate_catalogue(CATALOGUE)
