"""
Turn the catalogue into the things the model needs - the grid dtype, the
initialised grid itself, and the netCDF metadata for its variables.
"""

import numpy as np

from monarchs.variables.catalogue import CATALOGUE
from monarchs.variables.definitions import Dim, INPUT, InitContext

# dictionary to map a non-scalar Dim to the size
# argument that gives its length
_DIM_SIZE = {
    Dim.FIRN: "vert_grid",
    Dim.LAKE: "vert_grid_lake",
    Dim.LID: "vert_grid_lid",
    Dim.DIRECTIONS: "n_directions",
}


def build_dtype(vert_grid, vert_grid_lake, vert_grid_lid, n_directions=8):
    """
    Structured-array dtype for the model grid.

    We get this by checking each variable in the catalogue, find its datatype,
    and add it to the list of fields.

    Parameters
    ----------
    vert_grid : int
        Number of vertical grid points in the firn column. This, along with
        vert_grid_lake and vert_grid_lid are needed since their size determines
        the size of the vector variables dependent on the grid sizes.
    vert_grid_lake : int
        Number of vertical grid points in the lake.
    vert_grid_lid : int
        Number of vertical grid points in the lid.
    """
    sizes = {
        "vert_grid": vert_grid,
        "vert_grid_lake": vert_grid_lake,
        "vert_grid_lid": vert_grid_lid,
        "n_directions": n_directions,
    }
    fields = []
    for var in CATALOGUE:
        if var.dim is Dim.SCALAR:
            fields.append((var.name, var.dtype))
        else:
            fields.append((var.name, var.dtype, (sizes[_DIM_SIZE[var.dim]],)))
    return np.dtype(fields, align=True)


def make_grid(
    num_rows, num_cols, vert_grid, vert_grid_lake, vert_grid_lid, inputs=None
):
    """
    Build the model grid from the catalogue.

    ``inputs`` is a dict of {name: value}, which is defined either
    by the firn profile specified in the model setup script, or from
    the input DEM (at least one of these is required for the model to run).
    This means that it contains variables relating to the initial firn column,
    i.e. Sfrac, firn_depth, etc.
    """
    inputs = inputs or {}
    unknown = set(inputs) - {v.name for v in CATALOGUE}
    if unknown:
        raise ValueError(f"unknown grid variable(s) in inputs: {sorted(unknown)}")
    ctx = InitContext(
        num_rows, num_cols, vert_grid, vert_grid_lake, vert_grid_lid, inputs
    )
    dtype = build_dtype(vert_grid, vert_grid_lake, vert_grid_lid)
    grid = np.zeros((num_rows, num_cols), dtype=dtype)

    for var in CATALOGUE:
        if var.name in inputs:
            value = inputs[var.name]
        elif var.default_value is INPUT:
            continue  # not supplied so leave as zero or array of zeros
        elif callable(var.default_value):
            value = var.default_value(ctx)
        else:
            value = var.default_value
        grid[var.name][:] = value  # numpy broadcasts scalars over the shape
    return grid


def variable_metadata():
    """
    netCDF metadata for output-eligible variables.

    These all come from the catalogue, but if they are not present for
    a given variable then they are removed.
    """
    meta = {}
    for v in CATALOGUE:
        if not v.output:
            continue
        attrs = {
            "units": v.units,
            "long_name": v.long_name,
            "description": v.description,
        }
        # drop if empty string
        meta[v.name] = {k: val for k, val in attrs.items() if val != ""}
    return meta


def validate_catalogue():
    """Validate the catalogue in case of duplicates or badly characterised
    input parameters"""
    names = [v.name for v in CATALOGUE]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate variable names in catalogue: {dupes}"
    for v in CATALOGUE:
        assert v.dim is Dim.SCALAR or v.dim in _DIM_SIZE, f"{v.name}: bad dim {v.dim}"
