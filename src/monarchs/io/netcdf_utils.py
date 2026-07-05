"""
Utilities for dealing with netCDF files.

Both the time-series model output (``monarchs.io.output``) and the full-state
checkpoint (``monarchs.io.checkpoint``) map the NumPy structured model grid to
and from a netCDF file. They differ in *what* they write (a user-defined subset
over an unlimited time dimension vs. the entire grid state as a single snapshot),
but they share the same underlying operations: creating the grid dimensions,
deciding which dimensions a given variable uses, converting dtypes for netCDF
compatibility, and extracting a field from the grid as a dense array.

This module uses ``netCDF4`` so cannot be JIT-compiled with Numba.
"""

import numpy as np
from monarchs.core.utils import get_2d_grid


def create_grid_dimensions(data, grid, vert_grid_size=None, include_time=False):
    """
    Create the netCDF dimensions used to describe the model grid. We have the
    following dimensions:
    - x: horizontal grid dimension (number of columns)
    - y: horizontal grid dimension (number of rows)
    - time: unlimited dimension for time steps (only if ``include_time``)
    - vert_grid: vertical grid dimension for firn column
    - vert_grid_lid: vertical grid dimension for frozen lid
    - vert_grid_lake: vertical grid dimension for lake
    - direction: dimension for lateral flow (8 cardinal directions), used for
      outputting direction in which water moves in each timestep

    Parameters
    ----------
    data : netCDF4.Dataset
        The NetCDF dataset to which dimensions will be added.
    grid : np.ndarray
        Structured array representing the model grid.
    vert_grid_size : int or None, optional
        Size of the vertical (firn) grid. If None, the native grid size
        (``grid["vert_grid"][0][0]``) is used. The time-series output uses this
        to optionally interpolate onto a coarser vertical grid; the checkpoint
        always writes at native resolution.
    include_time : bool, optional
        Whether to create an unlimited ``time`` dimension. True for the
        time-series output, False for a single-snapshot checkpoint.

    Returns
    -------
    None (amends data object inplace)
    """
    if vert_grid_size is None:
        vert_grid_size = grid["vert_grid"][0][0]
    data.createDimension("x", size=len(grid))
    data.createDimension("y", size=len(grid[0]))
    if include_time:
        data.createDimension("time", None)
    data.createDimension("vert_grid", size=vert_grid_size)
    data.createDimension("vert_grid_lid", size=grid["vert_grid_lid"][0][0])
    data.createDimension("vert_grid_lake", size=grid["vert_grid_lake"][0][0])
    data.createDimension("direction", size=8)


def variable_dimensions(key, is_vector, include_time=False):
    """
    Determine the dimensions for a variable based on its key.
    Lake variables use the lake vertical grid, lid variables use the lid vertical
    grid, etc. Water direction uses a separate direction dimension, to represent
    one of the 8 cardinal directions in which water can flow. Scalar (per-cell)
    variables use only the horizontal ``(x, y)`` dimensions.

    Parameters
    ----------
    key : str
        Variable name.
    is_vector : bool
        Whether the variable has a vertical/direction axis (i.e. is 3D on the
        grid) as opposed to being a single value per cell.
    include_time : bool, optional
        Whether to prepend the ``time`` dimension (True for the time-series
        output, False for a single-snapshot checkpoint).

    Returns
    -------
    tuple
        Dimensions for the variable.
    """
    prefix = ("time",) if include_time else ()

    if not is_vector:
        return prefix + ("x", "y")

    if "water_direction" in key:
        return prefix + ("x", "y", "direction")
    if "lake" in key and "lake_depth" not in key:
        return prefix + ("x", "y", "vert_grid_lake")
    if "lid" in key and "lid_depth" not in key:
        return prefix + ("x", "y", "vert_grid_lid")

    return prefix + ("x", "y", "vert_grid")


def netcdf_dtype(var):
    """Convert boolean dtypes to 'b' for NetCDF compatibility."""
    return "b" if var.dtype == "bool" else var.dtype


def extract_field(grid, key):
    """Extract a field from the model grid as a dense array over all cells."""
    return get_2d_grid(grid, key, index="all")


def is_vector_field(var):
    """Whether an extracted field has a vertical/direction axis (3D on grid)."""
    shape = np.shape(var)
    return len(shape) > 2 and shape[-1] > 1
