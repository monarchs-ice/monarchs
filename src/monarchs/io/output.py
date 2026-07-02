"""
Time-series model output. This handles a user-defined subset of the model
variables (i.e. ones that are useful scientifically, rather than everything
needed to restart the model), written per timestep into an unlimited ``time``
dimension so that the evolution of the model over time can be analysed as
results.

Separate from ``monarchs.io.checkpoint``, which captures the full model state
for e.g. restarting a run.
"""

import numpy as np
from netCDF4 import Dataset  # pylint: disable=no-name-in-module
from monarchs.io import grid_serialisation as gs

# pylint: disable=duplicate-code
DEFAULT_VARS = (
    "firn_temperature",
    "Sfrac",
    "Lfrac",
    "firn_depth",
    "lake_depth",
    "lid_depth",
)
# pylint: enable=duplicate-code


def initialise_output(fname, grid, vars_to_save=DEFAULT_VARS, vert_grid_size=False):
    """
    Set up the NetCDF file for model output.

    Parameters
    ----------
    fname : str
        Filename for the output NetCDF file.
    grid : np.ndarray
        Structured array representing the model grid.
    vars_to_save : tuple, optional
        Variables to save in the output file. Default includes firn temperature,
        solid fraction, liquid fraction, firn depth, lake depth, and lid depth.
        See DEFAULT_VARS in the code for the full list.
    vert_grid_size : int or bool, optional
        Size of the vertical grid for interpolation. If False, uses the existing
        vertical grid size from the model grid. Default is False.
    Returns
    -------
    None.
    """
    with Dataset(fname, clobber=True, mode="w") as data:
        vert_grid_size = vert_grid_size or grid["vert_grid"][0][0]
        gs.create_grid_dimensions(
            data, grid, vert_grid_size=vert_grid_size, include_time=True
        )

        vars_loop = add_lat_long(vars_to_save)
        for key in vars_loop:
            var = gs.extract_field(grid, key)
            dtype = gs.netcdf_dtype(var)
            create_variable(data, key, var, dtype, grid, vert_grid_size=vert_grid_size)


def add_lat_long(vars_to_save):
    """
    Ensure that latitude and longitude are included in the list of variables to save.
    """
    dims = ["lat", "lon"]
    vars_loop = list(vars_to_save)
    for key in dims:
        if key not in vars_to_save:
            vars_loop.append(key)
    return vars_loop


# We need to pass everything in here, and using a config object
# would be overkill, so disable the pylint warnings.
# pylint: disable=too-many-arguments, too-many-positional-arguments
def create_variable(data, key, var, dtype, grid, vert_grid_size=20):
    """
    Create a variable in the NetCDF file with appropriate dimensions and data.
    Lat and long are variables with dimensions (x, y) only, as these don't change
    over time.
    Other variables may use the firn vertical grid, lid vertical grid, or
    lake vertical grid.
    Any variables that use the firn vertical grid (e.g. temperature)
    may need to be interpolated to a new vertical grid size.

    Parameters
    ----------
    data : netCDF4.Dataset
        The NetCDF dataset to which the variable will be added.
    key : str
        Variable name.
    var : np.ndarray
        2D or 3D variable data to be saved.
    dtype : str
        Data type for the variable.
    grid : np.ndarray
        Structured array representing the model grid.
    vert_grid_size : int
        Desired size of the vertical grid for interpolation.
    Returns
    -------
    None (amends data object inplace)
    """
    if key in ("lat", "lon"):
        var_write = data.createVariable(key, dtype, ("x", "y"))
        var_write[:] = var
        return

    if gs.is_vector_field(var):
        dims = gs.variable_dimensions(key, is_vector=True, include_time=True)
        if vert_grid_size != grid["vert_grid"][0][0] and "vert_grid" in dims:
            var = interpolate_model_output(grid, vert_grid_size, var)
        var_write = data.createVariable(key, dtype, dims)
        var_write[0] = var
    else:
        var_write = data.createVariable(key, dtype, ("time", "x", "y"))
        var_write[0] = var


# pylint: enable=too-many-arguments, too-many-positional-arguments


def interpolate_model_output(grid, vert_grid_size, var):
    """
    Interpolate a 3D variable to a new vertical grid size.

    Parameters
    ----------
    grid : np.ndarray
        Structured array representing the model grid.
    vert_grid_size : int
        Desired size of the vertical grid for interpolation.
    var : np.ndarray
        3D variable to interpolate, with shape (x, y, z).
    """
    new_var = np.zeros(
        (
            len(grid["firn_depth"]),
            len(grid["firn_depth"][0]),
            vert_grid_size,
        )
    )
    for i in range(len(grid["firn_depth"])):
        for j in range(len(grid["firn_depth"][i])):
            new_var[i][j] = np.interp(
                np.linspace(0, grid["firn_depth"][i][j], vert_grid_size),
                np.linspace(0, grid["firn_depth"][i][j], grid["vert_grid"][i][j]),
                var[i][j],
            )
    var = new_var
    return var


# We have many optional arguments here, and using a config object
# would be overkill, so disable the pylint warnings.
# pylint: disable=too-many-arguments, too-many-positional-arguments
def append_output(
    fname,
    grid,
    iteration,
    vars_to_save=(
        "firn_temperature",
        "Sfrac",
        "Lfrac",
        "firn_depth",
        "lake_depth",
        "lid_depth",
    ),
    hourly=False,
    t_step=0,
    vert_grid_size=False,
):
    """
    Update the NetCDF file with the current state of the model grid.

    Parameters
    ----------
    fname : str
        Filename for the output NetCDF file.
    grid : np.ndarray
        Structured array representing the model grid.
    iteration : int
        Current iteration number (i.e. day) of the model.
    vars_to_save : tuple, optional
        Variables to save in the output file.
    hourly : bool, optional
        Whether the model is running with hourly output (rather than daily).
        Default is False.
    t_step : int, optional
        Current timestep within the day (0-23 if hourly is True). Default is 0.
    vert_grid_size : int or bool, optional
        Size of the vertical grid for interpolation. If False, uses the existing
        vertical grid size from the model grid. Default is False.

    Returns
    -------
    None.
    """
    # Determine if we are indexing by day number or by hour.
    if not hourly:  # i.e. every day
        index = iteration
    else:
        index = iteration * 24 + t_step

    with Dataset(fname, clobber=True, mode="a") as data:
        for key in vars_to_save:
            var = gs.extract_field(grid, key)
            var_write = data.variables[key]
            if "vert_grid" in var_write.dimensions:
                if (
                    vert_grid_size != grid["vert_grid"][0][0]
                    and vert_grid_size is not False
                ):
                    var = interpolate_model_output(grid, vert_grid_size, var)

            var_write[index] = var


# pylint: enable=too-many-arguments, too-many-positional-arguments
