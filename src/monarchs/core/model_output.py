"""
Functions to handle output of model data.
"""

import numpy as np
from netCDF4 import Dataset
from src.monarchs.core.utils import get_2d_grid


def setup_output(
        fname,
        grid,
        vars_to_save=(
                "firn_temperature",
                "Sfrac",
                "Lfrac",
                "firn_depth",
                "lake_depth",
                "lid_depth",
        ),
        vert_grid_size=False,
):
    """
    Generate a netCDF file that we can use to save key diagnostics from our IceShelf grid over time.

    Parameters
    ----------
    fname : str
        Filename to save our output into.
    grid : List, or numba.typed.List
        grid of IceShelf objects that we want to convert into gridded netCDF data.
    vars_to_save : tuple, optional
        Tuple containing variables from our IceShelf objects that we want to save into our netCDF file.
    vert_grid_size : int, optional
        Desired size of the vertical grid to use.
         Default is False, which sets it to whatever the model vertical grid is specified as.

    Returns
    -------
    None

    Yields
    ------
    netCDF file with filename <fname>
    """

    with Dataset(fname, clobber=True, mode="w") as data:
        # Always save lat/long
        dims = ["lat", "lon"]
        vars_loop = list(vars_to_save)
        for key in dims:
            if key not in vars_to_save:
                vars_loop.append(key)
        data.createDimension("x", size=len(grid))
        data.createDimension("y", size=len(grid[0]))

        data.createDimension("time", None)

        if not vert_grid_size:
            vert_grid_size = grid[0][0].vert_grid

        data.createDimension("vert_grid", size=vert_grid_size)
        data.createDimension("vert_grid_lid", size=grid[0][0].vert_grid_lid)
        data.createDimension("vert_grid_lake", size=grid[0][0].vert_grid_lake)

        for key in vars_loop:
            var = get_2d_grid(grid, key, index="all")
            # Logic to ensure the correct dimensions are set
            if var.dtype == "bool":
                dtype = "b"
            else:
                dtype = var.dtype

            # Ensure we always write out lat/long.
            if key in dims:
                var_write = data.createVariable(key, dtype, ("x", "y"))
                var_write[:] = var

            elif len(np.shape(var)) > 2 and np.shape(var)[-1] > 1:
                if "lake" in key and "lake_depth" not in key and key not in dims:
                    var_write = data.createVariable(
                        key,
                        dtype,
                        (
                            "time",
                            "x",
                            "y",
                            "vert_grid_lake",
                        ),
                    )
                elif "lid" in key and "lid_depth" not in key and key not in dims:
                    var_write = data.createVariable(
                        key,
                        dtype,
                        (
                            "time",
                            "x",
                            "y",
                            "vert_grid_lid",
                        ),
                    )

                else:
                    new_var = np.zeros((len(grid), len(grid[0]), vert_grid_size))
                    # Interpolate to a smaller output size if needed.
                    if vert_grid_size != grid[0][0].vert_grid:
                        for i in range(len(grid)):
                            for j in range(len(grid[i])):
                                new_var[i][j] = np.interp(
                                    np.linspace(
                                        0, grid[i][j].firn_depth, vert_grid_size
                                    ),
                                    np.linspace(
                                        0,
                                        grid[i][j].firn_depth,
                                        grid[i][j].vert_grid,
                                    ),
                                    var[i][j],
                                )

                    var = new_var
                    var_write = data.createVariable(
                        key,
                        dtype,
                        (
                            "time",
                            "x",
                            "y",
                            "vert_grid",
                        ),
                    )
                    var_write[0] = var

            else:
                var_write = data.createVariable(key, dtype, ("time", "x", "y"))
                var_write[0] = var


def update_model_output(
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
    Update the netCDF file generated in setup_output with data from the grid at the current timestep.

    Parameters
    ----------
    fname : str
        Filename to save our output into.
    grid : List, or numba.typed.List
        grid of IceShelf objects that we want to convert into gridded netCDF data.
    iteration : int
        Current iteration (day) of the model (i.e. what time index to save the data into)
    vars_to_save : tuple, optional
        Tuple containing variables from our IceShelf objects that we want to save into our netCDF file.
    hourly : boolean, optional
        Determine whether to save output at the end of each day (default), or each hour. Currently hourly only works
        if running in 1D with Numba disabled. See `model_setup.py` documentation for details.
    t_step : int, optional
        Current timestep (hour), used to ensure we get the correct index for saving data.
    interpolate_output : bool, optional
        Flag that determines whether we need to interpolate the output on the model vertical grid to a smaller grid.
    vert_grid_size : int, optional
        Desired size of the vertical grid to use if interpolating.
         Default is False, which sets it to whatever the model vertical grid is specified as.

    Returns
    -------
    None.

    Yields
    ------
    Amended netCDF with filename fname
    """
    if not hourly:
        index = iteration
    else:
        index = (iteration * 24) + t_step

    with Dataset(fname, clobber=True, mode="a") as data:
        for key in vars_to_save:
            var = get_2d_grid(grid, key, index="all")  # get current situation
            var_write = data.variables[key]
            # Interpolate to a smaller output size if needed.
            if "vert_grid" in var_write.dimensions:
                if vert_grid_size != grid[0][0].vert_grid and vert_grid_size is not False:
                    new_var = np.zeros((len(grid), len(grid[0]), vert_grid_size))
                    for i in range(len(grid)):
                        for j in range(len(grid[i])):
                            new_var[i][j] = np.interp(
                                np.linspace(0, grid[i][j].firn_depth, vert_grid_size),
                                np.linspace(
                                    0,
                                    grid[i][j].firn_depth,
                                    grid[i][j].vert_grid,
                                ),
                                var[i][j],
                            )
                    var = new_var
            # append to time index corresponding to current iteration
            var_write[index] = var
