"""
Functions to handle dumping of model state, so that runs can be restarted
upon failure.
Separate from model_output, which just handles a user-defined subset of the
outputs (i.e. ones that are useful scientifically, rather than everything
 needed to restart the model).
"""

import os
import numpy as np
from netCDF4 import Dataset  # pylint: disable=no-name-in-module
from monarchs.core.utils import get_2d_grid


def dump_state(fname, grid, met_start_idx, met_end_idx):
    """
    MONARCHS can sometimes crash, or throw an error. This function allows for
    the model state to be saved into a file (name determined by
    <model_setup.reload_file>).
    This allows for restarting of the code from this saved state, which can be
    useful either to keep progress in the event of an error outside of
    MONARCHS' control, or to debug the cause of an error
    (e.g. by switching Numba and parallelisation off).
    Called by <main>, if the relevant flag is set in <model_setup.py>.

    Parameters
    ----------
    fname : str
        Filename we wish to save our data into.
    grid : numpy structured array
        Model grid containing our ice shelf.
    met_start_idx : int
        Index used to determine where in our grid of meteorological data we
        want to start from if we were to restart the model.
    met_end_idx : int
        As met_start_idx, but the ending index. These together create a slice
        across the current iteration,so we can continue from where we left
        off. These will be updated then in <main>.

    Returns
    -------
    None

    Yields
    ------
    netCDF file with filename <fname>.

    """
    folder_path = fname.rsplit("/", 1)[0]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with Dataset(fname, clobber=True, mode="w") as data:
        data.createDimension("vert_grid", size=grid["vert_grid"][0][0])
        data.createDimension("vert_grid_lid", size=grid["vert_grid_lid"][0][0])
        data.createDimension(
            "vert_grid_lake", size=grid["vert_grid_lake"][0][0]
        )
        data.createDimension("direction", size=8)
        data.createDimension("x", size=len(grid))
        data.createDimension("y", size=len(grid[0]))
        # keys = dir(grid[0][0])
        keys = list(grid.dtype.names)
        for key in keys:
            if not key.startswith("_"):
                var = get_2d_grid(grid, key, index="all")
                if var.dtype == "bool":
                    dtype = "b"
                else:
                    dtype = var.dtype
                if len(np.shape(var)) > 2 and np.shape(var)[-1] > 1:
                    if "lake" in key:
                        var_write = data.createVariable(
                            key, dtype, ("x", "y", "vert_grid_lake")
                        )
                    elif "lid" in key:
                        var_write = data.createVariable(
                            key, dtype, ("x", "y", "vert_grid_lid")
                        )
                    elif key == "water_direction":
                        var_write = data.createVariable(
                            key, dtype, ("x", "y", "direction")
                        )
                    else:
                        var_write = data.createVariable(
                            key, dtype, ("x", "y", "vert_grid")
                        )
                else:
                    var_write = data.createVariable(key, dtype, ("x", "y"))
                var_write[:] = var
        met_start_write = data.createVariable("met_start_idx", "i4")
        met_end_write = data.createVariable("met_end_idx", "i4")
        met_start_write[:] = met_start_idx
        met_end_write[:] = met_end_idx


def reload_from_dump(fname, dtype, keys="all"):
    """
    Loads the netCDF file containing the model state into a NumPy structured
    array.

    Parameters
    ----------
    fname : str
        Filename to load data from.
    dtype : np.dtype
        The dtype of the structured array to load the data into.
    keys : list or str, optional
        List of keys to load from the netCDF file.
        If "all", all keys will be loaded.

    Returns
    -------
    grid : np.ndarray
        NumPy structured array with data loaded from the netCDF file.
    met_start_idx : int
        Starting index for meteorological data.
    met_end_idx : int
        Ending index for meteorological data.
    iteration : int
        Iteration (day) to continue running the model from.
    """
    with Dataset(fname, mode="r") as data:
        # Determine which keys to load
        scalars = ["met_start_idx", "met_end_idx"]
        if keys == "all":
            desired_keys = [
                key for key in data.variables.keys() if key not in scalars
            ]
        else:
            desired_keys = keys

        # Create the structured array
        grid_shape = (len(data.dimensions["x"]), len(data.dimensions["y"]))
        grid = np.zeros(grid_shape, dtype=dtype)

        # Load data into the structured array
        for key in desired_keys:
            print(f"Loading key {key} into structured array")
            if len(data.variables[key].shape) > 2:
                grid[key] = data.variables[key][:, :, :].data
            else:
                grid[key] = data.variables[key][:, :].data

        # Load scalar values
        met_start_idx = int(data.variables["met_start_idx"][:].data)
        met_end_idx = int(data.variables["met_end_idx"][:].data)
        iteration = int(np.max(data.variables["day"][:].data))

    return grid, met_start_idx, met_end_idx, iteration
