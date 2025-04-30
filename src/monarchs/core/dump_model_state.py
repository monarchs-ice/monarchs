"""
Functions to handle dumping of model state, so that runs can be restarted upon failure.
Separate from model_output, which just handles a user-defined subset of the outputs (i.e. ones that are useful
scientifically, rather than everything needed to restart the model).
"""

import os
import numpy as np
from netCDF4 import Dataset
from monarchs.core.utils import get_2d_grid


def dump_state(fname, grid, met_start_idx, met_end_idx):
    """
    MONARCHS can sometimes crash, or throw an error. This function allows for the model state to be
    saved into a file (name determined by <model_setup.reload_file>). This allows for restarting
    of the code from this saved state, which can be useful either to keep progress in the event of an error outside
    of MONARCHS' control, or to debug the cause of an error (e.g. by switching Numba and parallelisation off).
    Called by <main>, if the relevant flag is set in <model_setup.py>.

    Parameters
    ----------
    fname : str
        Filename we wish to save our data into.
    grid : List, or numba.typed.List
        grid of IceShelf objects that we want to save progress information for.
    met_start_idx : int
        Index used to determine where in our grid of meteorological data we want to start from if we were
        to restart the model.
    met_end_idx : int
        As met_start_idx, but the ending index. These together create a slice across the current iteration,
        so we can continue from where we left off. These will be updated then in <main>.

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
        data.createDimension("vert_grid_lake", size=grid["vert_grid_lake"][0][0])
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


def reload_from_dump(fname, grid, keys="all"):
    """
    Loads in the netCDF file containing the model state, and loads the data into
    the format required for MONARCHS to function. This allows us to restart the model in the event of a failure.

    Called by <main> if the relevant flag is set in <model_setup.py>.

    Parameters
    ----------
    fname : str
        Filename we want to load our data from.
    grid : List, or numba.typed.List
        grid of IceShelf objects to update with the reloaded values
    keys: list, optional
        List of keys to load in from the netCDF file. If not specified, all keys will be loaded in.
    Returns
    -------
    grid : List, or numba.typed.List
        Updated grid of IceShelf objects, with values loaded in from our input netCDF.
    met_start_idx : int
        Starting index at which to slice our meteorological dataset at the current iteration.
    met_end_idx : int
        As above, but ending index.
    iteration : int
        Iteration (day) we wish to continue running the model from.
    """
    with Dataset(fname, mode="r") as data:
        scalars = ["met_start_idx", "met_end_idx"]
        bool_keys = [
            "melt",
            "exposed_water",
            "lake",
            "lid",
            "v_lid",
            "ice_lens",
            "has_had_lid",
            "reset_combine",
            "valid_cell",
        ]
        int_keys = [
            "ice_lens_depth",
            "x",
            "y",
            "rho_ice",
            "rho_water",
            "melt_hours",
            "lid_sfc_melt",
            "lid_melt_count",
            "t_step",
            "exposed_water_refreeze_counter",
            "L_ice",
            "iteration",
        ]
        if keys == "all":
            desired_keys = [key for key in data.variables.keys() if key not in scalars]
        else:
            desired_keys = keys
        print(f"monarchs.core.dump_model_state.reload_from_dump: ")
        for key in desired_keys:
            print(f"Loading in key {key} from progress netCDF file")
            if key == "log":
                continue
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if len(np.shape(data.variables[key][:])) > 2:
                        setattr(grid[i][j], key, data.variables[key][i, j, :].data)
                    else:
                        try:
                            setattr(grid[i][j], key, data.variables[key][i, j].data)
                        except AttributeError:
                            setattr(grid[i][j], key, data.variables[key][i, j])
                    if key in bool_keys:
                        var = getattr(grid[i][j], key)
                        setattr(grid[i][j], key, bool(var))
                    if key in int_keys:
                        var = getattr(grid[i][j], key)
                        setattr(grid[i][j], key, int(var))
        met_start_idx = data.variables["met_start_idx"][:].data
        met_end_idx = data.variables["met_end_idx"][:].data
        iteration = np.max(data.variables["day"][:].data)
    grid = np.transpose(grid)
    return grid, met_start_idx, met_end_idx, iteration
