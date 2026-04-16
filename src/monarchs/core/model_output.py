import numpy as np
from netCDF4 import Dataset
from monarchs.core.utils import get_2d_grid


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
        "daily_melt",
    ),
    vert_grid_size=False,
):
    with Dataset(fname, clobber=True, mode="w") as data:
        dims = ["lat", "lon"]
        vars_loop = list(vars_to_save)
        for key in dims:
            if key not in vars_to_save:
                vars_loop.append(key)
        data.createDimension("x", size=len(grid["firn_depth"]))
        data.createDimension("y", size=len(grid["firn_depth"][0]))
        data.createDimension("time", None)
        if not vert_grid_size:
            vert_grid_size = grid["vert_grid"][0][0]
        data.createDimension("vert_grid", size=vert_grid_size)
        data.createDimension("vert_grid_lid", size=grid["vert_grid_lid"][0][0])
        data.createDimension("vert_grid_lake", size=grid["vert_grid_lake"][0][0])
        data.createDimension("direction", size=8)
        for key in vars_loop:
            var = get_2d_grid(grid, key, index="all")
            if var.dtype == "bool":
                dtype = "b"
            else:
                dtype = var.dtype
            if key in dims:
                var_write = data.createVariable(key, dtype, ("x", "y"))
                var_write[:] = var

            elif len(np.shape(var)) > 2 and np.shape(var)[-1] > 1:
                print(key)
                if "water_direction" in key:
                    var_write = data.createVariable(
                        key, dtype, ("time", "x", "y", "direction")
                    )
                elif "lake" in key and "lake_depth" not in key and key not in dims:
                    var_write = data.createVariable(
                        key, dtype, ("time", "x", "y", "vert_grid_lake")
                    )
                elif "lid" in key and "lid_depth" not in key and key not in dims:
                    var_write = data.createVariable(
                        key, dtype, ("time", "x", "y", "vert_grid_lid")
                    )
                else:
                    new_var = np.zeros(
                        (
                            len(grid["firn_depth"]),
                            len(grid["firn_depth"][0]),
                            vert_grid_size,
                        )
                    )
                    if vert_grid_size != grid["vert_grid"][0][0]:
                        for i in range(len(grid["firn_depth"])):
                            for j in range(len(grid["firn_depth"][i])):
                                new_var[i][j] = np.interp(
                                    np.linspace(
                                        0, grid["firn_depth"][i][j], vert_grid_size
                                    ),
                                    np.linspace(
                                        0,
                                        grid["firn_depth"][i][j],
                                        grid["vert_grid"][i][j],
                                    ),
                                    var[i][j],
                                )
                    var = new_var
                    var_write = data.createVariable(
                        key, dtype, ("time", "x", "y", "vert_grid")
                    )
                    var_write[0] = var
            else:
                var_write = data.createVariable(key, dtype, ("time", "x", "y"))
                var_write[0] = var


def interpolate_model_output(grid, vert_grid_size, var):
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
                np.linspace(
                    0, grid["firn_depth"][i][j], vert_grid_size
                ),
                np.linspace(
                    0, grid["firn_depth"][i][j], grid["vert_grid"][i][j]
                ),
                var[i][j],
            )
    var = new_var
    return var

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
        "daily_melt",
    ),
    hourly=False,
    t_step=0,
    vert_grid_size=False,
):
    # Determine if we are indexing by day number or by hour.
    if not hourly:  # i.e. every day
        index = iteration
    else:
        index = iteration * 24 + t_step


    with Dataset(fname, clobber=True, mode="a") as data:
        for key in vars_to_save:
            var = get_2d_grid(grid, key, index="all")
            var_write = data.variables[key]
            if "vert_grid" in var_write.dimensions:
                if (
                    vert_grid_size != grid["vert_grid"][0][0]
                    and vert_grid_size is not False
                ):
                    var = interpolate_model_output(grid, vert_grid_size, var)

            var_write[index] = var
