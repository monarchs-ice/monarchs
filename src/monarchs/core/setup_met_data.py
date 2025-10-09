"""
TODO - docstrings, module-level docstring
"""

from netCDF4 import Dataset
import numpy as np
from monarchs.met_data.import_ERA5 import (
    ERA5_to_variables,
    interpolate_grid,
    grid_subset,
    get_met_bounds_from_DEM,
)


def setup_era5(model_setup, lat_array=False, lon_array=False):
    """
    Meteorological data input

    TODO - docstring
    """

    if model_setup.met_timestep == "hourly":
        index = 1
    elif model_setup.met_timestep == "three_hourly":
        index = 3
    elif model_setup.met_timestep == "daily":
        index = 24
    elif isinstance(model_setup.met_timestep, int):
        index = model_setup.met_timestep
    else:
        raise ValueError(
            f"monarchs.core.initial_conditions.setup_era5: met_timestep should"
            f" be an integer, 'hourly', 'three_hourly' or 'daily'. See"
            f" documentation for model_setup.met_timestep for details."
        )

    # Chunk the input data into years.
    # TODO - Make it so that we can pass yearly files in, controlled by a
    # TODO - model_setup parameter whether we have a - single file or multiple.
    chunk_size = 365
    model_years = max(1, model_setup.num_days // chunk_size + 1)
    if model_setup.num_days % chunk_size == 0:
        # If the number of days is a multiple of chunk_size (default 365),
        # we don't need an extra year.
        model_years -= 1
    for year in range(model_years):
        start_index = year * chunk_size * 24 / index
        timesteps_per_day = 24 / index

        total_days = model_setup.num_days
        ERA5_vars = ERA5_to_variables(
            model_setup.met_input_filepath,
            timesteps_per_day,
            total_days,
            start_index=start_index,
            chunk_size=chunk_size,
        )
        bounds = ["latmax", "latmin", "longmax", "longmin"]
        if all(hasattr(model_setup, attr) for attr in bounds) and all(
            ~np.isnan(getattr(model_setup, attr)) for attr in bounds
        ):
            ERA5_vars = grid_subset(
                ERA5_vars,
                model_setup.latmax,
                model_setup.latmin,
                model_setup.longmax,
                model_setup.longmin,
            )
        ERA5_grid = interpolate_grid(
            ERA5_vars, model_setup.row_amount, model_setup.col_amount
        )
        if (
            hasattr(model_setup, "lat_bounds")
            and model_setup.lat_bounds.lower() == "dem"
        ):
            ERA5_grid = get_met_bounds_from_DEM(
                model_setup,
                ERA5_grid,
                lat_array,
                lon_array,
                diagnostic_plots=model_setup.met_dem_diagnostic_plots,
            )
        ERA5_grid_path = model_setup.met_output_filepath

        if hasattr(model_setup, "radiation_forcing_factor"):
            if model_setup.radiation_forcing_factor not in [False, 1]:
                ERA5_grid["SW_surf"] *= model_setup.radiation_forcing_factor
                ERA5_grid["LW_surf"] *= model_setup.radiation_forcing_factor
                print(f"monarchs.core.initial_conditions.setup_era5: ")
                print(
                    "Scaling SW_surf and LW_surf by a factor of"
                    f" {model_setup.radiation_forcing_factor} for testing"
                )

        if index > 1:
            selected_keys = [
                key for key in ERA5_grid.keys() if key not in ["lat", "long"]
            ]
            for var in selected_keys:
                ERA5_grid[var] = np.repeat(ERA5_grid[var], index, axis=0)

        print(
            "monarchs.core.initial_conditions.setup_era5: Writing data for"
            f" year {year + 1} into netCDF..."
        )
        write_to_netcdf(
            ERA5_grid_path, ERA5_grid, model_setup, start_index=start_index
        )
    print(
        "monarchs.core.initial_conditions.setup_era5: Saved meteorological"
        f" data used for the model run into {ERA5_grid_path}"
    )
    return ERA5_grid_path


def write_to_netcdf(ERA5_grid_path, ERA5_grid, model_setup, start_index=0):

    start_index = int(start_index)
    end_index = int(start_index + len(ERA5_grid["SW_surf"]))

    if start_index == 0:
        mode = "w"
    else:
        mode = "a"

    with Dataset(ERA5_grid_path, mode) as f:
        if start_index == 0:
            f.createGroup("variables")
            f.createDimension("time", None)
            f.createDimension("column", model_setup.col_amount)
            f.createDimension("row", model_setup.row_amount)
        for key, value in ERA5_grid.items():
            if start_index == 0:
                if key in ["long", "lat", "time"]:
                    if key == "long":
                        var = f.createVariable(
                            "cell_longitude",
                            np.dtype("float64").char,
                            ("column", "row"),
                        )
                        var.long_name = "Longitude of grid cell"
                        var[start_index:end_index] = value
                    elif key == "lat":
                        var = f.createVariable(
                            "cell_latitude",
                            np.dtype("float64").char,
                            ("column", "row"),
                        )
                        var.long_name = "Latitude of grid cell"
                        var[start_index:end_index] = value
                    else:
                        continue
                var = f.createVariable(
                    key, np.dtype("float64").char, ("time", "column", "row")
                )
                var.long_name = key
                var[start_index:end_index] = value
            else:
                if key in ["long", "lat", "time"]:
                    continue
                var = f.variables[key]
                var[start_index:end_index] = value


def prescribed_met_data(model_setup):
    """
    Create a netCDF file using the prescribed met data defined in
    ``model_setup``.

    Returns
    -------
    None
    """
    met_data = model_setup.met_data
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    except ImportError:
        comm = None

    with Dataset(model_setup.met_output_filepath, "w", comm=comm) as f:
        f.createGroup("variables")
        f.createDimension(
            "time", model_setup.num_days * model_setup.t_steps_per_day
        )
        f.createDimension("column", model_setup.col_amount)
        f.createDimension("row", model_setup.row_amount)
        if "lat" not in met_data:
            if "latitude" in met_data:
                met_data["lat"] = met_data["latitude"]
            else:
                met_data["lat"] = np.zeros(
                    (model_setup.row_amount, model_setup.col_amount)
                )
        if "long" not in met_data:
            if "longitude" in met_data:
                met_data["long"] = met_data["longitude"]
            else:
                met_data["long"] = np.zeros(
                    (model_setup.row_amount, model_setup.col_amount)
                )
        if "snow_dens" not in met_data:
            met_data["snow_dens"] = np.broadcast_to(
                300
                * np.ones(model_setup.num_days * model_setup.t_steps_per_day),
                (
                    model_setup.row_amount,
                    model_setup.col_amount,
                    len(met_data["snow_dens"]),
                ),
            )
        for key, value in met_data.items():
            if key in ["long", "lat", "time"]:
                if key == "long":
                    var = f.createVariable(
                        "cell_longitude",
                        np.dtype("float64").char,
                        ("column", "row"),
                    )
                    var.long_name = "Longitude of grid cell"
                    var[:] = value
                if key == "lat":
                    var = f.createVariable(
                        "cell_latitude",
                        np.dtype("float64").char,
                        ("column", "row"),
                    )
                    var.long_name = "Latitude of grid cell"
                    var[:] = value
                else:
                    continue
            var = f.createVariable(
                key, np.dtype("float64").char, ("time", "column", "row")
            )
            var.long_name = key
            var[:] = value
        print(
            "monarchs.core.initial_conditions.setup_prescribed_data: Saved"
            " meteorological data used for the model run into"
            f" {model_setup.met_output_filepath}"
        )
