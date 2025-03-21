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
            f"monarchs.core.initial_conditions.setup_era5: "
            'met_timestep should be an integer, "hourly", "three_hourly" or "daily". See documentation for'
            " model_setup.met_timestep for details."
        )

    # convert the variable names from the netCDF to those used in this code
    ERA5_vars = ERA5_to_variables(model_setup.met_input_filepath)

    # Select some bounds - lat upper/lower, long upper/lower (in degrees, -90/90, -180/180)
    bounds = ["latmax", "latmin", "longmax", "longmin"]

    if all(hasattr(model_setup, attr) for attr in bounds) and (
        all(~np.isnan(getattr(model_setup, attr)) for attr in bounds)
    ):
        ERA5_vars = grid_subset(
            ERA5_vars,
            model_setup.latmax,
            model_setup.latmin,
            model_setup.longmax,
            model_setup.longmin,
        )

    # Interpolate to our grid size
    ERA5_grid = interpolate_grid(
        ERA5_vars, model_setup.row_amount, model_setup.col_amount
    )

    # Restrict to bounds of our input DEM if desired.
    if hasattr(model_setup, "lat_bounds") and model_setup.lat_bounds.lower() == "dem":
        ERA5_grid = get_met_bounds_from_DEM(
            model_setup,
            ERA5_grid,
            lat_array,
            lon_array,
            diagnostic_plots=model_setup.met_dem_diagnostic_plots,
        )

    # Save it so we don't need to keep the whole thing in memory
    ERA5_grid_path = model_setup.met_output_filepath

    # Arbitrary forcing of met data in case we want to do some testing.
    if hasattr(model_setup, "radiation_forcing_factor"):
        if model_setup.radiation_forcing_factor not in [False, 1]:
            ERA5_grid["SW_surf"] *= model_setup.radiation_forcing_factor
            ERA5_grid["LW_surf"] *= model_setup.radiation_forcing_factor
            print(f"monarchs.core.initial_conditions.setup_era5: ")
            print(
                f"Scaling SW_surf and LW_surf by a factor of {model_setup.radiation_forcing_factor} for testing"
            )
    # If our index is not 1, then we need to repeat each point <index> times, e.g. if our data is daily, then each
    # point actually corresponds to 24 model timesteps so we need to account for this

    if index > 1:
        selected_keys = [key for key in ERA5_grid.keys() if key not in ["lat", "long"]]
        for var in selected_keys:
            ERA5_grid[var] = np.repeat(ERA5_grid[var], index, axis=0)  # along time axis

    with Dataset(ERA5_grid_path, "w") as f:
        f.createGroup("variables")
        f.createDimension("time", len(ERA5_grid["SW_surf"]))
        f.createDimension("column", model_setup.col_amount)
        f.createDimension("row", model_setup.row_amount)

        for key, value in ERA5_grid.items():
            if key in ["long", "lat", "time"]:
                if key == 'long':
                    var = f.createVariable('cell_longitude',
                    np.dtype("float64").char, ("column", "row"))
                    var.long_name = 'Longitude of grid cell'
                    var[:] = value
                if key == 'lat':
                    var = f.createVariable('cell_latitude',
                     np.dtype("float64").char, ("column", "row"))
                    var.long_name = 'Latitude of grid cell'
                    var[:] = value
                else:
                    continue
            var = f.createVariable(
                key, np.dtype("float64").char, ("time", "column", "row")
            )
            var.long_name = key
            var[:] = value
        print(
            f"monarchs.core.initial_conditions.setup_era5: "
            f"Saved meteorological data used for the model run into {ERA5_grid_path}"
        )
    return ERA5_grid_path


def prescribed_met_data(model_setup):
    """
    Create a netCDF file using the prescribed met data defined in ``model_setup``.

    Returns
    -------
    None
    """

    met_data = model_setup.met_data


    from monarchs.met_data.metdata_class import initialise_met_data_grid
    with Dataset(model_setup.met_output_filepath, "w") as f:
        f.createGroup("variables")
        f.createDimension("time", model_setup.num_days * model_setup.t_steps_per_day)
        f.createDimension("column", model_setup.col_amount)
        f.createDimension("row", model_setup.row_amount)

        # As with real data, we need to create a netCDF input for MONARCHS to read. However,
        # we may need to extend the data to the number of days we actually want, if only a single
        # value is specified.
        # Most values should be specified, but we can assume some defaults for a couple of variables:
        if 'lat' not in met_data:
            # try and pre-empt people using "latitude" instead of "lat"
            if 'latitude' in met_data:
                met_data['lat'] = met_data['latitude']
            else:
                met_data['lat'] = np.zeros((model_setup.row_amount, model_setup.col_amount))
        if 'long' not in met_data:
            # as above for long
            if 'longitude' in met_data:
                met_data['long'] = met_data['longitude']
            else:
                met_data['long'] = np.zeros((model_setup.row_amount, model_setup.col_amount))
        if 'snow_dens' not in met_data:
            met_data['snow_dens'] = np.broadcast_to(300 * np.ones(model_setup.num_days * model_setup.t_steps_per_day),
                                            (model_setup.row_amount, model_setup.col_amount,
                                             len(met_data['snow_dens'])))

        # create the netCDF file
        for key, value in met_data.items():
            if key in ["long", "lat", "time"]:
                if key == 'long':
                    var = f.createVariable('cell_longitude',
                                           np.dtype("float64").char, ("column", "row"))
                    var.long_name = 'Longitude of grid cell'
                    var[:] = value
                if key == 'lat':
                    var = f.createVariable('cell_latitude',
                                           np.dtype("float64").char, ("column", "row"))
                    var.long_name = 'Latitude of grid cell'
                    var[:] = value
                else:
                    continue
            var = f.createVariable(
                key, np.dtype("float64").char, ("time", "column", "row")
            )
            var.long_name = key
            var[:] = value
        print(
            f"monarchs.core.initial_conditions.setup_prescribed_data: "
            f"Saved meteorological data used for the model run into {model_setup.met_output_filepath}"
        )