""" """

# TODO - docstrings, module-level docstring
from netCDF4 import Dataset  # pylint: disable=no-name-in-module
import numpy as np
from monarchs.met_data.import_ERA5 import (
    ERA5_to_variables,
    interpolate_grid,
    grid_subset,
    get_met_bounds_from_DEM,
)

# TODO - make CHUNKSIZE model_setup variable rather than constant?
CHUNKSIZE = 365  # days


def met_data_from_era5(model_setup, lat_array=False, lon_array=False,
                       interpolated=False):
    """
    Convert ERA5 input data into a netCDF in MONARCHS format.

    """
    #     TODO - docstring, refactor to break up with a few helper functions
    func_name = "monarchs.core.initial_conditions.met_data_from_era5"
    # Determine the timestep index - either a string (in which case convert to int),
    # or int directly
    timesteps = {"hourly": 1, "three_hourly": 3, "daily": 24}
    if model_setup.met_timestep in timesteps:
        index = timesteps[model_setup.met_timestep]
    elif isinstance(model_setup.met_timestep, int):
        index = model_setup.met_timestep
    else:
        raise ValueError(
            f"{func_name}: met_timestep should"
            " be an integer, 'hourly', 'three_hourly' or 'daily'. See"
            f" documentation for {func_name} for details."
        )
    # Chunk the input data into years.
    # TODO - Make it so that we can pass yearly files in, controlled by a
    # TODO - model_setup parameter whether we have a - single file or multiple.
    num_model_years = get_model_years(
        model_setup.num_days, chunk_size=CHUNKSIZE
    )
    for year in range(num_model_years):
        start_index = year * CHUNKSIZE * 24 / index

        era5_grid = process_year(
            model_setup, year, index, lat_array, lon_array
        )

        if index > 1:
            # Repeat each timestep index times to get to hourly data.
            repeat_to_make_hourly(era5_grid, index)

        print(
            f"{func_name}: Writing data for" f" year {year + 1} into netCDF..."
        )
        if interpolated:
            write_to_netcdf_full(
                model_setup.met_output_filepath,
                era5_grid,
                model_setup,
                start_index=start_index,
            )
        else:
            write_to_netcdf(
                model_setup.met_output_filepath,
                era5_grid,
                model_setup,
                start_index=start_index,
            )

    # end of loop over years
    print(
        f"{func_name}: Saved meteorological"
        f" data used for the model run into {model_setup.met_output_filepath}"
    )
    return model_setup.met_output_filepath


def process_year(model_setup, year, index, lat_array, lon_array):
    """
    Process a single year of ERA5 data, returning the coarse-resolution ERA5
    dict with index maps attached, ready for write_to_netcdf.

    For regular lat/lon grids, 1-D index maps are built from linearly-spaced
    fine coordinate vectors (matching what interpolate_grid previously produced).

    For polar stereographic grids (lat_bounds == "dem"), get_met_bounds_from_DEM
    already contains the nearest-neighbour logic; we simply extract the index
    maps it builds rather than re-deriving them.

    Parameters
    ----------
    model_setup : module
        Model configuration object.
    year : int
        Year index (0-based) within the full model run.
    index : int
        Number of hours per met timestep (1=hourly, 3=three-hourly, etc.)
    lat_array : ndarray or False
        2-D array of DEM latitudes, shape (num_rows, num_cols).
        Only used when model_setup.lat_bounds == "dem".
    lon_array : ndarray or False
        2-D array of DEM longitudes, shape (num_rows, num_cols).
        Only used when model_setup.lat_bounds == "dem".

    Returns
    -------
    era5_vars : dict
        Coarse ERA5 data dict with added keys:
            "lat_idx"  : int32 ndarray, shape (num_cols,) or (num_rows, num_cols)
            "lon_idx"  : int32 ndarray, shape (num_rows,) or (num_rows, num_cols)
            "fine_lat" : ndarray, shape (num_cols,)       or (num_rows, num_cols)
            "fine_lon" : ndarray, shape (num_rows,)       or (num_rows, num_cols)
    """
    start_index = year * CHUNKSIZE * 24 / index
    timesteps_per_day = 24 / index

    era5_vars = ERA5_to_variables(
        model_setup.met_input_filepath,
        timesteps_per_day,
        model_setup.num_days,
        start_index=start_index,
        chunk_size=CHUNKSIZE,
    )

    if has_user_defined_bounds(model_setup):
        era5_vars = grid_subset(
            era5_vars,
            model_setup.latmax,
            model_setup.latmin,
            model_setup.longmax,
            model_setup.longmin,
        )


    # get_met_bounds_from_DEM already contains the find_nearest loop
    # that builds the 2-D index maps.  We now get those maps back
    # directly rather than discarding them after the expansion.
    era5_vars, lat_idx, lon_idx = get_met_bounds_from_DEM(
        model_setup,
        era5_vars,
        lat_array,
        lon_array,
        diagnostic_plots=model_setup.met_dem_diagnostic_plots,
    )
    fine_lat = lat_array    # 2-D (num_rows, num_cols), geographic coords
    fine_lon = lon_array    # 2-D (num_rows, num_cols), geographic coords



    # attach maps so write_to_netcdf can persist them
    era5_vars["lat_idx"]  = lat_idx
    era5_vars["lon_idx"]  = lon_idx
    era5_vars["fine_lat"] = fine_lat
    era5_vars["fine_lon"] = fine_lon

    # optionally scale radiation for testing
    scale_by_factor(model_setup, era5_vars)

    return era5_vars

def write_to_netcdf(era5_grid_path, era5_grid, model_setup, start_index=0):
    """
    Write ERA5 data to the met netCDF at **coarse** resolution.

    On the first call (``start_index == 0``) the index maps and fine-grid
    lat / lon are written as static variables so that
    ``update_met_conditions`` can reconstruct the full model-grid arrays
    without storing one value per fine cell.

    Supports both 1-D index maps (separable regular grid) and 2-D index maps
    (e.g. from get_met_bounds_from_DEM). When 2-D, also writes
    cell_latitude and cell_longitude from fine_lat/fine_lon for the reader.
    """
    start_index = int(start_index)
    end_index = int(start_index + len(era5_grid["SW_surf"]))

    mode = "w" if start_index == 0 else "a"

    # Keys that require special handling or are stored separately.
    SKIP_KEYS = {"lat", "long", "time", "lat_idx", "lon_idx",
                 "fine_lat", "fine_lon", "coarse_lat", "coarse_lon"}

    with Dataset(era5_grid_path, mode) as f:
        if start_index == 0:
            # Use coarse_lat/coarse_lon when present (after get_met_bounds_from_DEM
            # overwrites lat/long with 2-D fine arrays)
            if "coarse_lat" in era5_grid and "coarse_lon" in era5_grid:
                coarse_lat_1d = era5_grid["coarse_lat"]
                coarse_lon_1d = era5_grid["coarse_lon"]
            else:
                coarse_lat_1d = np.asarray(era5_grid["lat"]).ravel()
                coarse_lon_1d = np.asarray(era5_grid["long"]).ravel()
                if coarse_lat_1d.ndim > 1 or coarse_lon_1d.ndim > 1:
                    raise ValueError(
                        "write_to_netcdf: expected 1-D coarse lat/lon. "
                        "If using get_met_bounds_from_DEM, it should set coarse_lat/coarse_lon."
                    )
            n_coarse_lat = len(coarse_lat_1d)
            n_coarse_lon = len(coarse_lon_1d)

            lat_idx = era5_grid["lat_idx"]
            lon_idx = era5_grid["lon_idx"]
            fine_lat = era5_grid["fine_lat"]
            fine_lon = era5_grid["fine_lon"]
            is_2d = lat_idx.ndim == 2

            f.createGroup("variables")
            f.createDimension("time", None)                        # unlimited
            f.createDimension("coarse_lat", n_coarse_lat)
            f.createDimension("coarse_lon", n_coarse_lon)
            f.createDimension("fine_row", model_setup.row_amount)
            f.createDimension("fine_col", model_setup.col_amount)

            # ── Static coordinate variables ───────────────────────────────
            v = f.createVariable("coarse_lat", "f8", ("coarse_lat",))
            v.long_name = "ERA5 coarse latitude"
            v[:] = coarse_lat_1d

            v = f.createVariable("coarse_lon", "f8", ("coarse_lon",))
            v.long_name = "ERA5 coarse longitude"
            v[:] = coarse_lon_1d

            if is_2d:
                v = f.createVariable("fine_lat", "f8", ("fine_row", "fine_col"))
                v.long_name = "Model grid latitude (fine)"
                v[:] = fine_lat

                v = f.createVariable("fine_lon", "f8", ("fine_row", "fine_col"))
                v.long_name = "Model grid longitude (fine)"
                v[:] = fine_lon

                v = f.createVariable("lat_idx", "i4", ("fine_row", "fine_col"))
                v.long_name = "Nearest coarse-lat index for each fine cell"
                v[:] = lat_idx

                v = f.createVariable("lon_idx", "i4", ("fine_row", "fine_col"))
                v.long_name = "Nearest coarse-lon index for each fine cell"
                v[:] = lon_idx

                v = f.createVariable("cell_latitude", "f8", ("fine_row", "fine_col"))
                v.long_name = "Latitude of grid cell"
                v[:] = fine_lat

                v = f.createVariable("cell_longitude", "f8", ("fine_row", "fine_col"))
                v.long_name = "Longitude of grid cell"
                v[:] = fine_lon
            else:
                v = f.createVariable("fine_lat", "f8", ("fine_col",))
                v.long_name = "Model grid latitude (fine)"
                v[:] = fine_lat

                v = f.createVariable("fine_lon", "f8", ("fine_row",))
                v.long_name = "Model grid longitude (fine)"
                v[:] = fine_lon

                v = f.createVariable("lat_idx", "i4", ("fine_col",))
                v.long_name = "Nearest coarse-lat index for each fine-grid column"
                v[:] = lat_idx

                v = f.createVariable("lon_idx", "i4", ("fine_row",))
                v.long_name = "Nearest coarse-lon index for each fine-grid row"
                v[:] = lon_idx

        # ── Time-varying met variables at coarse resolution ───────────────
        for key, value in era5_grid.items():
            if key in SKIP_KEYS:
                continue
            if start_index == 0:
                var = f.createVariable(
                    key, np.dtype("float64").char,
                    ("time", "coarse_lat", "coarse_lon"),
                )
                var.long_name = key
                var[start_index:end_index] = value
            else:
                f.variables[key][start_index:end_index] = value

def write_to_netcdf_full(era5_grid_path, era5_grid, model_setup, start_index=0):
    """
    Write the ERA5 data to a netCDF file.
    """
    #     TODO - docstring
    start_index = int(start_index)
    end_index = int(start_index + len(era5_grid["SW_surf"]))

    # If we are starting afresh, need to write the file from scratch,
    # else append
    if start_index == 0:
        mode = "w"
    else:
        mode = "a"

    with Dataset(era5_grid_path, mode) as f:
        # Initialise some variables if we are starting afresh
        if start_index == 0:
            f.createGroup("variables")
            f.createDimension("time", None)
            f.createDimension("column", model_setup.col_amount)
            f.createDimension("row", model_setup.row_amount)
        # Go through all of the items in the original ERA5 data, and create
        # variables in the new netCDF file.
        for key, value in era5_grid.items():
            if start_index == 0:
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
                elif key != "time":
                    var = f.createVariable(
                        key,
                        np.dtype("float64").char,
                        ("time", "column", "row"),
                    )
                    var.long_name = key
                    var[start_index:end_index] = value
            else:
                if key in ["long", "lat", "time"]:
                    continue
                var = f.variables[key]
                var[start_index:end_index] = value


def process_year_interpolated(model_setup, year, index, lat_array, lon_array):
    """
    Process a single year of ERA5 data, returning the interpolated grid in MONARCHS format.
    """
    #     TODO - docstring
    start_index = year * CHUNKSIZE * 24 / index
    timesteps_per_day = 24 / index

    era5_vars = ERA5_to_variables(
        model_setup.met_input_filepath,
        timesteps_per_day,
        model_setup.num_days,
        start_index=start_index,
        chunk_size=CHUNKSIZE,
    )

    # If we have specified some bounds in our configuration script, subset the
    # input data here.
    if has_user_defined_bounds(model_setup):
        era5_vars = grid_subset(
            era5_vars,
            model_setup.latmax,
            model_setup.latmin,
            model_setup.longmax,
            model_setup.longmin,
        )

    era5_grid = interpolate_grid(
        era5_vars, model_setup.row_amount, model_setup.col_amount
    )
    # If we are using a DEM to define our lat/long bounds, get these from the DEM now
    if getattr(model_setup, "lat_bounds", "").lower() == "dem":
        era5_grid = get_met_bounds_from_DEM(
            model_setup,
            era5_grid,
            lat_array,
            lon_array,
            diagnostic_plots=model_setup.met_dem_diagnostic_plots,
        )
    # Optionally scale our data by a forcing factor for testing purposes.
    scale_by_factor(model_setup, era5_grid)
    return era5_grid


def prescribed_met_data(model_setup):
    """
    Create a netCDF file using the prescribed met data defined in ``model_setup``
    rather than ERA5.
    """
    func_name = "monarchs.core.initial_conditions.setup_prescribed_data"
    met_data = model_setup.met_data

    def ensure_key(key, fallback_key=None, default_shape=None, fill_value=0):
        """Helper function - if a key doesn't exist, look for a fallback, else initialise to a
        default value."""
        if key not in met_data:
            met_data[key] = met_data.get(fallback_key) or np.full(
                default_shape, fill_value
            )

    # Try loading in keys, including fallbacks for common alternatives. If they don't exist,
    # set default values.
    ensure_key(
        "lat", "latitude", (model_setup.row_amount, model_setup.col_amount)
    )
    ensure_key(
        "long", "longitude", (model_setup.row_amount, model_setup.col_amount)
    )
    ensure_key(
        "snow_dens",
        default_shape=(
            model_setup.row_amount,
            model_setup.col_amount,
            model_setup.num_days * model_setup.t_steps_per_day,
        ),
        fill_value=300,
    )

    # Open a new dataset and create dimensions and variables.
    with Dataset(model_setup.met_output_filepath, "w") as f:
        f.createGroup("variables")
        f.createDimension(
            "time", model_setup.num_days * model_setup.t_steps_per_day
        )
        f.createDimension("column", model_setup.col_amount)
        f.createDimension("row", model_setup.row_amount)

        for key, value in met_data.items():
            # lat and long are functions of column and row only, not time
            if key == "long":
                var = f.createVariable(
                    "cell_longitude", "f8", ("column", "row")
                )
                var.long_name = "Longitude of grid cell"
                var[:] = value
            elif key == "lat":
                var = f.createVariable(
                    "cell_latitude", "f8", ("column", "row")
                )
                var.long_name = "Latitude of grid cell"
                var[:] = value
            # time is only a dimension, not a variable. everything else is a function of it, so
            # include it in the dims
            elif key != "time":
                var = f.createVariable(key, "f8", ("time", "column", "row"))
                var.long_name = key
                var[:] = value

    print(
        f"{func_name}: Saved meteorological data to {model_setup.met_output_filepath}"
    )


def scale_by_factor(model_setup, era5_grid):
    """
    Scale the shortwave and longwave radiation by a factor for testing purposes.
    """
    func_name = "monarchs.core.initial_conditions.scale_by_factor"
    if hasattr(model_setup, "radiation_forcing_factor"):
        if model_setup.radiation_forcing_factor not in [False, 1]:
            era5_grid["SW_surf"] *= model_setup.radiation_forcing_factor
            era5_grid["LW_surf"] *= model_setup.radiation_forcing_factor
            print(f"{func_name}: ")
            print(
                "Scaling SW_surf and LW_surf by a factor of"
                f" {model_setup.radiation_forcing_factor} for testing"
            )


def get_model_years(num_days, chunk_size=365):
    """Helper function to determine how many years of data we need to process
    based on the chunk size."""
    # For a 365-day chunk size, it will return 1 always.
    years = max(1, num_days // chunk_size + 1)
    # if the number of days is less than the chunk size, then fill the chunk
    return years - 1 if num_days % chunk_size == 0 else years


def has_user_defined_bounds(model_setup):
    """Returns True if the model_setup has valid lat/long bounds defined."""
    bounds = ["latmax", "latmin", "longmax", "longmin"]
    return all(
        hasattr(model_setup, attr) and not np.isnan(getattr(model_setup, attr))
        for attr in bounds
    )


def repeat_to_make_hourly(era5_grid, index):
    """If our timestep is e.g. 3-hourly, repeat each timestep index times to get to hourly data.
    to be used in MONARCHS"""
    for var in era5_grid:
        if var not in ["lat", "long"]:
            era5_grid[var] = np.repeat(era5_grid[var], index, axis=0)
