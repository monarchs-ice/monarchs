""" """

# TODO - docstrings, module-level docstring
from netCDF4 import Dataset  # pylint: disable=no-name-in-module
import numpy as np
from monarchs.met_data.import_ERA5 import (
    ERA5_to_variables,
    grid_subset,
    get_met_bounds_from_DEM,
)

# TODO - make CHUNKSIZE model_setup variable rather than constant?
CHUNKSIZE = 365  # days
MODULE_NAME = "monarchs.met_data.setup_met_data"

def met_data_from_era5(model_setup, lat_array=False, lon_array=False,
                       ):
    """
    Convert ERA5 input data into a netCDF in MONARCHS format.
    """
    # TODO - docstring, refactor to break up with a few helper functions
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
        start_index = year * CHUNKSIZE * 24 // index

        era5_grid = process_year(
            model_setup, year, index, lat_array, lon_array
        )

        if index > 1:
            # Repeat each timestep index times to get to hourly data.
            repeat_to_make_hourly(era5_grid, index)

        print(
            f"{func_name}: Writing data for" f" year {year + 1} into netCDF..."
        )

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

    This will write an index map if we are obtaining the limits of the
    meteorological data from the DEM (i.e. model_setup.lat_bounds == "dem"). This
    mapping allows MONARCHS gridpoints to select the nearest relevant data for the
    forcing without needing to save everything to disk or interpolate at runtime.

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
    start_index = year * CHUNKSIZE * 24 // index
    timesteps_per_day = 24 // index

    era5_vars = ERA5_to_variables(
        model_setup.met_input_filepath,
        timesteps_per_day,
        model_setup.num_days,
        start_index=start_index,
        chunk_size=CHUNKSIZE,
    )

    use_user_bounds = has_user_defined_bounds(model_setup)
    use_dem_bounds = getattr(model_setup, "lat_bounds", None) == "dem"

    if use_user_bounds and use_dem_bounds:
        raise ValueError(
            "monarchs.setup_met_data.process_year: choose either explicit lat/long bounds "
            "or lat_bounds='dem', not both."
            )

    # First restrict the coarse ERA5 grid if we are subsetting via the DEM.
    if use_user_bounds:
        era5_vars = grid_subset(
            era5_vars,
            model_setup.latmax,
            model_setup.latmin,
            model_setup.longmax,
            model_setup.longmin,
        )

    # Then decide how to map the coarse ERA5 grid onto the model grid.
    # DEM mode produces 2-D index maps; regular lat/lon mode produces 1-D maps.
    if use_dem_bounds:
        if lat_array is False or lon_array is False:
            raise ValueError(
                "process_year: lat_array and lon_array are required when "
                "lat_bounds='dem'."
                )

        era5_vars, lat_idx, lon_idx = get_met_bounds_from_DEM(
            model_setup,
            era5_vars,
            lat_array,
            lon_array,
            diagnostic_plots=model_setup.met_dem_diagnostic_plots,
        )
        fine_lat = lat_array
        fine_lon = lon_array

        era5_vars["lat_idx"] = lat_idx
        era5_vars["lon_idx"] = lon_idx
        era5_vars["fine_lat"] = fine_lat
        era5_vars["fine_lon"] = fine_lon

    else:
        # Regular lat/lon case: keep the current 1-D index-map convention used
        # by update_met_conditions:
        # - fine_lat has length col_amount
        # - fine_lon has length row_amount
        coarse_lat = np.asarray(era5_vars["lat"])
        coarse_lon = np.asarray(era5_vars["long"])

        if coarse_lat.ndim != 1 or coarse_lon.ndim != 1:
            raise ValueError(
                "process_year: expected 1-D ERA5 lat/long axes for regular-grid "
                "index mapping."
                )
        fine_lat = np.linspace(
            coarse_lat[-1], coarse_lat[0], model_setup.col_amount
                                            )
        fine_lon = np.linspace(
            coarse_lon[0], coarse_lon[-1], model_setup.row_amount
                                            )

        lat_idx = np.abs(
            coarse_lat[:, np.newaxis] - fine_lat[np.newaxis, :]
            ).argmin(axis=0).astype(np.int32)
        lon_idx = np.abs(
            coarse_lon[:, np.newaxis] - fine_lon[np.newaxis, :]
                    ).argmin(axis=0).astype(np.int32)

        era5_vars["lat_idx"] = lat_idx
        era5_vars["lon_idx"] = lon_idx
        era5_vars["fine_lat"] = fine_lat
        era5_vars["fine_lon"] = fine_lon


    # optionally scale radiation for testing
    scale_by_factor(model_setup, era5_vars)

    return era5_vars

def write_to_netcdf(era5_grid_path, era5_grid, model_setup, start_index=0):
    """
    Write ERA5 data to the met netCDF. This data is either index-mapped or
    interpolated to obtain values on the MONARCHS model grid.

    On the first call (``start_index == 0``) the index maps and fine-grid
    lat / lon are written as static variables so that
    ``update_met_conditions`` can reconstruct the full model-grid arrays
    without storing one value per fine cell.

    Supports both 1-D index maps (separable regular grid) and 2-D index maps
    (e.g. from get_met_bounds_from_DEM). When 2-D, also writes
    cell_latitude and cell_longitude from fine_lat/fine_lon for the reader.
    """
    routine_name = "write_to_netcdf"
    start_index = int(start_index)
    end_index = int(start_index + len(era5_grid["SW_surf"]))

    mode = "w" if start_index == 0 else "a"

    # check we have the index mapping vars
    required = ("lat_idx", "lon_idx", "fine_lat", "fine_lon")
    missing = [key for key in required if key not in era5_grid]
    if missing:
        raise ValueError(
            "write_to_netcdf requires index-mapped ERA5 data; "
            f"missing keys: {missing}"
        )


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
                coarse_lat_1d = np.asarray(era5_grid["lat"])
                coarse_lon_1d = np.asarray(era5_grid["long"])
                if coarse_lat_1d.ndim > 1 or coarse_lon_1d.ndim > 1:
                    raise ValueError(
                        f"{MODULE_NAME}.{routine_name}: expected 1-D coarse lat/lon. "
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

        # Time-varying met variables at coarse resolution
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
def prescribed_met_data(model_setup):
    """
    Create a netCDF file using the prescribed met data defined in ``model_setup``
    rather than ERA5.

    Full-grid prescribed data are written directly on the MONARCHS model grid:
    - time-varying variables use dimensions (time, row, column)
    - optional cell_latitude / cell_longitude use dimensions (row, column)
    """
    func_name = "prescribed_met_data"
    met_data = dict(model_setup.met_data)

    def ensure_key(key, fallback_key=None, default_shape=None, fill_value=0):
        """
        If ``key`` is missing, copy ``fallback_key`` if present;
        otherwise create a default array.
        """
        if key in met_data:
            return
        if fallback_key is not None and fallback_key in met_data:
            met_data[key] = met_data[fallback_key]
        else:
            met_data[key] = np.full(default_shape, fill_value)

    # if we have coords great, else ignore
    if "lat" not in met_data and "latitude" in met_data:
        met_data["lat"] = met_data["latitude"]
    if "long" not in met_data and "longitude" in met_data:
        met_data["long"] = met_data["longitude"]

    # check snow density exists
    ensure_key(
        "snow_dens",
        default_shape=(
            model_setup.num_days * model_setup.t_steps_per_day,
            model_setup.row_amount,
            model_setup.col_amount,
        ),
        fill_value=300,
    )

    with Dataset(model_setup.met_output_filepath, "w") as f:
        f.createGroup("variables")
        f.createDimension(
            "time", model_setup.num_days * model_setup.t_steps_per_day
        )
        f.createDimension("row", model_setup.row_amount)
        f.createDimension("column", model_setup.col_amount)

        for key, value in met_data.items():
            if key == "time":
                continue

            if key == "long":
                var = f.createVariable(
                    "cell_longitude", "f8", ("row", "column")
                )
                var.long_name = "Longitude of grid cell"
                var[:] = value

            elif key == "lat":
                var = f.createVariable(
                    "cell_latitude", "f8", ("row", "column")
                )
                var.long_name = "Latitude of grid cell"
                var[:] = value

            else:
                var = f.createVariable(key, "f8", ("time", "row", "column"))
                var.long_name = key
                var[:] = value

    print(
        f"{MODULE_NAME}.{func_name}: Saved meteorological data to "
        f"{model_setup.met_output_filepath}"
    )

def scale_by_factor(model_setup, era5_grid):
    """
    Scale the shortwave and longwave radiation by a factor for testing purposes.
    """
    func_name = "scale_by_factor"
    if hasattr(model_setup, "radiation_forcing_factor"):
        if model_setup.radiation_forcing_factor not in [False, 1]:
            era5_grid["SW_surf"] *= model_setup.radiation_forcing_factor
            era5_grid["LW_surf"] *= model_setup.radiation_forcing_factor
            print(f"{MODULE_NAME}.{func_name}: ")
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
    TIME_VARYING_KEYS = {
        "time",
        "wind",
        "temperature",
        "dew_point_temperature",
        "pressure",
        "snowfall",
        "SW_surf",
        "LW_surf",
        "snow_albedo",
        "snow_dens",
    }

    for var in TIME_VARYING_KEYS:
        if var in era5_grid:
            era5_grid[var] = np.repeat(era5_grid[var], index, axis=0)

