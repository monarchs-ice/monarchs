"""
Read-side of the MONARCHS met cache.

Loads one day of meteorological data at a time from the netCDF cache written
by ``monarchs.met_data.setup_met_data`` (which owns the write side of the
same format), expanding coarse ERA5 data onto the model grid via index maps
where present. Called by the driver each model day.
"""

import numpy as np
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

from monarchs.met_data.index_map import apply_index_map_expand
from monarchs.met_data.met_data_grid import initialise_met_data, get_spec


def get_snow_sum(met_data_grid, grid, snow_added):
    """
    Work out how much snow has been added to the model over the last day, and
    add it to the total amount of snow already added.
    """
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid["valid_cell"][i, j]:
                snow_array = (
                    met_data_grid["snow_dens"][:, i, j]
                    * met_data_grid["snowfall"][:, i, j]
                )
                snow_added += np.sum(snow_array)

    return snow_added


def met_window(day, t_steps_per_day, met_data_len):
    """
    Get the window of met data that we want to read in for a given day.
    Dependent on the number of timesteps in the day. Defaults to 24
    """
    start = day * t_steps_per_day % met_data_len
    end = (day + 1) * t_steps_per_day % met_data_len
    if start > end:
        end = met_data_len
    return start, end


def update_met_conditions(
    model_setup, grid, met_start_idx, met_end_idx, start=False, snow_added=0
):
    """
    Load meteorological data for one day from the met netCDF and optionally
    expand from coarse (ERA5) to fine (MONARCHS) grid using index maps if present.

    Parameters
    ----------
    model_setup
    grid
    met_start_idx
    met_end_idx
    start : bool, optional
        If True, wrap met_start_idx modulo met_data_len (e.g. for restart).
    snow_added : float, optional
        Running total of snow added (for mass accounting).

    Returns
    -------
    met_data_grid : structured array
    met_data_len : int
    snow_added : float
    """
    with Dataset(model_setup.met_output_filepath) as met_data:
        met_data_len = len(met_data.variables["temperature"])
        if start:
            met_start_idx = met_start_idx % met_data_len

        has_index_maps = (
            "lat_idx" in met_data.variables and "lon_idx" in met_data.variables
        )

        if has_index_maps:
            lat_idx = np.asarray(met_data.variables["lat_idx"][:], dtype=np.int32)
            lon_idx = np.asarray(met_data.variables["lon_idx"][:], dtype=np.int32)

            def _expand(var):
                """Read a coarse variable and expand it to the fine model grid."""
                out = apply_index_map_expand(
                    met_data.variables[var][met_start_idx:met_end_idx].data,
                    lat_idx,
                    lon_idx,
                )
                # 1-D index maps yield (time, len(lat_idx), len(lon_idx)); when
                # stored as (fine_col, fine_row) we need (time, fine_row, fine_col)
                if lat_idx.ndim == 1:
                    out = np.transpose(out, (0, 2, 1))
                return out

            def _read(var):
                return _expand(var)

            # Cell coordinates: coarse format may use cell_latitude/cell_longitude
            # or fine_lat/fine_lon (written by write_to_netcdf)
            if "cell_latitude" in met_data.variables:
                cell_lat = met_data.variables["cell_latitude"][:].data
                cell_lon = met_data.variables["cell_longitude"][:].data
            else:
                fine_lat = met_data.variables["fine_lat"][:].data
                fine_lon = met_data.variables["fine_lon"][:].data
                if fine_lat.ndim == 2:
                    cell_lat = fine_lat
                    cell_lon = fine_lon
                else:
                    # 1-D: fine_lat (fine_col,), fine_lon (fine_row,) -> (row, col)
                    cell_lat = np.broadcast_to(
                        fine_lat[np.newaxis, :],
                        (model_setup.row_amount, model_setup.col_amount),
                    ).copy()
                    cell_lon = np.broadcast_to(
                        fine_lon[:, np.newaxis],
                        (model_setup.row_amount, model_setup.col_amount),
                    ).copy()
        else:
            # full-grid: variables are already (time, row, col)
            def _read(var):
                return met_data.variables[var][met_start_idx:met_end_idx].data

            if (
                "cell_latitude" in met_data.variables
                and "cell_longitude" in met_data.variables
            ):
                cell_lat = met_data.variables["cell_latitude"][:].data
                cell_lon = met_data.variables["cell_longitude"][:].data
            # dummy values if no lat/long specified
            else:
                cell_lat = np.full(
                    (model_setup.row_amount, model_setup.col_amount), np.nan
                )
                cell_lon = np.full(
                    (model_setup.row_amount, model_setup.col_amount), np.nan
                )

        met_data_dtype = get_spec()
        met_data_grid = initialise_met_data(
            _read("snowfall"),
            _read("snow_dens"),
            _read("temperature"),
            _read("wind"),
            _read("pressure"),
            _read("dew_point_temperature"),
            _read("LW_surf"),
            _read("SW_surf"),
            cell_lat,
            cell_lon,
            model_setup.row_amount,
            model_setup.col_amount,
            met_data_dtype,
            model_setup.t_steps_per_day,
        )

        snow_added = get_snow_sum(met_data_grid, grid, snow_added)

        for key in met_data.variables.keys():
            if key not in (
                "cell_latitude",
                "cell_longitude",
                "lat_idx",
                "lon_idx",
                "coarse_lat",
                "coarse_lon",
                "fine_lat",
                "fine_lon",
            ):
                if met_end_idx > len(met_data[key]):
                    raise IndexError(
                        "monarchs.met_data.load.update_met_conditions:"
                        " met_end_idx > days *"
                        " hours, i.e. your grid of meteorological data is too"
                        " small for the number of timesteps you wish to run"
                    )

    return met_data_grid, met_data_len, snow_added
