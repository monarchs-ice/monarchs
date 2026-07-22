"""
Read-side of the MONARCHS met cache.

Loads one day of meteorological data at a time from the netCDF file written by
``monarchs.met_data.setup_met_data``. For large grids the file stores coarse
ERA5 data plus an index map, which is expanded to the model grid on read.
"""

import numpy as np
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

from monarchs.met_data.catalogue import MET_CATALOGUE
from monarchs.met_data.index_map import apply_index_map_expand
from monarchs.met_data.met_data_grid import initialise_met_data


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


def _met_reader(met_data, start, end):
    """
    Return ``read(name) -> (time, row, col)`` for one met window.

    When the file stores coarse ERA5 data plus index maps, ``read`` expands the
    coarse field onto the model grid; otherwise it reads the full-grid variable
    directly.
    """
    variables = met_data.variables
    if "lat_idx" in variables and "lon_idx" in variables:
        lat_idx = np.asarray(variables["lat_idx"][:], dtype=np.int32)
        lon_idx = np.asarray(variables["lon_idx"][:], dtype=np.int32)

        def read(name):
            # apply_index_map_expand returns (time, row, col) for both 1-D and
            # 2-D index maps
            return apply_index_map_expand(
                variables[name][start:end].data, lat_idx, lon_idx
            )

        return read

    def read(name):
        return variables[name][start:end].data

    return read


def _cell_coords(met_data, model_setup):
    """
    Per-cell (lat, lon) arrays of shape (row, col) for the model grid.

    Uses cell_latitude/cell_longitude when the file stores them (2-D index maps
    or prescribed data); otherwise reconstructs them from the 1-D fine_lat/
    fine_lon axes, or falls back to NaN when no coordinates were provided.
    """
    variables = met_data.variables
    rows, cols = model_setup.row_amount, model_setup.col_amount
    if "cell_latitude" in variables and "cell_longitude" in variables:
        return variables["cell_latitude"][:].data, variables["cell_longitude"][:].data
    if "fine_lat" in variables and "fine_lon" in variables:
        fine_lat = variables["fine_lat"][:].data
        fine_lon = variables["fine_lon"][:].data
        if fine_lat.ndim == 2:
            return fine_lat, fine_lon
        # 1-D: fine_lat (col,), fine_lon (row,) -> (row, col)
        cell_lat = np.broadcast_to(fine_lat[np.newaxis, :], (rows, cols)).copy()
        cell_lon = np.broadcast_to(fine_lon[:, np.newaxis], (rows, cols)).copy()
        return cell_lat, cell_lon
    return np.full((rows, cols), np.nan), np.full((rows, cols), np.nan)


def update_met_conditions(
    model_setup, grid, met_start_idx, met_end_idx, start=False, snow_added=0
):
    """
    Load one day of meteorological data from the met netCDF, expanding coarse
    (ERA5) data onto the fine (MONARCHS) model grid via index maps when present.

    Parameters
    ----------
    model_setup, grid
    met_start_idx, met_end_idx : int
        Timestep window to read.
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
        if met_end_idx > met_data_len:
            raise IndexError(
                "monarchs.met_data.load.update_met_conditions: met_end_idx"
                f" ({met_end_idx}) exceeds the {met_data_len} timesteps of met"
                " data available - the met grid is too small for the number of"
                " timesteps you wish to run."
            )

        read = _met_reader(met_data, met_start_idx, met_end_idx)
        cell_lat, cell_lon = _cell_coords(met_data, model_setup)

        # read every catalogue field from the file (their names match the file
        # variables); lat/lon come from the per-cell coordinates resolved above
        inputs = {
            var.name: read(var.name)
            for var in MET_CATALOGUE
            if var.name not in ("lat", "lon")
        }
        inputs["lat"] = cell_lat
        inputs["lon"] = cell_lon
        met_data_grid = initialise_met_data(
            inputs,
            model_setup.row_amount,
            model_setup.col_amount,
            model_setup.t_steps_per_day,
        )
        snow_added = get_snow_sum(met_data_grid, grid, snow_added)

    return met_data_grid, met_data_len, snow_added
