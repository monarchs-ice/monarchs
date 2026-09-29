"""
Loads one day of meteorological data at a time from the netCDF file written by
``monarchs.met_data.setup_met_data``. For large grids the file stores coarse
ERA5 data plus an index map, which is expanded to the (much larger)
model grid on read, which saves us writing an enormous netCDF file.
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
    snow = np.sum(met_data_grid["snow_dens"] * met_data_grid["snowfall"], axis=0)
    return snow_added + np.sum(snow[grid["valid_cell"]])


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


# define coordinates that we skip over in read_fields
_COORDS = ("lat", "lon")


def _read_fields(met_data, start, end):
    """
    Read every met field for one window as a (time, row, col) array.
    """
    variables = met_data.variables
    missing = [
        var.name
        for var in MET_CATALOGUE
        if var.name not in _COORDS and var.name not in variables
    ]
    if missing:
        raise KeyError(
            f"monarchs.met_data.load: met field(s) {missing} not found in"
            f" {met_data.filepath()}, which holds {sorted(variables)}. Met"
            " fields use the names in monarchs.met_data.catalogue."
        )

    lat_idx = lon_idx = None
    if "lat_idx" in variables:
        lat_idx = np.asarray(variables["lat_idx"][:], dtype=np.int32)
        lon_idx = np.asarray(variables["lon_idx"][:], dtype=np.int32)

    fields = {}
    for var in MET_CATALOGUE:
        if var.name in _COORDS:
            continue
        value = variables[var.name][start:end].data
        if lat_idx is not None:
            value = apply_index_map_expand(value, lat_idx, lon_idx)
        fields[var.name] = value
    return fields


def _cell_coords(met_data, model_setup):
    """
    Per-cell (lat, lon) arrays of shape (row, col), or NaN when the met data
    has no coordinates (e.g. user-defined forcing)
    """
    variables = met_data.variables
    if "cell_latitude" in variables:
        return variables["cell_latitude"][:].data, variables["cell_longitude"][:].data
    shape = (model_setup.row_amount, model_setup.col_amount)
    return np.full(shape, np.nan), np.full(shape, np.nan)


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

        fields = _read_fields(met_data, met_start_idx, met_end_idx)
        fields["lat"], fields["lon"] = _cell_coords(met_data, model_setup)
        met_data_grid = initialise_met_data(
            fields,
            model_setup.row_amount,
            model_setup.col_amount,
            model_setup.t_steps_per_day,
        )
        snow_added = get_snow_sum(met_data_grid, grid, snow_added)

    return met_data_grid, met_data_len, snow_added
