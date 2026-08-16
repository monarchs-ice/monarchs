"""
Model initialisation functions for MONARCHS.

Contains the functions responsible for setting up the initial model state:
loading the firn profile, building the met data, creating the model grid,
and handling restart from a checkpoint dump.
"""

import os
import warnings
import numpy as np
from monarchs.core import initial_conditions
from monarchs.io import read_checkpoint
from monarchs.variables import build_dtype
from monarchs.met_data import sources


def check_for_reload_from_dump(model_setup, grid, met_start_idx, met_end_idx):
    """
    Determine if the model needs to re-initialise parameters from a dump file.

    Parameters
    ----------
    model_setup
    grid
    met_start_idx
    met_end_idx

    Returns
    -------

    """
    # TODO - add support for reloading from pickle

    reload_name = model_setup.dump_filepath if model_setup.dump_filepath else ""

    if model_setup.reload_from_dump:
        print("Reloading state from dump...")

        if not os.path.exists(reload_name):
            first_iteration = 0
            warnings.warn(
                f"Reload/dump filepath {reload_name} does not exist - instead"
                " starting model from scratch. If you believe you do have a"
                " dump file, check that it is specified correctly in"
                " model_setup.py."
            )
            reload_dump_success = False
        else:
            (
                grid,
                met_start_idx,
                met_end_idx,
                first_iteration,
            ) = read_checkpoint(
                reload_name,
                build_dtype(
                    model_setup.vertical_points_firn,
                    model_setup.vertical_points_lake,
                    model_setup.vertical_points_lid,
                ),
            )
            print(
                f"Loading model state from dump file {reload_name} - first"
                " iteration = ",
                first_iteration,
            )
            reload_dump_success = True
    else:
        first_iteration = 0
        reload_dump_success = False
    return (
        grid,
        met_start_idx,
        met_end_idx,
        first_iteration,
        reload_dump_success,
    )


def initialise_model_data(model_setup):
    """
    Wrapper function that calls various initialisation functions to set up
    MONARCHS.
    """
    # Load in the initial firn profile (from a DEM or a user-defined firn depth).
    (
        firn_temperature,
        rho,
        firn_depth,
        valid_cells,
        dx,
        dy,
        lat_array,
        lon_array,
    ) = initial_conditions.initialise_firn_profile(
        model_setup, diagnostic_plots=model_setup.dem_diagnostic_plots
    )
    # DEM-derived coordinates are only used when lat_bounds == "dem". Otherwise
    # fall back to NaN and leave the grid lat/lon as default values from the catalogue
    use_dem_coords = model_setup.lat_bounds == "dem"
    if not use_dem_coords:
        lat_array = np.full((model_setup.row_amount, model_setup.col_amount), np.nan)
        lon_array = np.full((model_setup.row_amount, model_setup.col_amount), np.nan)

    # Write the met netCDF the run reads from, using whichever forcing source
    # the setup specifies (see monarchs.met_data.sources)
    sources.prepare_met_data(model_setup, lat_array, lon_array)

    # Write all of the initial ice shelf values into the model grid. Only pass
    # DEM-derived coordinates; without a DEM lat/lon fall back to the catalogue
    # default.
    grid_inputs = dict(valid_cell=valid_cells, size_dx=dx, size_dy=dy)
    if use_dem_coords:
        grid_inputs["lat"] = lat_array
        grid_inputs["lon"] = lon_array
    grid = initial_conditions.create_model_grid(
        model_setup, firn_depth, rho, firn_temperature, **grid_inputs
    )
    return grid
