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
from monarchs.core.model_grid import get_spec as get_iceshelf_spec
from monarchs.met_data import setup_met_data


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

    if hasattr(model_setup, "dump_filepath"):
        reload_name = model_setup.dump_filepath
    else:
        reload_name = ""

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
                get_iceshelf_spec(
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
    func_name = "monarchs.core.driver.initialise_model_data"
    # Load in the initial firn profile, either from a whole DEM, or a
    # user-defined subset
    if hasattr(model_setup, "lat_bounds") and model_setup.lat_bounds.lower() == "dem":
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
    else:
        (
            firn_temperature,
            rho,
            firn_depth,
            valid_cells,
            dx,
            dy,
            _,
            _,
        ) = initial_conditions.initialise_firn_profile(
            model_setup, diagnostic_plots=model_setup.dem_diagnostic_plots
        )
        lat_array = np.zeros((model_setup.row_amount, model_setup.col_amount)) * np.nan
        lon_array = np.zeros((model_setup.row_amount, model_setup.col_amount)) * np.nan

    # Set up meteorological data, from either ERA5-format input ("ERA5") or
    # user-defined values from their model configuration ("user_defined")
    if model_setup.met_data_source == "ERA5":
        setup_met_data_flag = True
        if model_setup.load_precalculated_met_data:
            print(f"{func_name}: Loading in pre-calculated MONARCHS format met data")
            # check the file actually exists first
            if not os.path.exists(model_setup.met_output_filepath):
                print(
                    f"{func_name}: Pre-calculated met data file"
                    f" {model_setup.met_output_filepath} does not exist."
                    " Calculating from raw ERA5 data instead."
                )
                setup_met_data_flag = True
            else:
                setup_met_data_flag = False

        if setup_met_data_flag:
            setup_met_data.met_data_from_era5(model_setup, lat_array, lon_array)
    elif model_setup.met_data_source == "user_defined":
        setup_met_data.prescribed_met_data(model_setup)

    # Write all of the initial ice shelf values into the model grid
    grid = initial_conditions.create_model_grid(
        model_setup,
        firn_depth,
        rho,
        firn_temperature,
        valid_cells=valid_cells,
        lats=lat_array,
        lons=lon_array,
        size_dx=dx,
        size_dy=dy,
    )
    return grid
