"""
Core functions used in the running of MONARCHS.
This module contains the core functions that drive the code
when it is executed.

When the code is run, it calls main(), which handles loading in the data,
calling core.loop_over_grid every model day
(interchangeably called "iteration").
The loop over each timestep (by default 1 hour) is handled
in the single-column physics steps.
main then calls the lateral movement functions, and handles saving the data,
both in terms of the model state (also known as a "dump"), and the
variables that the user wants to track over time.

"""

# TODO - docstrings

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pathos
from netCDF4 import Dataset  # pylint: disable=no-name-in-module
from monarchs.core import configuration, initial_conditions, setup_met_data
from monarchs.core.load_model_setup import get_model_setup
from monarchs.core.dump_model_state import dump_state, reload_from_dump
from monarchs.core.model_grid import get_spec as get_iceshelf_spec
from monarchs.core.model_output import setup_output, update_model_output
from monarchs.core.utils import (
    get_2d_grid,
    calc_grid_mass,
    check_grid_correctness,
    get_num_cores,
)
from monarchs.met_data.met_data_grid import initialise_met_data, get_spec
from monarchs.physics import lateral_movement

# dummy init value for Dask client which may be used later - this needs to be
# global
CLIENT = None


def setup_toggle_dict(model_setup):
    """
    Set up a dictionary of switches to determine the running of the model.
    These are accessed by each thread, so we need to set up a new object to
    hold these else we will run into errors.
    Additionally, the ModelSetup class is not a jitclass
    (and cannot be dynamically set to be one), so will not work with Numba.
    We therefore need a numba.typed.Dict object in this instance.

    Parameters
    ----------
    model_setup

    Returns
    -------

    """

    toggle_dict = {
        "parallel": model_setup.parallel,
        "use_numba": model_setup.use_numba,
        "snowfall_toggle": model_setup.snowfall_toggle,
        "firn_column_toggle": model_setup.firn_column_toggle,
        "firn_heat_toggle": model_setup.firn_heat_toggle,
        "lake_development_toggle": model_setup.lake_development_toggle,
        "lid_development_toggle": model_setup.lid_development_toggle,
        "percolation_toggle": model_setup.percolation_toggle,
        "spinup": model_setup.spinup,
        "perc_time_toggle": model_setup.perc_time_toggle,
        "densification_toggle": model_setup.densification_toggle,
        "ignore_errors": model_setup.ignore_errors,
        "use_mpi": model_setup.use_mpi,
    }

    if model_setup.use_numba:
        # in this case we need to convert to a Numba typed dict
        # pylint: disable=import-outside-toplevel
        from numba import types
        from numba.typed import Dict  # pylint: disable=no-name-in-module

        # pylint: enable=import-outside-toplevel
        num_dict = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.boolean,
        )
        for key, value in toggle_dict.items():
            num_dict[key] = value
        toggle_dict = num_dict

    return toggle_dict


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
            grid, met_start_idx, met_end_idx, first_iteration = (
                reload_from_dump(
                    reload_name,
                    get_iceshelf_spec(
                        model_setup.vertical_points_firn,
                        model_setup.vertical_points_lid,
                        model_setup.vertical_points_lake,
                    ),
                )
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


def get_snow_sum(met_data_grid, grid, met_start_idx, met_end_idx, snow_added):
    """
    Work out how much snow has been added to the model over the last day, and
    add it to the total amount of snow already added.
    """
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid["valid_cell"][i, j]:
                snow_array = (
                    met_data_grid["snow_dens"][met_start_idx:met_end_idx, i, j]
                    * met_data_grid["snowfall"][met_start_idx:met_end_idx, i, j]
                )

                snow_added += np.sum(snow_array)
    return snow_added


def update_met_conditions(
    model_setup, grid, met_start_idx, met_end_idx, start=False, snow_added=0
):
    """

    Parameters
    ----------
    model_setup
    grid
    met_start_idx
    met_end_idx

    Returns
    -------

    """
    with Dataset(model_setup.met_output_filepath) as met_data:
        met_data_len = len(met_data.variables["temperature"])
        if start:
            met_start_idx = met_start_idx % met_data_len

        met_data_dtype = get_spec()
        met_data_grid = initialise_met_data(
            met_data.variables["snowfall"][met_start_idx:met_end_idx].data,
            met_data.variables["snow_dens"][met_start_idx:met_end_idx].data,
            met_data.variables["temperature"][met_start_idx:met_end_idx].data,
            met_data.variables["wind"][met_start_idx:met_end_idx].data,
            met_data.variables["pressure"][met_start_idx:met_end_idx].data,
            met_data.variables["dew_point_temperature"][
                met_start_idx:met_end_idx
            ].data,
            met_data.variables["LW_surf"][met_start_idx:met_end_idx].data,
            met_data.variables["SW_surf"][met_start_idx:met_end_idx].data,
            met_data.variables["cell_latitude"][:].data,
            met_data.variables["cell_longitude"][:].data,
            model_setup.row_amount,
            model_setup.col_amount,
            met_data_dtype,
            model_setup.t_steps_per_day,
        )

        snow_added = get_snow_sum(
            met_data_grid, grid, met_start_idx, met_end_idx, snow_added
        )

        for key in met_data.variables.keys():
            if key not in ("cell_latitude", "cell_longitude"):
                if met_end_idx > len(met_data[key]):
                    raise IndexError(
                        "monarchs.core.driver.main: met_end_idx > days *"
                        " hours, i.e. your grid of meteorological data is too"
                        " small for the number of timesteps you wish to run"
                    )

    return met_data_grid, met_data_len, snow_added


def check_firn_met_consistency(grid, met_data_grid):
    """
    Check visually that the meteorological data input is mapped to the
    dem_utils input.
    """
    # pylint: disable=import-outside-toplevel
    from matplotlib import pyplot as plt
    from cartopy import crs as ccrs

    # pylint: enable=import-outside-toplevel
    projection = ccrs.PlateCarree()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211, projection=projection)
    ax1.set_title("Original vs regridded met data")
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)
    ax1.contourf(
        get_2d_grid(grid, "lon"),
        get_2d_grid(grid, "lat"),
        get_2d_grid(grid, "firn_depth"),
        transform=projection,
    )
    ax2 = fig1.add_subplot(212, projection=projection)
    ax2.coastlines()
    ax2.gridlines(draw_labels=True)
    ax2.contourf(
        get_2d_grid(grid, "lon"),
        get_2d_grid(grid, "lat"),
        get_2d_grid(met_data_grid, "temperature"),
        transform=projection,
    )


def print_model_end_of_timestep_messages(
    grid,
    day,
    total_mass_start,
    snow_added,
    catchment_outflow,
    tic,
    model_setup,
):
    """
    Messages to print out at the end of each model timestep (day).

    Parameters
    ----------
    grid
    day
    total_mass_start
    snow_added
    catchment_outflow
    tic

    Returns
    -------

    """
    toc = time.perf_counter()
    print("\n*******************************************\n")
    print("End of timestep diagnostics:")
    print(f"Total mass at end of iteration {day + 1} = ", calc_grid_mass(grid))
    if model_setup.snowfall_toggle and model_setup.catchment_outflow:
        print(
            "Original mass accounting for catchment outflow and snowfall = ",
            total_mass_start + snow_added - catchment_outflow,
        )
    elif model_setup.snowfall_toggle and not model_setup.catchment_outflow:
        print("Total snow added = ", snow_added)
        print(
            "Original mass accounting for snowfall = ",
            total_mass_start + snow_added,
        )
    elif model_setup.catchment_outflow:
        print(f"Model has lost {catchment_outflow} units of water")
        print(
            "Original mass accounting for catchment outflow = ",
            total_mass_start - catchment_outflow,
        )
    else:
        print("Original mass = ", total_mass_start)
    print(f"Time at end of day {day + 1} is {toc - tic:0.4f} seconds")
    print("Firn depth = ", get_2d_grid(grid, "firn_depth"))
    print("Lake depth = ", get_2d_grid(grid, "lake_depth"))
    print("Lid depth = ", get_2d_grid(grid, "lid_depth"))
    print("Number of lakes = ", np.sum(get_2d_grid(grid, "lake")))
    print("Number of lids = ", np.sum(get_2d_grid(grid, "lid")))
    # ensure that output is flushed to the console immediately rather than
    # being buffered.
    # Mostly a fix for output not updating when running with Slurm.
    sys.stdout.flush()


def main(model_setup, grid):
    """
    Main loop function. This calls loop_over_grid, which in turn calls
    timestep_loop and the other functions (vertical, lake, and lid),
    a total of <t_steps_per_day> times per <day>.
    This then calls lateral_movement (i.e. water movement) once per <day>.

    This also handles input/output via netCDF.

    Parameters
    ----------
    model_setup : module
        The model setup module imported by ``configuration.py``.
        This contains the following arguments used in this function:

            row_amount : int
                number of rows in the model grid.
            col_amount : int
                number of columns in the model grid.
            met_output_filepath : str
                path to netCDF file containing the meteorological data used to
                drive the model.
            days : int
                Number of days to run in total. [days]
            t_steps : int
                Number of hours to run in each day.
                This should most likely be 24. [h]
            lat_grid_size : float
                Size of each lateral grid cell, used to determine the lateral
                flow of water [m]
            timestep : int
                Amount of seconds to run in each t_step, most likely 3600. [s]
            output_filename : str, optional
                Name of the netCDF file used to save model output,
                if applicable.
            vars_to_save: tuple, optional
                Tuple containing the variables that you want to save from your
                model grid.
                These will be added to the output netCDF file
                defined in output_filename.

    grid : numpy structured array
        Model grid, containing the data specified in get_spec() of
        monarchs.core.model_grid.

    Returns
    -------
    grid : numpy structured array
        Model grid at the end of the run.

    """
    # TODO - split main up into a few different helper functions. Possibly
    # move some of the functions from above into utils also, or a new
    # module entirely.

    # If running in parallel across multiple nodes, then we first set up the
    # dask Client object
    if (
        model_setup.parallel
        and model_setup.dask_scheduler == "distributed"
        and not model_setup.use_numba
    ):
        print("Setting up Dask Client object...")
        # only import dask.distributed if we need it, which only happens once.
        # to improve portability, this avoids requiring dask.distributed to be
        # installed for the model to run. this is desirable, so remove
        # pylint check. also, Client exists within dask.distributed, but
        # the linter cannot see it, so disable that check also
        # pylint: disable=import-outside-toplevel, no-name-in-module
        from dask.distributed import Client

        # pylint: enable=import-outside-toplevel, no-name-in-module
        # make client global so we don't need to pass it around depending on
        # whether we use dask or not.
        # pylint: disable=global-statement
        global CLIENT
        # pylint: enable=global-statement
        CLIENT = Client()

    # pylint: disable=import-outside-toplevel
    if model_setup.use_numba:
        from numba import jit
        from monarchs.core.Numba.loop_over_grid import loop_over_grid_numba
        from numba import set_num_threads

        if model_setup.cores in ["all", False]:
            cores = pathos.helpers.cpu_count()
        else:
            cores = model_setup.cores
        set_num_threads(int(cores))
        loop_over_grid = jit(
            loop_over_grid_numba, parallel=model_setup.parallel, nopython=True
        )
    else:
        from monarchs.core.loop_over_grid import loop_over_grid
    # pylint: enable=import-outside-toplevel

    tic = time.perf_counter()
    met_start_idx = 0
    met_end_idx = model_setup.t_steps_per_day
    output_counter = 0
    if hasattr(model_setup, "output_grid_size"):
        output_grid_size = model_setup.output_grid_size
        if not output_grid_size:
            output_grid_size = grid["vert_grid"][0][0]
    else:
        output_grid_size = grid["vert_grid"][0][0]
    (
        grid,
        met_start_idx,
        met_end_idx,
        first_iteration,
        reload_dump_success,
    ) = check_for_reload_from_dump(
        model_setup, grid, met_start_idx, met_end_idx
    )
    print("firn depth = ", get_2d_grid(grid, "firn_depth"))
    print("valid_cell = ", get_2d_grid(grid, "valid_cell"))
    if model_setup.save_output and not reload_dump_success:
        setup_output(
            model_setup.output_filepath,
            grid,
            vars_to_save=model_setup.vars_to_save,
            vert_grid_size=output_grid_size,
        )
    if reload_dump_success:
        start = True
        output_counter = first_iteration
    else:
        start = False
    snow_added = 0
    met_data_grid, met_data_len, snow_added = update_met_conditions(
        model_setup,
        grid,
        met_start_idx,
        met_end_idx,
        start=start,
        snow_added=snow_added,
    )
    toggle_dict = setup_toggle_dict(model_setup)
    cores = get_num_cores(model_setup)
    total_mass_start = calc_grid_mass(grid)
    catchment_outflow = 0
    time_loop = range(first_iteration, model_setup.num_days)
    start = time.perf_counter()
    dt = 3600

    for day in time_loop:
        timestep_start = time.perf_counter()
        print("\n*******************************************\n")
        print(f"Start of model day {day + 1}\n")

        # pre-flatten and rearrange met_data_grid
        met_data_grid = met_data_grid.reshape(24, -1)
        met_data_grid = np.moveaxis(
            met_data_grid, 0, -1
        )  # move the first axis to the last axis

        if model_setup.single_column_toggle:
            grid = loop_over_grid(
                model_setup.row_amount,
                model_setup.col_amount,
                grid,
                dt,
                met_data_grid,
                model_setup.t_steps_per_day,
                toggle_dict,
                parallel=model_setup.parallel,
                ncores=cores,
                dask_scheduler=model_setup.dask_scheduler,
                client=CLIENT,
            )

        print("Single-column physics finished")
        print(f"Single column physics time: {time.perf_counter() - start:.2f}s")
        start_serial = time.perf_counter()
        if model_setup.dump_data_pre_lateral_movement:
            if model_setup.dump_format == "NETCDF4":
                dump_state(
                    model_setup.dump_filepath, grid, met_start_idx, met_end_idx
                )
            elif model_setup.dump_format == "pickle":
                with open(model_setup.dump_filepath, "wb") as outfile:
                    pickle.dump(grid, outfile)

        lat_start = time.perf_counter()
        if model_setup.lateral_movement_toggle:
            print("Moving water laterally...")
            # perc_toggle as its own variable for PEP8 (line too long)
            perc_toggle = model_setup.lateral_movement_percolation_toggle
            grid, current_iteration_outwater = lateral_movement.move_water(
                grid,
                model_setup.row_amount,
                model_setup.col_amount,
                model_setup.lateral_timestep,
                catchment_outflow=model_setup.catchment_outflow,
                flow_into_land=model_setup.flow_into_land,
                lateral_movement_percolation_toggle=perc_toggle,
                flow_speed_scaling=model_setup.flow_speed_scaling,
                outflow_proportion=model_setup.outflow_proportion,
            )
            if model_setup.catchment_outflow:
                catchment_outflow += current_iteration_outwater
        lat_end = time.perf_counter()
        start = time.perf_counter()
        check_grid_correctness(grid)

        print_model_end_of_timestep_messages(
            grid,
            day,
            total_mass_start,
            snow_added,
            catchment_outflow,
            tic,
            model_setup,
        )
        msg_end = time.perf_counter()
        print(f"Lateral movement time: {lat_end - lat_start:.2f}s")
        print(f"Checking grid consistency time: {msg_end - start:.2f}s")
        print(f"Diagnostic messages time: {time.perf_counter() - start:.2f}s")
        start = time.perf_counter()
        met_start_idx = (day + 1) * model_setup.t_steps_per_day % met_data_len
        met_end_idx = (day + 2) * model_setup.t_steps_per_day % met_data_len
        if met_start_idx > met_end_idx:
            met_end_idx = met_data_len
        met_data_grid, met_data_len, snow_added = update_met_conditions(
            model_setup,
            grid,
            met_start_idx,
            met_end_idx,
            snow_added=snow_added,
        )
        print(f"Updating met data time: {time.perf_counter() - start:.2f}s")
        start = time.perf_counter()
        if model_setup.dump_data and day % model_setup.dump_timestep == 0:
            print(f"Dumping model state to {model_setup.dump_filepath}...")
            if model_setup.dump_format == "NETCDF4":
                dump_state(
                    model_setup.dump_filepath, grid, met_start_idx, met_end_idx
                )
            elif model_setup.dump_format == "pickle":
                with open(model_setup.dump_filepath, "wb") as outfile:
                    pickle.dump(grid, outfile)
            print(
                f"Dumping model state time: {time.perf_counter() - start:.2f}s"
            )
        start = time.perf_counter()
        if model_setup.save_output and day % model_setup.output_timestep == 0:
            output_counter += 1
            update_model_output(
                model_setup.output_filepath,
                grid,
                output_counter,
                vars_to_save=model_setup.vars_to_save,
                vert_grid_size=output_grid_size,
            )
        print(f"Updating model output time: {time.perf_counter() - start:.2f}s")
        print(f"Serial time total: {time.perf_counter() - start_serial:.2f}s")
        print(
            f"Total time for day {day + 1}:"
            f" {time.perf_counter() - timestep_start:.2f}s"
        )
    print("\n*******************************************\n")
    print("MONARCHS has finished running successfully!")
    print("Total time taken = ", time.perf_counter() - tic)
    return grid


def initialise_model_data(model_setup):
    """
    Wrapper function that calls various initialisation functions to set up
    MONARCHS.
    """

    # Load in the initial firn profile, either from a whole dem_utils, or a
    # user-defined subset
    if (
        hasattr(model_setup, "lat_bounds")
        and model_setup.lat_bounds.lower() == "dem"
    ):
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
        firn_temperature, rho, firn_depth, valid_cells, dx, dy, _, _ = (
            initial_conditions.initialise_firn_profile(
                model_setup, diagnostic_plots=model_setup.dem_diagnostic_plots
            )
        )
        lat_array = (
            np.zeros((model_setup.row_amount, model_setup.col_amount)) * np.nan
        )
        lon_array = (
            np.zeros((model_setup.row_amount, model_setup.col_amount)) * np.nan
        )

    # Set up meteorological data, from either ERA5-format input ("ERA5") or
    # user-defined values from their model configuration ("user_defined")
    if model_setup.met_data_source == "ERA5":
        if model_setup.load_precalculated_met_data:
            print(
                "monarchs.core.driver.initialise: Loading in pre-calculated"
                " MONARCHS format met data"
            )
        else:
            setup_met_data.met_data_from_era5(model_setup, lat_array, lon_array)
    elif model_setup.met_data_source == "user_defined":
        setup_met_data.prescribed_met_data(model_setup)

    # Write all of the initial ice shelf values into the model grid
    grid = initial_conditions.create_model_grid(
        model_setup.row_amount,
        model_setup.col_amount,
        firn_depth,
        model_setup.vertical_points_firn,
        model_setup.vertical_points_lake,
        model_setup.vertical_points_lid,
        rho,
        firn_temperature,
        use_numba=model_setup.use_numba,
        valid_cells=valid_cells,
        lats=lat_array,
        lons=lon_array,
        size_dx=dx,
        size_dy=dy,
    )
    return grid


def monarchs():
    """
    Main function for running MONARCHS.
    This works a level above initialise, which handles the initial setup
    of the model configuration (as opposed to initialising the model data).
    """

    # Check the environment to see if we have MPI enabled.
    if os.environ.get("MONARCHS_MPI", None) is not None:
        mpi = True
        print("Setting MPI to True")
    else:
        mpi = False
    # If so, we need to get the model setup path from the environment
    # variable rather than loading it in, as it will otherwise attempt
    # to do so for every single MPI process.
    # TODO - obviate the need for this by making it run on proc 0 only
    if mpi:
        if os.environ.get("MONARCHS_MODEL_SETUP_PATH") is not None:
            model_setup_path = os.environ.get("MONARCHS_MODEL_SETUP_PATH")
        else:
            model_setup_path = "model_setup.py"
    # If not using MPI, just parse the input arguments and
    # load in the model setup path
    else:
        model_setup_path = configuration.parse_args()
    model_setup = get_model_setup(model_setup_path)

    # If we are using Numba optimisation, we need to ensure that
    # all physics functions are JIT-compiled before we start the model run.
    # use_numba may or may not be a model_setup attribute, so ignore
    # the linter errors here as it is entirely optional
    # pylint: disable=no-member
    if hasattr(model_setup, "use_numba") and model_setup.use_numba:
        configuration.jit_modules()
    # pylint: enable=no-member

    # Model configuration steps
    configuration.create_output_folders(model_setup)
    configuration.handle_incompatible_flags(model_setup)
    configuration.handle_invalid_values(model_setup)
    configuration.create_defaults_for_missing_flags(model_setup)

    # Set up the data, then run the model physics.
    grid = initialise_model_data(model_setup)
    grid = main(model_setup, grid)
    return grid
