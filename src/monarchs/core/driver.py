"""
Core functions used in the running of MONARCHS.
This module contains the core functions that drive the code
when it is executed.

When the code is run, it calls main(), which handles loading in the data,
calling core.loop_over_grid every model day (interchangeably called "iteration".
The loop over each timestep (by default 1 hour) is handled
in the single-column physics steps.
main then calls the lateral movement functions, and handles saving the data,
both in terms of the model state (also known as a "dump"), and the
variables that the user wants to track over time.
"""

# Import everything here. This is so that we don't then do the same steps for every parallel process for the first
# iteration (day)!

import os
import time
import numpy as np
import pathos
from netCDF4 import Dataset
from monarchs.core import configuration, initial_conditions, setup_met_data
from monarchs.core.dump_model_state import dump_state, reload_from_dump
from monarchs.core.model_output import setup_output, update_model_output
from monarchs.core.loop_over_grid import loop_over_grid
from monarchs.core.utils import get_2d_grid, calc_grid_mass, check_correct
from monarchs.met_data.metdata_class import initialise_met_data_grid
from monarchs.physics import lateral_functions

model_setup = configuration.model_setup

# if we want to use Numba - apply just-in-time compilation to the relevant
# functions
if hasattr(model_setup, 'use_numba') and model_setup.use_numba:
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


def setup_toggle_dict(model_setup):
    """
    Set up a dictionary of switches to determine the running of the model.
    These are accessed by each thread, so we need to set up a new object to hold these
    else we will run into errors. Additionally, the ModelSetup class is not a jitclass
    (and cannot be dynamically set to be one), so will not work with Numba.
    We therefore need a numba.typed.Dict object in this instance.

    Parameters
    ----------
    model_setup

    Returns
    -------

    """

    if model_setup.use_numba:
        import numba

        toggle_dict = numba.typed.Dict()
    else:
        toggle_dict = {}

    toggle_dict["parallel"] = model_setup.parallel
    toggle_dict["use_numba"] = model_setup.use_numba
    toggle_dict["snowfall_toggle"] = model_setup.snowfall_toggle
    toggle_dict["firn_column_toggle"] = model_setup.firn_column_toggle
    toggle_dict["firn_heat_toggle"] = model_setup.firn_heat_toggle
    toggle_dict["lake_development_toggle"] = model_setup.lake_development_toggle
    toggle_dict["lid_development_toggle"] = model_setup.lid_development_toggle
    toggle_dict["percolation_toggle"] = model_setup.percolation_toggle
    toggle_dict["spinup"] = model_setup.spinup
    toggle_dict["perc_time_toggle"] = model_setup.perc_time_toggle
    toggle_dict["densification_toggle"] = model_setup.densification_toggle
    toggle_dict["ignore_errors"] = model_setup.ignore_errors
    toggle_dict['use_mpi'] = model_setup.use_mpi

    return toggle_dict


def check_for_reload_from_dump(model_setup, grid, met_start_idx, met_end_idx):
    """
    Determine if the model needs to re-initialise parameters from a dump file.
    TODO - add support for reloading from pickle

    Parameters
    ----------
    model_setup
    grid
    met_start_idx
    met_end_idx

    Returns
    -------

    """
    if hasattr(model_setup, "dump_filepath"):
        reload_name = model_setup.dump_filepath
    else:
        reload_name = ""

    import warnings

    # If we are reloading from a failed run, then
    if model_setup.reload_from_dump:
        print("Reloading state from dump...")
        # If we have reload_from_dump specified but no dumpfile, warn the user
        if not os.path.exists(reload_name):
            first_iteration = 0
            warnings.warn(
                f"Reload/dump filepath {reload_name} does not exist - instead starting model from scratch. "
                f"If you believe you do have a dump file, check that it is specified correctly in model_setup.py."
            )
            reload_dump_success = False

        # Otherwise load in parameters from the dumpfile
        else:
            grid, met_start_idx, met_end_idx, first_iteration = reload_from_dump(
                reload_name, grid
            )
            print(
                f"Loading model state from dump file {reload_name} - first iteration = ",
                first_iteration,
            )
            # In this case - we should already have an output file, so we don't need to set it up.
            reload_dump_success = True

    else:
        first_iteration = 0
        reload_dump_success = False
        grid = grid
    # Return everything in both cases - if we haven't reloaded from dump we just return the initial state
    return grid, met_start_idx, met_end_idx, first_iteration, reload_dump_success


def get_num_cores(model_setup):
    """

    Parameters
    ----------
    model_setup

    Returns
    -------

    """
    if model_setup.cores in ["all", False] and model_setup.parallel:
        cores = pathos.helpers.cpu_count()
        print(f"Using all cores - {pathos.helpers.cpu_count()} detected")
    elif not model_setup.parallel:
        cores = 1
    else:
        cores = model_setup.cores
    return cores


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
        # print('Start idx for met data = ', met_start_idx)
        # Initial met conditions
        # call .data so we get a Numpy ndarray, not a numpy masked array. This is crucial for Numba support
        met_data_grid = initialise_met_data_grid(
            model_setup.row_amount,
            model_setup.col_amount,
            met_data.variables["snowfall"][met_start_idx:met_end_idx].data,
            met_data.variables["snow_dens"][met_start_idx:met_end_idx].data,
            met_data.variables["temperature"][met_start_idx:met_end_idx].data,
            met_data.variables["wind"][met_start_idx:met_end_idx].data,
            met_data.variables["pressure"][met_start_idx:met_end_idx].data,
            met_data.variables["dew_point_temperature"][met_start_idx:met_end_idx].data,
            met_data.variables["LW_surf"][met_start_idx:met_end_idx].data,
            met_data.variables["SW_surf"][met_start_idx:met_end_idx].data,
            met_data.variables['cell_latitude'][:].data,
            met_data.variables['cell_longitude'][:].data,
            model_setup.use_numba
        )
        # initialise total mass calculation - check how much snow was added during the next iteration and add it
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j].valid_cell:
                    snow_array = (np.array(met_data.variables["snow_dens"][met_start_idx:met_end_idx, i,j]) *
                    np.array(met_data.variables["snowfall"][met_start_idx:met_end_idx, i, j]))
                    snow_added += np.sum(snow_array)

        # Check to make sure we are not going out of bounds
        for key in met_data.variables.keys():
            if key != 'cell_latitude' and key != 'cell_longitude':
                if met_end_idx > len(met_data[key]):
                    raise IndexError(
                        "monarchs.core.driver.main: met_end_idx > days * hours, i.e. your grid of meteorological data "
                        "is too small for the number of timesteps you wish to run"
                    )
    return met_data_grid, met_data_len, snow_added


def check_firn_met_consistency(grid, met_data_grid):
    from matplotlib import pyplot as plt
    from cartopy import crs as ccrs
    projection = ccrs.PlateCarree()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211, projection=projection)
    ax1.set_title("Original vs regridded met data")
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)
    ax1.contourf(get_2d_grid(grid, 'lon'), get_2d_grid(grid, 'lat'),
                 get_2d_grid(grid, 'firn_depth'),transform=projection)
    ax2 = fig1.add_subplot(212, projection=projection)
    ax2.coastlines()
    ax2.gridlines(draw_labels=True)
    ax2.contourf(
        get_2d_grid(grid, 'lon'),
        get_2d_grid(grid, 'lat'),
        get_2d_grid(met_data_grid, 'temperature'),
        transform=projection
    )

def print_model_end_of_timestep_messages(
    grid, day, total_mass_start, snow_added, catchment_outflow, tic
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
    print(
        f"Total mass at end of iteration {day + 1} = ",
        calc_grid_mass(grid),
    )

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


def main(model_setup, grid):
    """
    Main loop function. This calls loop_over_grid, which in turn calls timestep_loop
    and the other functions (vertical, lake, and lid), a total of <t_steps_per_day>
    times per <day>. This then calls lateral_functions (i.e. water movement) once per <day>.

    This also handles input/output via netCDF.

    Parameters
    ----------
    model_setup : module
        The model setup module imported by ``configuration.py``.
        This contains the following arguments used in this function:

            row_amount : int
                number of rows in the grid of IceShelf objects we want to operate on.
            col_amount : int
                number of columns in the grid of IceShelf objects we want to operate on.
            met_output_filepath : str
                path to netCDF file containing the meteorological data used to drive the model.
            days : int
                Number of days to run in total. [days]
            t_steps : int
                Number of hours to run in each day. This should most likely be 24. [h]
            lat_grid_size : float
                Size of each lateral grid cell, used to determine the lateral flow of water [m]
            timestep : int
                Amount of seconds to run in each t_step, most likely 3600. [s]
            output_filename : str, optional
                Name of the netCDF file used to save model output, if applicable.
            vars_to_save: tuple, optional
                Tuple containing the variables that you want to save from your IceShelf grid. These will be added
                to the output netCDF file defined in output_filename.

    grid : List, or numba.typed.List
        grid of IceShelf objects, each representing a single column.

    Returns
    -------
    grid : List, or numba.typed.List
        grid of IceShelf objects, amended from the original state by the model

    """
    tic = time.perf_counter()
    met_start_idx = 0
    met_end_idx = model_setup.t_steps_per_day
    # very likely 24h, so we get one days worth of data each time
    output_counter = 0  # Index for output file
    # Check if we want to shrink output
    if hasattr(model_setup, "output_grid_size"):
        output_grid_size = model_setup.output_grid_size
        if not output_grid_size:
            output_grid_size = grid[0][0].vert_grid
    else:
        output_grid_size = grid[0][0].vert_grid

    # Check if we are reloading from a failed/stopped run, and load in the data if so
    grid, met_start_idx, met_end_idx, first_iteration, reload_dump_success = (
        check_for_reload_from_dump(model_setup, grid, met_start_idx, met_end_idx)
    )
    print('firn depth = ', get_2d_grid(grid, 'firn_depth'))
    print('valid_cell = ', get_2d_grid(grid, 'valid_cell'))

    # If we haven't successfully read in a dumpfile, then set up the model output (i.e. time series) file if requested.
    if model_setup.save_output and not reload_dump_success:
        setup_output(
            model_setup.output_filepath,
            grid,
            vars_to_save=model_setup.vars_to_save,
            vert_grid_size=output_grid_size,
        )

    # if loading in from a dump file - want to make sure that the met data is consistent with the firn data
    # i.e. don't set start True else we will start from index 0
    if reload_dump_success:
        start = True
    else:
        start = False
    # A few more setup steps - set up a grid of met data, with the correct start indices as read from the
    # dump file, and convert some of the model_setup switches into a dictionary so the model can read it later
    snow_added = 0
    met_data_grid, met_data_len, snow_added = update_met_conditions(
        model_setup, grid, met_start_idx, met_end_idx, start=start, snow_added=snow_added
    )
    toggle_dict = setup_toggle_dict(model_setup)

    # Check how many cores we want to use
    cores = get_num_cores(model_setup)

    # a couple of diagnostics here: starting with the total mass inside the model at the start (in arbitrary units)
    total_mass_start = calc_grid_mass(grid)
    # track water moving out of catchment area - only used if catchment_outflow flag enabled
    catchment_outflow = 0

    time_loop = range(first_iteration, model_setup.num_days)

    # Main model loop here.
    for day in time_loop:
        dt = 3600  # In seconds timestep for each iteration of heat eqn solver
        print("\n*******************************************\n")
        print(f"Start of model day {day + 1}\n")

        # Loop over the model grid. This effectively splits the problem into many
        # single-column problems, and loops over them.
        # It does so in parallel if the relevant model_setup variable is set.
        # The single column calculations are done t_steps_per_day times.
        if model_setup.single_column_toggle:
            if model_setup.parallel and not model_setup.use_numba:
                grid = loop_over_grid(
                    model_setup.row_amount,
                    model_setup.col_amount,
                    grid,
                    dt,
                    met_data_grid,
                    model_setup.t_steps_per_day,
                    toggle_dict,
                    parallel=model_setup.parallel,
                    use_mpi=model_setup.use_mpi,
                    ncores=cores,
                )
            else:
                loop_over_grid(
                    model_setup.row_amount,
                    model_setup.col_amount,
                    grid,
                    dt,
                    met_data_grid,
                    model_setup.t_steps_per_day,
                    toggle_dict,
                    parallel=model_setup.parallel,
                    use_mpi=model_setup.use_mpi,
                    ncores=cores,
                )
        print("Single-column physics finished")
        # Lateral movement of water - this is not an embarrassingly parallel process,
        # and is comparatively very fast, so we do it outside the parallelised block


        # Save progress information as netCDF - this allows us to reload in the file
        # if the code stops for whatever reason and continue from there. This is an
        # intermediate debugging step if the relevant model_setup variable is set to True.
        # This might want to be specified if you have issues with the lateral flow and
        # want to debug, in which case you don't want to have to run the single-column
        # physics all over again.
        if model_setup.dump_data_pre_lateral_movement:
            if model_setup.dump_format == 'NETCDF4':
                dump_state(model_setup.dump_filepath, grid, met_start_idx, met_end_idx)
            elif model_setup.dump_format == 'pickle':
                import pickle
                outfile = open(model_setup.dump_filepath, 'wb')
                pickle.dump(grid, outfile)
                outfile.close()
        # Handle the lateral movement, but only if the toggle is set to True.
        if model_setup.lateral_movement_toggle:
            print("Moving water laterally...")

            grid, current_iteration_outwater = lateral_functions.move_water(
                grid,
                model_setup.row_amount,
                model_setup.col_amount,
                model_setup.lateral_timestep,
                catchment_outflow=model_setup.catchment_outflow,
                flow_into_land=model_setup.flow_into_land,
                lateral_movement_percolation_toggle=model_setup.lateral_movement_percolation_toggle,
            )

            if model_setup.catchment_outflow:
                catchment_outflow += current_iteration_outwater

        # Error checking step
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                check_correct(grid[i][j])

        print_model_end_of_timestep_messages(
            grid, day, total_mass_start, snow_added, catchment_outflow, tic
        )
        # check_energy_conservation(grid) # TODO - WIP

        # Update our met data for the next timestep
        # slice via an index dependent on timesteps per day
        # - we pass this into update_met_conditions.
        met_start_idx = (day + 1) * model_setup.t_steps_per_day % (met_data_len)
        met_end_idx = (day + 2) * model_setup.t_steps_per_day % (met_data_len)
        if met_start_idx > met_end_idx:
            met_end_idx = met_data_len

        met_data_grid, met_data_len, snow_added = update_met_conditions(
            model_setup, grid, met_start_idx, met_end_idx, snow_added=snow_added
        )

        # Save progress information as netCDF - this allows us to reload in the file
        # if the code stops for whatever reason and continue from there.
        if model_setup.dump_data:
            print(f'Dumping model state to {model_setup.dump_filepath}...')
            if model_setup.dump_format == 'NETCDF4':
                dump_state(model_setup.dump_filepath, grid, met_start_idx, met_end_idx)
            elif model_setup.dump_format == 'pickle':
                import pickle
                outfile = open(model_setup.dump_filepath, 'wb')
                pickle.dump(grid, outfile)
                outfile.close()
        # Save model variables as a time series. Do this at the rate desired by the user
        # (i.e. if model_setup.output_timestep = 3, then do this every 3 timesteps).
        # By default, we save at every timestep.

        if model_setup.save_output and day % model_setup.output_timestep == 0:
            output_counter += 1
            update_model_output(
                model_setup.output_filepath,
                grid,
                output_counter,
                vars_to_save=model_setup.vars_to_save,
                vert_grid_size=output_grid_size,
            )

    print("\n*******************************************\n")
    print("MONARCHS has finished running successfully!")
    print("Total time taken = ", time.perf_counter() - tic)
    return grid


def initialise(model_setup):
    # Initialise firn column. Obtain the lat/long boundaries from the DEM if we want to use these.
    if hasattr(model_setup, "lat_bounds") and model_setup.lat_bounds.lower() == "dem":
        T_firn, rho, firn_depth, valid_cells, lat_array, lon_array, dx, dy = (
            initial_conditions.initialise_firn_profile(
                model_setup, diagnostic_plots=model_setup.dem_diagnostic_plots
            )
        )
    else:
        T_firn, rho, firn_depth, valid_cells, dx, dy = (
            initial_conditions.initialise_firn_profile(
                model_setup, diagnostic_plots=model_setup.dem_diagnostic_plots
            )
        )
        lat_array = np.zeros((model_setup.row_amount, model_setup.col_amount)) * np.nan
        lon_array = np.zeros((model_setup.row_amount, model_setup.col_amount)) * np.nan

    # Set up meteorological data and return the path to the grid actually used by MONARCHS
    if model_setup.met_data_source == 'ERA5':
        setup_met_data.setup_era5(model_setup, lat_array, lon_array)
    elif model_setup.met_data_source == 'user_defined':
        setup_met_data.prescribed_met_data(model_setup)


    # Initialise the model grid.
    grid = initial_conditions.create_model_grid(
        model_setup.row_amount,
        model_setup.col_amount,
        firn_depth,
        model_setup.vertical_points_firn,
        model_setup.vertical_points_lake,
        model_setup.vertical_points_lid,
        rho,
        T_firn,
        use_numba=model_setup.use_numba,
        valid_cells=valid_cells,
        lats=lat_array,
        lons=lon_array,
        size_dx=dx,
        size_dy=dy,
    )
    return grid


def monarchs():
    # Load in the model setup file from the filepath defined by the user args, or default 'model_setup.py'
    model_setup = configuration.model_setup
    configuration.create_output_folders(model_setup)
    # Handle incompatible flags in runscript
    configuration.handle_incompatible_flags(model_setup)
    configuration.create_defaults_for_missing_flags(model_setup)

    # Get the initial grid and run the DEM/met data interpolation steps
    grid = initialise(model_setup)

    # Run MONARCHS
    grid = main(model_setup, grid)

    return grid
