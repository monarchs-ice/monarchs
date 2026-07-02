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
import time
import pickle
import logging
import numpy as np
import pathos
from netCDF4 import Dataset  # pylint: disable=no-name-in-module
from monarchs.core import configuration
from monarchs.core.load_model_setup import get_model_setup
from monarchs.io import write_checkpoint, initialise_output, append_output
from monarchs.core.utils import (
    get_num_cores,
)
from monarchs.core.error_handling import (
    calc_grid_mass,
    check_grid_correctness,
    check_for_single_column_errors,
)

from monarchs.met_data.met_data_grid import initialise_met_data, get_spec
from monarchs.physics import lateral_movement
from monarchs.met_data.index_map import apply_index_map_expand
from monarchs.core.initialise import check_for_reload_from_dump, initialise_model_data
from monarchs.core.diagnostics import (
    print_model_end_of_timestep_messages,
)

logger = logging.getLogger(__name__)

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


def get_snow_sum(met_data_grid, grid, met_start_idx, met_end_idx, snow_added):
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

        snow_added = get_snow_sum(
            met_data_grid, grid, met_start_idx, met_end_idx, snow_added
        )

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
                        "monarchs.core.driver.main: met_end_idx > days *"
                        " hours, i.e. your grid of meteorological data is too"
                        " small for the number of timesteps you wish to run"
                    )

    return met_data_grid, met_data_len, snow_added


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

    """ Initialise parallelisation if required """
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
    # set up some helper variables

    tic = time.perf_counter()
    met_start_idx = 0
    met_end_idx = model_setup.t_steps_per_day
    output_counter = 0

    """ Check for dump file and setup model output files """

    # determine if we need to interpolate vector values (e.g. Sfrac)
    # in the output
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
    ) = check_for_reload_from_dump(model_setup, grid, met_start_idx, met_end_idx)

    if model_setup.save_output and not reload_dump_success:
        initialise_output(
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

    """ Load in meteorological data """
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

    """ Main model time loop """

    for day in time_loop:
        timestep_start = time.perf_counter()
        print("\n*******************************************\n")
        print(f"Start of model day {day + 1}\n")

        # pre-flatten and rearrange met_data_grid
        met_data_grid = met_data_grid.reshape(24, -1)
        met_data_grid = np.moveaxis(
            met_data_grid, 0, -1
        )  # move the first axis to the last axis

        """ Single-column physics """
        # Check that we access each cell exactly once during the single-column physics step
        visit_grid = np.copy(grid["visit_count"])
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

        """Check for any errors in the grid after the single-column physics step"""
        errflag = check_for_single_column_errors(grid)

        if errflag:
            raise RuntimeError(
                "monarchs.core.driver.main: Error flag raised during single-"
                "column physics step. See logs for details."
            )

        # Validation - check that valid cells are actually being
        # operated upon during the single-column physics step
        if model_setup.single_column_toggle:
            visit_flag = False
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if grid[i][j]["valid_cell"] and (
                        grid[i][j]["visit_count"] != visit_grid[i][j] + 1
                    ):
                        logger.error(
                            "i = %s j = %s old visit count = %s new visit count = %s",
                            i,
                            j,
                            visit_grid[i][j],
                            grid[i][j]["visit_count"],
                        )
                        visit_flag = True
            if visit_flag:
                raise ValueError(
                    "Cells not being visited in single-column physics step"
                )

            print("Single-column physics finished")
            print(f"Single column physics time: {time.perf_counter() - start:.2f}s")

        start_serial = time.perf_counter()
        if model_setup.dump_data_pre_lateral_movement:
            if model_setup.dump_format == "NETCDF4":
                write_checkpoint(
                    model_setup.dump_filepath, grid, met_start_idx, met_end_idx
                )
            elif model_setup.dump_format == "pickle":
                with open(model_setup.dump_filepath, "wb") as outfile:
                    pickle.dump(grid, outfile)

        """ Lateral movement """
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

        """ Validation checks and end-of-timestep messages """
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

        """ Update meteorological data for next day """
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

        """ Dump model state """
        start = time.perf_counter()
        if model_setup.dump_data and day % model_setup.dump_timestep == 0:
            print(f"Dumping model state to {model_setup.dump_filepath}...")
            if model_setup.dump_format == "NETCDF4":
                write_checkpoint(
                    model_setup.dump_filepath, grid, met_start_idx, met_end_idx
                )
            elif model_setup.dump_format == "pickle":
                with open(model_setup.dump_filepath, "wb") as outfile:
                    pickle.dump(grid, outfile)
            print(f"Dumping model state time: {time.perf_counter() - start:.2f}s")

        """ Model output """
        start = time.perf_counter()
        if model_setup.save_output and day % model_setup.output_timestep == 0:
            output_counter += 1
            append_output(
                model_setup.output_filepath,
                grid,
                output_counter,
                vars_to_save=model_setup.vars_to_save,
                vert_grid_size=output_grid_size,
            )
        print(f"Updating model output time: {time.perf_counter() - start:.2f}s")
        print(f"Serial time total: {time.perf_counter() - start_serial:.2f}s")
        print(
            f"Total time for day {day + 1}: {time.perf_counter() - timestep_start:.2f}s"
        )
        """ Extra checkpointing """
        if model_setup.dump_data:
            if model_setup.dump_checkpoint_frequency:
                dump_checkpoints = np.arange(
                    0,
                    model_setup.num_days,
                    model_setup.dump_checkpoint_frequency,
                )
                print(f"Writing model state as an extra checkpoint at timestep {day}")
                if day in dump_checkpoints:
                    write_checkpoint(
                        model_setup.dump_filepath + str(day),
                        grid,
                        met_start_idx,
                        met_end_idx,
                    )
    print("\n*******************************************\n")
    print("MONARCHS has finished running successfully!")
    print("Total time taken = ", time.perf_counter() - tic)
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
