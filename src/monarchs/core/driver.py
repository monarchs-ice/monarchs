"""
Core functions used in the running of MONARCHS.
This module contains the core functions that drive the code
when it is executed.

When the code is run, monarchs() loads and validates the configuration and
initial data, then calls run_model(), the model time loop. Each model day
("iteration"), run_model calls core.loop_over_grid for the single-column
physics (which loops over each timestep, by default 1 hour), then the
lateral movement functions, and handles saving the data - both the model
state (also known as a "dump"), and the variables that the user wants to
track over time.
"""

import time
import logging
import numpy as np
import pathos
from monarchs.core import configuration, kernels
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

from monarchs.physics import lateral_movement
from monarchs.met_data.load import update_met_conditions, met_window
from monarchs.core.initialise import check_for_reload_from_dump, initialise_model_data
from monarchs.core.diagnostics import (
    print_model_end_of_timestep_messages,
)

logger = logging.getLogger(__name__)

# dummy init value for Dask client which may be used later - this needs to be
# global
CLIENT = None


class Timer:
    """Context manager that prints '<label> time: X.XXs' on exit."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        print(f"{self.label} time: {time.perf_counter() - self.start:.2f}s")
        return False


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
        "perc_time_toggle": model_setup.perc_time_toggle,
        "densification_toggle": model_setup.densification_toggle,
        "ignore_errors": model_setup.ignore_errors,
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


def setup_parallelism(model_setup):
    """
    Select and prepare the grid-loop implementation for this run.

    Numba mode compiles the prange-based loop with the user's parallel flag;
    otherwise the Dask-based loop is used, with a distributed Client created
    if requested (stored in the module-global CLIENT, which process_chunk
    workers read).
    """
    # pylint: disable=import-outside-toplevel
    # dask path
    if (
        model_setup.parallel
        and model_setup.dask_scheduler == "distributed"
        and not model_setup.use_numba
    ):
        print("Setting up Dask Client object...")
        # only import dask.distributed if we need it - this avoids requiring
        # dask.distributed to be installed for the model to run
        from dask.distributed import Client  # pylint: disable=no-name-in-module

        global CLIENT  # pylint: disable=global-statement
        CLIENT = Client()
    # numba path
    if model_setup.use_numba:
        from monarchs.core.Numba.loop_over_grid import loop_over_grid_numba
        from numba import njit, set_num_threads

        if model_setup.cores in ["all", False]:
            cores = pathos.helpers.cpu_count()
        else:
            cores = model_setup.cores
        set_num_threads(int(cores))
        loop_over_grid = njit(parallel=model_setup.parallel)(loop_over_grid_numba)
    else:
        from monarchs.core.loop_over_grid import loop_over_grid
    # pylint: enable=import-outside-toplevel
    return loop_over_grid


def single_column_step(
    grid, loop_over_grid, met_data_grid, dt, model_setup, toggle_dict, cores
):
    """
    Run one day of single-column physics over the whole grid, then check for
    per-cell errors and verify that every valid cell was visited exactly once.
    """
    # pre-flatten and rearrange met_data_grid from
    # (t_steps_per_day, rows, cols) to (rows*cols, t_steps_per_day)
    met_data_grid = met_data_grid.reshape(model_setup.t_steps_per_day, -1)
    met_data_grid = np.moveaxis(met_data_grid, 0, -1)

    visit_grid = np.copy(grid["visit_count"])
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

    if check_for_single_column_errors(grid):
        raise RuntimeError(
            "monarchs.core.driver.single_column_step: Error flag raised during"
            " single-column physics step. See logs for details."
        )
    validate_visits(grid, visit_grid)
    print("Single-column physics finished")
    return grid


def validate_visits(grid, visit_grid):
    """Check that each valid cell was visited exactly once this day.
    This ensures that the model does not silently give incorrect results if
    a gridcell exits early due to e.g. a numerical error"""
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
        raise ValueError("Cells not being visited in single-column physics step")


def lateral_movement_step(grid, model_setup):
    """
    Run the daily lateral water movement. Returns the updated grid and the
    water lost from the catchment this day (0 unless catchment_outflow).
    """
    print("Moving water laterally...")
    grid, out_water = lateral_movement.move_water(
        grid,
        model_setup.row_amount,
        model_setup.col_amount,
        model_setup.lateral_timestep,
        catchment_outflow=model_setup.catchment_outflow,
        flow_into_land=model_setup.flow_into_land,
        lateral_movement_percolation_toggle=(
            model_setup.lateral_movement_percolation_toggle
        ),
        flow_speed_scaling=model_setup.flow_speed_scaling,
        outflow_proportion=model_setup.outflow_proportion,
    )
    if not model_setup.catchment_outflow:
        out_water = 0
    return grid, out_water


def write_outputs(
    model_setup, grid, day, output_counter, output_grid_size, met_start_idx, met_end_idx
):
    """
    End-of-day writes: restart checkpoint, time-series output, and any extra
    numbered checkpoints. Returns the updated output counter.
    """
    if model_setup.dump_data and day % model_setup.dump_timestep == 0:
        print(f"Dumping model state to {model_setup.dump_filepath}...")
        with Timer("Dumping model state"):
            write_checkpoint(
                model_setup.dump_filepath,
                grid,
                met_start_idx,
                met_end_idx,
                model_setup=model_setup,
            )

    if model_setup.save_output and day % model_setup.output_timestep == 0:
        with Timer("Updating model output"):
            output_counter += 1
            append_output(
                model_setup.output_filepath,
                grid,
                output_counter,
                vars_to_save=model_setup.vars_to_save,
                vert_grid_size=output_grid_size,
            )

    if (
        model_setup.dump_data
        and model_setup.dump_checkpoint_frequency
        and day % model_setup.dump_checkpoint_frequency == 0
    ):
        print(f"Writing model state as an extra checkpoint at timestep {day}")
        write_checkpoint(
            model_setup.dump_filepath + str(day),
            grid,
            met_start_idx,
            met_end_idx,
            model_setup=model_setup,
        )
    return output_counter


def run_model(model_setup, grid):
    """
    The model time loop. Each day this calls loop_over_grid (which in turn
    calls timestep_loop for the single-column physics, <t_steps_per_day>
    times), then lateral_movement (i.e. water movement), then handles
    checkpointing and output via netCDF.

    Parameters
    ----------
    model_setup : ModelSetup
        The loaded model configuration (see monarchs.core.load_model_setup).
    grid : numpy structured array
        Model grid, containing the data specified in get_spec() of
        monarchs.core.model_grid.

    Returns
    -------
    grid : numpy structured array
        Model grid at the end of the run.
    """
    loop_over_grid = setup_parallelism(model_setup)

    tic = time.perf_counter()
    met_start_idx = 0
    met_end_idx = model_setup.t_steps_per_day
    output_counter = 0
    # vertical grid size to interpolate vector outputs onto (native if unset)
    output_grid_size = model_setup.output_grid_size or grid["vert_grid"][0][0]

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
            model_setup=model_setup,
        )
    if reload_dump_success:
        output_counter = first_iteration

    snow_added = 0
    met_data_grid, met_data_len, snow_added = update_met_conditions(
        model_setup,
        grid,
        met_start_idx,
        met_end_idx,
        start=reload_dump_success,
        snow_added=snow_added,
    )
    toggle_dict = setup_toggle_dict(model_setup)
    cores = get_num_cores(model_setup)
    total_mass_start = calc_grid_mass(grid)
    catchment_outflow = 0
    # seconds per single-column timestep (3600 for the default 24 steps/day)
    dt = int(86400 / model_setup.t_steps_per_day)

    for day in range(first_iteration, model_setup.num_days):
        day_start = time.perf_counter()
        print("\n*******************************************\n")
        print(f"Start of model day {day + 1}\n")

        if model_setup.single_column_toggle:
            with Timer("Single column physics"):
                grid = single_column_step(
                    grid,
                    loop_over_grid,
                    met_data_grid,
                    dt,
                    model_setup,
                    toggle_dict,
                    cores,
                )

        serial_start = time.perf_counter()
        if model_setup.dump_data_pre_lateral_movement:
            write_checkpoint(
                model_setup.dump_filepath,
                grid,
                met_start_idx,
                met_end_idx,
                model_setup=model_setup,
            )

        if model_setup.lateral_movement_toggle:
            with Timer("Lateral movement"):
                grid, out_water = lateral_movement_step(grid, model_setup)
                catchment_outflow += out_water

        with Timer("Checking grid consistency"):
            check_grid_correctness(grid)
        with Timer("Diagnostic messages"):
            print_model_end_of_timestep_messages(
                grid,
                day,
                total_mass_start,
                snow_added,
                catchment_outflow,
                tic,
                model_setup,
            )

        with Timer("Updating met data"):
            met_start_idx, met_end_idx = met_window(
                day + 1, model_setup.t_steps_per_day, met_data_len
            )
            met_data_grid, met_data_len, snow_added = update_met_conditions(
                model_setup,
                grid,
                met_start_idx,
                met_end_idx,
                snow_added=snow_added,
            )

        output_counter = write_outputs(
            model_setup,
            grid,
            day,
            output_counter,
            output_grid_size,
            met_start_idx,
            met_end_idx,
        )
        print(f"Serial time total: {time.perf_counter() - serial_start:.2f}s")
        print(f"Total time for day {day + 1}: {time.perf_counter() - day_start:.2f}s")

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

    model_setup_path = configuration.parse_args()
    model_setup = get_model_setup(model_setup_path)

    # Compile the registered @kernel functions before the model run (a no-op
    # when use_numba is False). getattr, as defaults have not yet been applied
    # so use_numba may or may not be a model_setup attribute at this point.
    kernels.compile_all(getattr(model_setup, "use_numba", False))

    # Model configuration steps
    configuration.create_output_folders(model_setup)
    configuration.handle_incompatible_flags(model_setup)
    configuration.handle_invalid_values(model_setup)
    configuration.create_defaults_for_missing_flags(model_setup)

    # Set up the data, then run the model physics.
    grid = initialise_model_data(model_setup)
    grid = run_model(model_setup, grid)
    return grid
