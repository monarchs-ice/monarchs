"""
Core functions used in the running of MONARCHS.
This module contains the core functions that drive the code
when it is executed.

When the code is run, monarchs() loads and validates the configuration and
initial data, then calls run_model(), the model time loop. Each model day
("iteration"), run_model calls core.loop_over_grid for the single-column
physics (which loops over each timestep, by default 1 hour), then the
lateral movement functions, and handles saving the data - both the model
state (also known as a "dump" or checkpoint), and the variables that the
user wants to track over time.
"""

import time
import logging
import numpy as np
import pathos
from monarchs.core import configuration, kernels
from monarchs.core.load_model_setup import get_model_setup
from monarchs.config import SETTINGS, configure
from monarchs.io import write_checkpoint, initialise_output, append_output
from monarchs.core.utils import get_num_cores, Timer
from monarchs.core.error_handling import (
    calc_grid_mass,
    check_grid_correctness,
    check_for_single_column_errors,
)

from monarchs.physics import lateral
from monarchs.met_data.load import update_met_conditions, met_window
from monarchs.core.setup_run import check_for_reload_from_dump, initialise_model_data
from monarchs.core.diagnostics import (
    print_model_end_of_timestep_messages,
)

logger = logging.getLogger(__name__)


def setup_toggle_dict(model_setup):
    """
    Set up a dictionary of switches to determine the running of the model.
    These are accessed by each thread, so we need to set up a new object to
    hold these else we will run into errors.

    Parameters
    ----------
    model_setup

    Returns
    -------

    """
    # define toggle switches read into the physics kernels via the settings
    # config.
    toggles = [s.name for s in SETTINGS if s.kernel_toggle]

    toggle_dict = {name: getattr(model_setup, name) for name in toggles}

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

    Numba mode compiles the prange-based loop, parallel over the flattened
    grid when <parallel> is set. Without Numba the pure-Python loop is used,
    which is always serial.
    """
    # pylint: disable=import-outside-toplevel
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
    # timestep_loop bumps visit_count before it checks valid_cell, so cells the
    # loop skips still need theirs incremented
    grid["visit_count"][~grid["valid_cell"]] += 1
    grid = loop_over_grid(
        model_setup.row_amount,
        model_setup.col_amount,
        grid,
        dt,
        met_data_grid,
        model_setup.t_steps_per_day,
        toggle_dict,
        ncores=cores,
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
    bad = grid["valid_cell"] & (grid["visit_count"] != visit_grid + 1)
    if not bad.any():
        return
    # vectorised now rather than explicit double loop as this is not @kernel
    for i, j in np.argwhere(bad):
        logger.error(
            "i = %s j = %s old visit count = %s new visit count = %s",
            i,
            j,
            visit_grid[i][j],
            grid[i][j]["visit_count"],
        )
    raise ValueError("Cells not being visited in single-column physics step")


def lateral_movement_step(grid, model_setup):
    """
    Run the daily lateral water movement. Returns the updated grid and the
    water lost from the catchment this day (0 unless catchment_outflow).
    """
    print("Moving water laterally...")
    grid, out_water = lateral.move_water(
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
    model_setup,
    grid,
    day,
    output_counter,
    output_grid_size,
    met_start_idx,
    met_end_idx,
):
    """
    End-of-day writes: restart checkpoint, time-series output, and any extra
    numbered checkpoints. Returns the updated output counter.
    """
    dumping = model_setup.dump_data and day % model_setup.dump_timestep == 0
    saving = model_setup.save_output and day % model_setup.output_timestep == 0

    if dumping:
        print(f"Dumping model state to {model_setup.dump_filepath}...")
        with Timer("Dumping model state"):
            write_checkpoint(
                model_setup.dump_filepath,
                grid,
                met_start_idx,
                met_end_idx,
                model_setup=model_setup,
            )

    if saving:
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
    times), then lateral movement (i.e. water movement), then handles
    checkpointing and output via netCDF.

    Parameters
    ----------
    model_setup : ModelSetup
        The loaded model configuration (see monarchs.core.load_model_setup).
    grid : numpy structured array
        Model grid, containing the data specified by the variable catalogue
        in monarchs.variables (see build_dtype).

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

    # dump state at the end of the model run regardless of
    # dump frequency (if remainder !=0 wouldnt dump otherwise)
    final_day = model_setup.num_days - 1
    loop_ran = final_day >= first_iteration
    if (
        model_setup.dump_data
        and loop_ran
        and final_day % model_setup.dump_timestep != 0
    ):
        print(f"Writing final model state to {model_setup.dump_filepath}...")
        write_checkpoint(
            model_setup.dump_filepath,
            grid,
            met_start_idx,
            met_end_idx,
            model_setup=model_setup,
        )
    # likewise with output file
    if (
        model_setup.save_output
        and loop_ran
        and final_day % model_setup.output_timestep != 0
    ):
        print(f"Writing final model output to {model_setup.output_filepath}...")
        output_counter += 1
        append_output(
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


def monarchs():
    """
    Main function for running MONARCHS.
    This works a level above initialise, which handles the initial setup
    of the model configuration (as opposed to initialising the model data).
    """

    model_setup_path = configuration.parse_args()
    model_setup = get_model_setup(model_setup_path)

    # Validate the setup and freeze it into an immutable config for the run.
    model_setup = configure(model_setup)

    # Compile the registered @kernel functions before the model run (a no-op
    # when use_numba is False).
    kernels.compile_all(model_setup.use_numba)
    # Create output folders now that filepaths are defined.
    configuration.create_output_folders(model_setup)

    # Set up the data, then run the model physics.
    grid = initialise_model_data(model_setup)
    grid = run_model(model_setup, grid)
    return grid
