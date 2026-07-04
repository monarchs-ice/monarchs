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
from netCDF4 import Dataset  # pylint: disable=no-name-in-module
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
        "spinup": model_setup.spinup,
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
                        "monarchs.core.driver.update_met_conditions:"
                        " met_end_idx > days *"
                        " hours, i.e. your grid of meteorological data is too"
                        " small for the number of timesteps you wish to run"
                    )

    return met_data_grid, met_data_len, snow_added


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
