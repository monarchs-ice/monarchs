"""
Diagnostic and progress-reporting functions for the MONARCHS driver.

Contains functions that print end-of-timestep summaries and provide
visual consistency checks between the DEM and met-data grids.
"""

import sys
import time
import numpy as np
from monarchs.core.utils import get_2d_grid
from monarchs.core.error_handling import calc_grid_mass


def check_firn_met_consistency(grid, met_data_grid):
    """
    Check visually that the meteorological data input is mapped to the
    DEM input.
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
    if np.isnan(calc_grid_mass(grid)):
        raise ValueError(
            "Total mass of grid is NaN. This likely indicates that in the single-column physics "
            "a variable has become undefined due to a divide-by-zero. Check the logs for more details."
        )
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
    print("Firn depth = ", get_2d_grid(grid, "firn_depth", mask_invalid=True))
    print("Lake depth = ", get_2d_grid(grid, "lake_depth", mask_invalid=True))
    print("Lid depth = ", get_2d_grid(grid, "lid_depth", mask_invalid=True))
    print("Number of lakes = ", np.sum(get_2d_grid(grid, "lake")))
    print("Number of lids = ", np.sum(get_2d_grid(grid, "lid")))
    print("Max lake depth = ", np.max(get_2d_grid(grid, "lake_depth")))
    print("Max lid depth = ", np.max(get_2d_grid(grid, "lid_depth")))
    # ensure that output is flushed to the console immediately rather than
    # being buffered.
    # Mostly a fix for output not updating when running with Slurm.
    sys.stdout.flush()
