"""
Lateral water movement orchestrator for MONARCHS.

Water flows between grid cells based on differences in water level. Water will move
from high points to low points, to and from all grid cells except those with
frozen lids.

To make water move simultaneously, move_water has three passes. The first
calculates the initial state and water levels, the second reads this and
determines which water would move *from* the initial cell, and the final
goes through and applies these movements to the grid.
"""

import numpy as np
from monarchs.core.kernels import kernel, prange
from monarchs.physics.firn.percolation import percolate, calc_saturation
from monarchs.physics.lateral.water_level import update_water_level
from monarchs.physics.lateral.neighbours import find_biggest_neighbour
from monarchs.physics.lateral.transfer import move_to_neighbours


@kernel()
def move_water(
    grid,
    max_grid_row,
    max_grid_col,
    timestep,
    catchment_outflow=True,
    lateral_movement_percolation_toggle=True,
    flow_into_land=False,
    flow_speed_scaling=1.0,
    outflow_proportion=1.0,
):
    """
    Loop over the model grid and determine which cells water is allowed to move
    from/to, then perform this movement simultaneously.
     This is the main function used to perform the water movement.

    This movement can be from a lake to a lake, from a lake to an ice lens,
    or from an ice lens to another ice lens.
    In the case that an ice lens is present, water will move to and from the
    lowest point in the firn that is not saturated.

    At the end of the algorithm, a percolation step (without refreezing) is
    run, so that water ends up in the correct places in the column.
    Similarly, a pass of `calc_saturation` is also run, to ensure that cells
    that are over-full with water pool upward, so we don't end up with an
    unphysical state.

    Parameters
    ----------
    grid : np.ndarray
        Numpy structured array containing the model grid.
    max_grid_row : int
        Total number of grid rows in *grid* (used for bounds checking).
    max_grid_col : int
        Total number of grid columns (used for bounds checking).
    timestep : int
        Time taken for the lateral movement to occur. This is typically
        24 hours, i.e. 3600 * 24 seconds. [s]
    lateral_movement_percolation_toggle : bool, optional
        Boolean flag to determine whether to perform a percolation step after
        we do lateral water, to ensure that water is where it should be in the
        firn column.

    Returns
    -------
    grid : amended version of the model grid with water having moved laterally

    Raises
    ------
    ValueError
        If the total amount of water is not conserved, then the algorithm is
        not working correctly (as we are losing mass), so an unphysical state
        has been reached.
        This conservation calculation accounts for water lost to catchment
        outflow.
    """

    # Set up some values used later
    total_water = 0
    catchment_out_water = 0
    new_water = 0

    # Sanitise inputs as we expect floats - primarily so Numba doesn't complain
    # about types if a user specifies e.g. 1.
    flow_speed_scaling = float(flow_speed_scaling)
    outflow_proportion = float(outflow_proportion)

    # Perform some housekeeping. This consists of:
    # a) determining the water level in each cell,
    # b) creating temporary cells for our water to move in and out of,
    # so it can be performed simultaneously in parallel.
    dtype = grid.dtype
    temp_grid = np.zeros((len(grid), len(grid[0])), dtype=dtype)

    # First pass - work out water level differences between the grid cells
    for row in prange(max_grid_row):
        for col in range(max_grid_col):
            cell = grid[row][col]
            update_water_level(cell)
            total_water += np.sum(cell["water"]) + cell["lake_depth"]
            cell["water_direction"][:] = 0

            # Snapshot relevant fields into temp_grid so Phase 2 can read the
            # pre-movement state while writing proposed movements.
            temp_grid[row, col]["lake_depth"] = cell["lake_depth"]
            temp_grid[row, col]["lake"] = cell["lake"]
            temp_grid[row, col]["valid_cell"] = cell["valid_cell"]
            # Explicit level loops: avoids a shallow copy and is preferred by
            # Numba's auto-paralleliser.
            for level in range(len(grid[row, col]["water"])):
                temp_grid[row, col]["water"][level] = cell["water"][level]
                temp_grid[row, col]["saturation"][level] = cell["saturation"][level]
                temp_grid[row, col]["meltflag"][level] = cell["meltflag"][level]

    # Second pass - determine where water is moving to and from, and how much
    # to move. This and the prior loop can't be merged, as the water level will change
    # as water is added to it if not, particularly if the loops are parallelised.
    for row in prange(max_grid_row):
        for col in range(max_grid_col):
            cell = grid[row][col]
            if cell["valid_cell"] and (
                cell["ice_lens"]
                and (cell["water"] > 0).any()
                or cell["lake"]
                and cell["lake_depth"] > 0
            ):
                biggest_height_difference, max_dirs = find_biggest_neighbour(
                    cell,
                    grid,
                    col,
                    row,
                    max_grid_col,
                    max_grid_row,
                    catchment_outflow,
                    flow_into_land=flow_into_land,
                )
                split = len(max_dirs)
                if biggest_height_difference > 0:
                    catchment_out_water += move_to_neighbours(
                        grid,
                        temp_grid,
                        max_dirs,
                        col,
                        row,
                        split,
                        catchment_outflow,
                        biggest_height_difference,
                        timestep,
                        flow_into_land=flow_into_land,
                        flow_speed_scaling=flow_speed_scaling,
                        outflow_proportion=outflow_proportion,
                    )

    # Now we have the values calculated in our temporary grid,
    # update the values of <grid>, i.e. performing our movement in one step.
    # TODO - Not parallelsed at the moment since race conditions are possible
    # in this step. Need to rework to a different scheme to make parallel.
    for row in range(max_grid_row):
        for col in range(0, max_grid_col):
            cell = grid[row][col]
            temporary_cell = temp_grid[row][col]

            # Round water and lake depth to 10 decimal places, which gives a
            # bit of robustness to floating-point errors. This is only an issue
            # in a case where we expect some symmetry, e.g. the idealised
            # 10x10 Gaussian lakes test case.
            cell["water"] = np.around(temporary_cell["water"], 10)
            cell["lake_depth"] = np.around(temporary_cell["lake_depth"], 10)
            cell["saturation"] = temporary_cell["saturation"]
            cell["meltflag"] = temporary_cell["meltflag"]
            new_water += np.sum(cell["water"]) + cell["lake_depth"]
            # Sometimes numerical noise can result in water being negative.
            # In this case, if it is within some tolerance, we can accept it
            # as noise, else an error is raised.
            tol = -1e-8
            if (cell["water"] < tol).any():
                print(cell["water"])
                print(cell["row"], cell["column"])
                print(cell["valid_cell"])
                print(cell["lake"])
                if cell["valid_cell"]:
                    print(cell["water"])
                    print(np.where(cell["water"] < 0))
                    raise ValueError("cell.water is negative")
                else:  # if in an invalid cell, we don't care, just zero it
                    cell["water"][:] = 0
            if (cell["water"] < 0).any():
                set_to_zeros = np.where(cell["water"] < 0)
                cell["water"][set_to_zeros] = 0

            if cell["valid_cell"]:
                # Once all water has moved, update the Lfrac of each cell based
                # on where the water now is.
                # The water level is calculated at the beginning of the next
                # call to move_water, and is not used elsewhere in the code.
                cell["Lfrac"] = cell["water"] / (cell["firn_depth"] / cell["vert_grid"])
                # We have put all the water at one level in the firn -
                # we need to percolate it to make sure that
                # the water fills out the available pore space.
                if lateral_movement_percolation_toggle:
                    # assume the percolation happens within 1 hour.
                    percolate(cell, 3600, lateral_refreeze_flag=True)
                    # Perform an extra saturation calculation so we don't
                    # end up with unphysical liquid fraction
                    for k in np.arange(cell["vert_grid"])[::-1]:
                        calc_saturation(cell, k, end=True)

    print("\nLateral water movement diagnostics:")
    print("Starting water total = ", total_water)
    print("Finishing water total = ", new_water)
    print("Difference in water total = ", total_water - new_water)
    if catchment_outflow:
        print("Catchment outflow = ", catchment_out_water)
    # Raise an error if we "create" or lose water from somewhere.
    # This is arbitrarily set to 0.01 for now - it should be nonzero as
    # otherwise it will trigger when we have e.g. floating point errors.
    if abs(total_water - new_water - catchment_out_water) > 0.0001:
        raise ValueError(
            "monarchs.physics.lateral_functions.move_water:Too much water has"
            " appeared out of nowhere during lateral functions"
        )
    return grid, catchment_out_water
