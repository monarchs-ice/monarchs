"""
Functions for determining how water moves to and from different types
of gridcells.
"""

from monarchs.core.kernels import kernel
from monarchs.core.utils import find_nearest
from monarchs.physics.lateral.directions import _DIR_ROW, _DIR_COL
from monarchs.physics.lateral.available_water import (
    water_fraction,
    calc_available_water_lake,
    calc_available_water_ice_lens,
    calc_catchment_outflow,
)


@kernel()
def move_from_lake(
    cell,
    grid,
    temporary_cell,
    temporary_neighbour,
    row,
    col,
    n_s_index,
    w_e_index,
    water_frac,
    split,
):
    """
    Move water from a lake source cell into the neighbour in direction
    (n_s_index, w_e_index). The amount is set by the available lake water; the
    destination layer depends on the neighbour's state. Returns the updated
    (temporary_cell, temporary_neighbour).

    Parameters
    ----------
    cell
    grid
    temporary_cell
    temporary_neighbour
    row
    col
    n_s_index
    w_e_index
    water_frac
    split

    Returns
    -------

    """
    neighbour = grid[row + n_s_index][col + w_e_index]
    water_to_move, _, _, _ = calc_available_water_lake(
        cell, water_frac, split, neighbour, outflow=False
    )
    # Simplest case - lake water into lake
    if grid[row + n_s_index][col + w_e_index]["lake"]:
        temporary_neighbour["lake_depth"] += water_to_move

    # In case where neighbour cell is not a lake, and is lower down,
    # then the water moves to the top of that cell
    elif (
        cell["firn_depth"] + cell["lake_depth"]
        > grid[row + n_s_index][col + w_e_index]["firn_depth"]
    ):
        temporary_neighbour["water"][0] += water_to_move
    # Otherwise we need to move it to a specific point in the neighbour cell
    # corresponding to the topmost saturated cell above the ice lens.
    else:
        move_to_index = find_nearest(
            grid[row + n_s_index][col + w_e_index]["vertical_profile"][::-1],
            cell["water_level"],
        )
        temporary_neighbour["water"][move_to_index] += water_to_move

    # Whatever outcome it was, we need to remove the moved water from the lake
    temporary_cell["lake_depth"] -= water_to_move

    # Fix floating-point errors before sanity checking
    if 0 > temporary_cell["lake_depth"] > -1e-12:
        temporary_cell["lake_depth"] = 0
        print("Fixed floating point error in lake depth")
    if temporary_cell["lake_depth"] < 0:
        print("After = ", temporary_cell["lake_depth"])
        print("Before = ", cell["lake_depth"])
        raise ValueError(
            "Moving water has caused lake depth to go below 0 - in the central cell"
        )
    return temporary_cell, temporary_neighbour


@kernel()
def move_from_ice_lens(
    cell,
    grid,
    temporary_cell,
    temporary_neighbour,
    row,
    col,
    n_s_index,
    w_e_index,
    water_frac,
    split,
):
    """
    Move water from an ice-lens source cell into the neighbour in direction
    (n_s_index, w_e_index). The amount and the source layers are set by the
    available ice-lens water. Returns the updated (temporary_cell,
    temporary_neighbour).
    """
    neighbour = grid[row + n_s_index][col + w_e_index]
    (
        water_to_move,
        lowest_water_level,
        move_from_index,
        top_saturation_level,
    ) = calc_available_water_ice_lens(cell, water_frac, split, neighbour, outflow=False)
    # TODO | Rather than using water level, we actually need to account for
    #      | the available pore space.
    #      | This will be something along the lines of calculating
    #      | 1 - Sfrac - Lfrac for the selected indices and weighting based on
    #      | that, rather than just water_depth.
    #      | Should normalise between different timesteps, so not a priority
    #      | right now, but possible for future development.

    # This is the most complicated case - moving from a cell with an ice lens
    # to a cell with or without an ice lens, and that has no lake.
    # We need to move water from the correct vertical layer
    # of cell into the correct vertical layer of neighbour_cell.
    # We do this in the loop after this one, so that we can check that we have
    # enough water to move from each vertical layer of the central cell.
    if not grid[row + n_s_index][col + w_e_index]["lake"]:
        move_to_index = find_nearest(
            grid[row + n_s_index][col + w_e_index]["vertical_profile"][::-1],
            lowest_water_level,
        )
    else:
        move_to_index = 0

    # We now need to update the amount of water in the initial firn column.
    # Water can only be deducted from the area above lowest_water_level.
    for idx in range(top_saturation_level, move_from_index + 1):
        # If more water in cell than we can move, then subtract that amount
        # from the current cell
        # JE - added factor of split so water is evenly moved from the bottom
        if cell["water"][idx] / (split + 1) > water_to_move:
            temporary_cell["water"][idx] -= water_to_move
            if not grid[row + n_s_index][col + w_e_index]["lake"]:
                temporary_neighbour["water"][move_to_index] += water_to_move
                temporary_neighbour["meltflag"][move_to_index] = 1
            else:
                temporary_neighbour["lake_depth"] += water_to_move
            temporary_cell["saturation"][idx] = 0
            water_to_move = 0
        # Otherwise - remove all of it from that cell and go up one.
        else:
            temporary_cell["water"][idx] -= cell["water"][idx] / (split + 1)
            # decrement the quota by exactly the amount moved
            water_to_move -= cell["water"][idx] / (split + 1)
            if not grid[row + n_s_index][col + w_e_index]["lake"]:
                temporary_neighbour["water"][move_to_index] += cell["water"][idx] / (
                    split + 1
                )
                temporary_neighbour["meltflag"][move_to_index] = 1
            else:
                temporary_neighbour["lake_depth"] += cell["water"][idx] / (split + 1)
            temporary_cell["saturation"][idx] = 0
    return temporary_cell, temporary_neighbour


@kernel()
def handle_edge_cases(
    grid,
    cell,
    temporary_cell,
    row,
    col,
    n_s_index,
    w_e_index,
    water_frac,
    split,
    outflow_proportion,
    catchment_outflow=False,
):
    """
    Handles the edge cases where we are trying to move water out of the model
    domain.

    Parameters
    ----------
    grid
    cell
    temporary_cell
    row
    col
    n_s_index
    w_e_index
    water_frac
    split
    outflow_proportion
    catchment_outflow

    Returns
    -------

    """
    if (
        row + n_s_index == -1
        or col + w_e_index == -1
        or row + n_s_index >= len(grid)
        or col + w_e_index >= len(grid[0])
    ):
        # edge case handling - since Python handles "-1" as
        # a valid index this will impose periodic boundary conditions rather
        # than raise an error, which is not the behaviour we want!
        # If we are moving out of the catchment area, we will get an IndexError
        # when trying to read neighbour_cell.
        # Instead, in this case, just determine how much water will move out of
        # the catchment area.
        # We return straight out of the function in this case as we only want
        # to move out one way.
        if catchment_outflow:
            water_out = calc_catchment_outflow(
                cell,
                temporary_cell,
                water_frac,
                split,
                outflow_proportion=outflow_proportion,
            )
            return water_out, True
        # if catchment outflow not enabled:
        raise ValueError(
            "Issue with lateral movement - trying to move to a"
            " non-existent grid cell and <catchment_outflow> is not"
            " enabled"
        )
    else:
        return 0, False


@kernel()
def handle_invalid_neighbour_cell(
    grid,
    cell,
    temporary_cell,
    row,
    col,
    n_s_index,
    w_e_index,
    flow_into_land,
    split,
    water_frac,
    outflow_proportion,
):
    """

    Parameters
    ----------
    grid
    cell
    temporary_cell
    row
    col
    n_s_index
    w_e_index
    flow_into_land
    split
    water_frac
    outflow_proportion

    Returns
    -------

    """
    # If the neighbour cell is invalid, we aren't interested in flowing water
    # into it...
    neighbour = grid[row + n_s_index][col + w_e_index]
    if neighbour["valid_cell"]:
        return 0, False
    # ...unless water can flow into the land (i.e. through crevices etc.) - in
    # which case run the catchment outflow algorithm, then return out of the
    # function.
    if flow_into_land:
        # how much drains away?
        water_out = calc_catchment_outflow(
            cell,
            temporary_cell,
            water_frac,
            split,
            outflow_proportion=outflow_proportion,
        )
        if water_out > 0:
            print(f"Moved {float(water_out)} units of water into the land")
            return water_out, False
        else:
            # no water moved, but still invalid, so skip
            return 0, True

    else:
        return 0, True


@kernel()
def move_to_neighbours(
    grid,
    temp_grid,
    biggest_neighbours,
    col,
    row,
    split,
    catchment_outflow,
    biggest_height_difference,
    timestep,
    flow_into_land=False,
    flow_speed_scaling=1.0,
    outflow_proportion=1.0,
):
    """
    Moves water from the central cell (grid[row][col]) to the neighbours
    indicated by *biggest_neighbours*.

    Parameters
    ----------
    grid : numpy structured array
        Model grid representing the ice shelf.
    temp_grid : numpy structured array
        Working copy of the grid used to accumulate water movements before
        they are applied simultaneously at the end of ``move_water``.
    biggest_neighbours : np.ndarray[int32]
        Direction indices (0=NW … 7=W; see module ``_DIR_ROW``/``_DIR_COL``)
        of the neighbours with the largest downhill water-level difference.
    col, row : int
        Column and row indices of the source cell.
    split : int
        Number of directions water is being split across (len(biggest_neighbours)).
    catchment_outflow : bool
        Whether out-of-domain flow is allowed.
    biggest_height_difference : float
        Largest water-level difference, used in ``water_fraction``.
    timestep : int
        Lateral movement timestep in seconds.
    flow_into_land : bool
        Whether water may flow into invalid (land) cells.
    flow_speed_scaling : float
        Multiplicative scaling factor for flow speed.
    outflow_proportion : float
        Fraction of available water allowed to leave the domain per step.

    Returns
    -------
    total_water_out : float
        Volume of water that left the model domain this step.

    Raises
    ------
    ValueError
        If an inconsistent model state is encountered (e.g. negative lake
        depth, or water present in a cell with no lake or ice lens).
    """
    cell = grid[row][col]
    cell["water_direction"][:] = 0  # clear water direction
    total_water_out = 0.0

    # Cells with a frozen lid cannot release water laterally.
    if cell["lid"]:
        return 0.0

    # biggest_neighbours holds the target direction indices (ascending); move
    # water into each in turn.
    for d in biggest_neighbours:
        cell["water_direction"][d] = 1
        n_s_index = _DIR_ROW[d]  # row offset
        w_e_index = _DIR_COL[d]  # column offset
        temporary_cell = temp_grid[row][col]
        water_frac = water_fraction(
            cell,
            biggest_height_difference,
            timestep,
            d,
            flow_speed_scaling=flow_speed_scaling,
        )

        # Check domain boundary before attempting to access the neighbour.
        water_out, edge_flag = handle_edge_cases(
            grid,
            cell,
            temporary_cell,
            row,
            col,
            n_s_index,
            w_e_index,
            water_frac,
            split,
            outflow_proportion,
            catchment_outflow=catchment_outflow,
        )
        if edge_flag:
            return water_out

        # Check that the neighbour cell is a valid ice-shelf cell.
        land_water_out, invalid_flag = handle_invalid_neighbour_cell(
            grid,
            cell,
            temporary_cell,
            row,
            col,
            n_s_index,
            w_e_index,
            flow_into_land,
            split,
            water_frac,
            outflow_proportion,
        )
        total_water_out += land_water_out
        if invalid_flag or land_water_out > 0:
            continue
        water_out += land_water_out

        """Logic for valid cells"""
        if 0 <= row + n_s_index < len(grid) and 0 <= col + w_e_index < len(grid[0]):
            temporary_neighbour = temp_grid[row + n_s_index][col + w_e_index]

        # Cells with a lid in either source or destination are skipped.
        if cell["lid"] or grid[row + n_s_index][col + w_e_index]["lid"]:
            continue

        # Dispatch on the source cell state: a lake drains from its surface, an
        # ice lens from its saturated firn layers.
        if cell["lake"]:
            temporary_cell, temporary_neighbour = move_from_lake(
                cell,
                grid,
                temporary_cell,
                temporary_neighbour,
                row,
                col,
                n_s_index,
                w_e_index,
                water_frac,
                split,
            )
        elif cell["ice_lens"] and not cell["lake"]:
            temporary_cell, temporary_neighbour = move_from_ice_lens(
                cell,
                grid,
                temporary_cell,
                temporary_neighbour,
                row,
                col,
                n_s_index,
                w_e_index,
                water_frac,
                split,
            )
        else:
            # No ice lens and no lake: water percolates downward only.
            raise ValueError("This should not be happening...")

    return total_water_out
