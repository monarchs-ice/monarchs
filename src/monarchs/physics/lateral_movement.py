"""
TODO - Module docstring, PEP8 compliance for comments
"""

import numpy as np
from monarchs.core.utils import find_nearest
from monarchs.physics.percolation import percolation, calc_saturation
from numba.typed import Dict
from numba import types

try:
    from numba import prange
except ImportError:
    # in case we don't have Numba - use the standard range function
    prange = range


def update_water_level(cell):
    """
    Determine the water level of a single cell, so we can determine where water
    flows laterally to and from.
    This is determined by the presence of lakes, lids or ice lenses within the
    firn column.
    If there is no lake, lid or ice lens, then the entire grid cell is free for
    water to move into it.
    If there is no lake or lid, but there is an ice lens then the water level
    is the level of the highest point at which we have saturated firn.
    If a cell has a lake, but no lid, then the water level is the height of
    that lake.
    Finally, if we have a lid, we set the water level to be arbitrarily high,
    as we are not currently interested in water flow from a frozen lid as it is
    too complicated to model.
    Called in <move_water>.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.

    Returns
    -------

    """

    if not cell["valid_cell"]:
        # Invalid cell - not interested so set water level to something
        # unreasonably high
        cell["water_level"] = 1e10
        return

    elif not cell["lake"] and not cell["lid"]:
        if cell["ice_lens"]:

            # We find the water level by the topmost bit of saturated firn
            # above the ice lens.
            if not np.any(
                cell["saturation"][: cell["ice_lens_depth"] + 1] > 0
            ):
                top_saturation_depth = cell["ice_lens_depth"]
            else:
                top_saturation_depth = np.argmax(
                    cell["saturation"][: cell["ice_lens_depth"] + 1] > 0
                )
            cell["water_level"] = cell["vertical_profile"][::-1][
                top_saturation_depth
            ]

        # Otherwise, water is free to percolate all the way to the bottom,
        # so it doesn't move laterally from here.
        else:
            cell["water_level"] = 0

        # cell.water is only used for the lateral movement. So we first need to
        # update it based on Lfrac,
        # which is used in the rest of MONARCHS.
        cell["water"] = cell["Lfrac"] * (
            cell["firn_depth"] / cell["vert_grid"]
        )

    # Add lake depth into water for the purposes of moving it around if a lake
    # is present.
    elif cell["lake"] and not cell["lid"]:
        cell["water_level"] = cell["lake_depth"] + cell["firn_depth"]
        # Determine the water level from the water on top + the firn depth.
        cell["water"] = cell["Lfrac"] * (
            cell["firn_depth"] / cell["vert_grid"]
        )

    elif cell["lake_depth"] > 0.1 and not cell["lid"]:
        # Same again - account for a bug where lake switch doesn't activate
        cell["water_level"] = cell["lake_depth"] + cell["firn_depth"]
        cell["water"] = cell["Lfrac"] * (
            cell["firn_depth"] / cell["vert_grid"]
        )
        cell["water"][0] += cell.lake_depth

    elif cell["lid"]:
        # shouldn't matter, as water can't move from a lid
        cell["water_level"] = (
            cell["lid_depth"] + cell["firn_depth"] + cell["lake_depth"]
        )
        cell["water"] = cell["Lfrac"] * (
            cell["firn_depth"] / cell["vert_grid"]
        )


def get_neighbour_water_levels(
    cell, grid, col, row, max_grid_col, max_grid_row
):
    neighbours = Dict.empty(
        key_type=types.unicode_type, value_type=types.float64
    )
    # if not on top edge of domain - init N, possibly NE/NW
    if row > 0:
        # north neighbour = -1 in row index (i.e. when selecting rows,
        # previous row is the cell directly above)
        neighbours["N"] = (
            cell["water_level"] - grid["water_level"][row - 1][col]
        )
        if col < max_grid_col - 1:  # if not at right edge - init NE
            neighbours["NE"] = (
                cell["water_level"] - grid["water_level"][row - 1][col + 1]
            )
        # if not at left edge - init NW
        if col > 0:
            neighbours["NW"] = (
                cell["water_level"] - grid["water_level"][row - 1][col - 1]
            )
    # if at top edge - N/NW/NE all False
    else:
        neighbours["N"] = -999
        neighbours["NE"] = -999
        neighbours["NW"] = -999

    # if not on bottom edge of domain - init S, possibly SE/SW
    if row < max_grid_row - 1:
        neighbours["S"] = (
            cell["water_level"] - grid["water_level"][row + 1][col]
        )
        if col < max_grid_col - 1:  # if not at right edge - init SE
            neighbours["SE"] = (
                cell["water_level"] - grid["water_level"][row + 1][col + 1]
            )
        if col > 0:  # if not at left edge - init SW
            neighbours["SW"] = (
                cell["water_level"] - grid["water_level"][row + 1][col - 1]
            )
    # if at bottom edge - S/SW/SE all False
    else:
        neighbours["S"] = -999
        neighbours["SE"] = -999
        neighbours["SW"] = -999
    # if not at left edge - init W - NW/SW have already been initialised if
    # possible to do so
    if col > 0:
        neighbours["W"] = (
            cell["water_level"] - grid["water_level"][row][col - 1]
        )
    else:  # if at left edge - W/SW/NW all False
        neighbours["W"] = -999
        neighbours["SW"] = -999
        neighbours["NW"] = -999
    if col < max_grid_col - 1:  # As above, for the right edge
        neighbours["E"] = (
            cell["water_level"] - grid["water_level"][row][col + 1]
        )
    else:
        neighbours["E"] = -999
        neighbours["SE"] = -999
        neighbours["NE"] = -999
    return neighbours


def find_biggest_neighbour(
    cell,
    grid,
    col,
    row,
    max_grid_col,
    max_grid_row,
    catchment_outflow,
    flow_into_land=False,
):
    """
    Finds the points in grid surrounding a central point cell that have the
    highest difference in water level.
    Called in <move_water>.

    Parameters
    ----------
    catchment_outflow: bool
        If True, then water that reaches the edge of the catchment area will
        flow outward, if it is in the lowest cell locally.
    cell: numpy structured array
       Element of our model grid.
    grid: List or numba.typed.List()
        List containing multiple instances of IceShelf objects.
    i: int
        Iteration index along the x-axis of grid.
    j: int
        Iteration index along the y-axis of grid.
    max_grid_row: int
        Number of x points in grid.
    max_grid_col: int
        Number of y points in grid.

    Returns
    -------
    biggest_height_difference: List
        List containing the values of the largest difference(s) in water level
        between cell and its neighbours.
    max_list: List
        List containing the indices corresponding to the largest neighbour(s).
    """
    neighbours = get_neighbour_water_levels(
        cell, grid, col, row, max_grid_col, max_grid_row
    )
    # Ensure that if catchment_outflow is set, water that reaches the edge of
    # the catchment area will flow outward, if it is in the lowest cell
    # locally. This way, water will preferentially stay in the model.
    if catchment_outflow and max(neighbours.values()) <= 0:
        for key in neighbours.keys():
            if neighbours[key] == -999:
                neighbours[key] = 9999

    # if flow_into_land is True, then if the cell is at a local minimum aside
    # from invalid cells, then it will flow into the land. This is motivated by
    # the appearance of lakes on the ice shelf-land boundary in testing runs,
    # which is not seen in the validation data. This is placed after the
    # initial loop, since this should only occur if the water level is a local
    # minimum.
    if flow_into_land:
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                try:
                    code = []
                    if i == -1:
                        code.append("N")
                    elif i == 1:
                        code.append("S")
                    if j == -1:
                        code.append("W")
                    elif j == 1:
                        code.append("E")

                    code_str = "".join(code)
                    if (
                        0 <= row + i < grid.shape[0]
                        and 0 <= col + j < grid.shape[1]
                    ):
                        neighbour_cell = grid[row + i][col + j]
                    else:
                        continue
                    # If the water level is a local minimum, then we want to
                    # flow water into the land.

                    if max(neighbours.values()) <= 0:
                        if not neighbour_cell["valid_cell"]:
                            neighbours[code_str] = 9998

                    # Otherwise, ensure we *don't* flow water into land as it
                    # has places it can go within the model.
                    elif not neighbour_cell["valid_cell"]:
                        neighbours[code_str] = -9999

                except Exception:
                    continue

    # Find neighbour with the biggest height difference in water level
    biggest_height_difference = max(neighbours.values())

    # check a new list of indices/value pairs to see which one corresponds
    # to the maximum calculated above
    # max list is a list of the indices of m that have the largest value
    # e,g. if the NW and SE neighbours have the biggest value,
    # max_list = [0, 7]
    # max_list = [
    #     idx for idx, val in enumerate(neighbour_list) if
    #     (val == biggest_height_difference and val > 0)
    # ]
    max_list = [
        k for k, v in neighbours.items() if v == biggest_height_difference
    ]

    # A fix for the symmetric test case. What was happening was that there were
    # tiny rounding errors. This caused water to preferentially move one way
    # over the other, which is bad for a symmetric case! This fixes the issue.
    # Append any that are within a rounding error
    # (i.e. if the difference is 0.0000000001, then we want to include it)
    for k, v in neighbours.items():
        if biggest_height_difference - v < 1e-08:
            if k not in max_list:
                max_list.append(k)

    return biggest_height_difference, max_list


def water_fraction(cell, m, timestep, direction, flow_speed_scaling=1.0):
    """
    Determine the fraction of water that is allowed to move through the solid
    firn, dependent on its density.
    Called in <move_water>.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    m : float
        difference in heights between this cell and the cell the water is
        moving to
    timestep : int
        time in which the water can move - default 3600 (set by user in
        runscript) [s]
    direction: str
        direction of the water flow - used to determine the "cell size", as
        these can be non-square.

    Returns
    -------
    water_frac : np.ndarray or float
        Fraction of the total water that is allowed to move. Either a single
        number if a lake is present, or a Numpy array of length cell.vert_grid
        if not. M is biggest height difference
    """
    # first calculate the distance the water has to move
    if direction in ["NW", "NE", "SW", "SE"]:
        cell_size = np.sqrt(cell["size_dx"] ** 2 + cell["size_dy"] ** 2)
    elif direction in ["N", "S"]:
        cell_size = cell["size_dy"]
    elif direction in ["E", "W"]:
        cell_size = cell["size_dx"]
    else:
        raise ValueError("Direction not recognised")
    # TODO - should cell_size be divided by 2? Since water is moving from the
    # TODO - centre of the cell. But if we consider that it is moving from
    # TODO - centre to centre, is probably fine assuming size of adjacent cells
    # TODO - is the same, but could implement a fix at some point.

    Big_pi = -2.53 * 10**-10  # hydraulic permeability (m^2)
    eta = 1.787 * 10**-3  # viscosity(Pa/s)
    cell["rho"] = (
        cell["Sfrac"] * cell["rho_ice"] + cell["Lfrac"] * cell["rho_water"]
    )

    # This block ensures that u is not greater than 1, which would be
    # unphysical.
    # if lake, then we only are interested in rho_water so u is a float
    # (not an array)
    if cell["lake"]:
        # if in a lake all water moves
        water_frac = np.array([1.0])
    # otherwise we look at the density of a specific point in the firn so u
    # is an array
    else:
        cell_density = cell["rho"]
        u = (
            Big_pi / eta * (m / cell_size) * cell_density * -9.8
        )  # flow speed (m/s)
        water_frac = u * timestep / cell_size
        water_frac[np.where(water_frac > 1)] = 1

        # JE - added flow_speed_scaling variable here.
        water_frac = water_frac * flow_speed_scaling
    return water_frac


def calc_available_water_lake(
    cell, water_frac, split, neighbour_cell, outflow=False
):
    if outflow:  # outflow case, so most of the calculation is irrelevant
        water_to_move = float(
            cell["lake_depth"]
        )  # if flowing out, whole lake drains away.
        return water_to_move, 0, 0, 0

    # Otherwise, proceed as normal
    # enough water should move such that we equalise water level,
    # accounting for moving to multiple cells.
    # [0] since we want a float not an array for Numba
    water_to_move = (
        water_frac
        * ((cell["water_level"] - neighbour_cell["water_level"]) / (split + 1))
    )[0]

    # Ensure we have enough water to fill our "quota" -
    # else everything goes but no more.
    # multiply by split since we want the total water that can move.
    # water frac is always 1 if the central cell is a lake
    # so doesn't pose a problem
    if cell["lake_depth"] < water_to_move * (split + 1):
        water_to_move = float(cell["lake_depth"] / (split + 1))
    if water_to_move < 0:
        raise ValueError("Water to move from lake is less than 0")
    return water_to_move, 0, 0, 0


def calc_available_water_ice_lens(
    cell, water_frac, split, neighbour_cell, outflow=False
):
    """
    Determine the amount of water that can be moved from one cell to the next.

    Parameters
    ----------
    cell
    water_frac
    split
    neighbour_cell
    outflow

    Returns
    -------

    """
    # central cell - (split * water_to_move) = outside cell + water_to_move
    # > water_to_move * split = central_cell - outside_cell
    # > water_to_move = central_cell - outside_cell / (split)
    # only the case if outside cells are equal so far - maybe change in future
    # We need to move enough water for the water levels to equalise.
    # If lake depth < difference between water levels, then all needs to move.

    cell["vertical_profile"] = np.linspace(
        0, cell["firn_depth"], cell["vert_grid"]
    )

    if outflow:
        lowest_water_level = cell["vertical_profile"][::-1][
            cell["ice_lens_depth"]
        ]
        vp = cell["vertical_profile"][::-1]
        move_from_index = find_nearest(vp, lowest_water_level)
        water_to_move = np.sum(cell["water"][: move_from_index + 1])
        return water_to_move, 0, 0, 0

    # Determine the highest point of the water - this is needed later
    top_saturation_level = np.argmax(
        cell["saturation"][: cell["ice_lens_depth"] + 1] > 0
    )
    # The water moves from the lowest level it can physically do so
    # i.e. the ice lens depth:
    if (
        cell["vertical_profile"][::-1][cell["ice_lens_depth"]]
        > neighbour_cell["water_level"]
    ):
        lowest_water_level = cell["vertical_profile"][::-1][
            cell["ice_lens_depth"]
        ]
    else:  # or the midpoint of the water depths
        lowest_water_level = cell["vertical_profile"][::-1][
            find_nearest(
                cell["vertical_profile"][::-1],
                (cell["water_level"] + neighbour_cell["water_level"])
                / (split + 1),
            )
        ]
    vp = cell["vertical_profile"][::-1]
    move_from_index = find_nearest(vp, lowest_water_level)
    water_to_move = (
        water_frac[move_from_index]
        * (cell["water_level"] - neighbour_cell["water_level"])
        / (split + 1)
    )

    # If saturated firn doesn't hold enough water to fill the "quota",
    # then all of it moves but no more
    # +1 so we include the one where we move from
    if (
        np.sum(cell["water"][: move_from_index + 1]) / (split + 1)
        < water_to_move
    ):
        # i.e. if more water is allowed to move than we have in the cell
        water_to_move = np.sum(cell["water"][: move_from_index + 1]) / (
            split + 1
        )
    return (
        water_to_move,
        lowest_water_level,
        move_from_index,
        top_saturation_level,
    )


def calc_catchment_outflow(
    cell, temporary_cell, water_frac, split, outflow_proportion=1.0
):
    """
    Calculate the amount of water that can be moved from a cell to the land
    (i.e. outside of the model domain).

    Parameters
    ----------
    cell
    temporary_cell
    water_frac
    split

    Returns
    -------

    """
    # call cell twice as "cell" is the neighbour cell in this instance -
    # it isn't actually used but lets us jit-compile with Numba
    if cell["lake"]:
        water_to_move, _, __, ___ = calc_available_water_lake(
            cell, water_frac, split, cell, outflow=True
        )
    elif cell["ice_lens"] and not cell["lid"]:
        (
            water_to_move,
            lowest_water_level,
            move_from_index,
            top_saturation_level,
        ) = calc_available_water_ice_lens(
            cell, water_frac, split, cell, outflow=True
        )
    else:
        raise ValueError(
            "Trying to calculate outflow from a cell that is neither a "
            "lake or has an ice lens"
        )
    # Whether we are moving from a lake or ice lens, scale by the outflow
    # proportion.
    water_to_move = water_to_move * outflow_proportion
    # either: remove water from the lake directly.
    if cell["lake"] and not cell["lid"]:
        water_out = water_to_move
        if cell["lake_depth"] < water_to_move:
            water_to_move = cell["lake_depth"]
            temporary_cell["lake_depth"] = 0
            water_out = water_to_move
            water_to_move = 0
        else:
            temporary_cell["lake_depth"] -= water_to_move
            if temporary_cell["lake_depth"] < 0:
                # account for rounding errors
                if temporary_cell["lake_depth"] < -1e-12:
                    raise ValueError("Lake depth has gone below 0")
                else:
                    temporary_cell["lake_depth"] = 0
    # otherwise, loop through the column and remove water from it,
    # going from the top.
    else:
        water_out = 0
        try:
            for _l in range(0, cell["ice_lens_depth"] + 1):
                if cell["water"][_l] > water_to_move:
                    temporary_cell["water"][_l] -= water_to_move
                    temporary_cell["saturation"][_l] = 0
                    water_out += water_to_move
                    water_to_move = 0
                else:
                    water_to_move -= cell["water"][_l]
                    water_out += cell["water"][_l]
                    temporary_cell["water"][_l] -= cell["water"][_l]
        except Exception:
            print(_l)
            print(water_out)
            print(cell["ice_lens_depth"])
            raise Exception
    return water_out


def move_from_lake(
    cell,
    grid,
    temporary_cell,
    temporary_neighbour,
    row,
    col,
    n_s_index,
    w_e_index,
    water_to_move,
):
    """

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
    water_to_move

    Returns
    -------

    """
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
            "Moving water has caused lake depth to go below 0 - in the central"
            " cell"
        )
    return temporary_cell, temporary_neighbour


def move_from_ice_lens(
    cell,
    grid,
    temporary_cell,
    temporary_neighbour,
    split,
    row,
    col,
    n_s_index,
    w_e_index,
    lowest_water_level,
    move_from_index,
    top_saturation_level,
    water_to_move,
):
    # TODO | Rather than using water level, we actually need to account for
    # TODO | the available pore space.
    # TODO | This will be something along the lines of calculating
    # TODO | 1 - Sfrac - Lfrac for the selected indices and weighting based on
    # TODO | that, rather than just water_depth.
    # TODO | Should normalise between different timesteps, so not a priority
    # TODO | right now, but possible for future development.

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
            water_to_move -= cell["water"][idx] / split
            if not grid[row + n_s_index][col + w_e_index]["lake"]:
                temporary_neighbour["water"][move_to_index] += cell["water"][
                    idx
                ] / (split + 1)
                temporary_neighbour["meltflag"][move_to_index] = 1
            else:
                temporary_neighbour["lake_depth"] += cell["water"][idx] / (
                    split + 1
                )
            temporary_cell["saturation"][idx] = 0
    return temporary_cell, temporary_neighbour


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
        or row + n_s_index == len(grid)
        or col + w_e_index == len(grid[0])
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
            return water_out
        else:
            raise ValueError(
                "Issue with lateral movement - trying to move to a"
                " non-existent grid cell and <catchment_outflow> is not"
                " enabled"
            )
    else:
        return 0


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
    if (
        not grid[row + n_s_index][col + w_e_index]["valid_cell"]
        and not flow_into_land
    ):
        # return -1 to indicate that we want to `continue` the loop in
        # move_water rather than return out
        return -1

    # ...unless water can flow into the land (i.e. through crevices etc.) - in
    # which case run the catchment outflow algorithm, then return out of the
    # function.
    elif (
        not grid[row + n_s_index][col + w_e_index]["valid_cell"]
        and flow_into_land
    ):
        water_out = calc_catchment_outflow(
            cell,
            temporary_cell,
            water_frac,
            split,
            outflow_proportion=outflow_proportion,
        )
        if water_out > 0:
            print(f"Moved {float(water_out)} units of water into the land")
            return water_out
        if temporary_cell["lake_depth"] < 0:
            print(
                "temporary_cell['lake_depth'] ="
                f" {temporary_cell['lake_depth']}, col = {col}, row = {row}"
            )
            print(
                f"split = {split}, water_frac = {water_frac}, cell.lake_depth"
                f" = {cell['lake_depth']}"
            )
        else:
            return -1

    else:
        return 0


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
    Moves water from the central cell (grid[i][j]) to the neighbours with the
    lowest water level.
    The specific neighbour is associated with an index [i + m][j + n] with
    m, n = [-1, 0, 1].
    e.g. SW neighbour is associated with the m = +1, n = -1.
    How the movement is handled is dependent on the features of the cell and
    its neighbour.
    Called in <move_water>.

    Parameters
    ----------
    grid : numpy structured array
        Model grid representing the ice shelf.
    biggest_neighbours : List
        List of the largest neighbour cells calculated from
        find_biggest_neighbour
    i : int
        current row index
    j : int
        current column index
    biggest_height_difference : float
        largest difference in water level between the central cell and its
        neighbours. Used in `water_fraction`.
    split : int
        Number determining whether water is split between two or more neighbour
        grid cells, if there are cells of equal water_level

    Returns
    -------
    None (amends instance of grid passed into it)

    Raises
    ------
    ValueError
        If model reaches a state that should not be able to occur, then raise
        this error.
        This can occur if the lake depth goes below 0, or if there is water to
        move in a grid cell where water should not be able to move from
        (i.e. no lake or ice lens)
    """
    # recall that the "column" index is our *x* coordinate, not y, and we index
    # by [row, col], i.e. [y, x] in the grid. And that we start indexing from
    # the top left, not the bottom left!
    all_neighbours = Dict()
    all_neighbours["NW"] = np.array([-1, -1])
    all_neighbours["N"] = np.array([-1, 0])
    all_neighbours["NE"] = np.array([-1, 1])
    all_neighbours["E"] = np.array([0, 1])
    all_neighbours["SE"] = np.array([1, 1])
    all_neighbours["S"] = np.array([1, 0])
    all_neighbours["SW"] = np.array([1, -1])
    all_neighbours["W"] = np.array([0, -1])

    cell = grid[row][col]
    cell["water_direction"][:] = 0  # clear water direction

    # for each of the neighbours we wish to move water to, which depends on the
    # difference in water level between the central cell and the cells at each
    # cardinal point
    for idx, neighbour in enumerate(all_neighbours.keys()):
        if neighbour in biggest_neighbours:
            if cell["lid"]:
                return 0
            cell["water_direction"][idx] = 1
            n_s_index = all_neighbours[neighbour][0]  # i.e. row index
            w_e_index = all_neighbours[neighbour][1]  # i.e. col index
            temporary_cell = temp_grid[row][col]
            water_frac = water_fraction(
                cell,
                biggest_height_difference,
                timestep,
                neighbour,
                flow_speed_scaling=flow_speed_scaling,
            )

            # Before actually moving water, we first need to find out how much
            # water can actually move.
            # This try/except block happens in all non-lid cases, and
            # determines how much water can move from the central cell to the
            # neighbour cell. If the cell is at a boundary, and the neighbour
            # cell is outside of the domain, then we don't want to move water
            # unless catchment_outflow is True.
            water_out = handle_edge_cases(
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
            # If we returned water from handle_edge_cases, water will not move
            # further so we return this amount to move_water.
            # This amount will always be positive.
            if water_out > 0:
                return water_out
            # Also check that neighbour_cell is actually a valid cell.
            # If it isn't, then we don't want to move water to it,
            # unless flow_into_land is True.
            water_out += handle_invalid_neighbour_cell(
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
            if water_out > 0:
                return water_out
            # if handle_invalid_neighbour_cell returns -1, then don't flow to
            # this particular (invalid) cell, but there may be a neighbour
            # that *is* valid
            elif water_out == -1 and idx != len(biggest_neighbours) - 1:
                continue
            # If we get to the end and all possible neighbours have been
            # exhausted, then simply return 0 so we don't do anything.
            # This fixes a bug where if no valid neighbours are found, then
            # an extremely large amount of water is moved.
            elif water_out == -1:
                return 0
            # Determine the neighbour and create a temporary version of it to
            # hold info while we move to/from other cells
            if 0 <= row + n_s_index < len(grid) and 0 <= col + w_e_index < len(
                grid[0]
            ):
                temporary_neighbour = temp_grid[row + n_s_index][
                    col + w_e_index
                ]

            # If there is a lid in either cell - too complicated to model, and
            # physically unlikely, so we ignore movement in this case.
            if cell["lid"] or grid[row + n_s_index][col + w_e_index]["lid"]:
                continue

            # Case where the central cell has a lake.
            if cell["lake"]:
                water_to_move, _, _, _ = calc_available_water_lake(
                    cell,
                    water_frac,
                    split,
                    grid[row + n_s_index][col + w_e_index],
                )
                temporary_cell, temporary_neighbour = move_from_lake(
                    cell,
                    grid,
                    temporary_cell,
                    temporary_neighbour,
                    row,
                    col,
                    n_s_index,
                    w_e_index,
                    water_to_move,
                )

            elif cell["ice_lens"] and not cell["lake"]:
                # in this case we need to return a bit more info, hence the
                # tuple
                (
                    water_to_move,
                    lowest_water_level,
                    move_from_index,
                    top_saturation_level,
                ) = calc_available_water_ice_lens(
                    cell,
                    water_frac,
                    split,
                    grid[row + n_s_index][col + w_e_index],
                )

                temporary_cell, temporary_neighbour = move_from_ice_lens(
                    cell,
                    grid,
                    temporary_cell,
                    temporary_neighbour,
                    split,
                    row,
                    col,
                    n_s_index,
                    w_e_index,
                    lowest_water_level,
                    move_from_index,
                    top_saturation_level,
                    water_to_move,
                )
            else:  # If no ice lens, then no water should be able to move as
                # it should preferentially percolate downwards.
                raise ValueError("This should not be happening...")

    return 0


def move_water(
    grid,
    max_grid_col,
    max_grid_row,
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
        Number of grid rows
    max_grid_col : int
        Number of grid columns
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
    for row in prange(max_grid_row):
        for col in range(max_grid_col):
            cell = grid[row][col]
            update_water_level(cell)
            total_water += np.sum(cell["water"]) + cell["lake_depth"]

            # temporary cell specification
            temp_grid[row, col]["lake_depth"] = cell["lake_depth"]
            temp_grid[row, col]["lake"] = cell["lake"]
            # need to explicitly loop rather than using array syntax
            # else it will create a shallow copy, and explicit loops are better
            # for Numba anyway
            for level in range(len(grid[row, col]["water"])):
                temp_grid[row, col]["water"][level] = cell["water"][level]
                temp_grid[row, col]["saturation"][level] = cell["saturation"][
                    level
                ]
                temp_grid[row, col]["meltflag"][level] = cell["meltflag"][
                    level
                ]

    # Now loop through the cells again, this time actually performing the
    # movement step. This and the prior loop cannot be merged, since the
    # temporary grid needs to be fully populated and the water levels
    # determined everywhere before we start moving water around, particularly
    # as both loops are done in parallel.
    for row in prange(max_grid_row):
        for col in range(max_grid_col):
            cell = grid[row][col]
            # clear water direction - this will be updated later
            cell["water_direction"][:] = 0
            # don't bother trying moving water unless we have
            # a) a cell that is actually running physics, and
            # b) that cell actually has water in it!
            if cell["valid_cell"] and (
                cell["ice_lens"]
                and (cell["water"] > 0).any()
                or cell["lake"]
                and cell["lake_depth"] > 0
            ):
                # For each element of grid, determine the neighbour with the
                # biggest (positive or negative) difference in water level.
                biggest_height_difference, max_list = find_biggest_neighbour(
                    cell,
                    grid,
                    col,
                    row,
                    max_grid_col,
                    max_grid_row,
                    catchment_outflow,
                    flow_into_land=flow_into_land,
                )
                # If more than one cell is equally lower than central the water
                # is split between them
                split = len(max_list)
                # Water moves if one of surrounding grid is lower than
                # central cell
                if biggest_height_difference > 0:
                    # Now calculate the amount of water to move.
                    # Note that this is in the form of temporary arrays,
                    # as we need to avoid race conditions (i.e. order matters).
                    # The actual water level is updated in the next loop.
                    catchment_out_water += move_to_neighbours(
                        grid,
                        temp_grid,
                        max_list,
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
    # update the values of <grid>, i.e. performing our movement in one step
    for row in prange(0, max_grid_row):
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
                raise ValueError("cell.water is negative")
            elif (cell["water"] < 0).any():
                set_to_zeros = np.where(cell["water"] < 0)
                cell["water"][set_to_zeros] = 0

            if cell["valid_cell"]:
                # Once all water has moved, update the Lfrac of each cell based
                # on where the water now is.
                # The water level is calculated at the beginning of the next
                # call to move_water, and is not used elsewhere in the code.
                cell["Lfrac"] = cell["water"] / (
                    cell["firn_depth"] / cell["vert_grid"]
                )
                # We have put all the water at one level in the firn -
                # we need to percolate it to make sure that
                # the water fills out the available pore space.
                if lateral_movement_percolation_toggle:
                    # assume the percolation happens within 1 hour.
                    percolation(cell, 3600, lateral_refreeze_flag=True)
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
