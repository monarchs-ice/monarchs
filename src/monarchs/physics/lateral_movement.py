"""
Lateral water movement module for MONARCHS.

Water flows between grid cells based on differences in water level.
Movement can occur from lakes, or from saturated firn above an ice lens,
to any of the 8 surrounding cells (N, NE, E, SE, S, SW, W, NW).

Parallelism notes
-----------------
The water-level initialisation loop (Phase 1 of ``move_water``) is safe to
parallelise with ``prange`` because each iteration writes only to its own
cell in ``temp_grid``.

The movement loop (Phase 2) writes to *neighbour* cells in ``temp_grid``.
Running it with ``prange`` over rows creates a race condition: two rows
processed simultaneously can both write to the same intermediate-row cell.
Phase 2 is therefore deliberately kept serial (plain ``range``).

To re-enable parallelism in Phase 2 in the future, a 3×3 tile-colouring
(9-colour) scheme should be used: cells sharing the same colour are at
least 3 apart in every dimension, so no two same-colour cells can share a
neighbour. Running 9 sequential ``prange`` passes over the 9 colours is
both correct and fully parallel.
"""

# TODO - PEP8 compliance for comments, function docstrings
import numpy as np
from monarchs.core.kernels import kernel, prange
from monarchs.core.utils import find_nearest
from monarchs.physics.firn.percolation import percolate, calc_saturation
from monarchs.physics.constants import rho_ice, rho_water

# ---------------------------------------------------------------------------
# Direction constants
# ---------------------------------------------------------------------------
# The 8 compass directions are stored as parallel row-offset and col-offset
# arrays.  The index ordering matches the original Dict insertion order used
# in move_to_neighbours, so that ``cell["water_direction"][d]`` retains its
# existing meaning:
#   0=NW  1=N  2=NE  3=E  4=SE  5=S  6=SW  7=W
_DIR_ROW = np.array([-1, -1, -1, 0, 1, 1, 1, 0], dtype=np.int32)
_DIR_COL = np.array([-1, 0, 1, 1, 1, 0, -1, -1], dtype=np.int32)
_N_DIRS = 8
# Even indices (0,2,4,6) are diagonal; odd indices (1,3,5,7) are cardinal.


@kernel()
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
        cell["water"][:] = 0
        cell["saturation"][:] = 0
        cell["lake_depth"] = 0
        cell["Lfrac"][:] = 0
        cell["water_level"] = 1e11
        return

    elif not cell["lake"] and not cell["lid"]:
        if cell["ice_lens"]:
            # We find the water level by the topmost bit of saturated firn
            # above the ice lens.
            if not np.any(cell["saturation"][: cell["ice_lens_depth"] + 1] > 0):
                top_saturation_depth = cell["ice_lens_depth"]
            else:
                top_saturation_depth = np.argmax(
                    cell["saturation"][: cell["ice_lens_depth"] + 1] > 0
                )
            cell["water_level"] = cell["vertical_profile"][::-1][top_saturation_depth]

        # Otherwise, water is free to percolate all the way to the bottom,
        # so it doesn't move laterally from here.
        else:
            cell["water_level"] = 0

        # cell.water is only used for the lateral movement. So we first need to
        # update it based on Lfrac,
        # which is used in the rest of MONARCHS.
        cell["water"] = cell["Lfrac"] * (cell["firn_depth"] / cell["vert_grid"])

    # Add lake depth into water for the purposes of moving it around if a lake
    # is present.
    elif cell["lake"] and not cell["lid"]:
        cell["water_level"] = cell["lake_depth"] + cell["firn_depth"]
        # Determine the water level from the water on top + the firn depth.
        cell["water"] = cell["Lfrac"] * (cell["firn_depth"] / cell["vert_grid"])

    elif cell["lake_depth"] > 0.1 and not cell["lid"]:
        # Same again - account for a bug where lake switch doesn't activate
        cell["water_level"] = cell["lake_depth"] + cell["firn_depth"]
        cell["water"] = cell["Lfrac"] * (cell["firn_depth"] / cell["vert_grid"])
        cell["water"][0] += cell.lake_depth

    elif cell["lid"]:
        # shouldn't matter, as water can't move from a lid
        cell["water_level"] = (
            cell["lid_depth"] + cell["firn_depth"] + cell["lake_depth"]
        )
        cell["water"] = cell["Lfrac"] * (cell["firn_depth"] / cell["vert_grid"])


@kernel()
def get_neighbour_water_levels(cell, grid, col, row, max_grid_col, max_grid_row):
    """
    Return the water-level differences between *cell* and each of its 8
    neighbours as a fixed-size float64 array indexed by direction
    (0=NW … 7=W; see module-level ``_DIR_ROW``/``_DIR_COL``).

    Out-of-bounds neighbours are given a sentinel value of -999.0 so that
    they are never chosen as a flow destination.
    """
    diffs = np.full(_N_DIRS, -999.0)
    for d in range(_N_DIRS):
        nr = row + _DIR_ROW[d]
        nc = col + _DIR_COL[d]
        if 0 <= nr < max_grid_row and 0 <= nc < max_grid_col:
            diffs[d] = cell["water_level"] - grid["water_level"][nr][nc]
    return diffs


@kernel()
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
    Return the largest water-level height difference between *cell* and its
    neighbours, together with the direction indices (0-7) that achieve it.

    Parameters
    ----------
    catchment_outflow : bool
        If True, out-of-bounds neighbours are treated as valid outflow
        destinations when the cell is a local water-level minimum.
    flow_into_land : bool
        If True, invalid (land) neighbours may receive water when the cell
        is a local minimum, but are blocked when other valid neighbours exist.

    Returns
    -------
    biggest_height_difference : float
    max_dirs : np.ndarray[int32]
        Array of direction indices (subset of 0-7) that are within floating-
        point rounding tolerance of the maximum height difference.
    """
    diffs = get_neighbour_water_levels(cell, grid, col, row, max_grid_col, max_grid_row)

    # Allow outflow at domain boundaries when the cell is a local minimum.
    if catchment_outflow and np.max(diffs) <= 0:
        for d in range(_N_DIRS):
            if diffs[d] == -999.0:
                diffs[d] = 9999.0

    # Optionally allow (or block) flow into invalid (land) cells.
    if flow_into_land:
        current_max = np.max(diffs)
        for d in range(_N_DIRS):
            nr = row + _DIR_ROW[d]
            nc = col + _DIR_COL[d]
            if 0 <= nr < max_grid_row and 0 <= nc < max_grid_col:
                if current_max <= 0 and not grid[nr][nc]["valid_cell"]:
                    # Cell is a local minimum: allow flow into land.
                    diffs[d] = 9998.0
                elif not grid[nr][nc]["valid_cell"]:
                    # Other valid destinations exist: block flow into land.
                    diffs[d] = -9999.0

    biggest_height_difference = np.max(diffs)

    # Collect all directions within floating-point rounding tolerance of the
    # maximum.  A fixed-size buffer avoids heap allocation inside @njit.
    max_dirs = np.zeros(_N_DIRS, dtype=np.int32)
    n_max = 0
    for d in range(_N_DIRS):
        if biggest_height_difference - diffs[d] < 1e-8:
            max_dirs[n_max] = d
            n_max += 1

    return biggest_height_difference, max_dirs[:n_max]


@kernel()
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
    direction : int
        Direction index (0=NW, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W).
        Even indices are diagonal; odd indices are cardinal.
        Used to determine the effective cell size for flow calculations.

    Returns
    -------
    water_frac : np.ndarray or float
        Fraction of the total water that is allowed to move. Either a single
        number if a lake is present, or a Numpy array of length cell.vert_grid
        if not. M is biggest height difference
    """
    # Determine effective cell size based on direction type.
    # Even index → diagonal (NW, NE, SE, SW); odd → cardinal.
    # Among cardinal directions: _DIR_COL == 0 → N or S (use size_dy);
    #                            _DIR_ROW == 0 → E or W (use size_dx).
    if direction % 2 == 0:  # diagonal
        cell_size = np.sqrt(cell["size_dx"] ** 2 + cell["size_dy"] ** 2)
    elif _DIR_COL[direction] == 0:  # N or S
        cell_size = cell["size_dy"]
    else:  # E or W
        cell_size = cell["size_dx"]
    # TODO - should cell_size be divided by 2? Since water is moving from the
    #      - centre of the cell. But if we consider that it is moving from
    #      - centre to centre, is probably fine assuming size of adjacent cells
    #      - is the same, but could implement a fix at some point.

    big_pi = -2.53 * 10**-10  # hydraulic permeability (m^2)
    eta = 1.787 * 10**-3  # viscosity(Pa/s)
    cell["rho"] = cell["Sfrac"] * rho_ice + cell["Lfrac"] * rho_water

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
        u = big_pi / eta * (m / cell_size) * cell_density * -9.8  # flow speed (m/s)
        water_frac = u * timestep / cell_size
        water_frac[np.where(water_frac > 1)] = 1

        # JE - added flow_speed_scaling variable here.
        water_frac = water_frac * flow_speed_scaling
    water_frac = np.clip(water_frac, a_min=0, a_max=1)
    return water_frac


@kernel()
def calc_available_water_lake(cell, water_frac, split, neighbour_cell, outflow=False):
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
        if water_to_move > -1e-8:
            water_to_move = 0
        else:
            print("Water to move = ", water_to_move)
            raise ValueError("Water to move from lake is less than 0")
    return water_to_move, 0, 0, 0


@kernel()
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

    cell["vertical_profile"] = np.linspace(0, cell["firn_depth"], cell["vert_grid"])

    if outflow:
        lowest_water_level = cell["vertical_profile"][::-1][cell["ice_lens_depth"]]
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
        lowest_water_level = cell["vertical_profile"][::-1][cell["ice_lens_depth"]]
    else:  # or the midpoint of the water depths
        lowest_water_level = cell["vertical_profile"][::-1][
            find_nearest(
                cell["vertical_profile"][::-1],
                (cell["water_level"] + neighbour_cell["water_level"]) / (split + 1),
            )
        ]
    vp = cell["vertical_profile"][::-1]
    move_from_index = find_nearest(vp, lowest_water_level)
    water_to_move = (
        water_frac[move_from_index]
        * (cell["water_level"] - neighbour_cell["water_level"])
        / (split + 1)
    )

    if water_to_move < 0:
        if water_to_move > -1e-8:
            water_to_move = 0
        else:
            raise ValueError(f"Negative water in move_to_ice_lens: {water_to_move}")
    # If saturated firn doesn't hold enough water to fill the "quota",
    # then all of it moves but no more
    # +1 so we include the one where we move from
    if np.sum(cell["water"][: move_from_index + 1]) / (split + 1) < water_to_move:
        # i.e. if more water is allowed to move than we have in the cell
        water_to_move = np.sum(cell["water"][: move_from_index + 1]) / (split + 1)

    return (
        water_to_move,
        lowest_water_level,
        move_from_index,
        top_saturation_level,
    )


@kernel()
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
        ) = calc_available_water_ice_lens(cell, water_frac, split, cell, outflow=True)
    else:
        raise ValueError(
            "Trying to calculate outflow from a cell that is neither a "
            "lake or has an ice lens"
        )
    # Whether we are moving from a lake or ice lens, scale by the outflow
    # proportion.
    water_to_move = water_to_move * outflow_proportion
    water_out = 0.0
    # either: remove water from the lake directly.
    if cell["lake"] and not cell["lid"]:
        current_lake_depth = temporary_cell["lake_depth"]

        if cell["lake_depth"] < water_to_move:
            water_out = current_lake_depth
            temporary_cell["lake_depth"] = 0
        else:
            temporary_cell["lake_depth"] -= water_to_move
            water_out = water_to_move
        if temporary_cell["lake_depth"] < 0:
            # account for rounding errors
            if temporary_cell["lake_depth"] < -1e-12:
                raise ValueError("Lake depth has gone below 0")
            temporary_cell["lake_depth"] = 0
    # otherwise, loop through the column and remove water from it,
    # going from the top.
    else:
        for _l in range(0, cell["ice_lens_depth"] + 1):
            layer_share = cell["water"][_l] / (split + 1)
            current_actual = temporary_cell["water"][_l]

            if layer_share > current_actual:
                layer_share = current_actual

            if layer_share > water_to_move:
                amount_to_remove = water_to_move
                water_to_move = 0
            else:
                amount_to_remove = layer_share
                water_to_move -= layer_share
            # actually remove the water
            temporary_cell["water"][_l] -= amount_to_remove
            water_out += amount_to_remove

            if temporary_cell["water"][_l] <= 1e-12:
                temporary_cell["water"][_l] = 0
                temporary_cell["saturation"][_l] = 0
            # stop looping if we've flowed all the water
            if water_to_move <= 0:
                break
    return water_out


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
            "Moving water has caused lake depth to go below 0 - in the central cell"
        )
    return temporary_cell, temporary_neighbour


@kernel()
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

    for d in range(_N_DIRS):
        # Skip directions that are not in the set of downhill neighbours.
        is_target = False
        for k in range(len(biggest_neighbours)):
            if biggest_neighbours[k] == d:
                is_target = True
                break
        if not is_target:
            continue

        # Cells with a frozen lid cannot release water laterally.
        if cell["lid"]:
            return 0.0

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

        is_outflow = False
        # Case 1: source cell has a lake.
        if cell["lake"]:
            water_to_move, _, _, _ = calc_available_water_lake(
                cell,
                water_frac,
                split,
                grid[row + n_s_index][col + w_e_index],
                outflow=is_outflow,
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

        # Case 2: source cell has an ice lens (and no lake).
        elif cell["ice_lens"] and not cell["lake"]:
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
                outflow=is_outflow,
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
        else:
            # No ice lens and no lake: water percolates downward only.
            raise ValueError("This should not be happening...")

    return total_water_out


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
    # ---------------------------------------------------------------------------
    # Phase 1: initialise water levels and build temp_grid snapshot.
    # Safe to parallelise with prange: each iteration writes only to its own
    # cell.  water_direction is also zeroed here so the diagnostic field is
    # clean at the start of every move_water call.
    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # Phase 2: compute water movements using the 9-colour (3×3 tile) scheme.
    #
    # Why 9 colours?
    # Each cell can write to any of its 8 diagonal/cardinal neighbours in
    # temp_grid.  Two cells processed simultaneously must not share a
    # neighbour, otherwise we get a write-write race condition.  The cells of
    # a single (phase_r, phase_c) colour are spaced ≥3 apart in every
    # dimension, so their 8-neighbour sets are always disjoint.
    #
    # Each phase is a prange over rows, skipping rows that do not belong to the
    # current colour.  The 2/3 of iterations that hit ``continue`` execute only
    # two cheap modulo comparisons — the overhead is negligible.
    #
    # Thread-level parallelism (prange → actual threads) would require
    # move_water to be compiled with @njit(parallel=True). It is currently
    # registered as a plain @kernel (see monarchs.core.kernels), so prange
    # behaves as range and the phases run serially; enabling parallel=True is a
    # deliberate future step (verify the 9-colour scheme is race-free first).
    # ---------------------------------------------------------------------------
    for phase_r in range(3):
        for phase_c in range(3):
            for row in prange(max_grid_row):
                if row % 3 != phase_r:
                    continue
                for col in range(max_grid_col):
                    if col % 3 != phase_c:
                        continue
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
