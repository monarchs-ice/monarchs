"""
Functions for determining which neighbours we need to move water into.
"""

import numpy as np
from monarchs.core.kernels import kernel
from monarchs.physics.lateral.directions import _DIR_ROW, _DIR_COL, _N_DIRS


@kernel()
def get_neighbour_water_levels(cell, grid, col, row, max_grid_col, max_grid_row):
    """
    Get an array of water level differences between the grid cell and its neighbours.
    Previously this was a whole load of if/else logic based on the direction.
    Now we just loop over the possible directions defined in ``lateral.directions``.

    If a neighbour is beyond the edge, then it is given a value of -999 so that it can't be
    flowed into or from.
    """

    # default to -999
    diffs = np.full(_N_DIRS, -999.0)
    # then for each direction, assign the difference if the neighbour is in bounds
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

    # Ensure that if catchment_outflow is set, water that reaches the edge of
    # the catchment area will flow outward, if it is in the lowest cell
    # locally. This way, water will preferentially stay in the model.
    if catchment_outflow and np.max(diffs) <= 0:
        for d in range(_N_DIRS):
            if diffs[d] == -999.0:
                diffs[d] = 9999.0

    # if flow_into_land is True, then if the cell is at a local minimum aside
    # from invalid cells, then it will flow into the land. This is motivated by
    # the appearance of lakes on the ice shelf land-boundary in testing runs,
    # which is not seen in the validation data. This is placed after the
    # initial loop, since this should only occur if the water level is a local
    # minimum.
    if flow_into_land:
        current_max = np.max(diffs)
        for d in range(_N_DIRS):
            nr = row + _DIR_ROW[d]
            nc = col + _DIR_COL[d]
            if 0 <= nr < max_grid_row and 0 <= nc < max_grid_col:
                # If the water level is a local minimum, then we want to
                # flow water into the land if at the edge.
                if current_max <= 0 and not grid[nr][nc]["valid_cell"]:
                    diffs[d] = 9998.0
                # Otherwise, ensure we *don't* flow water into land as it
                # has places it can go within the model.
                elif not grid[nr][nc]["valid_cell"]:
                    diffs[d] = -9999.0

    # Find neighbour with the biggest height difference in water level
    biggest_height_difference = np.max(diffs)

    # Check a new list of indices/value pairs to see which one corresponds
    # to the maximum calculated above
    # now implemented as arrays rather than pure Python with dicts
    max_dirs = np.zeros(_N_DIRS, dtype=np.int32)
    n_max = 0

    # A fix for the symmetric test case. What was happening was that there were
    # tiny rounding errors. This caused water to preferentially move one way
    # over the other, which is bad for a symmetric case! This fixes the issue.
    # Check if any directions are within a rounding error of the maximum
    # and include it if so
    for d in range(_N_DIRS):
        if biggest_height_difference - diffs[d] < 1e-8:
            max_dirs[n_max] = d
            n_max += 1

    return biggest_height_difference, max_dirs[:n_max]
