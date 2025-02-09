# All functions related to the lateral transport of meltwater in the MONARCHS model
import numpy as np
from monarchs.core.utils import find_nearest
from monarchs.physics.percolation_functions import percolation, calc_saturation


def update_water_level(cell):
    """
    Determine the water level of a single IceShelf object, so we can determine where water flows laterally to and from.
    This is determined by the presence of lakes, lids or ice lenses within the firn column.
    If there is no lake, lid or ice lens, then the entire grid cell is free for water to move into it.
    If there is no lake or lid, but there is an ice lens then the water level is the level of the highest point
    at which we have saturated firn.
    If a cell has a lake, but no lid, then the water level is the height of that lake.
    Finally, if we have a lid, we set the water level to be arbitrarily high, as we are not currently
    interested in water flow from a frozen lid as it is too complicated to model.
    Called in <move_water>.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object that we want to determine the water level of.

    Returns
    -------

    """
    # cell.water is only used for the lateral movement. So we first need to update it based on Lfrac,
    # which is used in the rest of MONARCHS.
    if not cell.lake and not cell.lid:
        if cell.ice_lens:
            # We find the water level by the topmost bit of saturated firn above the ice lens.

            if not np.any(cell.saturation[: cell.ice_lens_depth + 1] > 0):
                top_saturation_depth = cell.ice_lens_depth
            else:
                top_saturation_depth = np.argmax(
                    cell.saturation[: cell.ice_lens_depth + 1] > 0
                )

            cell.water_level = cell.vertical_profile[::-1][top_saturation_depth]

        # Otherwise, water is free to percolate all the way to the bottom, so it doesn't move laterally from here.
        else:
            cell.water_level = 0
        # Update the water content based on the Lfrac of the firn
        cell.water = cell.Lfrac * (cell.firn_depth / cell.vert_grid)

    # Add lake depth into water for the purposes of moving it around if a lake is present.
    elif cell.lake and not cell.lid:
        # cell.water[0] = cell.water[0] + cell.lake_depth
        # Determine the water level from the water on top + the firn depth.
        cell.water_level = cell.lake_depth + cell.firn_depth
        cell.water = cell.Lfrac * (cell.firn_depth / cell.vert_grid)
    elif cell.lake_depth > 0.1 and not cell.lid:
        # Same again - account for a bug where lake switch doesn't activate
        cell.water_level = cell.lake_depth + cell.firn_depth
        cell.water = cell.Lfrac * (cell.firn_depth / cell.vert_grid)

    elif cell.lid:
        # cell.water_level = (cell.water_level + cell.firn_depth
        #                     + cell.lake_depth + cell.lid_depth)
        cell.water_level = 999
        cell.water = cell.Lfrac * (cell.firn_depth / cell.vert_grid)


def get_neighbour_water_levels(cell, grid, col, row, max_grid_col, max_grid_row):
    # if not on left edge of domain - init W, possibly NW/NW
    neighbours = {}

    if row > 0:
        neighbours['W'] = cell.water_level - grid[col][row - 1].water_level

        if col < max_grid_col - 1:  # if not at right edge - init NE
            neighbours['SW'] = cell.water_level - grid[col + 1][row - 1].water_level
        if col > 0:  # if not at left edge - init NW
            neighbours['NW'] = cell.water_level - grid[col - 1][row - 1].water_level
    else:  # if at left edge - W/NW/SW all False
        neighbours['W'] = -999
        neighbours['NW'] = -999
        neighbours['SW'] = -999

    # if not on right edge of domain - init E, possibly SE/NE
    if row < max_grid_row - 1:
        neighbours['E'] = cell.water_level - grid[col][row + 1].water_level

        if col < max_grid_col - 1:  # if not at bottom - init SE
            neighbours['SE'] = cell.water_level - grid[col + 1][row + 1].water_level
        if col > 0:  # if not at top - init NE
            neighbours['NE'] = cell.water_level - grid[col - 1][row + 1].water_level

    else:  # if at left edge - E/NE/SE all False
        neighbours['E'] = -999
        neighbours['SE'] = -999
        neighbours['NE'] = -999

    if col > 0:  # if not at top edge - initialise north neighbour
        neighbours['N'] = cell.water_level - grid[col - 1][row].water_level
    else:
        neighbours['N'] = -999
        neighbours['NE']= -999
        neighbours['NW'] = -999

    # if not at bottom edge - initialise south neighbour
    if col < max_grid_col - 1:
        neighbours['S'] = cell.water_level - grid[col + 1][row].water_level
    else:
        neighbours['S'] = -999
        neighbours['SE'] = -999
        neighbours['SW'] = -999
    return neighbours


def find_biggest_neighbour(
        cell, grid, col, row, max_grid_col, max_grid_row, catchment_outflow, flow_into_land=False
):
    """
    Finds the points in grid surrounding a central point cell that have the highest difference
    in water level.
    Called in <move_water>.

    Parameters
    ----------
    cell: IceShelf
        IceShelf object corresponding to a point in grid.
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
        List containing the values of the largest difference(s) in water level between cell and its neighbours.
    max_list: List
        List containing the indices corresponding to the largest neighbour(s).

    all_neighbours = {
        "1": [-1, -1],  # NW
        "2": [-1, 0],  # N
        "3": [-1, 1],  # NE
        "4": [0, -1],  # W
        "5": [0, 1],  # E
        "6": [1, -1],  # SW
        "7": [1, 0],  # S
        "8": [1, 1],  # SE
    }
    """

    neighbours = get_neighbour_water_levels(cell, grid, col, row, max_grid_col, max_grid_row)
    # Ensure that if catchment_outflow is set, water that reaches the edge of the
    # catchment area will flow outward, if it is in the lowest cell locally.
    # This way, water will preferentially stay in the model.
    if catchment_outflow and max(neighbours.values()) <= 0:
        for key in neighbours.keys():
            if neighbours[key] == -999:
                neighbours[key] = 9999  # set to 9999 so that this overrides flow_into_land

    # if flow_into_land is True, then if the cell is at a local mininum aside
    # from invalid cells, then it will flow into the land. This is motivated by the appearance
    # of lakes on the ice shelf-land boundary in testing runs, which is not seen in the validation
    # data. This is placed after the initial loop, since this should only occur if the water level
    # is a local minimum.
    if flow_into_land and max(neighbours.values()) <= 0:
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                neighbour_cell = grid[col + i][row + j]
                if not neighbour_cell.valid_cell:
                    neighbour_cell.water_level = -999

        # run the algorithm again based on the new water levels.
        neighbours = get_neighbour_water_levels(cell, grid, col, row, max_grid_col, max_grid_row)

    # Find neighbour with the biggest height difference in water level
    biggest_height_difference = max(neighbours.values())

    # check a new list of indices/value pairs to see which one corresponds
    # to the maximum calculated above
    # max list is a list of the indices of m that have the largest value
    # e,g. if the NW and SE neighbours have the biggest value,
    # max_list = [0, 7]
    # max_list = [
    #     idx for idx, val in enumerate(neighbour_list) if (val == biggest_height_difference and val > 0)
    # ]
    max_list = [k for k, v in neighbours.items() if v == biggest_height_difference]

    return biggest_height_difference, max_list


def water_fraction(cell, m, cell_size, timestep):
    """
    Determine the fraction of water that is allowed to move through the solid firn, dependent on its density.
    Called in <move_water>.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object that water is moving through
    m : float
        difference in heights between this cell and the cell the water is moving to
    cell_size : float
        size of each lateral grid cell [m]
    timestep : int
        time in which the water can move - default 3600 (set by user in runscript) [s]

    Returns
    -------
    water_frac : np.ndarray or float
        Fraction of the total water that is allowed to move. Either a single number if a lake is present,
        or a Numpy array of length cell.vert_grid if not. M is biggest height difference
    """
    Big_pi = -2.53 * 10 ** -10  # hydraulic permeability (m^2)
    eta = 1.787 * 10 ** -3  # viscosity(Pa/s)
    if cell.lake:
        return 1  # if in a lake all water moves

    cell_density = cell.rho

    u = Big_pi / eta * m / cell_size * cell_density * -9.8  # flow speed (m/s)

    # This block ensures that u is not greater than 1, which would be unphysical.
    if (
            cell.lake
    ):  # if lake, then we only are interested in rho_water so u is a float (not an array)
        u = min(u, 1)
    else:  # otherwise we look at the density of a specific point in the firn so u is an array
        u[np.where(u > 1)] = 1  # All water can move in one timestep

    water_frac = u * timestep / cell_size

    # water_move = cell.water * water_frac
    return water_frac


def calc_available_water(cell, water_frac, split, neighbour_cell=False, outflow=False):
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
    # > water_to_move * split + 1 = central_cell - outside_cell
    # > water_to_move = central_cell - outside_cell / (split + 1)
    # only the case if outside cells are equal so far - maybe change in future
    # We need to move enough water for the water levels to equalise.
    # If lake depth < difference between water levels, then all needs to move.
    if cell.lake:
        if outflow:  # outflow case, so most of the calculation is irrelevant
            water_to_move = cell.lake_depth  # if flowing out, whole lake drains away.
            return water_to_move

        # Otherwise, proceed as normal
        # enough water should move such that we equalise water level, accounting for moving to multiple cells.
        water_to_move = water_frac * (
                (cell.water_level - neighbour_cell.water_level) / (split + 1)
        )
        # Ensure we have enough water to fill our "quota" - else everything goes but no more.
        # multiply by split + 1 since we want the total water that can move.
        # water frac is always 1 if the central cell is a lake
        # so doesn't pose a problem
        if cell.lake_depth < water_to_move * (split):
            water_to_move = cell.lake_depth / (split)
        return water_to_move

    elif cell.ice_lens and not cell.lake:
        if outflow:
            lowest_water_level = cell.vertical_profile[::-1][cell.ice_lens_depth]
            vp = cell.vertical_profile[::-1]
            move_from_index = find_nearest(vp, lowest_water_level)
            water_to_move = np.sum(cell.water[: move_from_index + 1])
            return water_to_move
        # Determine the highest point of the water - this is needed later
        top_saturation_level = np.argmax(cell.saturation[: cell.ice_lens_depth + 1] > 0)

        # The water moves from the lowest level it can physically do so
        # i.e. the ice lens depth:
        if cell.vertical_profile[::-1][cell.ice_lens_depth] > neighbour_cell.water_level:
            lowest_water_level = cell.vertical_profile[::-1][cell.ice_lens_depth]

        else:  # or the midpoint of the water depths
            lowest_water_level = cell.vertical_profile[::-1][
                find_nearest(
                    cell.vertical_profile[::-1],
                    (cell.water_level + neighbour_cell.water_level) / (split + 1),
                )
            ]

        vp = cell.vertical_profile[::-1]
        move_from_index = find_nearest(vp, lowest_water_level)

        water_to_move = (
                water_frac[move_from_index]
                * ((cell.water_level - neighbour_cell.water_level) / 2)
                / (split + 1)
        )

        # If saturated firn doesn't hold enough water to fill the "quota", then all of it moves but no more
        # +1 so we include the one where we move from
        if (
                np.sum(cell.water[: move_from_index + 1]) / (split + 1) < water_to_move
        ):  # i.e. if more water is allowed to move than we have in the cell
            water_to_move = np.sum(cell.water[: move_from_index + 1]) / (split + 1)
        return (
            water_to_move,
            lowest_water_level,
            move_from_index,
            top_saturation_level,
        )
    else:
        return 0  # water should not be able to move from this cell


def calc_catchment_outflow(cell, temporary_cell, water_frac, split):
    water_to_move = calc_available_water(
        cell, water_frac, split, outflow=True
    )
    if cell.lake:  # remove water from the lake directly.
        water_out = np.copy(water_to_move)
        if cell.lake_depth < water_to_move:
            water_to_move = cell.lake_depth
            temporary_cell.lake_depth -= water_to_move
            water_out = water_to_move
        else:
            temporary_cell.lake_depth -= water_to_move
    else:  # Otherwise, loop through the column and remove water from it, going from the top.
        water_out = 0
        try:
            for _l in range(0, cell.ice_lens_depth + 1):
                if cell.water[_l] > water_to_move:
                    temporary_cell.water[_l] -= water_to_move
                    temporary_cell.saturation[_l] = 0
                    water_out += water_to_move
                    water_to_move = 0
                else:
                    water_to_move -= cell.water[_l]
                    water_out += cell.water[_l]
                    temporary_cell.water[_l] -= cell.water[_l]

        except IndexError:
            print(_l)
            print(water_out)
            print(cell.ice_lens_depth)
            raise IndexError

    return water_out


def move_to_neighbours(
        grid,
        temp_grid,
        biggest_neighbours,
        col,
        row,
        water_frac,
        split,
        catchment_outflow,
        flow_into_land=False
):
    """
    Moves water from the central cell (grid[i][j]) to the neighbours with the lowest water level.
    The specific neighbour is associated with an index [i + m][j + n] with m, n = [-1, 0, 1].
    e.g. SW neighbour is associated with the m = -1, n = +1.
    How the movement is handled is dependent on the features of the cell and its neighbour.
    Called in <move_water>.

    Parameters
    ----------
    grid : List, or numba.typed.List
        grid of IceShelf objects
    biggest_neighbours : List
        List of the largest neighbour cells calculated from find_biggest_neighbour
    i : int
        current row index
    j : int
        current column index
    water_frac : float
        Fraction of water that is allowed to move, from water_fraction
    split : int
        Number determining whether water is split between two or more neighbour grid cells, if there are cells of
        equal water_level

    Returns
    -------
    None (amends instance of grid passed into it)

    Raises
    ------
    ValueError
        If model reaches a state that should not be able to occur, then raise this error.
        This can occur if the lake depth goes below 0, or if there is water to move in a grid cell where water
        should not be able to move from (i.e. no lake or ice lens)
    """

    all_neighbours = {
        "NW": [-1, -1],  # NW
        "N": [-1, 0],  # N
        "NE": [-1, 1],  # NE
        "W": [0, -1],  # W
        "E": [0, 1],  # E
        "SW": [1, -1],  # SW
        "S": [1, 0],  # S
        "SE": [1, 1],  # SE
    }

    # for each of the neighbours we wish to move water to, which depends on the difference
    # in water level between the central cell and the cells at each cardinal point

    for idx, neighbour in enumerate(all_neighbours.keys()):
        if neighbour in biggest_neighbours:
            n_s_index = all_neighbours[neighbour][0]
            w_e_index = all_neighbours[neighbour][1]
            cell = grid[col][row]
            temporary_cell = temp_grid[col][row]

            if cell.lid:
                return 0

            try:
                if (
                        col + n_s_index == -1 or row + w_e_index == -1
                ):  # edge case handling - since Python handles "-1" as
                    # a valid index this will impose periodic boundary conditions rather than raise an error, which
                    # is not the behaviour we want!
                    raise IndexError
                neighbour_cell = grid[col + n_s_index][row + w_e_index]
                temporary_neighbour = temp_grid[col + n_s_index][row + w_e_index]
                # If cell is invalid, we aren't interested in flowing water
                if not neighbour_cell.valid_cell and not flow_into_land:
                    continue

                # unless water can flow into the land (i.e. through crevices etc.) - in which case
                # do the catchment outflow algorithm.
                elif not neighbour_cell.valid_cell and flow_into_land:
                    water_out = calc_catchment_outflow(cell, temporary_cell, water_frac, split)
                    if water_out > 0:
                        print(f'Moved {water_out} units of water into the land')
                    if temporary_cell.lake_depth < 0:
                        print(f'Lake depth = {temporary_cell.lake_depth}, i = {col}, j = {row}')
                    return water_out

                if cell.lid or neighbour_cell.lid:
                    # Too complicated to model, so we just ignore this for now
                    continue

                if cell.lake:
                    water_to_move = calc_available_water(
                        cell, water_frac, split, neighbour_cell=neighbour_cell
                    )
                elif cell.ice_lens and not cell.lake:  # in this case we need to return a bit more info

                    water_to_move, lowest_water_level, move_from_index, top_saturation_level \
                        = calc_available_water(
                        cell, water_frac, split, neighbour_cell=neighbour_cell
                    )

            # If we are moving out of the catchment area, we will get an IndexError when trying to read neighbour_cell.
            # Instead, in this case, just determine how much water will move out of the catchment area.
            # We return straight out of the function in this case as we only want to move out one way.
            except IndexError:
                if catchment_outflow:
                    water_out = calc_catchment_outflow(cell, temporary_cell, water_frac, split)
                    return water_out
                else:
                    raise ValueError(
                        "Issue with lateral movement - trying to move to a non-existent grid cell and "
                        "<catchment_outflow> is not enabled"
                    )
            # Case where the central cell has a lake.
            if cell.lake:
                if neighbour_cell.lake:  # Simplest case - lake water into lake
                    temporary_neighbour.lake_depth += water_to_move
                # Otherwise we need to move it to a specific point in the neighbour cell corresponding to the
                # topmost saturated cell above the ice lens.
                else:
                    # Find the point at which we need to add the water -
                    # either surface
                    if cell.firn_depth + cell.lake_depth > neighbour_cell.firn_depth:
                        temporary_neighbour.water[0] += water_to_move

                    else:  # or somewhere along firn profile corresponding to cell.water_level
                        move_to_index = find_nearest(
                            neighbour_cell.vertical_profile[::-1],
                            cell.water_level,
                        )
                        temporary_neighbour.water[move_to_index] += water_to_move

                # Whatever outcome it was, we need to remove the moved water from the lake
                temporary_cell.lake_depth -= water_to_move

                # Fix floating-point errors before sanity checking
                if 0 > temporary_cell.lake_depth > -0.001:
                    temporary_cell.lake_depth = 0

                if temporary_cell.lake_depth < 0:
                    print("After = ", temporary_cell.lake_depth)
                    print("Before = ", cell.lake_depth)
                    raise ValueError(
                        "Moving water has caused lake depth to go below 0 - in the central cell"
                    )

            elif cell.ice_lens and not cell.lake:  # water is moving from an ice lens
                # TODO | Rather than using water level, we actually need to account for the available pore space.
                # TODO | This will be something along the lines of calculating 1 - Sfrac - Lfrac for the selected
                # TODO | indices and weighting based on that, rather than just water_depth.
                # TODO | Should normalise between different timesteps, so not a priority right now, but possible
                # TODO | for future development.

                # This is the most complicated case - moving from a cell with an ice lens to
                # a cell with or without an ice lens, and that has no lake.
                # We need to move water from the correct vertical layer
                # of cell into the correct vertical layer of neighbour_cell. We do this in the loop after
                # this one, so that we can check that we have enough water to move from each vertical
                # layer of the central cell.
                if not neighbour_cell.lake:
                    move_to_index = find_nearest(
                        neighbour_cell.vertical_profile[::-1],
                        lowest_water_level,
                    )
                else:
                    move_to_index = 0
                # We now need to update the amount of water in the initial firn column.
                # Water can only be deducted from the area above lowest_water_level.

                for idx in range(top_saturation_level, move_from_index + 1):
                    # If more water in cell than we can move, then subtract that amount from the current cell
                    if cell.water[idx] > water_to_move:
                        temporary_cell.water[idx] -= water_to_move
                        if not neighbour_cell.lake:
                            temporary_neighbour.water[move_to_index] += water_to_move
                            temporary_neighbour.meltflag[move_to_index] = 1
                        else:
                            temporary_neighbour.lake_depth += water_to_move

                        temporary_cell.saturation[idx] = 0
                    # Otherwise - remove all of it from that cell and go up one.
                    else:
                        water_to_move -= cell.water[idx]
                        if not neighbour_cell.lake:
                            temporary_neighbour.water[move_to_index] += cell.water[idx]
                            temporary_neighbour.meltflag[move_to_index] = 1
                        else:
                            temporary_neighbour.lake_depth += cell.water[idx]
                        temporary_cell.water[idx] = 0

            else:  # If no ice lens, then no water should be able to move as
                # it should preferentially percolate downwards.
                raise ValueError("This should not be happening...")
    return 0


class TemporaryCell:
    """
    Define a temporary class for us to store info on where water has moved to and from.
    This allows us to do the lateral movement "simultaneously", rather than moving it iteratively
    (causing the end result to be determined by which cells you start moving water from)

    Attributes
    ----------
    lake_depth : float
        lake depth of the IceShelf we are mirroring
    water : array_like, float, dimension(cell.vert_grid)
        water content of the IceShelf
    saturation : array_like, bool, dimension(cell.vert_grid)
        boolean array determining whether the IceShelf is saturated at each vertical point
    """

    def __init__(self, lake_depth, water, saturation, meltflag):
        self.lake_depth = lake_depth
        self.water = water
        self.saturation = saturation
        self.meltflag = meltflag


def move_water(
        grid,
        max_grid_col,
        max_grid_row,
        cell_size,
        timestep,
        catchment_outflow=True,
        lateral_movement_percolation_toggle=True,
        flow_into_land=False
):
    """
    Loop over our grid and determine which cells water is allowed to move from/to, then perform this movement
    simultaneously. This is the main function used to perform the water movement, and calls the previously
    defined functions.

    Parameters
    ----------
    grid : List, or numba.typed.List
        Grid of IceShelf objects
    max_grid_row : int
        Number of grid rows
    max_grid_col : int
        Number of grid columns
    cell_size : float
        size of each lateral grid cell [m]
    timestep : int
        Time taken for the lateral movement to occur. This is typically 24 hours, i.e. 3600 * 24 seconds. [s]
    lateral_movement_percolation_toggle : bool, optional
        Boolean flag to determine whether to perform a percolation step after we do
        lateral water, to ensure that water is where it should be in the firn column.

    Returns
    -------
    grid : amended version of the IceShelf grid with water having moved laterally

    Raises
    ------
    ValueError
        If water has appeared out of nowhere, then the algorithm is not working correctly and the code should stop.
        This should not occur, so please get in touch with the developers, so we can source and fix the issue.
    """

    water_level_to_sort = np.zeros((max_grid_col, max_grid_row))

    # First, we need to determine the water level.
    total_water = 0
    catchment_out_water = 0
    for col in range(0, max_grid_col):
        for row in range(0, max_grid_row):
            cell = grid[col][row]
            update_water_level(cell)
            water_level_to_sort[col, row] = cell.water_level
            total_water += np.sum(cell.water) + cell.lake_depth

    # Create temporary cells for our water to move in and out of. This is required so that the water all moves
    # simultaneously.

    temp_grid = []
    for col in range(max_grid_col):
        _l = []
        for row in range(max_grid_row):
            _l.append(
                TemporaryCell(
                    grid[col][row].lake_depth,
                    grid[col][row].water,
                    grid[col][row].saturation,
                    grid[col][row].meltflag
                )
            )
        temp_grid.append(_l)

    # Now loop through the cells again, this time actually performing the movement step.
    for col in range(max_grid_col):
        for row in range(max_grid_row):
            cell = grid[col][row]
            if (cell.ice_lens and (cell.water > 0).any()) or (cell.lake and cell.lake_depth > 0):
                # Get the points with the largest difference in heights
                # TODO | Is the thing we are really interested in just "if water level > neighbour level, move water"?
                # TODO | So rather than moving all to one lower gridcell, move equally to all adjacent grid cells
                # TODO | that are lower. Or is this too complicated?
                # TODO | Too complicated for now - maybe revisit?
                biggest_height_difference, max_list = find_biggest_neighbour(
                    cell,
                    grid,
                    col,
                    row,
                    max_grid_col,
                    max_grid_row,
                    catchment_outflow,
                    flow_into_land=flow_into_land
                )
                # If more than one cell is equally lower than central the water is split between them
                split = len(max_list)
                # print('Split = ', split)
                if (
                        biggest_height_difference > 0
                ):  # Water moves if one of surrounding grid is lower than central cell
                    water_frac = water_fraction(
                        cell, biggest_height_difference, cell_size, timestep
                    )

                    # Now calculate the amount of water to move.
                    # Note that this is in the form of temporary arrays, as we need to avoid
                    # race conditions (i.e. order matters).
                    # The actual water level is updated in the next loop.
                    catchment_out_water += move_to_neighbours(
                        grid,
                        temp_grid,
                        max_list,
                        col,
                        row,
                        water_frac,
                        split,
                        catchment_outflow,
                        flow_into_land=flow_into_land
                    )

            if (cell.water < 0).any():
                raise ValueError("cell.water is negative")
    new_water = 0
    # Loop over our grid, set the values to the corresponding temporary values,
    # i.e. performing our movement in one step
    for col in range(0, max_grid_col):
        for row in range(0, max_grid_row):
            cell = grid[col][row]
            temporary_cell = temp_grid[col][row]
            cell.water = temporary_cell.water
            cell.lake_depth = temporary_cell.lake_depth

            cell.saturation = temporary_cell.saturation
            cell.meltflag = temporary_cell.meltflag
            new_water += np.sum(cell.water) + cell.lake_depth

            if cell.valid_cell:
                # Once all water has moved, update the Lfrac of each cell based on where the water now is.
                # The water level is calculated at the beginning of the next call to move_water, and is not
                # used anywhere else in the code.
                cell.Lfrac = cell.water / (cell.firn_depth / cell.vert_grid)
                # We have put all the water at one level in the firn - we need to percolate it to make sure that
                # the water fills out the available pore space.
                if lateral_movement_percolation_toggle:
                    # assume the percolation happens within 1 hour.
                    percolation(cell, 3600, lateral_refreeze_flag=True)
                    for k in range(cell.vert_grid)[::-1]:
                        calc_saturation(cell, k, end=True)

            update_water_level(cell)
    print("\nLateral water movement diagnostics:")
    print("Starting water total = ", total_water)
    print("Finishing water total = ", new_water)
    print("Difference in water total = ", total_water - new_water)
    if catchment_outflow:
        print("Catchment outflow = ", catchment_out_water)
    # Raise an error if we "create" or lose water from somewhere. This is arbitrarily set to 0.01 for now - it should be
    # nonzero as otherwise it will trigger when we have e.g. floating point errors.
    if abs(total_water - new_water - catchment_out_water) > 0.0001:
        raise ValueError(
            "monarchs.physics.lateral_functions.move_water:"
            "Too much water has appeared out of nowhere during lateral functions"
        )
    return grid, catchment_out_water
