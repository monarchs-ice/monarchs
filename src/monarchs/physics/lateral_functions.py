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
    if not cell["valid_cell"]:
        cell["water_level"] = 999
        return
    elif not cell["lake"] and not cell["lid"]:
        if cell["ice_lens"]:
            if not np.any(cell["saturation"][: cell["ice_lens_depth"] + 1] > 0):
                top_saturation_depth = cell["ice_lens_depth"]
            else:
                top_saturation_depth = np.argmax(
                    cell["saturation"][: cell["ice_lens_depth"] + 1] > 0
                )
            cell["water_level"] = cell["vertical_profile"][::-1][top_saturation_depth]
        else:
            cell["water_level"] = 0
        cell["water"] = cell["Lfrac"] * (cell["firn_depth"] / cell["vert_grid"])
    elif cell["lake"] and not cell["lid"]:
        cell["water_level"] = cell["lake_depth"] + cell["firn_depth"]
        cell["water"] = cell["Lfrac"] * (cell["firn_depth"] / cell["vert_grid"])
    elif cell["lake_depth"] > 0.1 and not cell["lid"]:
        cell["water_level"] = cell["lake_depth"] + cell["firn_depth"]
        cell["water"] = cell["Lfrac"] * (cell["firn_depth"] / cell["vert_grid"])
    elif cell["lid"]:
        cell["water_level"] = 999
        cell["water"] = cell["Lfrac"] * (cell["firn_depth"] / cell["vert_grid"])


def get_neighbour_water_levels(cell, grid, col, row, max_grid_col, max_grid_row):
    neighbours = {}
    if row > 0:
        neighbours["N"] = cell["water_level"] - grid["water_level"][row - 1][col]
        if col < max_grid_col - 1:
            neighbours["NE"] = (
                cell["water_level"] - grid["water_level"][row - 1][col + 1]
            )
        if col > 0:
            neighbours["NW"] = (
                cell["water_level"] - grid["water_level"][row - 1][col - 1]
            )
    else:
        neighbours["N"] = -999
        neighbours["NE"] = -999
        neighbours["NW"] = -999
    if row < max_grid_row - 1:
        neighbours["S"] = cell["water_level"] - grid["water_level"][row + 1][col]
        if col < max_grid_col - 1:
            neighbours["SE"] = (
                cell["water_level"] - grid["water_level"][row + 1][col + 1]
            )
        if col > 0:
            neighbours["SW"] = (
                cell["water_level"] - grid["water_level"][row + 1][col - 1]
            )
    else:
        neighbours["S"] = -999
        neighbours["SE"] = -999
        neighbours["SW"] = -999
    if col > 0:
        neighbours["W"] = cell["water_level"] - grid["water_level"][row][col - 1]
    else:
        neighbours["W"] = -999
        neighbours["SW"] = -999
        neighbours["NW"] = -999
    if col < max_grid_col - 1:
        neighbours["E"] = cell["water_level"] - grid["water_level"][row][col + 1]
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
        "2": [-1, 0],  # W
        "3": [-1, 1],  # SW
        "4": [0, -1],  # S
        "5": [0, 1],  # N
        "6": [1, -1],  # SE
        "7": [1, 0],  # E
        "8": [1, 1],  # NE
    }
    """
    neighbours = get_neighbour_water_levels(
        cell, grid, col, row, max_grid_col, max_grid_row
    )
    if catchment_outflow and max(neighbours.values()) <= 0:
        for key in neighbours.keys():
            if neighbours[key] == -999:
                neighbours[key] = 9999
    if flow_into_land:
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                try:
                    code = ""
                    if i == -1:
                        code += "N"
                    elif i == 1:
                        code += "S"
                    if j == -1:
                        code += "W"
                    elif j == 1:
                        code += "E"
                    neighbour_cell = grid[row + i][col + j]
                    if max(neighbours.values()) <= 0:
                        if not neighbour_cell["valid_cell"]:
                            neighbours[code] = 9998
                    elif not neighbour_cell["valid_cell"]:
                        neighbours[code] = -9999
                except IndexError:
                    continue
    biggest_height_difference = max(neighbours.values())
    max_list = [k for k, v in neighbours.items() if v == biggest_height_difference]
    for k, v in neighbours.items():
        if biggest_height_difference - v < 1e-08:
            if k not in max_list:
                max_list.append(k)
    return biggest_height_difference, max_list


def water_fraction(cell, m, timestep, direction):
    """
    Determine the fraction of water that is allowed to move through the solid firn, dependent on its density.
    Called in <move_water>.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object that water is moving through
    m : float
        difference in heights between this cell and the cell the water is moving to
    timestep : int
        time in which the water can move - default 3600 (set by user in runscript) [s]
    direction: str
        direction of the water flow - used to determine the "cell size", as these can be non-square.

    Returns
    -------
    water_frac : np.ndarray or float
        Fraction of the total water that is allowed to move. Either a single number if a lake is present,
        or a Numpy array of length cell.vert_grid if not. M is biggest height difference
    """
    if direction in ["NW", "NE", "SW", "SE"]:
        cell_size = np.sqrt(cell["size_dx"] ** 2 + cell["size_dy"] ** 2)
    elif direction in ["N", "S"]:
        cell_size = cell["size_dy"]
    elif direction in ["E", "W"]:
        cell_size = cell["size_dx"]
    else:
        raise ValueError("Direction not recognised")
    Big_pi = -2.53 * 10**-10
    eta = 1.787 * 10**-3
    cell["rho"] = cell["Sfrac"] * cell["rho_ice"] + cell["Lfrac"] * cell["rho_water"]
    if cell["lake"]:
        water_frac = 1
    else:
        cell_density = cell["rho"]
        u = Big_pi / eta * (m / cell_size) * cell_density * -9.8
        water_frac = u * timestep / cell_size
        water_frac[np.where(water_frac > 1)] = 1
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
    if cell["lake"]:
        if outflow:
            water_to_move = cell["lake_depth"]
            return water_to_move
        water_to_move = water_frac * (
            (cell["water_level"] - neighbour_cell["water_level"]) / (split + 1)
        )
        if cell["lake_depth"] < water_to_move * (split + 1):
            water_to_move = cell["lake_depth"] / (split + 1)
        return water_to_move
    elif cell["ice_lens"] and not cell["lake"]:
        if outflow:
            lowest_water_level = cell["vertical_profile"][::-1][cell["ice_lens_depth"]]
            vp = cell["vertical_profile"][::-1]
            move_from_index = find_nearest(vp, lowest_water_level)
            water_to_move = np.sum(cell["water"][: move_from_index + 1])
            return water_to_move
        top_saturation_level = np.argmax(
            cell["saturation"][: cell["ice_lens_depth"] + 1] > 0
        )
        if (
            cell["vertical_profile"][::-1][cell["ice_lens_depth"]]
            > neighbour_cell["water_level"]
        ):
            lowest_water_level = cell["vertical_profile"][::-1][cell["ice_lens_depth"]]
        else:
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
        if np.sum(cell["water"][: move_from_index + 1]) / (split + 1) < water_to_move:
            water_to_move = np.sum(cell["water"][: move_from_index + 1]) / (split + 1)
        return (
            water_to_move,
            lowest_water_level,
            move_from_index,
            top_saturation_level,
        )
    else:
        raise ValueError("Water should not be able to flow from this cell")


def calc_catchment_outflow(cell, temporary_cell, water_frac, split):
    water_to_move = calc_available_water(cell, water_frac, split, outflow=True)
    if cell["lake"]:
        water_out = np.copy(water_to_move)
        if cell["lake_depth"] < water_to_move:
            water_to_move = cell["lake_depth"]
            temporary_cell.lake_depth = 0
            water_out = water_to_move
            water_to_move = 0
        else:
            temporary_cell.lake_depth -= water_to_move
            if temporary_cell.lake_depth < 0:
                raise ValueError("Lake depth has gone below 0")
    else:
        water_out = 0
        try:
            for _l in range(0, cell["ice_lens_depth"] + 1):
                if cell["water"][_l] > water_to_move:
                    temporary_cell.water[_l] -= water_to_move
                    temporary_cell.saturation[_l] = 0
                    water_out += water_to_move
                    water_to_move = 0
                else:
                    water_to_move -= cell["water"][_l]
                    water_out += cell["water"][_l]
                    temporary_cell.water[_l] -= cell["water"][_l]
        except IndexError:
            print(_l)
            print(water_out)
            print(cell["ice_lens_depth"])
            raise IndexError
    return water_out


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
):
    """
    Moves water from the central cell (grid[i][j]) to the neighbours with the lowest water level.
    The specific neighbour is associated with an index [i + m][j + n] with m, n = [-1, 0, 1].
    e.g. SW neighbour is associated with the m = +1, n = -1.
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
    biggest_height_difference : float
        largest difference in water level between the central cell and its neighbours. Used in `water_fraction`.
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
        "NW": [-1, -1],
        "N": [-1, 0],
        "NE": [-1, 1],
        "E": [0, 1],
        "SE": [1, 1],
        "S": [1, 0],
        "SW": [1, -1],
        "W": [0, -1],
    }
    for idx, neighbour in enumerate(all_neighbours.keys()):
        if neighbour in biggest_neighbours:
            cell = grid[row][col]
            if cell["lid"]:
                return 0
            n_s_index = all_neighbours[neighbour][0]
            w_e_index = all_neighbours[neighbour][1]
            temporary_cell = temp_grid[row][col]
            water_frac = water_fraction(
                cell, biggest_height_difference, timestep, neighbour
            )
            try:
                if (
                    row + n_s_index == -1
                    or col + w_e_index == -1
                    or row + n_s_index >= len(grid)
                    or col + w_e_index >= len(grid[0])
                ):
                    raise IndexError
                neighbour_cell = grid[row + n_s_index][col + w_e_index]
                temporary_neighbour = temp_grid[row + n_s_index][col + w_e_index]
                if not neighbour_cell["valid_cell"] and not flow_into_land:
                    continue
                elif not neighbour_cell["valid_cell"] and flow_into_land:
                    water_out = calc_catchment_outflow(
                        cell, temporary_cell, water_frac, split
                    )
                    if water_out > 0:
                        print(f"Moved {water_out} units of water into the land")
                    if temporary_cell.lake_depth < 0:
                        print(
                            f"temporary_cell.lake_depth = {temporary_cell.lake_depth}, col = {col}, row = {row}"
                        )
                        print(
                            f"split = {split}, water_frac = {water_frac}, cell.lake_depth = {cell['lake_depth']}"
                        )
                    return water_out
                if cell["lid"] or neighbour_cell["lid"]:
                    continue
                if cell["lake"]:
                    water_to_move = calc_available_water(
                        cell, water_frac, split, neighbour_cell=neighbour_cell
                    )
                elif cell["ice_lens"] and not cell["lake"]:
                    (
                        water_to_move,
                        lowest_water_level,
                        move_from_index,
                        top_saturation_level,
                    ) = calc_available_water(
                        cell, water_frac, split, neighbour_cell=neighbour_cell
                    )
            except IndexError:
                if catchment_outflow:
                    water_out = calc_catchment_outflow(
                        cell, temporary_cell, water_frac, split
                    )
                    return water_out
                else:
                    raise ValueError(
                        "Issue with lateral movement - trying to move to a non-existent grid cell and <catchment_outflow> is not enabled"
                    )
            if cell["lake"]:
                if neighbour_cell["lake"]:
                    temporary_neighbour.lake_depth += water_to_move
                elif (
                    cell["firn_depth"] + cell["lake_depth"]
                    > neighbour_cell["firn_depth"]
                ):
                    temporary_neighbour.water[0] += water_to_move
                else:
                    move_to_index = find_nearest(
                        neighbour_cell["vertical_profile"][::-1], cell["water_level"]
                    )
                    temporary_neighbour.water[move_to_index] += water_to_move
                temporary_cell.lake_depth -= water_to_move
                if 0 > temporary_cell.lake_depth > -1e-12:
                    temporary_cell.lake_depth = 0
                    print("Fixed floating point error in lake depth")
                if temporary_cell.lake_depth < 0:
                    print("After = ", temporary_cell.lake_depth)
                    print("Before = ", cell["lake_depth"])
                    raise ValueError(
                        "Moving water has caused lake depth to go below 0 - in the central cell"
                    )
            elif cell["ice_lens"] and not cell["lake"]:
                if not neighbour_cell["lake"]:
                    move_to_index = find_nearest(
                        neighbour_cell["vertical_profile"][::-1], lowest_water_level
                    )
                else:
                    move_to_index = 0
                for idx in range(top_saturation_level, move_from_index + 1):
                    if cell["water"][idx] / (split + 1) > water_to_move:
                        temporary_cell.water[idx] -= water_to_move
                        if not neighbour_cell["lake"]:
                            temporary_neighbour.water[move_to_index] += water_to_move
                            temporary_neighbour.meltflag[move_to_index] = 1
                        else:
                            temporary_neighbour.lake_depth += water_to_move
                        temporary_cell.saturation[idx] = 0
                        water_to_move = 0
                    else:
                        temporary_cell.water[idx] -= cell["water"][idx] / (split + 1)
                        water_to_move -= cell["water"][idx] / split
                        if not neighbour_cell["lake"]:
                            temporary_neighbour.water[move_to_index] += cell["water"][
                                idx
                            ] / (split + 1)
                            temporary_neighbour.meltflag[move_to_index] = 1
                        else:
                            temporary_neighbour.lake_depth += cell["water"][idx] / (
                                split + 1
                            )
                        temporary_cell.saturation[idx] = 0
            else:
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

    def __init__(self, lake_depth, water, saturation, meltflag, lake):
        self.lake_depth = lake_depth
        self.water = water
        self.saturation = saturation
        self.meltflag = meltflag
        self.lake = lake


def move_water(
    grid,
    max_grid_col,
    max_grid_row,
    timestep,
    catchment_outflow=True,
    lateral_movement_percolation_toggle=True,
    flow_into_land=False,
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
    total_water = 0
    catchment_out_water = 0
    from monarchs.core.utils import get_2d_grid

    for row in range(0, max_grid_row):
        for col in range(0, max_grid_col):
            cell = grid[row][col]
            update_water_level(cell)
            total_water += np.sum(cell["water"]) + cell["lake_depth"]
    temp_grid = []

    for row in range(max_grid_row):
        _l = []

        for col in range(max_grid_col):
            _l.append(
                TemporaryCell(
                    np.copy(grid["lake_depth"][row][col]),
                    np.copy(grid["water"][row][col]),
                    np.copy(grid["saturation"][row][col]),
                    np.copy(grid["meltflag"][row][col]),
                    np.copy(grid["lake"][row][col]),
                )
            )
        temp_grid.append(_l)

    for row in range(max_grid_row):
        for col in range(max_grid_col):
            cell = grid[row][col]

            if cell["valid_cell"] and (
                cell["ice_lens"]
                and (cell["water"] > 0).any()
                or cell["lake"]
                and cell["lake_depth"] > 0
            ):
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
                split = len(max_list)

                if biggest_height_difference > 0:
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
                    )

            if (cell["water"] < 0).any():
                raise ValueError("cell.water is negative")
    new_water = 0

    for row in range(0, max_grid_row):
        for col in range(0, max_grid_col):

            cell = grid[row][col]
            temporary_cell = temp_grid[row][col]
            cell["water"] = np.around(temporary_cell.water, 10)
            cell["lake_depth"] = np.around(temporary_cell.lake_depth, 10)
            cell["saturation"] = temporary_cell.saturation
            cell["meltflag"] = temporary_cell.meltflag
            new_water += np.sum(cell["water"]) + cell["lake_depth"]
            if cell["valid_cell"]:
                cell["Lfrac"] = cell["water"] / (cell["firn_depth"] / cell["vert_grid"])
                if lateral_movement_percolation_toggle:
                    percolation(cell, 3600, lateral_refreeze_flag=True)
                    for k in range(cell["vert_grid"])[::-1]:
                        calc_saturation(cell, k, end=True)
            update_water_level(cell)

    print('Water level at end of timestep: ', get_2d_grid(grid, 'water_level'))
    print("\nLateral water movement diagnostics:")
    print("Starting water total = ", total_water)
    print("Finishing water total = ", new_water)
    print("Difference in water total = ", total_water - new_water)
    if catchment_outflow:
        print("Catchment outflow = ", catchment_out_water)
    if abs(total_water - new_water - catchment_out_water) > 0.0001:
        raise ValueError(
            "monarchs.physics.lateral_functions.move_water:Too much water has appeared out of nowhere during lateral functions"
        )
    return grid, catchment_out_water
