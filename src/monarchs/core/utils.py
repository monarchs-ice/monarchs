import numpy as np
from functools import wraps

def do_not_jit(function):
    """
    An empty function used to decorate functions that we do not wish to jit-compile.
    Parameters
    ----------
    function - function to be decorated

    Returns
    -------
    wrapped function

    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper

@do_not_jit
def get_2d_grid(grid, attr, index=False):
    """
    Helper function to get a printout of a variable from a 2D grid of IceShelf
    objects.

    Parameters
    ----------
    grid : List, or nb.typed.List
        the grid of IceShelf instances
    attr : str
        the attribute you want to print out, e.g:
        get_2d_grid(grid, 'firn_depth') will print out the firn depth
        for each point on the grid at index <index>
    index : int, optional
        the index (i.e. height) at which you want to print out.
        Defaults to False, which the code interprets as the surface
        (index 0).

    Returns
    -------
    None. (but will print to stdout)

    """
    # if no index specified, get surface values
    if not index:
        index = 0
    var = [None] * len(grid)
    for row in range(len(grid)):
        var[row] = [None] * len(grid[0])
        for col in range(len(grid[0])):
            var[row][col] = getattr(
                grid[row][col], attr
            )

    # get all values
    if index == "all":
        return np.array(var)
    # otherwise
    else:
        # ensure that we get the whole array regardless of whether it is a vector or scalar
        try:
            return np.array(var)[:, :, index]
        except IndexError:
            return np.array(var)[:, :]


def find_nearest(a, a0):
    """Obtain index of element in array `a` closest to the scalar value `a0`"""
    idx = np.abs(a - a0).argmin()
    return idx


def calc_grid_mass(grid):
    """
    Calculate the total mass inside the grid, to check for mass conservation.
    Only do this for valid cells, since invalid cells will have constant mass as no physics is run on them.

    Parameters
    ----------
    grid : List, or numba.typed.List
        grid containing IceShelf objects.
    Returns
    -------
    total_mass : float
        Total mass of the whole grid, in arbitrary units.
    """
    total_mass = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            cell = grid[i][j]
            if cell.valid_cell:
                total_mass += calc_mass_sum(cell)
    return total_mass


def check_correct(cell):
    """
    Sanity checking to ensure model is still in a physically correct state. This ensures things such as the
    solid and liquid fraction being below 1 and less than 0 within a grid cell, firn depth and lake depth being
    non-negative, etc.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object that we want to check.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any of the conditions are met, raise ValueError as the model has reached an unphysical state.
        See the code body for details.
    """
    if cell.lake_depth < 0:
        print(f"monarchs.core.utils.check_correct: ")
        print("Lake depth = ", cell.lake_depth)
        raise ValueError("Lake depth must not be negative \n")
    if cell.firn_depth < 0:
        print(
            "Error: column = ",
            cell.column,
            ", row = ",
            cell.row,
            ", firn_depth = ",
            cell.firn_depth,
            "\n",
        )
        # cell.firn_depth = 0
        raise ValueError(f"monarchs.core.utils.check_correct: " "All firn has melted.")

    # For the following - use +/- 0.01 as the bounding maximum so that we have a "grace period" for the model
    # to correct itself, and to be more robust to small numerical errors or rounding errors.

    # Sfrac
    if np.any(cell.Sfrac[cell.Sfrac < -0.01]) or np.any(cell.Sfrac[cell.Sfrac > 1.01]):
        print(
            f"{np.max(cell.Sfrac)} at level {np.where((cell.Sfrac > 1) | (cell.Sfrac < 0))},"
            f" x = {cell.column}, y = {cell.row}\n"
        )
        print("Minimum Sfrac = ", np.min(cell.Sfrac))
        raise ValueError(
            f"monarchs.core.utils.check_correct: "
            "Solid fraction must be between 0 and 1 \n"
        )

    # Lfrac
    if np.any(cell.Lfrac[cell.Lfrac < -0.01]) or np.any(cell.Lfrac[cell.Lfrac > 1.01]):
        print(np.max(cell.Lfrac))
        print(np.min(cell.Lfrac))
        print(np.where(cell.Lfrac < -0.01))
        print(cell.Lfrac)
        raise ValueError(
            f"monarchs.core.utils.check_correct: "
            "Lfrac error - either above 1 or below 0"
        )
    # Sfrac + Lfrac
    total = cell.Lfrac + cell.Sfrac
    if np.any(total > 1.01):
        # print(total)
        print(f"monarchs.core.utils.check_correct: ")
        print(f"{np.max(total)} at level {np.where(total > 1)} \n")
        print("Sfrac :", cell.Sfrac[np.where(total > 1)])
        print("Lfrac :", cell.Lfrac[np.where(total > 1)])
        print(
            "Sfrac + Lfrac:",
            cell.Lfrac[np.where(total > 1)] + cell.Sfrac[np.where(total > 1)],
        )
        raise ValueError(
            f"monarchs.core.utils.check_correct: "
            "Sum of liquid and solid fraction must be less than 1 \n"
        )

def calc_mass_sum(cell):
    """
    Calculate the total mass in the grid cell in its current state.
    This is the sum of the firn column mass, the mass in the frozen lake, and the mass in the lid or virtual lid.
    This is in arbitrary units, since it is calculating the columnar mass independent of the lateral grid size, but is
    useful for consistency checking, i.e. ensuring the model does not gain or lose mass outside the known mechanisms
    (snowfall and losing water out of the catchment area laterally).

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        Instance of an IceShelf class from our model grid.

    Returns
    -------
    total_mass : float
        Amount of mass in the system, in arbitrary units.
    """
    total_mass = (
        np.sum((cell.Sfrac * cell.rho_ice * (cell.firn_depth / cell.vert_grid)))
        + np.sum((cell.Lfrac * cell.rho_water * (cell.firn_depth / cell.vert_grid)))
        + (cell.lake_depth * cell.rho_water)
        + (cell.lid_depth * cell.rho_ice)
        + (cell.v_lid_depth * cell.rho_ice)
    )
    return total_mass


def add_random_water(grid, max_grid_row, max_grid_col):
    """
    For testing - create water randomly inside the grid.

    Parameters
    ----------
    grid : List, or numba.typed.List
        grid of IceShelf objects we want to add water to
    max_grid_row : int
        Number of rows in the grid
    max_grid_col : int
        Number of columns in the grid

    Returns
    -------
    None
    """

    rand_water = np.random.rand(max_grid_row, max_grid_col) / 10
    for j in range(max_grid_col):
        for i in range(max_grid_row):
            grid[i][j].water[0] = grid[i][j].water[0] + rand_water[i][j]
            grid[i][j].water_level = grid[i][j].water[0] + grid[i][j].firn_depth


def add_edge_water(grid, max_grid_row, max_grid_col):
    """
    For testing - create water randomly at the edges of the grid.

    Parameters
    ----------
    grid : List, or numba.typed.List
        grid of IceShelf objects we want to add water to
    max_grid_row : int
        Number of rows in the grid
    max_grid_col : int
        Number of columns in the grid

    Returns
    -------
    None
    """

    edge_water = 2
    for i in range(max_grid_col):
        for j in range(max_grid_row):
            if i < 4 or max_grid_row - i < 4:
                grid[i][j].water[0] += edge_water
            if j < 4 or max_grid_col - j < 4:
                grid[i][j].water[0] += edge_water
            grid[i][j].water_level = grid[i][j].lake_depth + grid[i][j].firn_depth

def check_energy_conservation(grid):
    energy = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            cell = grid[i][j]
            if cell.valid_cell:
                cp_ice = 1000 * (7.16 * 10 ** (-3) * cell.firn_temperature + 0.138)
                cp = (cell.cp_water * cell.Lfrac + (1004 * (1 - cell.Sfrac - cell.Lfrac)) + (cp_ice * cell.Sfrac))
                cell.rho = cell.Sfrac * cell.rho_ice + cell.Lfrac * cell.rho_water
                energy += cell.rho * cell.firn_temperature * cp
    print('Total energy = ', np.sum(energy))

def spinup(cell, x, args):
    """
    Attempt to force the model into a solvable state if the initial conditions are unsuitable.
    TODO - This is a definite work in progress. Need to update this further.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        cell that we want to spin up.
    x : array_like, dimension(vert_grid)
        Initial guess temperature [K]
    args : array_like
        list of arguments to pass to the solver. See extract_args for details.
    Returns
    -------
    None (amends the instance of cell passed to it)

    Raises
    ------
    ValueError
        If solution does not converge, then rather than attempting to continue it raises an error. The user should
        re-consider their initial conditions in this case.
    """
    # days = 0
    # success = False
    # new_args = args
    # while success is False and days < 200:
    #     # print(new_args)
    #     root, fvec, success, info = solver(x, new_args)
    #     cell.firn_temperature = root
    #     # reduce SW/LW/air temp a bit to simulate cooling
    #     T_air = new_args[5] - 0.1
    #     LW_in = new_args[3] - 1
    #     SW_in = new_args[4] - 1
    #     new_args = [
    #         args[0],
    #         args[1],
    #         args[2],
    #         LW_in,
    #         SW_in,
    #         T_air,
    #         args[6],
    #         args[7],
    #         args[8],
    #     ]
    #     days = days + 1
    # if success is False:
    #     # print('Solution did not converge. Your initial conditions should be re-considered.')
    #     raise ValueError(
    #         "Solution did not converge. Your initial conditions should be re-considered."
    #     )


