"""
Utilities module containing various helper functions or wrappers.
"""

import contextlib
from functools import wraps
import numpy as np
import pathos
from monarchs.physics.constants import rho_ice, rho_water, cp_water

try:
    from numba import prange, objmode
except ImportError:
    # fallback if numba is not installed
    prange = range  # pylint: disable=invalid-name

    @contextlib.contextmanager
    def objmode(*args, **kwargs):
        """Dummy context manager for objmode when numba is not installed."""
        yield


def do_not_jit(function):
    """
    An empty function used to decorate functions that we do not wish to
    jit-compile.
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


def calc_mass_sum(cell):
    """
    Calculate the total mass in the grid cell in its current state.
    This is the sum of the firn column mass, the mass in the frozen lake, and
    the mass in the lid or virtual lid.
    This is in arbitrary units, since it is calculating the columnar mass
    independent of the lateral grid size, but is useful for consistency
    checking, i.e. ensuring the model does not gain or lose mass outside the
    known mechanisms (snowfall and losing water out of the catchment area
    laterally).

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.

    Returns
    -------
    total_mass : float
        Amount of mass in the system, in arbitrary units.
    """
    total_mass = (
        np.sum(cell["Sfrac"] * rho_ice * (cell["firn_depth"] / cell["vert_grid"]))
        + np.sum(cell["Lfrac"] * rho_water * (cell["firn_depth"] / cell["vert_grid"]))
        + cell["lake_depth"] * rho_water
        + cell["lid_depth"] * rho_ice
        + cell["v_lid_depth"] * rho_ice
    )
    return total_mass


def get_2d_grid(grid, attr, index=False, mask_invalid=False):
    """
    Helper function to get a printout of a variable from the model grid,
    at a user-specified index.

    Parameters
    ----------
    grid : List, or nb.typed.List
        the model grid
    attr : str
        the attribute you want to print out, e.g:
        get_2d_grid(grid, "firn_depth") will print out the firn depth
        for each point on the grid at index <index>
    index : int, optional
        the index (i.e. height) at which you want to print out.
        Defaults to False, which the code interprets as the surface
        (index 0).
    mask_invalid : bool, optional
        If True, mask out values where cell['valid_cell'] is False (set to np.nan).

    Returns
    -------
    None. (but will print to stdout)

    """

    if index is False:
        index = 0

    out = []

    if not isinstance(grid, np.ndarray):
        for row in grid:
            out_row = []
            for cell in row:
                value = getattr(cell, attr)

                if mask_invalid and (not getattr(cell, "valid_cell")):
                    if isinstance(value, np.ndarray):
                        value = np.full(value.shape, np.nan, dtype=float)
                    else:
                        value = np.nan

                out_row.append(value)
            out.append(out_row)

    else:
        valid_mask = None
        dtype_names = getattr(grid.dtype, "names", None)
        if mask_invalid and dtype_names is not None and "valid_cell" in dtype_names:
            valid_mask = grid["valid_cell"]

        for row_idx, row in enumerate(grid):
            out_row = []
            for col_idx, _ in enumerate(row):
                value = grid[attr][row_idx][col_idx]

                if valid_mask is not None and (not valid_mask[row_idx][col_idx]):
                    if isinstance(value, np.ndarray):
                        value = np.full(value.shape, np.nan, dtype=float)
                    else:
                        value = np.nan

                out_row.append(value)
            out.append(out_row)

    arr = np.array(out)

    if index == "all":
        return arr

    try:
        return arr[:, :, index]
    except IndexError:
        return arr[:, :]


def find_nearest(a, a0):
    """Obtain index of element in array `a` closest to the scalar value `a0`"""
    idx = np.abs(a - a0).argmin()
    return idx


def add_random_water(grid, max_grid_row, max_grid_col):
    """
    For testing - create water randomly inside the grid.

    Parameters
    ----------
    grid : List, or numba.typed.List
        model grid
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
            grid["water"][i][j][0] = grid["water"][i][j][0] + rand_water[i][j]
            grid["water_level"][i][j] = (
                grid["water"][i][j][0] + grid["firn_depth"][i][j]
            )


def add_edge_water(grid, max_grid_row, max_grid_col):
    """
    For testing - create water randomly at the edges of the grid.

    Parameters
    ----------
    grid : List, or numba.typed.List
        model grid
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
                grid["water"][i][j][0] += edge_water
            if j < 4 or max_grid_col - j < 4:
                grid["water"][i][j][0] += edge_water
            grid["water_level"][i][j] = (
                grid["lake_depth"][i][j] + grid["firn_depth"][i][j]
            )


def check_energy_conservation(grid):
    """ """
    #     TODO - WIP.
    energy = 0
    for row in grid:
        for cell in row:
            if cell["valid_cell"]:
                cp_ice = 1000 * (7.16 * 10**-3 * cell["firn_temperature"] + 0.138)
                cp = (
                    cp_water * cell["Lfrac"]
                    + 1004 * (1 - cell["Sfrac"] - cell["Lfrac"])
                    + cp_ice * cell["Sfrac"]
                )
                cell["rho"] = cell["Sfrac"] * rho_ice + cell["Lfrac"] * rho_water
                energy += cell["rho"] * cell["firn_temperature"] * cp
    print("Total energy = ", np.sum(energy))


def spinup(cell, x, args):
    """
    Attempt to force the model into a solvable state if the initial conditions
    are unsuitable.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
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
        If solution does not converge, then rather than attempting to continue
        it raises an error. The user should re-consider their initial
        conditions in this case.
    """
    #     TODO - This is a work in progress. Need to update this further.
    return cell, x, args


def get_num_cores(model_setup):
    """

    Parameters
    ----------
    model_setup

    Returns
    -------

    """
    # TODO - docstring, possibly remove pathos dependency
    if model_setup.cores in ["all", False] and model_setup.parallel:
        if not model_setup.use_mpi:
            cores = pathos.helpers.cpu_count()
        else:
            cores = pathos.helpers.cpu_count()
        print(f"Using all cores - {pathos.helpers.cpu_count()} detected")
    elif not model_setup.parallel:
        cores = 1
    else:
        cores = model_setup.cores
    return cores


try:
    import psutil
    import functools
    import os
except ImportError:
    print(
        "monarchs.core.utils: psutil module not found. Memory profiling will"
        " not be available. To suppress this warning, install psutil with"
        " 'python -m pip install psutil'."
    )


def memory_tracker(label=""):
    """
    Decorator to measure memory usage before and after a function call.
    Prints the difference in MB.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1e6  # in MB
            print(f"[{label}] Memory before: {mem_before:.2f} MB")

            result = func(*args, **kwargs)

            mem_after = process.memory_info().rss / 1e6
            print(f"[{label}] Memory after:  {mem_after:.2f} MB")
            print(f"[{label}] Memory delta:  {mem_after - mem_before:.2f} MB")
            return result

        return wrapper

    return decorator
