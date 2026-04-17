import numpy as np
import contextlib
from monarchs.core.utils import calc_mass_sum

try:
    from numba import prange, objmode
except ImportError:
    # fallback if numba is not installed
    prange = range  # pylint: disable=invalid-name

    @contextlib.contextmanager
    def objmode(*args, **kwargs):
        """Dummy context manager for objmode when numba is not installed."""
        yield


def check_for_mass_conservation(cell, original_mass, new_mass, routine_name, tol=1e-6):
    """
    Check for mass conservation in the grid. We use this instead of
    using standard asserts or if cond: raise ValueError, since
    these can often fail silently if using Numba + parallel processing.

    Parameters
    ----------
    grid : List, or numba.typed.List
        the model grid
    original_mass : float
        The original mass of the grid to compare against.
    new_mass : float
        The new mass of the grid to compare against.
    routine_name: str
        Name of the routine calling this function, for error reporting.
    tol : float, optional
        Tolerance for mass conservation check. Default is 1e-6.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If mass conservation is violated beyond the specified tolerance.
    """

    mass_diff = abs(original_mass - new_mass)
    errflag = 0
    if mass_diff >= tol:
        # Print error (cast row/col to int to avoid <object> print issues)
        r = int(cell["row"])
        c = int(cell["column"])
        print(f"{routine_name} - ERROR:")
        print("mass not conserved at [", r, ",", c, "] !!!")
        print("    Difference:", mass_diff)
        print("    Original:", original_mass)
        print("    New:", new_mass)
        cell["error_flag"] = 1
        errflag = 1
    return errflag


def generic_error(cell, routine_name, message):
    """
    Generic error handling function to set error flag and print message,
    based on a specific error the user is encountering.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid where the error occurred.
    routine_name : str
        Name of the routine where the error occurred.
    message : str
        Error message to be printed.

    Returns
    -------
    None
    """
    with objmode:
        r = int(cell["row"])
        c = int(cell["column"])
        print(f"{routine_name} - ERROR at [{r}, {c}]: {message}")
        cell["error_flag"] = 1


def calc_grid_mass(grid):
    """
    Calculate the total mass inside the grid, to check for mass conservation.
    Only do this for valid cells, since invalid cells will have constant mass
    as no physics is run on them.

    Parameters
    ----------
    grid : List, or numba.typed.List
        the model grid
    Returns
    -------
    total_mass : float
        Total mass of the whole grid, in arbitrary units.
    """
    total_mass = 0

    for row in grid:
        for cell in row:
            if cell["valid_cell"]:
                total_mass += calc_mass_sum(cell)
    return total_mass


def check_for_single_column_errors(grid):
    """
    Check the grid for any cells that have error_status set to 1,
    indicating an error has occurred in that cell.

    Parameters
    ----------
    grid : List, or numba.typed.List
        the model grid

    Returns
    -------
    bool
        True if any cell has error_flag == 1, False otherwise.
    """
    flag = False
    for row in grid:
        for cell in row:
            if cell["error_flag"] == 1:
                print("----------------------------------------")
                print("monarchs.core.utils.check_for_single_column_errors:")
                print(
                    "Error detected in cell at [",
                    int(cell["row"]),
                    ",",
                )
                print(" ", int(cell["column"]), "]")
                print("after the single-column physics step. ")
                print("Check the output logs for details on the error.")
                flag = True
    return flag


def check_correct(cell):
    """
    Sanity checking to ensure model is still in a physically correct state.
    This ensures things such as the solid and liquid fraction being below 1
    and less than 0 within a grid cell, firn depth and lake depth being
    non-negative, etc.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we want to check.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any of the conditions are met, raise ValueError as the model has
        reached an unphysical state.
        See the code body for details.
    """
    func_name = "monarchs.core.utils.check_correct"
    if cell["lake_depth"] < -1e-12:  # account for rounding errors
        generic_error(cell, func_name, "Lake depth < -1e-12")
    if cell["lake_depth"] < 0:
        cell["lake_depth"] = 0

    if cell["firn_depth"] < 0:
        generic_error(cell, func_name, "Firn depth below 0")
    if np.any(cell["Sfrac"][cell["Sfrac"] < -0.01]) or np.any(
        cell["Sfrac"][cell["Sfrac"] > 1.01]
    ):
        print(f"""{np.max(cell["Sfrac"])} at level
            {np.where((cell["Sfrac"] > 1) | (cell["Sfrac"] < 0))},
             x = {cell["column"]}, y = {cell["row"]}
""")
        print("Minimum Sfrac = ", np.min(cell["Sfrac"]))
        generic_error(cell, func_name, "Sfrac must be between 0 and 1")
    # Set total to look at all but the top layer. The top layer can sometimes
    # become oversaturated, but this is allowed provided that meltflag is True
    # for that cell (i.e. the water will percolate in the next timestep).
    # This can sometimes occur due to the regridding, particularly for small
    # cell spacing.
    total = cell["Lfrac"][1:] + cell["Sfrac"][1:]
    if np.any(total > 1.01):
        print(f"{func_name}: ")
        print(f"{np.max(total)} at level {np.where(total > 1)} \n")
        print("Sfrac :", cell["Sfrac"][np.where(total > 1)])
        print("Lfrac :", cell["Lfrac"][np.where(total > 1)])
        print(
            "Sfrac + Lfrac:",
            cell["Lfrac"][np.where(total > 1)] + cell["Sfrac"][np.where(total > 1)],
        )
        generic_error(
            cell,
            func_name,
            "Sum of Sfrac and Lfrac below surface must be <= 1",
        )
    if cell["Sfrac"][0] + cell["Lfrac"][0] > 1.01 and not cell["meltflag"][0]:
        total_top = cell["Sfrac"][0] + cell["Lfrac"][0]
        print(f"{func_name}: ")
        print(f"{total_top} at level 0 \n")
        print("Sfrac :", cell["Sfrac"][0])
        print("Lfrac :", cell["Lfrac"][0])
        print(
            "Sfrac + Lfrac:",
            cell["Lfrac"][0] + cell["Sfrac"][0],
        )
        generic_error(
            cell,
            func_name,
            "Sum of Sfrac and Lfrac at surface must be < 1 unless water can percolate",
        )


def check_grid_correctness(grid):
    """
    Wraps check_correct for each cell in the grid.
    We do not run in parallel, as there are issues with error
    handling inside numba prange loops.

    Parameters
    ----------
    grid

    Returns
    -------

    """

    for i in range(len(grid)):  # pylint: disable=not-an-iterable
        for j in range(len(grid[0])):
            check_correct(grid[i][j])
