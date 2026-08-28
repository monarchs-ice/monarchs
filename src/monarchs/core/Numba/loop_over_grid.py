"""
This module overloads core.loop_over_grid with a Numba implementation,
which parallelises the loop over the grid using Numba's prange.
"""

import numpy as np
from monarchs.physics.timestep import timestep_loop
import numba
from numba import prange


# disable unused-argument as this is to make the overloading work
# pylint: disable=unused-argument
def loop_over_grid_numba(
    row_amount,
    col_amount,
    grid,
    dt,
    met_data,
    t_steps_per_day,
    toggle_dict,
    ncores="all",
):
    # pylint: enable=unused-argument
    """
    This function wraps timestep_loop, allowing for it to be
    run in parallel over an arbitrarily sized grid. The grid
    is flattened, so for an NxN grid you don't need a multiple
    of N processors in order to use all the available cores.

    If model_setup.use_numba is True, then the pure
    Python loop_over_grid is overloaded with this function.

    Parameters
    ----------
    row_amount: Number of rows in <grid>.
    col_amount: Number of columns in <grid>.t_steps_per_day
    grid: numba.typed.List
        Nested list containing the instances of the IceShelf class for each
        x and y point. Vertical (z) information is stored within each class
        instance.
    met_data: numpy structured array
        Grid containing the met data associated with the model grid.
    t_steps_per_day: int
        Number of timesteps to run each day.
    toggle_dict : dict
        Dictionary of toggle switches to be fed into MONARCHS, that determine
        certain things about the model (such as whether to run certain
        physical processes).
    ncores:
        Number of cores to use. Default "all", in which case it will use
        numba.config.NUMBA_DEFAULT_NUM_THREADS threads (i.e. all of them that
        Numba can detect on the system).

    Returns
    -------
    None. The function amends the instance of <grid> passed to it.
          No need to reshape flat_grid back into np.shape(grid), as each
          element of flat_grid is a pointer to each element of grid, i.e.
          operating on flat_grid changes the corresponding element of grid
    """
    if isinstance(ncores, int):
        nthreads = ncores
    else:
        # disable pylint warnings as Numba is compiled and the linter
        # cannot see its members

        nthreads = numba.config.NUMBA_DEFAULT_NUM_THREADS  # pylint: disable=no-member

    numba.set_num_threads(nthreads)
    # was previously flatten, use reshape as it avoids a memory copy,
    # which doubles the memory use of the model and messes up locality
    # don't need to reshape back as the original elements in [row][col] are
    # pointers to the same elements in the flattened array
    flat_grid = grid.reshape(row_amount * col_amount)
    # only run timesteps in valid cells - get the indices of these
    cell_order = np.flatnonzero(flat_grid["valid_cell"])
    # disable linting for prange not being an iterable as it is
    # when running with use_numba=True as it is decorated with
    # jit in driver.py at runtime (this lets us determine whether
    # the user wants to run in parallel or not, e.g. if running
    # on a shared machine without a scheduler)
    # pylint: disable=not-an-iterable
    for i in prange(cell_order.shape[0]):
        # this basically handles chunking for valid cells only
        j = cell_order[i]
        timestep_loop(
            flat_grid[j],
            dt,
            met_data[j],
            t_steps_per_day,
            toggle_dict,
        )
    # flat_grid is a view, so grid has already been updated in place
    return grid
