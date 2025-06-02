import numba
from numba import prange, types
from numba.typed import Dict
from monarchs.physics.timestep import timestep_loop
import numpy as np
from memory_profiler import profile

@profile
def loop_over_grid_numba(
    row_amount,
    col_amount,
    grid,
    dt,
    met_data,
    t_steps_per_day,
    toggle_dict,
    parallel=False,
    use_mpi=False,
    ncores="all",
    dask_scheduler=None,
    client=None,
):
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
    met_data: numba.typed.List
        Nested list of the same shape as grid, containing instances of the
        MetData class for each x and y point.
    t_steps_per_day: int
        Number of timesteps to run each day.
    toggle_dict : dict
        Dictionary of toggle switches to be fed into MONARCHS, that determine certain
        things about the model (such as whether to run certain physical processes).
    parallel, use_mpi:
        Dummy arguments so that we can overload the regular loop_over_grid with this
        Numba implementation, since these are needed there.
    ncores:
        Number of cores to use. Default 'all', in which case it will use
        numba.config.NUMBA_DEFAULT_NUM_THREADS threads (i.e. all of them that
        Numba can detect on the system).
    dask_scheduler, client: Dummy arguments for compatibility

    Returns
    -------
    None. The function amends the instance of <grid> passed to it.
          No need to reshape flat_grid back into np.shape(grid), as each element
          of flat_grid is a pointer to each element of grid, i.e. operating on
          flat_grid changes the corresponding element of grid
    """
    if isinstance(ncores, int):
        nthreads = ncores
    else:
        nthreads = numba.config.NUMBA_DEFAULT_NUM_THREADS

    numba.set_num_threads(nthreads)
    # append everything to a new 1D instance of a Numba typed list, flatten, and loop over that.
    flat_grid = grid.flatten()
    # met_data_grid = met_data.reshape(24, -1)  # use reshape as want to pass the 24 timesteps
    # met_data_grid = np.moveaxis(met_data_grid, 0, -1)  # move the first axis to the last axis
    print('Grid nbytes = ', grid.nbytes)
    print('Flat grid nbytes = ', flat_grid.nbytes)

    for i in prange(row_amount * col_amount):
        print(f"Index: {i}, flat_grid[i] type: {type(flat_grid[i])}, met_data[i] type: {type(met_data[i])}")
        print(f"Flat grid shape: {flat_grid.shape}, met_data shape: {met_data.shape}")
        print(f'Grid shape: {grid.shape}, row_amount: {row_amount}, col_amount: {col_amount}')
        timestep_loop(
            flat_grid[i],
            dt,
            met_data[i],
            t_steps_per_day,
            toggle_dict,
        )
    return np.reshape(flat_grid, (row_amount, col_amount))   # reshape