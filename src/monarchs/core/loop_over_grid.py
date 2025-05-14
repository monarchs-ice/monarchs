"""
Grid-looping module. This either runs sequentially, or runs through in parallel
(effectively like a collapsed OpenMP parallel do loop).
core.loop_over_grid sets up a 2D grid of IceShelf objects (defined in
core.iceshelf_class), flattens this grid and then runs timestep_loop()
in parallel (using either pathos.Pool or numba.prange, since the problem is
embarassingly parallel), unless model_setup.parallel = False.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from monarchs.physics.timestep import timestep_loop
from monarchs.core.utils import get_2d_grid
import time

def process_chunk(start_idx, chunk, met_data_chunk, dt, toggle_dict, t_steps_per_day):
    results = []
    for offset, (cell, met_data), in enumerate(zip(chunk, met_data_chunk)):
        val = timestep_loop(cell, dt, met_data, t_steps_per_day, toggle_dict)
        results.append((start_idx + offset, val))
    return results

def chunk_grid(flat_grid, met_data_grid, chunk_size):
    for i in range(0, len(flat_grid), chunk_size):
        yield i, flat_grid[i:i + chunk_size], met_data_grid[i:i + chunk_size]


def loop_over_grid(
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
):
    """
    This function wraps timestep_loop, allowing for it to be
    run in parallel over an arbitrarily sized grid.

    This version works with the Pure Python implementation, i.e.
    with use_numba=False.

    The respective Numba version of loop_over_grid is found in
    the respective folders in /core.

    There is no need to reshape flat_grid back into np.shape(grid), as each element
    of flat_grid is a pointer to each element of grid, i.e. operating on
    flat_grid changes the corresponding element of grid (and likewise for log_grid).

    Parameters
    ----------
    row_amount : int
    Number of rows in <grid>.
    col_amount : int
        Number of columns in <grid>.
    grid : list, or numba.typed.List
        Nested list containing the instances of the IceShelf class for each
        x and y point. Vertical (z) information is stored within each class
        instance.
    met_data : list, or numba.typed.List
        Nested list of the same shape as grid, containing instances of the
        MetData class for each x and y point.
    parallel : bool, optional
        Flag to determine whether the model is to be run in parallel across
        multiple cores or not.
        Default False.
        Defined in the model setup script.
    use_numba : bool, optional
        Flag to determine whether to use Numba optimisations or not.
        Default False.
        Defined in the model setup script.
    use_mpi : bool, optional
        Flag to determine whether to use MPI for parallelisation.
        Default False.
        Defined in the model setup script.
    cores :  int
        How many cores to use if running in parallel.
        Defined in the model setup script. If this is set to 'all' or False there, then all cores will be used.
        This is determined by logic in `driver.main`.
        If you set a number greater than the number of cores on the machine
        you are running on, then all will be used.

    For all other arguments see documentation for timestep_loop.

    Returns
    -------
    None. The function amends the instance of <grid> passed to it.

    Raises
    ------
    ValueError
        if one parallel process fails and returns None, then ValueError is returned to ensure the code
        stops rather than continuing fruitlessly.
    """
    flat_grid = grid.flatten()

    met_data_grid = met_data.reshape(24, -1)  # use reshape as want to pass the 24 timesteps
    toggle_dict_grid = []
    x0 = get_2d_grid(grid, "column")

    if parallel:
        chunksize = 10
        # Reshape met data to (row * col, t_steps_per_day)
        met_data_grid = np.moveaxis(met_data_grid, 0, -1)
        # idx, cell, dt, met_data, t_steps_per_day, toggle_dict
        # We don't want to operate on invalid cells, as this adds overhead and can mess up the load balancing.
        valid_cells = np.where(flat_grid[:, "valid_cell"] == True)[0]

        with ProcessPoolExecutor(max_workers=ncores) as pool:
            futures = [
                pool.submit(process_chunk, i, chunk, met_data_chunk, dt, toggle_dict, t_steps_per_day)
                for i, chunk, met_data_chunk in chunk_grid(flat_grid, met_data_grid, chunksize)
            ]
            results = []
            for f in as_completed(futures):
                results.extend(f.result())

        for idx, val in results:
            flat_grid[idx] = val

        # Reshape back to original grid shape
        grid = flat_grid.reshape(grid.shape)

        xnew = get_2d_grid(grid, "column")

        assert (xnew == x0).all()
        return grid

    # Sequential version - with inplace modification
    else:
        for i in range(row_amount * col_amount):
            timestep_loop(
                flat_grid[i], dt, met_data_grid[:, i], t_steps_per_day, toggle_dict
            )
        xnew = get_2d_grid(grid, "column")
        assert (xnew == x0).all()
        grid[:] = flat_grid.reshape(grid.shape)
        return grid
