"""
Grid-looping module. This either runs sequentially, or runs through in parallel
(effectively like a collapsed OpenMP parallel do loop).
core.loop_over_grid sets up a 2D grid of IceShelf objects (defined in
core.iceshelf_class), flattens this grid and then runs timestep_loop()
in parallel (using either pathos.Pool or numba.prange, since the problem is
embarassingly parallel), unless model_setup.parallel = False.
"""

from multiprocessing import shared_memory, Pool
import numpy as np
from monarchs.physics.timestep import timestep_loop
from monarchs.core.utils import get_2d_grid


def timestep_worker(args):
    (
        i,
        shm_name,
        shape,
        dtype_descr,
        dt,
        met_data,
        t_steps_per_day,
        toggle_dict,
    ) = args

    # Reconstruct dtype and connect to shared memory
    dtype = np.dtype(dtype_descr)
    shm = shared_memory.SharedMemory(name=shm_name)
    grid_view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    # Access the cell directly and modify in-place
    timestep_loop(grid_view[i], dt, met_data, t_steps_per_day, toggle_dict)

    shm.close()


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
    for i in range(col_amount):
        for j in range(row_amount):
            toggle_dict_grid.append(toggle_dict)


    if parallel:
        dt = [dt] * len(flat_grid)
        t_steps_per_day = [t_steps_per_day] * len(flat_grid)

        shm = shared_memory.SharedMemory(create=True, size=flat_grid.nbytes)
        shared_array = np.ndarray(
            flat_grid.shape, dtype=flat_grid.dtype, buffer=shm.buf
        )
        np.copyto(shared_array, flat_grid)

        arg_list = [
            (
                i,
                shm.name,
                shared_array.shape,
                shared_array.dtype.descr,
                dt[i],
                met_data_grid[:, i],
                t_steps_per_day[i],
                toggle_dict_grid[i],
            )
            for i in range(np.shape(flat_grid)[0])
        ]
        with Pool(ncores) as pool:
            pool.map(timestep_worker, arg_list)

        np.copyto(flat_grid, shared_array)
        grid[:] = flat_grid.reshape(grid.shape)

        shm.close()
        shm.unlink()

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
