"""
Grid-looping module. This either runs sequentially, or runs through in parallel
(effectively like a collapsed OpenMP parallel do loop).
core.loop_over_grid sets up a 2D grid of IceShelf objects (defined in
core.iceshelf_class), flattens this grid and then runs timestep_loop()
in parallel (using either pathos.Pool or numba.prange, since the problem is
embarassingly parallel), unless model_setup.parallel = False.
"""
from pathos.pools import ParallelPool as Pool
import numpy as np
from monarchs.physics.timestep import timestep_loop
from monarchs.core.utils import get_2d_grid


def loop_over_grid(row_amount, col_amount, grid, dt, met_data,
    t_steps_per_day, toggle_dict, parallel=False, use_mpi=False, ncores='all'):
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
    flat_grid = []
    met_data_grid = []
    toggle_dict_grid = []
    x0 = get_2d_grid(grid, 'column')
    for i in range(col_amount):
        for j in range(row_amount):
            flat_grid.append(grid[i][j])
            met_data_grid.append(met_data[i][j])
            toggle_dict_grid.append(toggle_dict)
    if parallel:
        dt = [dt] * len(flat_grid)
        t_steps_per_day = [t_steps_per_day] * len(flat_grid)
        if use_mpi:
            from mpi4py import MPI
            from mpi4py.futures import MPIPoolExecutor, wait
            COMM = MPI.COMM_WORLD
            iceshelf = []
            with MPIPoolExecutor() as executor:
                for i in range(len(flat_grid)):
                    print('i = ', i)
                    iceshelf.append(executor.submit(timestep_loop,
                        flat_grid[i], dt[i], met_data_grid[i],
                        t_steps_per_day[i], toggle_dict_grid[i]))
            wait(iceshelf)
            iceshelf = [i.result() for i in iceshelf]
        else:
            with Pool(nodes=ncores, maxtasksperchild=1) as p:
                res = p.map(timestep_loop, flat_grid, dt, met_data_grid,
                    t_steps_per_day, toggle_dict_grid)
            iceshelf = np.array(res)
        for i in range(len(flat_grid)):
            if iceshelf[i] is None:
                raise ValueError(
                    """Timestep_loop returned None for at least one grid cell. 
 Check the logs to diagnose potential errors, or run without parallel = True."""
                    )
            flat_grid[i] = iceshelf[i]
        xnew = get_2d_grid(grid, 'column')
        assert (xnew == x0).all()
        return np.reshape(flat_grid, np.shape(grid))
    else:
        for i in range(row_amount * col_amount):
            timestep_loop(flat_grid[i], dt, met_data_grid[i],
                t_steps_per_day, toggle_dict)
        xnew = get_2d_grid(grid, 'column')
        assert (xnew == x0).all()
