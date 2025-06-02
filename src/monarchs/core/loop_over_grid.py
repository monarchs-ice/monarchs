"""
Grid-looping module. This either runs sequentially, or runs through in parallel
(effectively like a collapsed OpenMP parallel do loop).
core.loop_over_grid sets up a 2D grid of IceShelf objects (defined in
core.iceshelf_class), flattens this grid and then runs timestep_loop()
in parallel (using either ProcessPoolExecutor or Dask, since the problem is
embarrassingly parallel), unless model_setup.parallel = False.
"""

import numpy as np
from monarchs.physics.timestep import timestep_loop
import time
from dask import delayed, compute
from memory_profiler import profile

def process_chunk(original_indices, chunk, met_data_chunk, dt, toggle_dict, t_steps_per_day):
    """
    Process a chunk of cells and return results with their original indices.
    """
    results = []
    for offset, (cell, met_data) in enumerate(zip(chunk, met_data_chunk)):
        val = timestep_loop(cell, dt, met_data, t_steps_per_day, toggle_dict)
        results.append((original_indices[offset], val))
    return results


def chunk_grid(flat_grid, met_data_grid, chunk_size):
    """
    Chunk the grid and met_data for valid cells only.

    Parameters
    ----------
    flat_grid : np.ndarray
        Flattened grid of IceShelf objects.
    met_data_grid : np.ndarray
        Flattened grid of met_data corresponding to the grid.
    chunk_size : int
        Approximate size of each chunk.

    Yields
    ------
    tuple
        (original_indices, chunk, met_data_chunk) for each chunk.
    """
    valid_indices = np.where(flat_grid["valid_cell"] == True)[0]
    valid_cells = flat_grid[valid_indices]
    valid_met_data = met_data_grid[valid_indices]
    for i in range(0, len(valid_cells), chunk_size):
        yield valid_indices[i:i + chunk_size], valid_cells[i:i + chunk_size], valid_met_data[i:i + chunk_size]

@profile
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
        dask_scheduler='processes',
        client=None
):
    """
    This function wraps timestep_loop, allowing for it to be
    run in parallel over an arbitrarily sized grid.

    Parameters
    ----------
    row_amount : int
        Number of rows in <grid>.
    col_amount : int
        Number of columns in <grid>.
    grid : list, or np.ndarray
        Nested list containing the instances of the IceShelf class for each
        x and y point. Vertical (z) information is stored within each class
        instance.
    met_data : list, or np.ndarray
        Nested list of the same shape as grid, containing instances of the
        MetData class for each x and y point.
    parallel : bool, optional
        Flag to determine whether the model is to be run in parallel across
        multiple cores or not.
        Default False.
    use_mpi : bool, optional
        Flag to determine whether to use MPI for parallelisation.
        Default False.
    ncores : int or str
        Number of cores to use if running in parallel. If 'all', all available cores will be used.
    use_dask : bool, optional
        Flag to determine whether to use Dask for parallelism.
        Default False.

    Returns
    -------
    None. The function amends the instance of <grid> passed to it.
    """
    flat_grid = grid.flatten()
    met_data_grid = met_data
    # met_data_grid = met_data.reshape(24, -1)  # use reshape as want to pass the 24 timesteps
    # met_data_grid = np.moveaxis(met_data_grid, 0, -1)  # move the first axis to the last axis
    if parallel:
        chunksize = max(1, len(flat_grid) // (ncores * 2))  # Dynamic chunk size for load balancing

        start_submit = time.time()

        # Use Dask for parallelism
        if dask_scheduler != 'distributed':
            tasks = [
                delayed(process_chunk)(indices, chunk, met_data_chunk, dt, toggle_dict, t_steps_per_day)
                for indices, chunk, met_data_chunk in chunk_grid(flat_grid, met_data_grid, chunksize)
            ]
            end_submit = time.time()
            print(f"Submission time: {end_submit - start_submit:.2f}s")
            start_compute = time.time()
            results = compute(*tasks, scheduler=dask_scheduler)  # Use "threads" for I/O-bound tasks
            end_compute = time.time()
            print(f"Execution time: {end_compute - start_compute:.2f}s")
            print(f"Total time (submit + exec): {end_compute - start_submit:.2f}s")
            results = [item for sublist in results for item in sublist]  # Flatten results
            # Update the grid with results
            for idx, val in results:
                flat_grid[idx] = val

            # Reshape back to original grid shape
            grid = flat_grid.reshape(grid.shape)
            return grid
        else:
            scattered_chunks = [
                client.scatter((indices, chunk, met_data_chunk))
                for indices, chunk, met_data_chunk in chunk_grid(flat_grid, met_data_grid, chunksize)
            ]

        futures = [
            client.submit(process_chunk, indices, chunk, met_data_chunk, dt, toggle_dict, t_steps_per_day)
            for (indices, chunk, met_data_chunk) in scattered_chunks
        ]

        results = client.gather(futures)
        results = [item for sublist in results for item in sublist]  # Flatten

        # Update flat_grid
        for idx, val in results:
            flat_grid[idx] = val

        # Reshape back to original grid shape
        grid = flat_grid.reshape((row_amount, col_amount))
        return grid



    # Sequential version - with inplace modification
    else:
        for i in range(row_amount * col_amount):
            timestep_loop(
                flat_grid[i], dt, met_data_grid[i], t_steps_per_day, toggle_dict
            )
        grid[:] = flat_grid.reshape(grid.shape)
        return grid