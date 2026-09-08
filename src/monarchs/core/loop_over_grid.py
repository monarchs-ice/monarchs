"""
Grid-looping module. Runs the single-column physics over every cell in turn,
at the moment with no parallelism, as pure-Python is debug-only.
"""

from monarchs.physics.timestep import timestep_loop


# pylint: disable=too-many-arguments,unused-argument
def loop_over_grid(
    row_amount,
    col_amount,
    grid,
    dt,
    met_data,
    t_steps_per_day,
    toggle_dict,
    ncores="all",
):
    """
    Run ``timestep_loop`` over every cell of the grid, one at a time.

    Parameters
    ----------
    row_amount : int
        Number of rows in <grid>.
    col_amount : int
        Number of columns in <grid>.
    grid : numpy structured array
        Model grid containing the ice shelf parameters.
    dt : float
        Timestep in seconds (usually 3600 * t_steps_per_day).
    met_data : numpy structured array
        Grid containing the met data associated with the model grid.
    t_steps_per_day : int
        Number of timesteps in a model iteration (usually 24)
    toggle_dict : dict
        Dictionary of toggles to turn model features on and off.
    ncores : int or str, optional
        Dummy argument needed to keep consistent shape with the
        parallel Numba loop_over_grid.

    Returns
    -------
    None. The function amends the instance of <grid> passed to it.
    """
    flat_grid = grid.flatten()
    for i in range(row_amount * col_amount):
        timestep_loop(
            flat_grid[i],
            dt,
            met_data[i],
            t_steps_per_day,
            toggle_dict,
        )
    grid[:] = flat_grid.reshape(grid.shape)
    return grid
