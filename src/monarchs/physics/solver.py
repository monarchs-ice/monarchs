"""
This module contains functions used to solve the various heat/surface energy balance equations,
using the Scipy.optimize.fsolve implementation.
This is rather unwieldy, but allows us to select a solver conditionally, and do so both in the main model code
using the relevant flag in ``model_setup.py``, but also in our test suite.
This is in part a tradeoff between usability of the model, and code clarity. This approach was chosen to maximise
usability, so that the different solvers can be generated according to the value of a single Boolean.
"""

import numpy as np
from scipy.optimize import fsolve, root
from monarchs.physics import heateqn


def firn_heateqn_solver(x, args, fixed_sfc=False, solver_method="hybr"):
    """
    scipy.optimize.fsolve-compatible solver function to be used within the model.
    Solves physics.heateqn.
    This loads in the relevant arguments from the cell, packages them
    into an array form via args_array, and passes them into the hybrd solver.


    Called in <firn_column>.

    Parameters
    ----------
    x : array_like, float, dimension(cell.vert_grid)
        numpy array containing the initial estimate of the firn column temperature
    args : array_like
        Numpy array containing arguments to the heat equation.
        See <firn_column> for info on the contents of this array.
    fixed_sfc : bool, optional
        Boolean flag to determine whether to use the fixed surface form of the heat equation.

    Returns
    -------
    root : array_like, float, dimension(cell.vert_grid)
        Vector containing the calculated firn column temperature, either after successful completion or
        at the end of the final iteration for an unsuccessful solution. [K]
    infodict : dict
        A dictionary of optional outputs, such as the number of Jacobian iterations, etc.
        See scipy.optimize.fsolve documentation for more details on the content of infodict.
    ier : int
        An integer flag. Set to 1 if a solution was found, otherwise refer to mesg for more information.
    mesg : str
        If no solution is found, mesg details the cause of failure.


    """
    cell = args[0]
    dt = args[1]
    dz = args[2]
    LW_in = args[3]
    SW_in = args[4]
    T_air = args[5]
    p_air = args[6]
    T_dp = args[7]
    wind = args[8]

    if fixed_sfc:
        sol = np.array([273.15])
        infodict = {}
        ier = 1
        mesg = "Fixed surface temperature"
        T_tri = heateqn.propagate_temperature(cell, dz, dt, 273.15, N=1)
        T = np.concatenate((np.array([273.15]), T_tri))
        # print('T fixed sfc = ', T)
    else:
        N = 50
        x = x[:N]
        # N = cell['vert_grid']
        # If N is set to equal vert_grid, then when we solve for the surface temperature, we effectively
        # solve for the whole column, so should just return that at the end of the function.

        soldict = root(
        heateqn.heateqn,
        x,
        args=(
            cell,
            LW_in,
            SW_in,
            T_air,
            p_air,
            T_dp,
            wind,
            dz,
            dt,
         ),
        method=solver_method,
          )

        if not soldict.success:
            print(f"Root-finding for surface temperature failed - "
                  f"returning original guess. row = {cell['row']}, col = {cell['column']}")

        if N == cell['vert_grid']:
            return soldict.x, soldict.success, soldict.message, soldict.success

        sol = soldict.x
        ier = soldict.success
        mesg = soldict.message
        infodict = soldict.success
        #print(sol)
        # Take our root-finding algorithm output (from first N layers),
        # use it as the top boundary condition to the tridiagonal solver,
        # then concatenate the two
        # Now use tridiagonal solver to solve the heat equation once we have the surface temp

        T_tri = heateqn.propagate_temperature(cell, dz, dt, sol[-1], N=N)
        T = np.concatenate((sol[:], T_tri))
        # print('T free = ', T)

    T = np.around(T, decimals=8)
    #print('Sol0 = ', sol[0])
    #print('T = ', T)

    return T, infodict, ier, mesg

def lake_formation_eqn(x, args):
    """
    Scipy-compatible form of the lake formation version of the surface temperature equation.
    Called in get_lake_solver.

    Parameters
    ----------
    x : array_like, float, dimension(vert_grid_lake)
        Initial estimate of the lake temperature. [K]
    args : array_like
        Array of input arguments to be extracted into the relevant variables
        (firn_depth, vert_grid, Q, k, and T1).

    Returns
    -------
    output : float
        Estimate of the surface lake temperature [K].
    """
    firn_depth = args[0]
    vert_grid = args[1]
    Q = args[2]
    k = args[3]
    T1 = args[4]
    output = np.array([
        -0.98 * 5.670373 * 10**-8 * x[0] ** 4
        + Q
        - k * (-T1 + x[0]) / (firn_depth / vert_grid)]
    )
    return output


def lake_development_eqn(x, args):
    """
    Scipy-compatible form of the lake development version of the surface temperature equation.
    Called in get_lake_solver.

    Parameters
    ----------
    x : array_like, float, dimension(vert_grid_lake)
        Initial estimate of the lake temperature. [K]

    args : array_like
        Array of input arguments to be extracted into the relevant variables
        (J, Q, vert_grid_lake and lake_temperature).

    Returns
    -------
    output : float
        Estimate of the surface lake temperature [K].
    """
    J = args[0]
    Q = args[1]
    vert_grid_lake = args[2]
    lake_temperature = np.zeros(int(vert_grid_lake))
    for i in range(len(lake_temperature)):
        lake_temperature[i] = args[3 + i]
    T_core = lake_temperature[int(vert_grid_lake / 2)]
    output = np.array([
        -0.98 * 5.670373 * 10**-8 * x[0] ** 4
        + Q
        + np.sign(T_core - x[0]) * 1000 * 4181 * J * abs(T_core - x[0]) ** (4 / 3)]
    )
    return output



def lake_solver(x, args, formation=False):
    """
    Scipy version of the lake solver.
    Solves core.choose_solver.dev_eqn or core.choose_solver.form_eqn, which are defined
    in the body of get_lake_surface_energy_equations.

    Parameters
    ----------
    x : array_like, float, dimension(cell.vert_grid_lake)
        Initial estimate of the lake temperature profile. We only use the first element in the solver here
        (in both the lake formation and lake development cases).
    args : array_like
        Array of input arguments to the solver. See documentation for form_eqn and dev_eqn for details.
    formation : bool
        Flag to determine whether we want the lake *formation* equation, or the lake *development* equation.
        Defaults to the development equation, unless specified.

    Returns
    -------
    root : float,
        Calculated lake surface temperature, either after successful completion or
        at the end of the final iteration for an unsuccessful solution. [K]
    infodict : dict
        A dictionary of optional outputs, such as the number of Jacobian iterations, etc.
        See scipy.optimize.fsolve documentation for more details on the content of infodict.
    ier : int
        An integer flag. Set to 1 if a solution was found, otherwise refer to mesg for more information.
    mesg : str
        If no solution is found, mesg details the cause of failure.
    """
    if formation:
        eqn = lake_formation_eqn
    else:
        eqn = lake_development_eqn
    root, ier, mesg, infodict = fsolve(eqn, x, args=args, full_output=True)
    root = np.around(root, decimals=8)
    return root, infodict, ier, mesg


def sfc_energy_virtual_lid(x,  args):
    """

    Parameters
    ----------
    x
    args

    Returns
    -------
    output : float
    """
    Q = args[0]
    k_v_lid = args[1]
    lake_depth = args[2]
    vert_grid_lake = args[3]
    v_lid_depth = args[4]
    lake_temperature = np.zeros(int(vert_grid_lake))
    # hacky way to assign lake_temperature as the cfunc doesn't like slicing normally
    for i in np.arange(len(lake_temperature)):
        lake_temperature[i] = args[5 + i]

    # set output[0] rather than just output as solution doesn't converge otherwise as it expects
    # an array
    output = np.zeros(1)
    output[0] = (
        -0.98 * 5.670373 * (10**-8) * (x[0] ** 4)
        + Q
        - k_v_lid
        * (-lake_temperature[1] + x[0])
        / (lake_depth / ((vert_grid_lake) / 2) + v_lid_depth)
    )

    return output


def sfc_energy_lid(x, args):
    """

    Parameters
    ----------
    x
    args

    Returns
    -------
    output : float

    """
    Q = args[0]
    k_lid = args[1]
    lid_depth = args[2]
    vert_grid_lid = args[3]
    sub_T = args[4]
    output = np.zeros(1)
    output[0] = (
        -0.98 * 5.670373 * 10**-8 * x[0] ** 4
        + Q
        - k_lid * (-sub_T + x[0]) / (lid_depth / vert_grid_lid)
    )
    return output


def lid_seb_solver(x, args, v_lid=False):
    """
    scipy.optimise.fsolve version of the lid surface energy balance solver.
    Solves core.choose_solver.sfc_energy_lid or core.choose_solver.sfc_energy_vlid, which are defined
    in the body of get_lid_surface_energy_equations.
    Called in lid_functions.virtual_lid and lid_functions.lid_development.

    Parameters
    ----------
    x : array_like, float, dimension(cell.vert_grid_lid)
        Initial estimate of the lid temperature. We only use the first value (i.e. the surface value). [K]
    args : array_like
        Array of arguments to pass into the function.
    v_lid : bool, optional
        Flag that asks whether we want to solve the surface energy balance for a virtual lid (if True), or
        a true lid (if False). Default False.

    Returns
    -------
    root : float
        the calculated lid surface temperature, either after successful completion or
        at the end of the final iteration for an unsuccessful solution. [K]
    infodict : dict
        A dictionary of optional outputs, such as the number of Jacobian iterations, etc.
        See scipy.optimize.fsolve documentation for more details on the content of infodict.
    ier : int
        An integer flag. Set to 1 if a solution was found, otherwise refer to mesg for more information.
    mesg : str
        If no solution is found, mesg details the cause of failure.
    """
    if v_lid:
        eqn = sfc_energy_virtual_lid
    else:
        eqn = sfc_energy_lid
    root, infodict, ier, mesg = fsolve(eqn, x, args=args, full_output=True)
    root = np.around(root, decimals=8)
    return root, infodict, ier, mesg


def lid_heateqn_solver(x, args):
    """
    scipy.optimize.fsolve version of the lid heat equation solver. Solves physics.heateqn_lid.
    Called in lid_functions.lid_development.

    Parameters
    ----------
    x : array_like, float, dimension(cell.vert_grid_lid)
        Initial guess at the lid column temperature. [K]
    args : array_like
        Array of arguments to pass into the function. See args_array for details.

    Returns
    -------
    root : array_like, float, dimension(cell.vert_grid_lid)
        Vector containing the calculated lid column temperature, either after successful completion or
        at the end of the final iteration for an unsuccessful solution. [K]
    infodict : dict
        A dictionary of optional outputs, such as the number of Jacobian iterations, etc.
        See scipy.optimize.fsolve documentation for more details on the content of infodict.
    ier : int
        An integer flag. Set to 1 if a solution was found, otherwise refer to mesg for more information.
    mesg : str
        If no solution is found, mesg details the cause of failure.
    """
    eqn = heateqn.heateqn_lid
    cell = args[0]
    dt = args[1]
    dz = args[2]
    LW_in = args[3]
    SW_in = args[4]
    T_air = args[5]
    p_air = args[6]
    T_dp = args[7]
    wind = args[8]
    Sfrac_lid = args[-2]
    k_lid = args[-1]
    args = (cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind, k_lid, Sfrac_lid)
    root, infodict, ier, mesg = fsolve(eqn, x, args=args, full_output=True)
    root = np.around(root, decimals=8)
    return root, infodict, ier, mesg
