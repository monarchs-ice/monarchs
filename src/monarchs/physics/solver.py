"""
This module contains functions used to solve the various heat/surface energy balance equations,
using the Scipy.optimize.fsolve implementation.
This is rather unwieldy, but allows us to select a solver conditionally, and do so both in the main model code
using the relevant flag in ``model_setup.py``, but also in our test suite.
This is in part a tradeoff between usability of the model, and code clarity. This approach was chosen to maximise
usability, so that the different solvers can be generated according to the value of a single Boolean.
"""
import numpy as np
from scipy.optimize import root, fsolve, OptimizeResult
from monarchs.physics import heateqn


def firn_heateqn_solver(x, args, fixed_sfc=False, solver_method='hybr'):
    """
    scipy.optimize.fsolve-compatible solver function to be used within the model.
    Solves physics.heateqn.
    This loads in the relevant arguments from the cell, packages them
    into an array form via args_array, and passes them into the hybrd solver.
    While this format is not necessary to use scipy.optimize.fsolve, it is done in this way to
    ensure that we can switch between the NumbaMinpack and scipy versions of hybrd with a single
    toggle (model_setup.use_numba).

    Called in <firn_column>.

    Parameters
    ----------
    x : array_like, float, dimension(core.iceshelf_class.IceShelf.vert_grid)
        numpy array containing the initial estimate of the firn column temperature
    args : array_like
        Numpy array containing arguments to the heat equation.
        See <firn_column> for info on the contents of this array.
    fixed_sfc : bool, optional
        Boolean flag to determine whether to use the fixed surface form of the heat equation.

    Returns
    -------
    root : array_like, float, dimension(core.iceshelf_class.IceShelf.vert_grid)
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
    x_copy = np.copy(x)
    # precompute some values
    rho = cell['Sfrac'] * 913 + cell['Lfrac'] * 1000
    k_ice = np.zeros(np.shape(cell['firn_temperature']))
    k_ice[cell['firn_temperature'] < 273.15] = 1000 * (0.00224 + 5.975e-06 *
                                                       (273.15 - cell['firn_temperature'][cell['firn_temperature'] <
                                                                                          273.15]) ** 1.156)
    k_ice[cell['firn_temperature'] >= 273.15] = 2.24
    k = cell['Sfrac'] * k_ice + (1 - cell['Sfrac'] - cell['Lfrac']) * cell['k_air'] + cell['Lfrac'] * cell[
        'k_water']
    cp_ice = 7.16 * cell['firn_temperature'] + 138
    cp = cell['Sfrac'] * cp_ice + (1 - cell['Sfrac'] - cell['Lfrac']) * cell['cp_air'] + cell['Lfrac'] * cell[
        'cp_water']
    kappa = k / (cp * rho)

    def find_surface_temperature(cell, LW_in, SW_in, T_air, p_air, T_dp, wind):
        # Initial guess for T_sfc (can be close to the expected value)
        initial_guess = 273.15  # Example initial guess (e.g., melting point)

        # Use root-finding to solve for surface temperature
        result = root(heateqn.surface_temperature_residual, initial_guess, args=(cell, LW_in, SW_in, T_air, p_air, T_dp, wind))


        if not result.success:
            raise ValueError("Root-finding for surface temperature failed.")

        soldict = result  # Surface temperature solution
        return soldict

    if not fixed_sfc:
        soldict = find_surface_temperature(cell, LW_in, SW_in, T_air, p_air, T_dp, wind)

        sol = soldict.x
        ier = soldict.success
        mesg = soldict.message
        infodict = soldict.success

        if not fixed_sfc and (sol > 320):
            print('x0 = ', sol[:10])
            raise ValueError('Surface temperature too high - heateqn')
        if fixed_sfc and (sol > 273.151):
            print('x0 = ', sol[:10])
            raise ValueError('Surface temperature too high - heateqn_fixedsfc')
        if np.isnan(sol) and soldict.success:
            print('x0 = ', sol)
            for key in soldict.keys():
                print(f'{key}: {soldict[key]}')
            raise ValueError('NaN in root - heateqn')
    else:
        sol = 273.15
        infodict = {}
        ier = 1
        mesg = 'Fixed surface temperature'
    #print('Surface temperature = ', sol)
    # Now use tridiagonal solver to solve the heat equation once we have the surface temp
    T = heateqn.propagate_temperature(cell, dt, dz, kappa, sol)
    #print('Temperature profile = ', T[:10])
    return T, infodict, ier, mesg


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
    output = -0.98 * 5.670373 * 10 ** -8 * x[0] ** 4 + Q + np.sign(T_core -
                                                                   x[0]) * 1000 * 4181 * J * abs(T_core - x[0]) ** (
                         4 / 3)
    return output


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
    output = -0.98 * 5.670373 * 10 ** -8 * x[0] ** 4 + Q - k * (-T1 + x[0]) / (
            firn_depth / vert_grid)
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
    return root, infodict, ier, mesg


def sfc_energy_virtual_lid(x, args):
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
    for i in range(len(lake_temperature)):
        lake_temperature[i] = args[5 + i]
    output = -0.98 * 5.670373 * 10 ** -8 * x[0] ** 4 + Q - k_v_lid * (-
                                                                      lake_temperature[-2] + x[0]) / (
                         lake_depth / (vert_grid_lake / 2) +
                         v_lid_depth)
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
    output = -0.98 * 5.670373 * 10 ** -8 * x[0] ** 4 + Q - k_lid * (-sub_T +
                                                                    x[0]) / (lid_depth / vert_grid_lid)
    return output


def lid_seb_solver(x, args, v_lid=False):
    """
    scipy.optimise.fsolve version of the lid surface energy balance solver.
    Solves core.choose_solver.sfc_energy_lid or core.choose_solver.sfc_energy_vlid, which are defined
    in the body of get_lid_surface_energy_equations.
    Called in lid_functions.virtual_lid and lid_functions.lid_development.

    Parameters
    ----------
    x : array_like, float, dimension(core.iceshelf_class.IceShelf.vert_grid_lid)
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
    return root, infodict, ier, mesg


def lid_heateqn_solver(x, args):
    """
    scipy.optimize.fsolve version of the lid heat equation solver. Solves physics.heateqn_lid.
    Called in lid_functions.lid_development.

    Parameters
    ----------
    x : array_like, float, dimension(core.iceshelf_class.IceShelf.vert_grid_lid)
        Initial guess at the lid column temperature. [K]
    args : array_like
        Array of arguments to pass into the function. See args_array for details.

    Returns
    -------
    root : array_like, float, dimension(core.iceshelf_class.IceShelf.vert_grid_lid)
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
    args = (cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind, k_lid,
            Sfrac_lid)
    root, infodict, ier, mesg = fsolve(eqn, x, args=args, full_output=True)
    return root, infodict, ier, mesg
