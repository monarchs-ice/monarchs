"""
This module contains functions used to solve the various heat/surface energy balance equations,
using the NumbaMinpack implementation.
This is rather unwieldy, but allows us to select a solver conditionally, and do so both in the main model code
using the relevant flag in ``model_setup.py``, but also in our test suite.
This is in part a tradeoff between usability of the model, and code clarity. This approach was chosen to maximise
usability, so that the different solvers can be generated according to the value of a single Boolean.
"""

import numpy as np
from NumbaMinpack import hybrd, minpack_sig
from numba import cfunc, jit

import monarchs.physics.Numba.heateqn_nb as hnb

heqlid = hnb.heateqn_lid.address
heq = hnb.heateqn.address


@jit(nopython=True, fastmath=False)
def args_array(
    cell,
    dt,
    dz,
    LW_in,
    SW_in,
    T_air,
    p_air,
    T_dp,
    wind,
    Tsfc=273.15,
    lid=False,
    Sfrac_lid=np.array([np.nan]),
    k_lid=np.nan,
    fixed_sfc=True,
    N=50
):
    """
    Convert the variables from an instance of the IceShelf class into the
    format required to run NumbaMinpack.hybrd, a root finder written in Fortran
    compatible with Numba jitted functions. This should result
    in some speedup compared to using scipy's equivalent function fsolve.

    We need to convert any non-arrays into 1d arrays with one element to
    concatenate them into this vector, hence the "np.array([var])" syntax. This
    is because Numba is compiled rather than interpreted; the code expects to
    see arrays rather than floats and cannot infer types like regular Python.

    Called whenever the various heat equation implemenations are used,
    notably in heateqn, heateqn_lid and heateqn_fixedsfc, which are in turn called by the relevant solver functions
    (see get_firn_heateqn_solver, get_lid_solvers).

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object containing the relevant information on the temperature, etc. of the firn column
    dt : int
        timestep in seconds [s]
    dz : float
        size of each vertical grid cell [m]
    LW_in : array_like, float, dimension(cell.vert_grid)
        Surface downwelling longwave radiation [W m^-2]
    SW_in : array_like, float, dimension(cell.vert_grid)
        Surface downwelling shortwave radiation [W m^-2]
    T_air : array_like, float, dimension(cell.vert_grid)
        Surface air temperature [K]
    p_air : array_like, float, dimension(cell.vert_grid)
        Surface air pressure [Pa]
    T_dp :  array_like, float, dimension(cell.vert_grid)
        Dewpoint temperature [K]
    wind : array_like, float, dimension(cell.vert_grid)
        Wind speed [m s^-1]
    fixed_sfc : bool, optional
        Boolean flag to determine whether to use the fixed surface heat equation.
    Tsfc : float, optional
        Surface temperature to fix the surface of the firn column to, if applicable (i.e. if using fixed_sfc=True)
    lid : bool, optional
        Boolean flag determining whether we want the lid heat equation solver.
    Sfrac_lid : array_like, float, optional, dimension(cell.vert_grid)
        Solid fraction of the frozen lid, if applicable. Default to np.array([np.nan]), as array type,
        and not used in non-lid cases.
    k_lid : float, optional
        thermal conducitvity of the frozen lid, if applicable. Default np.nan, as not used in non-lid cases.

    Returns
    -------
    args : array_like
        Numpy array containing the arguments we want to pass into the heat equation solver.

    """
    if lid:
        args = np.hstack(
            (
                np.array([cell.vert_grid_lid]),
                cell.lid_temperature,
                Sfrac_lid,
                np.array([k_lid]),
                np.array([cell.cp_air]),
                np.array([cell.cp_water]),
                np.array([dt]),
                np.array([dz]),
                np.array([cell.melt]),
                np.array([cell.exposed_water]),
                np.array([cell.lid]),
                np.array([cell.lake]),
                np.array([cell.lake_depth]),
                np.array([LW_in]),
                np.array([SW_in]),
                np.array([T_air]),
                np.array([p_air]),
                np.array([T_dp]),
                np.array([wind]),
            )
        )
    # test = cell.firn_temperature
    else:
        args = np.hstack(
            (
                np.array([N]),
                cell.firn_temperature[:N],
                cell.Sfrac[:N],
                cell.Lfrac[:N],
                np.array([cell.k_air]),
                np.array([cell.k_water]),
                np.array([cell.cp_air]),
                np.array([cell.cp_water]),
                np.array([dt]),
                np.array([dz]),
                np.array([cell.melt]),
                np.array([cell.exposed_water]),
                np.array([cell.lid]),
                np.array([cell.lake]),
                np.array([cell.lake_depth]),
                np.array([LW_in]),
                np.array([SW_in]),
                np.array([T_air]),
                np.array([p_air]),
                np.array([T_dp]),
                np.array([wind]),
            )
        )


    return args


@jit(nopython=True, fastmath=False)
def firn_heateqn_solver(x, args, fixed_sfc=False, solver_method="hybr"):
    """
    Numba-compatible solver function to be used within the model.
    Solves physics.Numba.heateqn.
    This loads in the relevant arguments from the cell, packages them
    into an array form via args_array, and passes them into the hybrd solver.

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
    fvec : array_like, float, dimension(core.iceshelf_class.IceShelf.vert_grid)
        Vector containing the function evaluated at root, i.e. the raw output.
    success : bool
        Boolean flag determining whether the solution converged, or whether there was an error.
    info : int
        Integer flag containing information on the status of the solution.
        From the Minpack hybrd documentation:
        !!  * ***info = 0*** improper input parameters.
        !!  * ***info = 1*** relative error between two consecutive iterates
        !!    is at most `xtol`.
        !!  * ***info = 2*** number of calls to `fcn` has reached or exceeded
        !!    `maxfev`.
        !!  * ***info = 3*** `xtol` is too small. no further improvement in
        !!    the approximate solution `x` is possible.
        !!  * ***info = 4*** iteration is not making good progress, as
        !!    measured by the improvement from the last
        !!    five jacobian evaluations.
        !!  * ***info = 5*** iteration is not making good progress, as
        !!    measured by the improvement from the last
        !!    ten iterations.

    """

    N = 50  # number of cells at top to use in hybrd implementation
    x = x[:N]

    # cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind = [arg for arg in args]
    cell = args[0]
    dt = args[1]
    dz = args[2]
    LW_in = args[3]
    SW_in = args[4]
    T_air = args[5]
    p_air = args[6]
    T_dp = args[7]
    wind = args[8]
    args = args_array(
        cell,
        dt,
        dz,
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
        fixed_sfc=fixed_sfc,
        N=N
    )
    
    # we only return root in the model (hence why in driver.py we index the function by [0],
    # but the other info is useful for testing so can be used if calling solver directly
    if fixed_sfc:
        sol = np.array([273.15])
        fvec = np.array([1.0])
        success = 1
        info = 1
        T_fixed = hnb.propagate_temperature(cell, dz, dt, 273.15, N=1)
        T = np.concatenate((np.array([273.15]), T_fixed))
        # print('T fixed sfc = ', T)
    else:
        sol, fvec, success, info = hybrd(heq, x, args)
        #sol = np.around(sol, decimals=8)

        # Take our root-finding algorithm output (from first N layers),
        # use it as the top boundary condition to the tridiagonal solver,
        # then concatenate the two
        T_tri = hnb.propagate_temperature(cell, dz, dt, sol[-1], N=N)
        T = np.concatenate((sol[:], T_tri))
        # print('T free = ', T)

    T = np.around(T, decimals=8)
    #print('Sol0 = ', sol[0])
    #print('T = ', T)
    return T, fvec, success, info



@jit(nopython=True, fastmath=False)
def lake_formation_eqn(x, output, args):
    """
    Numba-compatible form of the lake formation version of the surface temperature equation.
    Called in get_lake_solver.

    Parameters
    ----------
    x : array_like, float, dimension(vert_grid_lake)
        Initial estimate of the lake temperature. [K]
    output : array_like, float, dimension(vert_grid_lake)
        Output array containing the lake temperature. We only actually return the first element of this array.
        This may be possible to set to float, along with x, but Numba works in mysterious ways and this seems
        to compile and work.
    args : array_like
        Array of input arguments to be extracted into the relevant variables
        (firn_depth, vert_grid, Q, k, and T1).

    Returns
    -------
    None.
    """
    firn_depth = args[0]
    vert_grid = args[1]
    Q = args[2]
    k = args[3]
    T1 = args[4]
    # set output[0] rather than just output else we will just return our initial guess.
    output[0] = (
        -0.98 * 5.670373 * 10**-8 * x[0] ** 4
        + Q
        - k * (-T1 + x[0]) / (firn_depth / vert_grid)
    )


@jit(nopython=True, fastmath=False)
def lake_development_eqn(x, output, args):
    """
    Numba-compatible form of the lake development version of the surface temperature equation.
    Called in get_lake_solver.

    Parameters
    ----------
    x : array_like, float, dimension(vert_grid_lake)
        Initial estimate of the lake temperature. [K]
    output : array_like, float, dimension(vert_grid_lake)
        Output array containing the lake temperature. We only actually return the first element of this array.
        This may be possible to set to float, along with x, but Numba works in mysterious ways and this seems
        to compile and work.
    args : array_like
        Array of input arguments to be extracted into the relevant variables
        (J, Q, vert_grid_lake and lake_temperature).

    Returns
    -------
    None.
    """

    J = args[0]  # float, turbulent heat flux factor, equal to 1.907 E-5. [m s^-1 K^-(1/3)]
    Q = args[1]  # float, Surface energy flux [W m^-2]
    vert_grid_lake = args[2]  # float, number of vertical levels in the lake profile
    # array_like, float, dimension(vert_grid_lake)
    lake_temperature = np.zeros(int(vert_grid_lake))
    # have to use a loop as Numba doesn't like slicing like args[3:] with cfuncs
    for i in range(len(lake_temperature)):
        lake_temperature[i] = args[3 + i]
    # get core temperature via central point of lake
    T_core = lake_temperature[int(vert_grid_lake / 2)]

    # set output[0] rather than just output else we will just return our initial guess.
    # (i.e. solution does not converge and the code fails)
    output[0] = (
        -0.98 * 5.670373 * 10**-8 * x[0] ** 4
        + Q
        + np.sign(T_core - x[0]) * 1000 * 4181 * J * abs(T_core - x[0]) ** (4 / 3)
    )


dev_eqn_cfunc = cfunc(minpack_sig)(lake_development_eqn)
form_eqn_cfunc = cfunc(minpack_sig)(lake_formation_eqn)
dev_eqnaddress = dev_eqn_cfunc.address
form_eqnaddress = form_eqn_cfunc.address


@jit(nopython=True, fastmath=False)
def lake_solver(x, args, formation=False):
    """
    NumbaMinpack version of the lake solver.
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
    root : float
        Calculated lake surface temperature, either after successful completion or
        at the end of the final iteration for an unsuccessful solution. [K]
    fvec : array_like, float, dimension(core.iceshelf_class.IceShelf.vert_grid)
        Vector containing the function evaluated at root, i.e. the raw output.
    success : bool
        Boolean flag determining whether the solution converged, or whether there was an error.
    info : int
        Integer flag containing information on the status of the solution.
    """
    # load in from the compiled function address in memory rather than the Python function object
    if formation:
        eqn = form_eqnaddress

    else:
        eqn = dev_eqnaddress

    root, fvec, success, info = hybrd(eqn, x, args)
    root = np.around(root, decimals=8)
    return root, fvec, success, info


@jit(nopython=True, fastmath=False)
def sfc_energy_virtual_lid(x, output, args):
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
    output[0] = (
        -0.98 * 5.670373 * (10**-8) * (x[0] ** 4)
        + Q
        - k_v_lid
        * (-lake_temperature[1] + x[0])
        / (lake_depth / ((vert_grid_lake) / 2) + v_lid_depth)
    )



@jit(nopython=True, fastmath=False)
def sfc_energy_lid(x, output, args):
    """

    Parameters
    ----------
    x
    output
    args

    Returns
    -------

    """

    Q = args[0]
    k_lid = args[1]
    lid_depth = args[2]
    vert_grid_lid = args[3]
    sub_T = args[4]
    output[0] = (
        -0.98 * 5.670373 * 10**-8 * x[0] ** 4
        + Q
        - k_lid * (-sub_T + x[0]) / (lid_depth / vert_grid_lid)
    )

sfc_energy_virtual_lid = cfunc(minpack_sig)(sfc_energy_virtual_lid)
sfc_energy_lid = cfunc(minpack_sig)(sfc_energy_lid)
sfc_energy_vlid_address = sfc_energy_virtual_lid.address
sfc_energy_lid_address = sfc_energy_lid.address


@jit(nopython=True, fastmath=False)
def lid_seb_solver(x, args, v_lid=False):
    """
    NumbaMinpack version of the lid surface energy balance solver.
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
    fvec : float
        Vector containing the function evaluated at root, i.e. the raw output.
    success : bool
        Boolean flag determining whether the solution converged, or whether there was an error.
    info : int
        Integer flag containing information on the status of the solution.
        From the Minpack hybrd documentation:
        !!  * ***info = 0*** improper input parameters.
        !!  * ***info = 1*** relative error between two consecutive iterates
        !!    is at most `xtol`.
        !!  * ***info = 2*** number of calls to `fcn` has reached or exceeded
        !!    `maxfev`.
        !!  * ***info = 3*** `xtol` is too small. no further improvement in
        !!    the approximate solution `x` is possible.
        !!  * ***info = 4*** iteration is not making good progress, as
        !!    measured by the improvement from the last
        !!    five jacobian evaluations.
        !!  * ***info = 5*** iteration is not making good progress, as
        !!    measured by the improvement from the last
        !!    ten iterations.
    """
    if v_lid:
        eqn = sfc_energy_vlid_address

    else:
        eqn = sfc_energy_lid_address

    root, fvec, success, info = hybrd(eqn, x, args)

    root = np.around(root, decimals=8)
    return root, fvec, success, info


@jit(nopython=True, fastmath=False)
def lid_heateqn_solver(x, args):
    """
    NumbaMinpack version of the lid heat equation solver. Solves physics.Numba.heateqn.

    Called in lid_functions.lid_development.

    Parameters
    ----------
    x : array_like, float, dimension(vert_grid)
        Initial guess at the lid column temperature. [K]
    args : array_like
        Array of arguments to pass into the function. See args_array for details.

    Returns
    -------
    root : array_like, float, dimension(core.iceshelf_class.IceShelf.vert_grid_lid)
        Vector containing the calculated lid column temperature, either after successful completion or
        at the end of the final iteration for an unsuccessful solution. [K]
    fvec : array_like, float, dimension(core.iceshelf_class.IceShelf.vert_grid_lid)
        Vector containing the function evaluated at root, i.e. the raw output.
    success : bool
        Boolean flag determining whether the solution converged, or whether there was an error.
    info : int
        Integer flag containing information on the status of the solution.
        From the Minpack hybrd documentation:
        !!  * ***info = 0*** improper input parameters.
        !!  * ***info = 1*** relative error between two consecutive iterates
        !!    is at most `xtol`.
        !!  * ***info = 2*** number of calls to `fcn` has reached or exceeded
        !!    `maxfev`.
        !!  * ***info = 3*** `xtol` is too small. no further improvement in
        !!    the approximate solution `x` is possible.
        !!  * ***info = 4*** iteration is not making good progress, as
        !!    measured by the improvement from the last
        !!    five jacobian evaluations.
        !!  * ***info = 5*** iteration is not making good progress, as
        !!    measured by the improvement from the last
        !!    ten iterations.
    """
    eqn = heqlid
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
    args = args_array(
        cell,
        dt,
        dz,
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
        lid=True,
        k_lid=k_lid,
        Sfrac_lid=Sfrac_lid,
    )
    # print(f'Lid heat equation solver, loaded in args... col = {cell.column}, row = {cell.row}')
    # for idx, arg in enumerate(args):
    #     print(f'index = {idx}')
    #     print(arg)
    # print('Temp = ', cell.lid_temperature)
    # print('args shape = ', np.shape(args))
    root, fvec, success, info = hybrd(eqn, x, args)

    root = np.around(root, decimals=8)
    return root, fvec, success, info
