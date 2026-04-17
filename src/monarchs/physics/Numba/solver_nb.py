"""
This module contains functions used to solve the various heat/surface energy
balance equations, using the NumbaMinpack implementation.
This is rather unwieldy, but allows us to select a solver conditionally, and
do so both in the main model codeusing the relevant flag in ``model_setup.py``,
but also in our test suite.
This is in part a tradeoff between usability of the model, and code clarity.
This approach was chosen to maximise usability, so that the different solvers
can be generated according to the value of a single Boolean.
"""

import numpy as np
from NumbaMinpack import hybrd, minpack_sig
from numba import cfunc, jit
from monarchs.physics.Numba.extract_args import pack_args
import monarchs.physics.Numba.heateqn_nb as hnb
from monarchs.physics.Numba.surface_energy_balance import lake_formation_eqn, lake_development_eqn, sfc_energy_lid, sfc_energy_virtual_lid


# load in the compiled function address in memory rather than the
# Python function object
heq = hnb.heateqn.address
heqlid = hnb.heateqn_lid.address
dev_eqn_cfunc = cfunc(minpack_sig)(lake_development_eqn)
form_eqn_cfunc = cfunc(minpack_sig)(lake_formation_eqn)
dev_eqnaddress = dev_eqn_cfunc.address
form_eqnaddress = form_eqn_cfunc.address





sfc_energy_virtual_lid = cfunc(minpack_sig)(sfc_energy_virtual_lid)
sfc_energy_lid = cfunc(minpack_sig)(sfc_energy_lid)
sfc_energy_vlid_address = sfc_energy_virtual_lid.address
sfc_energy_lid_address = sfc_energy_lid.address


################
# MAIN SOLVERS #
################

@jit(nopython=True, fastmath=False)
def solve_firn_heateqn(cell, met_data, dt, dz, fixed_sfc=False, solver_method="hybr"):
    """
    Numba-compatible solver function to be used within the model.
    Solves physics.Numba.heateqn.
    This loads in the relevant arguments from the cell, packages them
    into an array form via pack_args, and passes them into the hybrd solver.

    Called in <firn_column>.

    Parameters
    ----------
    cell: numpy structured array
        Element of the model grid we are operating on.
    met_data : dict
        Dictionary containing the meteorological data for the current timestep.
        See documentation in firn_column for details.
    dt : int
        Number of seconds in the current timestep [s]
    dz : float
        Size of each vertical point in the cell. [m]
    fixed_sfc : bool, optional
        Boolean flag to determine whether to use the fixed surface form of the
        heat equation.

    Returns
    -------
    root : array_like, float, dimension(core.iceshelf_class.IceShelf.vert_grid)
        Vector containing the calculated firn column temperature, either after
        successful completion or
        at the end of the final iteration for an unsuccessful solution. [K]
    fvec : array_like, float, dimension(core.iceshelf_class.IceShelf.vert_grid)
        Vector containing the function evaluated at root, i.e. the raw output.
    success : bool
        Boolean flag determining whether the solution converged, or whether
        there was an error.
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

    N = cell["vert_grid"]
    x = cell["firn_temperature"]
    # Need to pack arguments into a single array, so we can pass them into the
    # solver (as NumbaMinpack expects arguments as a vector) - we unpack these
    # later in the heat equation solver

    args = pack_args(
        cell,
        met_data,
        dt,
        dz,
        N=N,
    )
    T = np.empty_like(cell["firn_temperature"])

    # fixed surface temperature case - just use the tridiagonal solver as we have a
    # fixed surface BC
    if fixed_sfc:
        fvec = np.array([1.0])
        success = 1
        info = 1
        T_fixed = hnb.propagate_temperature(cell, dz, dt, 273.15, N=1)
        T[0] = 273.15
        T[1:] = T_fixed[:]
        print('T fixed = ', T_fixed[:20])
    # else, use hybrd to solve for the top N layers, then use the tridiagonal
    # solver for the rest of the column.
    else:
        sol, fvec, success, info = hybrd(heq, x, args)
        if N == len(cell["firn_temperature"]):
            T[:N] = sol
            print('T free = ', T[:5])
        else:
            T_tri = hnb.propagate_temperature(cell, dz, dt, sol[-1], N=N)
            T[:N] = sol
            T[N:] = T_tri[:]

    T = np.around(T, decimals=8)
    return T, fvec, success, info


@jit(nopython=True, fastmath=False)
def lid_heateqn_solver(cell, met_data, dt, dz):
    """
    NumbaMinpack version of the lid heat equation solver.
    Solves physics.Numba.heateqn. Called in lid_functions.lid_development.

    Parameters
    ----------
    x : array_like, float, dimension(vert_grid)
        Initial guess at the lid column temperature. [K]
    args : array_like
        Array of arguments to pass into the function.
        See pack_args for details.

    Returns
    -------
    root : array_like, float,
          dimension(core.iceshelf_class.IceShelf.vert_grid_lid)
        Vector containing the calculated lid column temperature, either after
        successful completion or at the end of the final iteration for an
        unsuccessful solution. [K]
    fvec : array_like, float,
          dimension(core.iceshelf_class.IceShelf.vert_grid_lid)
        Vector containing the function evaluated at root, i.e. the raw output.
    success : bool
        Boolean flag determining whether the solution converged, or whether
        there was an error.
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

    x = cell["lid_temperature"]

    args = pack_args(
        cell,
        met_data,
        dt,
        dz,
    )

    root, fvec, success, info = hybrd(eqn, x, args)

    root = np.around(root, decimals=8)
    return root, fvec, success, info


@jit(nopython=True, fastmath=False)
def lake_seb_solver(cell, met_data, dt, dz, formation=False):
    """
    NumbaMinpack version of the lake solver.
    Solves core.choose_solver.dev_eqn or core.choose_solver.form_eqn, which are
    defined in the body of get_lake_surface_energy_equations.

    Parameters
    ----------
    x : array_like, float, dimension(cell.vert_grid_lake)
        Initial estimate of the lake temperature profile. We only use the first
         element in the solver here (in both the lake formation and lake
         development cases).
    args : array_like
        Array of input arguments to the solver. See documentation for form_eqn
        and dev_eqn for details.
    formation : bool
        Flag to determine whether we want the lake *formation* equation,
        or the lake *development* equation.
        Defaults to the development equation, unless specified.
    Returns
    -------
    root : float
        Calculated lake surface temperature, either after successful completion
        or at the end of the final iteration for an unsuccessful solution. [K]
    fvec : array_like, float, dimension(core.iceshelf_class.IceShelf.vert_grid)
        Vector containing the function evaluated at root, i.e. the raw output.
    success : bool
        Boolean flag determining whether the solution converged, or whether
         there was an error.
    info : int
        Integer flag containing information on the status of the solution.
    """
    # load in from the compiled function address in memory rather than the
    # Python function object
    if formation:
        eqn = form_eqnaddress

    else:
        eqn = dev_eqnaddress
    if formation:
        x = np.array([met_data["temperature"]])
    else:
        x = np.array([cell["lake_temperature"][0]])

    args = pack_args(cell, met_data, dt, dz)

    root, fvec, success, info = hybrd(eqn, x, args)
    root = np.around(root, decimals=8)

    return root, fvec, success, info

@jit(nopython=True, fastmath=False)
def lid_seb_solver(cell, met_data, dt, dz, k_lid, Sfrac_lid=None):
    """
    NumbaMinpack version of the lid surface energy balance solver.
    Solves .sfc_energy_lid or  sfc_energy_virtual_lid.

    Called in virtual_lid and lid.lid_development.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    met_data : dict
        Dictionary containing the meteorological data for the current timestep.
        See firn_column for details.
    dt : int
        Number of seconds in the current timestep [s]
    dz : float
        Size of each vertical point in the cell. [m]

    Returns
    -------
    root : float
        the calculated lid surface temperature, either after successful
        completion or at the end of the final iteration for an unsuccessful
        solution. [K]
    fvec : float
        Vector containing the function evaluated at root, i.e. the raw output.
    success : bool
        Boolean flag determining whether the solution converged, or whether
        there was an error.
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
    if cell["v_lid"]:
        eqn = sfc_energy_vlid_address

    else:
        eqn = sfc_energy_lid_address

    if not Sfrac_lid:
        Sfrac_lid = np.array([1])

    args = pack_args(cell, met_data, dt, dz, k_lid=k_lid, Sfrac_lid=Sfrac_lid)
    x = np.array([cell['virtual_lid_temperature']])
    root, fvec, success, info = hybrd(eqn, x, args)

    root = np.around(root, decimals=8)
    return root, fvec, success, info
