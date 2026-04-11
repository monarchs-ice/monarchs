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
from monarchs.physics import surface_fluxes
from monarchs.physics.constants import emissivity, stefan_boltzmann
from monarchs.physics.Numba.surface_energy_balance import lake_formation_eqn, lake_development_eqn, sfc_energy_lid, sfc_energy_virtual_lid


# load in the compiled function address in memory rather than the
# Python function object
heq = hnb.heateqn.address
heq_fixed_sfc = hnb.heateqn_fixed_sfc.address
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
def _thomas_inplace(a, b, c, d, b_work, d_work, x_out):
    n = d.shape[0]
    for i in range(n):
        b_work[i] = b[i]
        d_work[i] = d[i]
    for i in range(1, n):
        m = a[i - 1] / b_work[i - 1]
        b_work[i] -= m * c[i - 1]
        d_work[i] -= m * d_work[i - 1]
    x_out[n - 1] = d_work[n - 1] / b_work[n - 1]
    for i in range(n - 2, -1, -1):
        x_out[i] = (d_work[i] - c[i] * x_out[i + 1]) / b_work[i]


@jit(nopython=True, fastmath=False)
def _assemble_newton_system_inplace(x, T_old, kappa, k0, dz, dt, q_val, dq_val, a, b, c, rhs):
    n = x.shape[0]
    fac = dt / (dz * dz)
    eps = emissivity
    sigma = stefan_boltzmann

    f0 = k0 * (x[0] - x[1]) / dz - (q_val - eps * sigma * x[0] ** 4)
    b[0] = k0 / dz - (dq_val - 4.0 * eps * sigma * x[0] ** 3)
    c[0] = -k0 / dz
    rhs[0] = -f0
    max_abs_resid = abs(f0)

    for i in range(1, n - 1):
        alpha = fac * kappa[i]
        fi = T_old[i] - x[i] + alpha * (x[i + 1] - 2.0 * x[i] + x[i - 1])
        a[i - 1] = alpha
        b[i] = -1.0 - 2.0 * alpha
        c[i] = alpha
        rhs[i] = -fi
        if abs(fi) > max_abs_resid:
            max_abs_resid = abs(fi)

    i = n - 1
    alpha = fac * kappa[i]
    fi = T_old[i] - x[i] + alpha * (-x[i] + x[i - 1])
    a[i - 1] = alpha
    b[i] = -1.0 - alpha
    rhs[i] = -fi
    if abs(fi) > max_abs_resid:
        max_abs_resid = abs(fi)

    return max_abs_resid


@jit(nopython=True, fastmath=False)
def _assemble_fixed_sfc_system_inplace(x, T_old, kappa, dz, dt, a, b, c, rhs):
    n = x.shape[0]
    fac = dt / (dz * dz)

    f0 = x[0] - 273.15
    b[0] = 1.0
    c[0] = 0.0
    rhs[0] = -f0
    max_abs_resid = abs(f0)

    for i in range(1, n - 1):
        alpha = fac * kappa[i]
        fi = T_old[i] - x[i] + alpha * (x[i + 1] - 2.0 * x[i] + x[i - 1])
        a[i - 1] = alpha
        b[i] = -1.0 - 2.0 * alpha
        c[i] = alpha
        rhs[i] = -fi
        if abs(fi) > max_abs_resid:
            max_abs_resid = abs(fi)

    i = n - 1
    alpha = fac * kappa[i]
    fi = T_old[i] - x[i] + alpha * (-x[i] + x[i - 1])
    a[i - 1] = alpha
    b[i] = -1.0 - alpha
    rhs[i] = -fi
    if abs(fi) > max_abs_resid:
        max_abs_resid = abs(fi)

    return max_abs_resid


@jit(nopython=True, fastmath=False)
def _newton_full_column(cell, met_data, dt, dz, fixed_sfc, tol=1e-10, maxiter=12):
    N = len(cell["firn_temperature"])
    T_old = cell["firn_temperature"][:N]
    Sfrac = cell["Sfrac"][:N]
    Lfrac = cell["Lfrac"][:N]
    k, kappa = hnb.get_k_and_kappa(T_old, Sfrac, Lfrac, hnb.cp_air, hnb.cp_water, hnb.k_air, hnb.k_water)

    x = T_old.copy()
    if fixed_sfc:
        x[0] = 273.15

    a = np.empty(N - 1, dtype=np.float64)
    b = np.empty(N, dtype=np.float64)
    c = np.empty(N - 1, dtype=np.float64)
    rhs = np.empty(N, dtype=np.float64)
    b_work = np.empty(N, dtype=np.float64)
    d_work = np.empty(N, dtype=np.float64)
    delta = np.empty(N, dtype=np.float64)

    dT_fd = 1e-3
    resid_tol = max(tol, tol * max(1.0, np.max(np.abs(T_old))))

    converged = False
    info = 2

    for _ in range(maxiter):
        if fixed_sfc:
            max_abs_resid = _assemble_fixed_sfc_system_inplace(
                x, T_old, kappa, dz, dt, a, b, c, rhs
            )
        else:
            q = surface_fluxes.sfc_flux(
                cell["albedo"],
                cell["lid"],
                cell["lake"],
                met_data["LW_down"],
                met_data["SW_down"],
                met_data["temperature"],
                met_data["surf_pressure"],
                met_data["dew_point_temperature"],
                met_data["wind"],
                x[0],
            )
            q_p = surface_fluxes.sfc_flux(
                cell["albedo"],
                cell["lid"],
                cell["lake"],
                met_data["LW_down"],
                met_data["SW_down"],
                met_data["temperature"],
                met_data["surf_pressure"],
                met_data["dew_point_temperature"],
                met_data["wind"],
                x[0] + dT_fd,
            )
            dq = (q_p - q) / dT_fd
            max_abs_resid = _assemble_newton_system_inplace(
                x, T_old, kappa, k[0], dz, dt, q, dq, a, b, c, rhs
            )

        _thomas_inplace(a, b, c, rhs, b_work, d_work, delta)
        x += delta

        if np.max(np.abs(delta)) < tol or max_abs_resid < resid_tol:
            converged = True
            info = 1
            break

    if fixed_sfc:
        x[0] = 273.15

    fvec = np.zeros(N)
    return x, fvec, converged, info


@jit(nopython=True, fastmath=False)
def solve_firn_heateqn(
    cell,
    met_data,
    dt,
    dz,
    fixed_sfc=False,
    solver_method="hybr",
    toggle_dict=None,
):
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

    N = len(cell["firn_temperature"])  # default: full column

    use_newton_solver = False
    minpack_n_50 = False
    if toggle_dict is not None:
        use_newton_solver = toggle_dict["use_newton_solver"]
        minpack_n_50 = toggle_dict["minpack_n_50"]

    # For the MINPACK hybrd path, allow N=50 (nonlinear top) + tridiagonal tail
    if not use_newton_solver and minpack_n_50:
        N = 50

    x = cell["firn_temperature"][:N]

    args = pack_args(
        cell,
        met_data,
        dt,
        dz,
        N=N,
    )

    T = np.empty_like(cell["firn_temperature"])


    if use_newton_solver:
        T, fvec, success, info = _newton_full_column(
            cell, met_data, dt, dz, fixed_sfc=fixed_sfc
        )
    elif fixed_sfc:
        x_fixed = x.copy()
        x_fixed[0] = 273.15
        sol, fvec, success, info = hybrd(heq_fixed_sfc, x_fixed, args)
        T[:N] = sol
        T[0] = 273.15
    else:
        sol, fvec, success, info = hybrd(heq, x, args)
        if N == len(cell["firn_temperature"]):
            T[:N] = sol
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
