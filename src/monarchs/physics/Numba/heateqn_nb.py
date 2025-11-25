"""
Functions used to solve the heat equation using NumbaMinpack's hybrd function.
This gives significant performance boosts over scipy's fsolve(hybrd = True).

For the version of heateqn used with Scipy's optimize.fsolve, see
/physics/heateqn.

These are kept separate as there is significantly more complexity required to
get the inputs in the correct format for this version, and the requirement to
not return anything (whereas scipy.optimize.fsolve requires a return value)

"""

# TODO - docstrings, possibility of using numba.overload
import numpy as np
from numba import cfunc, njit
from NumbaMinpack import minpack_sig
from monarchs.physics.Numba import extract_args
from monarchs.physics import surface_fluxes


@njit
def get_k_and_kappa(T, sfrac, lfrac, cp_air, cp_water, k_air, k_water):
    # precompute some values
    rho = sfrac * 917 + lfrac * 1000
    k_ice = np.zeros(np.shape(T), dtype=np.float64)
    k_ice[T < 273.15] = 1000 * (
        2.24e-03 + 5.975e-06 * ((273.15 - T[T < 273.15]) ** 1.156)
    )
    k_ice[T >= 273.15] = 2.24
    k = sfrac * k_ice + (1 - sfrac - lfrac) * k_air + lfrac * k_water
    cp_ice = 7.16 * T + 138
    cp = sfrac * cp_ice + (1 - sfrac - lfrac) * cp_air + lfrac * cp_water
    kappa = k / (cp * rho)
    return k, kappa


@njit
def propagate_temperature(cell, dz, dt, T_bc_top, N=10):
    T_old = cell["firn_temperature"][N:]
    Sfrac = cell["Sfrac"][N:]
    Lfrac = cell["Lfrac"][N:]
    cp_air = cell["cp_air"]
    cp_water = cell["cp_water"]
    k_air = cell["k_air"]
    k_water = cell["k_water"]
    k, kappa = get_k_and_kappa(
        T_old, Sfrac, Lfrac, cp_air, cp_water, k_air, k_water
    )
    # total number of layers in the firn column
    total_len = np.shape(cell["firn_temperature"])[0]
    # number of layers below the nonlinear region
    n = total_len - N

    # Initialize diagonals and RHS
    A = np.zeros(n - 1, dtype=np.float64)  # sub-diagonal (lower)
    B = np.zeros(n, dtype=np.float64)  # main diagonal
    C = np.zeros(n - 1, dtype=np.float64)  # super-diagonal (upper)
    D = np.zeros(n, dtype=np.float64)  # RHS vector

    factor = np.float64(dt / dz ** 2)

    # First row: connect to top nonlinear region
    i = 0
    alpha = factor * kappa[i]
    B[i] = 1 + 2 * alpha
    C[i] = -alpha
    D[i] = T_old[i] + alpha * T_bc_top

    # Interior rows
    # First row: connect to top nonlinear region
    i = 0
    alpha = factor * kappa[i]
    B[i] = 1 + 2 * alpha
    C[i] = -alpha
    D[i] = T_old[i] + alpha * T_bc_top

    # Interior rows
    for i in np.arange(1, n - 1):
        alpha = factor * kappa[i]
        A[i - 1] = -alpha
        B[i] = 1 + 2 * alpha
        C[i] = -alpha
        D[i] = T_old[i]

    # zero-flux (neumann) boundary condition
    i = n - 1
    alpha = factor * kappa[i]
    A[i - 1] = -alpha
    B[i] = 1 + alpha
    D[i] = T_old[i]
    T_new = solve_tridiagonal(A, B, C, D)
    return T_new


@njit
def solve_tridiagonal(a, b, c, d):
    """
    a: sub-diagonal (len n-1), A[i, i-1]
    b: main diagonal (len n), A[i, i]
    c: super-diagonal (len n-1), A[i, i+1]
    d: RHS (len n)
    """
    n = np.shape(d)[0]
    ac, bc, cc, dc = a, b, c, d

    # Forward elimination
    for i in np.arange(1, n):
        m = ac[i - 1] / bc[i - 1]
        bc[i] -= m * cc[i - 1]
        dc[i] -= m * dc[i - 1]

    # Back substitution
    x = np.zeros(n)
    x[-1] = dc[-1] / bc[-1]
    for i in np.arange(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]

    return x


def heateqn(x, output, args):
    """
    Heat equation function to be passed into a solver (e.g. hybrd). This uses
    the finite difference method, which is combined with the hybrd root-finding
    algorithm.

    Parameters
    ----------
    x : float64[:]
        Starting estimate for the roots of the PDE we are trying to solve.
        This is by default cell.firn_temperature.

    output : float64[:]
        Output array, i.e. the values of x such that F(x) = 0.
        This array is handled automatically by the solver, so does not need
        to be initialised and explicitly entered as an argument.

    args : float64[:]
        Vector describing arguments to the system of equations being solved.
        This comprises many of the attributes of an IceShelf instance, and
        some physical variables such as specific humidity, wind etc.
        See extract_args for more details.

    Returns
    -------
    None. (the output is handled by the solver)

    """

    # separate "args" into the relevant variables
    (
        T,
        Sfrac,
        Lfrac,
        k_air,
        k_water,
        cp_air,
        cp_water,
        dt,
        dz,
        melt,
        exposed_water,
        lid,
        lake,
        lake_depth,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
    ) = extract_args.extract_args_firn(args)

    N = np.int32(args[0])

    epsilon = 0.98
    sigma = 5.670374e-8

    Q = surface_fluxes.sfc_flux(
        melt,
        exposed_water,
        lid,
        lake,
        lake_depth,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        x[0],
    )

    k, kappa = get_k_and_kappa(
        T, Sfrac, Lfrac, cp_air, cp_water, k_air, k_water
    )
    # Surface temperature equation (residual)
    output[0] = k[0] * ((x[0] - x[1]) / dz) - (Q - epsilon * sigma * x[0] ** 4)

    # Calculate the temperature profile for the first 10 layers
    for idx in np.arange(1, N - 1, 1):
        output[idx] = (
            T[idx]
            - x[idx]
            + dt
            * (kappa[idx] / dz ** 2)
            * (x[idx + 1] - 2 * x[idx] + x[idx - 1])
        )

    output[N - 1] = (
        T[N - 1]
        - x[N - 1]
        + dt * (kappa[N - 1] / dz ** 2) * (-x[N - 1] + x[N - 2])
    )


def heateqn_lid(x, output, args):
    """
    Heat equation solver function for a frozen lid. To be passed into
    NumbaMinpack's hybrd solver.

    Parameters
    ----------
    x : array_like, float64,
          dimension(core.iceshelf_class.IceShelf.vert_grid_lid)
        Starting estimate for the roots of the PDE we are trying to solve.
        This is by default cell.firn_temperature.

    output : array_like, float64,
          dimension(core.iceshelf_class.IceShelf.vert_grid_lid)
        Output array, i.e. the values of x such that F(x) = 0.
        This array is handled automatically by the solver, so does not need
        to be initialised and explicitly entered as an argument.

    args : array_like
        Vector describing arguments to the system of equations being solved.
        This comprises many of the attributes of an IceShelf instance, and
        some physical variables such as specific humidity, wind etc.
        See extract_args_lid for more details.

    Returns
    -------
    None (output is handled by the solver)

    """
    vert_grid_lid = np.int32(args[0])

    # separate "args" into the relevant variables
    (
        lid_temperature,
        Sfrac_lid,
        k_lid,
        cp_air,
        cp_water,
        dt,
        dz,
        melt,
        exposed_water,
        lid,
        lake,
        lake_depth,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
    ) = extract_args.extract_args_lid(args)

    k_lid = 1000 * (
        2.24e-03 + 5.975e-06 * ((273.15 - lid_temperature) ** 1.156)
    )

    cp_ice = 1000 * (0.00716 * lid_temperature + 0.138)
    cp = Sfrac_lid * cp_ice
    rho = 917
    # thermal diffusivity [m^2 s^-1]
    kappa = k_lid / (cp * rho)
    epsilon = 0.98
    sigma = 5.670374 * (10 ** -8)
    tau_ice = 1.5
    Q = surface_fluxes.sfc_flux(
        melt,
        exposed_water,
        lid,
        lake,
        lake_depth,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        x[0],
    )
    output[0] = k_lid[0] * ((x[0] - x[1]) / dz) - (
        Q - epsilon * sigma * x[0] ** 4
    )
    # SW radiation baked into surface flux already

    for idx in np.arange(1, vert_grid_lid - 1):
        z_depth = idx * dz
        flux_in = sw_in * 0.6 * np.exp(-tau_ice * z_depth)
        flux_out = sw_in * 0.6 * np.exp(-tau_ice * z_depth + 1)
        sw_absorbed_in_layer = flux_in - flux_out
        # convert source to temperature change contribution: S / (rho * cp)
        # units: [W/m3] / ([kg/m3] * [J/kg/K]) = [J/s/m3] / [J/m3/K] = K/s
        dT_solar = sw_absorbed_in_layer / (rho * cp[idx])
        output[idx] = (
            lid_temperature[idx]
            - x[idx]
            + dt
            * (
                (
                    (kappa[idx])
                    * (x[idx + 1] - 2 * x[idx] + x[idx - 1])
                    / dz ** 2
                )
                + dT_solar
            )
        )

    output[-1] = x[vert_grid_lid - 1] - 273.15


heateqn = cfunc(minpack_sig)(heateqn)
heateqn_lid = cfunc(minpack_sig)(heateqn_lid)
