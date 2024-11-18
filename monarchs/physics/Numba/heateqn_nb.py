import numpy as np
from numba import cfunc
from NumbaMinpack import minpack_sig

from monarchs.physics.Numba.extract_args import (
    extract_args,
    extract_args_fixedsfc,
    extract_args_lid,
)
from monarchs.physics.surface_fluxes import sfc_flux

"""
Functions used to solve the heat equation using NumbaMinpack's hybrd function.
This gives significant performance boosts over scipy's fsolve(hybrd = True).

For the version of heateqn used with Scipy's optimize.fsolve, see /physics/heateqn.

These are kept separate as there is significantly more complexity required to get the 
inputs in the correct format for this version, and the requirement to not return anything
(whereas scipy.optimize.fsolve requires a return value)
"""


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

    vert_grid = np.int32(args[0])
    # output = np.zeros(vert_grid)
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
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
    ) = extract_args(args)

    k_ice = np.zeros(np.shape(T))
    # output = np.zeros(np.shape(T))
    for i in np.arange(len(T)):
        # If surface temperature is above 273.15, then we assume that T is at 273.15
        if T[i] < 273.15:
            k_ice[i] = 1000 * (2.24e-3 + 5.975e-6 * ((273.15 - T[i]) ** 1.156))
        else:
            k_ice[i] = 2.24
    # if np.isnan(k_ice).any():
    #     for i in range(len(k_ice)):
    #         k_ice[i] = k_ice[i].real
    #     print(np.max(T))
    # raise ValueError('k_ice = nan \n')

    k = Sfrac * k_ice + (1 - Sfrac - Lfrac) * k_air + Lfrac * k_water

    cp_ice = 7.16 * T + 138

    cp = Sfrac * cp_ice + (1 - Sfrac - Lfrac) * cp_air + Lfrac * cp_water
    rho = Sfrac * 913 + Lfrac * 1000
    kappa = k / (cp * rho)  # thermal diffusivity [m^2 s^-1]

    epsilon = 0.98
    sigma = 5.670374 * (10**-8)

    Q = sfc_flux(
        melt,
        exposed_water,
        lid,
        lake,
        lake_depth,
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
        x[0],
    )

    # print(x[0])
    output[0] = k[0] * ((x[0] - x[1]) / dz) - (Q - epsilon * sigma * x[0] ** 4)
    for idx in np.arange(1, vert_grid - 1):
        # idx = np.int32(idx)
        output[idx] = (
            T[idx]
            - x[idx]
            + dt * (kappa[idx] / dz**2) * (x[idx + 1] - 2 * x[idx] + x[idx - 1])
        )

    output[vert_grid - 1] = (
        T[vert_grid - 1]
        - x[vert_grid - 1]
        + dt * (kappa[vert_grid - 1] / dz**2) * (-x[vert_grid - 1] + x[vert_grid - 2])
    )


def heateqn_fixedsfc(x, output, args):
    """
    Heat equation function to be passed into a solver (e.g. hybrd), but
    assuming a fixed surface temperature. This is split into a separate function
    to heateqn since the arguments are different (trying to combine them
    results in issues with Numba).

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
        See extract_args_fixedsfc for more details.

    Returns
    -------
    None. (the output is handled by the solver)

    """
    vert_grid = np.int32(args[0])
    # output = np.zeros(vert_grid)
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
        T_sfc,
    ) = extract_args_fixedsfc(args)

    k_ice = np.zeros(np.shape(T))

    for i in range(len(T)):
        # If surface temperature is above 273.15, then we assume that T is at 273.15
        if T[i] < 273.15:
            k_ice[i] = 1000 * (2.24e-3 + 5.975e-6 * ((273.15 - T[i]) ** 1.156))
        else:
            k_ice[i] = 2.24

    k = Sfrac * k_ice + (1 - Sfrac - Lfrac) * k_air + Lfrac * k_water

    cp_ice = 7.16 * T + 138
    cp = Sfrac * cp_ice + (1 - Sfrac - Lfrac) * cp_air + Lfrac * cp_water
    rho = Sfrac * 913 + Lfrac * 1000

    kappa = k / (cp * rho)  # thermal diffusivity [m^2 s^-1]

    # fixed surface temperature
    output[0] = x[0] - T_sfc

    for idx in np.arange(1, vert_grid - 1):
        output[idx] = (
            T[idx]
            - x[idx]
            + dt * (kappa[idx] / dz**2) * (x[idx + 1] - 2 * x[idx] + x[idx - 1])
        )

    output[vert_grid - 1] = (
        T[vert_grid - 1]
        - x[vert_grid - 1]
        + dt * (kappa[vert_grid - 1] / dz**2) * (-x[vert_grid - 1] + x[vert_grid - 2])
    )


def heateqn_lid(x, output, args):
    """
    Heat equation solver function for a frozen lid. To be passed into NumbaMinpack's hybrd solver.

    Parameters
    ----------
    x : array_like, float64, dimension(core.iceshelf_class.IceShelf.vert_grid_lid)
        Starting estimate for the roots of the PDE we are trying to solve.
        This is by default cell.firn_temperature.

    output : array_like, float64, dimension(core.iceshelf_class.IceShelf.vert_grid_lid)
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
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
    ) = extract_args_lid(args)

    # print('here')
    # k_ice = (1000 * 2.24 * 0.001 + 5.975 * 0.000001 * (273.15 - cell.lid_temperature) ** 1.156)
    # k = cell.Sfrac_lid * k_ice + (1 - cell.Sfrac_lid - cell.Lfrac_lid) * cell.k_air + cell.Lfrac_lid * cell.k_water
    cp_ice = 1000 * (0.00716 * lid_temperature + 0.138)
    cp = Sfrac_lid * cp_ice + (1 - Sfrac_lid) * cp_air
    rho = 913
    kappa = k_lid / (cp * rho)  # thermal diffusivity [m^2 s^-1]
    epsilon = 0.98
    sigma = 5.670373 * (10**-8)
    Q = sfc_flux(
        melt,
        exposed_water,
        lid,
        lake,
        lake_depth,
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
        x[0],
    )
    output[0] = k_lid * (x[0] - x[1]) / dz - (Q - epsilon * sigma * x[0] ** 4)

    for idx in np.arange(1, vert_grid_lid - 1):
        output[idx] = (
            lid_temperature[idx]
            - x[idx]
            + dt * (kappa[idx] / dz**2) * (x[idx + 1] - 2 * x[idx] + x[idx - 1])
        )

    output[-1] = x[vert_grid_lid - 1] - 273.15  # fix boundary temperature to 273.15


heateqn = cfunc(minpack_sig)(heateqn)
heateqn_fixedsfc = cfunc(minpack_sig)(heateqn_fixedsfc)
heateqn_lid = cfunc(minpack_sig)(heateqn_lid)
