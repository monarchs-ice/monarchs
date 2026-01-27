"""
Numba-compatible surface energy balance equations for lake/lid formation and
development.
"""

import numpy as np
from numba import jit
from monarchs.physics.surface_fluxes import sfc_flux
from monarchs.physics.constants import emissivity, stefan_boltzmann, k_water, k_air
from monarchs.physics.Numba import extract_args
######################
# EQUATIONS TO SOLVE #
######################

@jit(nopython=True, fastmath=False)
def lake_formation_eqn(x, output, args):
    """
    Numba-compatible form of the lake formation version of the surface
    temperature equation.

    Parameters
    ----------
    x : array_like, float, dimension(vert_grid_lake)
        Initial estimate of the lake temperature. [K]
    output : array_like, float, dimension(vert_grid_lake)
        Output array containing the lake temperature. We only actually return
        the first element of this array.
        This may be possible to set to float, along with x, but Numba works in
        mysterious ways and this seems to compile and work.
    args : array_like
        Array of input arguments to be extracted into the relevant variables
        (firn_depth, vert_grid, Q, k, and T1).

    Returns
    -------
    None.
    """
    (vert_grid, firn_depth, dt, dz, melt, albedo, exposed_water,
     lid, lid_depth, virtual_lid, virtual_lid_depth, lake, lake_depth, snow_on_lid) = extract_args.extract_scalars(args)
    (lw_in, sw_in, air_temp, p_air,
     dew_point_temperature, wind)  = extract_args.extract_met_data(args)
    firn_temperature, Sfrac, Lfrac = extract_args.extract_firn_arrays(args)
    k = 1000 * (
                2.24 * 10 ** -3
                + 5.975
                * 10 ** -6
                * (273.15 - firn_temperature[0]) ** 1.156
            )
    k = Sfrac[0] * k + Lfrac[0] * k_water + (1 - Sfrac[0] - Lfrac[0]) * k_air
    T1 = firn_temperature[1]
    Q = sfc_flux(
        albedo,
        lid,
        lake,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        firn_temperature[0],
    )
    # set output[0] rather than just output else we will just return our
    # initial guess.
    output[0] = (
            -emissivity * stefan_boltzmann * x[0] ** 4
            + Q
            - k * (x[0] - T1) / (firn_depth / vert_grid)
    )


@jit(nopython=True, fastmath=False)
def lake_development_eqn(x, output, args):
    """
    Numba-compatible form of the lake development version of the surface
    temperature equation.

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
    J = 0.1 * (9.8 * 5 * 10 ** -5 * (1.19 * 10 ** -7) ** 2 / 10 ** -6) ** (
            1 / 3
    )
    (vert_grid, firn_depth, dt, dz, melt, albedo, exposed_water,
     lid, lid_depth, virtual_lid, virtual_lid_depth, lake, lake_depth, snow_on_lid) = extract_args.extract_scalars(args)
    (lw_in, sw_in, air_temp, p_air,
     dew_point_temperature, wind)  = extract_args.extract_met_data(args)

    lake_temperature, vert_grid_lake = extract_args.extract_lake_variables(args)

    Q = sfc_flux(
        albedo,
        lid,
        lake,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        x[0],
    )

    T_core = lake_temperature[int(vert_grid_lake / 2)]
    # fluxes +ve downwards - if T_sfc > T_core, we have a flux downward
    # which cools the lake surface - negative sign
    # we have Q (surface flux, downwards) + longwave radiation (upwards, so
    # negative) + turbulent flux (downwards if T_sfc < T_core, upwards if T_sfc >
    # T_core).
    output[0] = np.array(
        [
            -emissivity * (stefan_boltzmann) * (x[0] ** 4)
            + Q
            - np.sign(x[0] - T_core)
            * 1000
            * 4181
            * J
            * abs(x[0] - T_core) ** (4 / 3)
        ]
    )


@jit(nopython=True, fastmath=False)
def sfc_energy_virtual_lid(x, output, args):

    (vert_grid, firn_depth, dt, dz, melt, albedo, exposed_water,
     lid, lid_depth, virtual_lid, virtual_lid_depth,
     lake, lake_depth, snow_on_lid) = extract_args.extract_scalars(args)

    (lw_in, sw_in, air_temp, p_air,
     dew_point_temperature, wind) = extract_args.extract_met_data(args)
    (lid_temperature, Sfrac_lid,
     k_v_lid, vert_grid_lid, v_lid_depth, v_lid_temp) = extract_args.extract_lid_variables(args)

    Q = sfc_flux(
        albedo,
        lid,
        lake,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        x[0],
    )

    # set output[0] rather than just output as solution doesn't converge
    # otherwise as it expects an array
    # sign convention = fluxes positive downwards
    T_interface = 273.15
    effective_thickness = max(v_lid_depth, 1e-5)
    # conductive flux from virtual lid into interface - positive downward
    # (i.e. if lid is colder than interface, heat flows up, out of interface),
    # represented by a negative flux
    conduction_flux = - k_v_lid * (x[0] - T_interface) / effective_thickness
    # so then with this positive downwards convention, we sum
    # the surface fluxes (Q) downwards, + the conductive flux downwards,
    # + the longwave radiation (which upwards, so negative here)
    output[0] = (
        -emissivity * stefan_boltzmann * (x[0] ** 4)
        + Q
        + conduction_flux
    )


@jit(nopython=True, fastmath=False)
def sfc_energy_lid(x, output, args):
    """

    Parameters
    ----------
    x - array_like, float, dimension(vert_grid_lid)
        Initial estimate of the lid surface temperature. [K].
        Use the air temperature for this - as we only run this when initialising the lid.
    args

    Returns
    -------
    output : float

    """

    (vert_grid, firn_depth, dt, dz, melt, albedo, exposed_water,
     lid, lid_depth, virtual_lid, virtual_lid_depth, lake, lake_depth, snow_on_lid) = extract_args.extract_scalars(args)
    (lw_in, sw_in, air_temp, p_air,
     dew_point_temperature, wind) = extract_args.extract_met_data(args)
    (lid_temperature, Sfrac_lid,
     k_lid, vert_grid_lid, v_lid_depth, v_lid_temp) = extract_args.extract_lid_variables(args)

    sub_T = lid_temperature[0]

    Q = sfc_flux(
        albedo,
        lid,
        lake,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        x[0],
    )

    output[0] = (
        -emissivity * stefan_boltzmann * x[0] ** 4
        + Q
        - k_lid * (x[0] - sub_T) / (lid_depth / vert_grid_lid)
    )
