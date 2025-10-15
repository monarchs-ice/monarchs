"""
Module containing functions relating to the firn column. Some physics is
contained in percolation.py.

"""

# for Numba compatibility - need to use broad exceptions, not specific ones
# pylint: disable=broad-exception-raised, raise-missing-from
# TODO - flesh out module-level docstring
import numpy as np
from monarchs.physics import percolation, surface_fluxes, solver, regrid_column
from monarchs.core import utils


def firn_column(
    cell,
    dt,
    dz,
    lw_in,
    sw_in,
    air_temp,
    p_air,
    dew_point_temperature,
    wind,
    toggle_dict,
):
    """
    Perform the various processes applied to the firn each timestep where we
    don't have exposed water at the surface.

    The logic works as follows:
    Solve heat equation for firn

    If surface temperature is above melting point of water
    (i,e melting takes place):

    - Determine the height change as a result of the melting
    - Regrid everything to take account for this deformation
    - Re-solve heat equation, now with fixed surface temperature
    - Calculate the amount of water added to the surface as a result of melt
    - Percolate that water down, taking into account any lids or lakes that may
    - have formed.

    Otherwise:
    - Update cell temperature and continue.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    dt : int
        Number of seconds in the current timestep [s]
    dz : float
        Height of each vertical point in the cell. [m]
    lw_in : float
        Downwelling longwave radiation at the current timestep. [W m^-2]
    sw_in : float
        Downwelling shortwave radiation at the current timestep. [W m^-2]
    air_temp : float
        Surface air temperature at the current timestep. [K]
    p_air : float
        Surface air pressure at the current timestep. [Pa]
    dew_point_temperature : float
        Dewpoint temperature of the air at the surface at the current
        timestep. [K]
    wind : float
        Wind speed at the surface at the current timestep. [m s^-1]
    toggle_dict : dict
        Dictionary containing some switches that affect the running of
        the model.

    Returns
    -------
    None (amends cell inplace)
    """
    # calculate mass before any melting occurs to use for a diagnostic later
    original_mass = utils.calc_mass_sum(cell)
    percolation_toggle = toggle_dict["percolation_toggle"]
    perc_time_toggle = toggle_dict["perc_time_toggle"]
    # initial guess for the heat equation solver
    x = cell["firn_temperature"]

    # Solve the 1D heat equation. This needs to account for the top boundary
    # condition, which is driven by the meteorology/surface fluxes, so we need
    # to pass these variables into the solver. Since we are using a Numba cfunc
    # to solve the heat equation, we need to pack the arguments into a
    # single vector.
    args = [
        cell,
        dt,
        dz,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
    ]
    # Hard-coding "hybr" as the MINPACK solver method, as other methods are not
    # supported by the Numba cfunc.
    # If you are running with Scipy, you could consider changing this or adding
    # it as a namelist parameter.
    root, _, success, _ = solver.solve_firn_heateqn(
        x, args, fixed_sfc=False, solver_method="hybr"
    )
    # If the solver didn't fail (e.g. due to too many iterations), and we have
    # a surface temperature above the freezing point, then melt will occur.
    # Since the firn column has a fixed boundary condition (273.15 K),
    # recalculate the firn column temperature and regrid the column to account
    # for the height change that occurs due to melting.
    if (root[0] > 273.15) and success:
        dz = cell["firn_depth"] / cell["vert_grid"]
        args = (
            cell,
            dt,
            dz,
            lw_in,
            sw_in,
            air_temp,
            p_air,
            dew_point_temperature,
            wind,
        )
        root_fs, _, success_fixedsfc, _ = solver.solve_firn_heateqn(
            x, args, fixed_sfc=True, solver_method="hybr"
        )
        # if *this* solver works, then update the firn temperature and regrid
        # the firn column, accounting for the melt.
        if success_fixedsfc:
            cell["firn_temperature"] = root_fs
            cell["meltflag"][0] = 1
            cell["melt"] = True
            height_change = calc_height_change(
                cell,
                dt,
                lw_in,
                sw_in,
                air_temp,
                p_air,
                dew_point_temperature,
                wind,
                root,
            )

            if np.isnan(height_change):
                raise ValueError(
                    "Height change is NaN - likely due to unrealistic"
                    " meteorological data."
                )

            regrid_column.regrid_after_melt(cell, height_change)
        elif not success_fixedsfc and not toggle_dict["ignore_errors"]:
            raise ValueError(
                "Heat equation solver failed to converge when surface"
                " temperature was fixed."
            )

    # If the surface temperature is below the melting point, then we simply
    # update the firn temperature provided the solver didn't fail.
    elif success:
        cell["firn_temperature"] = root
        cell["melt"] = False

    # Run the percolation algorithm to ensure that any meltwater moves down the
    # column (refreezing as it goes).
    if percolation_toggle:
        percolation.percolate(cell, dt, perc_time_toggle=perc_time_toggle)
    # Update density to reflect newly calculated solid/liquid fractions
    cell["rho"] = (
        cell["Sfrac"] * cell["rho_ice"] + cell["Lfrac"] * cell["rho_water"]
    )

    # Test for mass conservation - if we lose mass, then there is an issue with
    # the regridding.
    new_mass = utils.calc_mass_sum(cell)
    assert abs(original_mass - new_mass) < 1.5 * 10**-7


def calc_height_change(
    cell,
    timestep,
    lw_in,
    sw_in,
    air_temp,
    p_air,
    dew_point_temperature,
    wind,
    surface_temp,
):
    """
    Determine the amount of firn height change that arises due to melting.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    timestep : float
        Number of seconds in each timestep. [s]
    lw_in : float
        Downwelling longwave radiation at the current timestep. [W m^-2]
    sw_in : float
        Downwelling shortwave radiation at the current timestep. [W m^-2]
    air_temp : float
        Surface air temperature at the current timestep. [K]
    p_air : float
        Surface air pressure at the current timestep. [Pa]
    dew_point_temperature : float
        Dewpoint temperature of the air at the surface at the current
        timestep. [K]
    wind : float
        Wind speed at the surface at the current timestep. [m s^-1]
    surf_T : float
        Calculated surface temperature of the firn from the initial (non-fixed
        surface) implementation of the heat equation. [K]

    Returns
    -------
    dHdt : float
        Change in firn height as a result of melting. [m]
    """
    epsilon = 0.98
    sigma = 5.670373 * 10**-8
    dz = cell["firn_depth"] / cell["vert_grid"]
    L_fus = 334000
    if (
        cell["firn_temperature"][0] > 273.14999999
        and cell["firn_temperature"][0] < 273.151
    ):
        cell["firn_temperature"][0] = 273.15
    if (
        cell["firn_temperature"][1] > 273.14999999
        and cell["firn_temperature"][1] < 273.151
    ):
        cell["firn_temperature"][1] = 273.15
    k_sfc = (
        1000 * 2.24 * 10**-3
        + 5.975
        * 10**-6
        * (
            273.15
            - (cell["firn_temperature"][0] + cell["firn_temperature"][1]) / 2
        )
        ** 1.156
    )
    Q = surface_fluxes.sfc_flux(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["lake_depth"],
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        surface_temp[0],
    )

    # Strictly speaking, since the MONARCHS grid is defined from the surface
    # downward, this is actually the "wrong" way round. It is simpler
    # conceptually for this function to consider the
    # height change rather than depth change, so dz is kept positive here.
    dtdz = (cell["firn_temperature"][0] - cell["firn_temperature"][1]) / dz
    dHdt = (
        (
            Q
            - epsilon
            * sigma
            * (
                1.5
                * (
                    cell["firn_temperature"][0]
                    - 0.5 * cell["firn_temperature"][1]
                )
            )
            ** 4
            - k_sfc * dtdz
        )
        / (cell["rho_ice"] * (cell["Sfrac"][0] * L_fus))
        * timestep
    )
    cell["firn_boundary_change"] += dHdt

    # dHdt = 0.01
    if 0 > dHdt > -0.01:
        dHdt = 0
    elif dHdt < -0.01:
        raise ValueError(
            "Height change during melt is negative, and outside the bounds of"
            " a numerical error"
        )
    elif np.isnan(dHdt):
        print("...")
        raise ValueError("Height change during melt is NaN")
    return dHdt
