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
from monarchs.core.error_handling import (
    check_for_mass_conservation,
    generic_error,
)
from monarchs.physics.constants import (
    rho_ice,
    rho_water,
    emissivity,
    stefan_boltzmann,
)

MODULE_NAME = "monarchs.physics.firn_column"


def firn_column(
    cell,
    dt,
    dz,
    met_data,
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
    met_data : dict
        Dictionary containing the meteorological data for the current timestep.
        This contains the following keys:
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
    routine_name = f"{MODULE_NAME}.firn_column"
    # calculate mass before any melting occurs to use for a diagnostic later
    original_mass = utils.calc_mass_sum(cell)
    percolation_toggle = toggle_dict["percolation_toggle"]
    perc_time_toggle = toggle_dict["perc_time_toggle"]
    # initial guess for the heat equation solver

    # Solve the 1D heat equation. This needs to account for the top boundary
    # condition, which is driven by the meteorology/surface fluxes, so we need
    # to pass these variables into the solver.

    # Hard-coding "hybr" as the MINPACK solver method, as other methods are not
    # supported by the Numba cfunc.
    # If you are running with Scipy, you could consider changing this or adding
    # it as a namelist parameter.

    # Update cell albedo
    cell["albedo"] = surface_fluxes.sfc_albedo(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["v_lid"],
        cell["lake_depth"],
        cell["snow_on_lid"],
    )
    root, _, success, _ = solver.solve_firn_heateqn(
        cell, met_data, dt, dz, fixed_sfc=False, solver_method="hybr"
    )
    # If the solver didn't fail (e.g. due to too many iterations), and we have
    # a surface temperature above the freezing point, then melt will occur.
    # Since the firn column has a fixed boundary condition (273.15 K),
    # recalculate the firn column temperature and regrid the column to account
    # for the height change that occurs due to melting.
    if (root[0] > 273.15) and success:
        dz = cell["firn_depth"] / cell["vert_grid"]

        root_fs, _, success_fixedsfc, _ = solver.solve_firn_heateqn(
            cell, met_data, dt, dz, fixed_sfc=True, solver_method="hybr"
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
                met_data,
                root,
            )
            # print('height change:', height_change)
            if np.isnan(height_change):
                message = (
                    "Height change is NaN - likely due to unrealistic "
                    "meteorological data."
                )
                generic_error(cell, routine_name, message)

            regrid_column.regrid_after_melt(cell, height_change)
        elif not success_fixedsfc and not toggle_dict["ignore_errors"]:
            message = (
                "Heat equation solver failed to converge when surface"
                " temperature was fixed."
            )
            generic_error(cell, routine_name, message)

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
    cell["rho"] = cell["Sfrac"] * rho_ice + cell["Lfrac"] * rho_water

    # Test for mass conservation - if we lose mass, then there is an issue with
    # the regridding.
    new_mass = utils.calc_mass_sum(cell)
    check_for_mass_conservation(cell, original_mass, new_mass, routine_name)


def calc_height_change(
    cell,
    timestep,
    met_data,
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
    met_data : dict
        Dictionary containing the meteorological data for the current timestep.
        See firn_column for details.
    surf_T : float
        Calculated surface temperature of the firn from the initial (non-fixed
        surface) implementation of the heat equation. [K]

    Returns
    -------
    dHdt : float
        Change in firn height as a result of melting. [m]
    """
    routine_name = f"{MODULE_NAME}.calc_height_change"
    epsilon = emissivity
    sigma = stefan_boltzmann

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
        * (273.15 - (cell["firn_temperature"][0] + cell["firn_temperature"][1]) / 2)
        ** 1.156
    )
    # Update cell albedo
    cell["albedo"] = surface_fluxes.sfc_albedo(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["v_lid"],
        cell["lake_depth"],
        cell["snow_on_lid"],
    )
    # now calculate surface flux
    Q = surface_fluxes.sfc_flux(
        cell["albedo"],
        cell["lid"],
        cell["lake"],
        met_data["LW_down"],
        met_data["SW_down"],
        met_data["temperature"],
        met_data["surf_pressure"],
        met_data["dew_point_temperature"],
        met_data["wind"],
        cell["firn_temperature"][0],
    )

    # Strictly speaking, since the MONARCHS grid is defined from the surface
    # downward, this is actually the "wrong" way round. It is simpler
    # conceptually for this function to consider the
    # height change rather than depth change, so dz is kept positive here.
    dtdz = (cell["firn_temperature"][0] - cell["firn_temperature"][1]) / dz

    # # If the surface is very empty, just melt one layer
    # if cell["Sfrac"][0] < 0.1:
    #     dHdt = cell["firn_depth"] / cell["vert_grid"]
    dHdt = (
        (Q - epsilon * sigma * (cell["firn_temperature"][0]) ** 4 - k_sfc * dtdz)
        / (rho_ice * (cell["Sfrac"][0] * L_fus))
        * timestep
    )
    # Melt maximum one layer per timestep to avoid instability
    if dHdt > (cell["firn_depth"] / cell["vert_grid"]):
        dHdt = cell["firn_depth"] / cell["vert_grid"]
    cell["firn_boundary_change"] += dHdt

    if 0 > dHdt > -0.01:
        dHdt = 0
    elif dHdt < -0.01:
        message = (
            "Height change during melt is negative, and outside the bounds of"
            " a numerical error"
        )
        generic_error(cell, routine_name, message)
    elif np.isnan(dHdt):
        message = "Height change during melt is NaN"
        generic_error(cell, routine_name, message)
    return dHdt
