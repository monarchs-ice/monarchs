"""
Module handling snow accumulation on the surface.
Uses conservative regridding to maintain mass balance when adding
new layers to the top of the firn column.
"""

import numpy as np
from monarchs.core import utils

from monarchs.physics.regrid_column import conservative_regrid

MODULE_NAME = "monarchs.physics.snow_accumulation"

def snowfall(cell, snow_depth, snow_rho, snow_T):
    """
    After melting occurs, subtract the amount of melting from the firn height,
    convert it into meltwater, and interpolate the entire column to the new
    vertical profile accounting for this height change.
    This meltwater is either converted into surface liquid water fraction,
    or if there is a lake, into lake height.
    code Code


    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    height_change : float
        Change in the firn height as a result of melting. [m]
    lake : bool, optional
        Flag to determine whether a lake is present or not. This is contained
        here so that we can re-use the bulk of this algorithm, but with some
        small changes to reflect the different situation that occurs when a
        lake is present.

    Returns
    -------
    None
    """
    routine_name = f"{MODULE_NAME}.snowfall"

    # 0. trivial checks
    if snow_depth <= 0:
        return

    # handle Lids and Lakes (no regridding of firn)
    if cell["lid"]:
        # add to lid depth
        cell["lid_depth"] += snow_depth * snow_rho / cell["rho_water"]
        return
    elif cell["lake"] and not cell["lid"]:
        # add to lake depth
        cell["lake_depth"] += snow_depth * snow_rho / cell["rho_water"]
        return

    # dry firn/ice lens

    # calculate mass
    current_mass = utils.calc_mass_sum(cell)
    added_mass = snow_depth * snow_rho
    expected_new_mass = current_mass + added_mass

    # sanitize Input Temperature
    if snow_T > 273.15:
        snow_T = 273.15


    nz = int(cell["vert_grid"])
    old_depth = float(cell["firn_depth"])

    # old edges start at 0 - shift them down by 'snow_depth'.
    old_edges = np.linspace(0, old_depth, nz + 1)
    shifted_old_edges = old_edges + snow_depth
    source_edges = np.concatenate((np.array([0.0]), shifted_old_edges))

    snow_sfrac = snow_rho / cell["rho_ice"]
    # ensure Sfrac doesn't exceed 1 (e.g. if input rho > 917)
    if snow_sfrac > 1.0:
        snow_sfrac = 1.0

    # combine solid fraction for both snow and old firn
    # assume fresh snow is dry
    source_Sfrac = np.concatenate((np.array([snow_sfrac]), cell["Sfrac"]))
    source_Lfrac = np.concatenate(
        (np.array([0.0]), cell["Lfrac"]))

    # temperature interpolation - use centres rather than edges
    old_centers = 0.5 * (old_edges[:-1] + old_edges[1:])
    shifted_old_centers = old_centers + snow_depth

    # new snow center is at half the snow depth
    snow_center = snow_depth / 2.0

    source_centers = np.concatenate((np.array([snow_center]), shifted_old_centers))
    source_temps = np.concatenate((np.array([snow_T]), cell["firn_temperature"]))

    # new grid (after height change)
    new_total_depth = old_depth + snow_depth
    target_edges = np.linspace(0, new_total_depth, nz + 1)
    target_centers = 0.5 * (target_edges[:-1] + target_edges[1:])

    # conservative regridding
    new_Sfrac = conservative_regrid(source_edges, source_Sfrac, target_edges)
    new_Lfrac = conservative_regrid(source_edges, source_Lfrac, target_edges)

    # linear interpolation in temperature
    new_T = np.interp(target_centers, source_centers, source_temps)

    # update the cell
    cell["firn_depth"] = new_total_depth
    cell["Sfrac"] = new_Sfrac
    cell["Lfrac"] = new_Lfrac
    cell["firn_temperature"] = new_T
    cell["vertical_profile"] = np.linspace(0, new_total_depth, nz)

    # clip Sfrac and Lfrac if regridding causes them to exceed physical limits
    cell["Sfrac"] = np.clip(cell["Sfrac"], 0, 1)
    cell["Lfrac"] = np.clip(cell["Lfrac"], 0, 1)

    # mass checks
    final_mass = utils.calc_mass_sum(cell)
    tol = max(1e-7, 1e-10 * expected_new_mass)

    if abs(final_mass - expected_new_mass) > tol:
        message = f"Mass conservation failed in snowfall.\n" \
                  f"Expected: {expected_new_mass}\n" \
                  f"Actual:   {final_mass}\n" \
                  f"Diff:     {final_mass - expected_new_mass}"
        utils.generic_error(cell, routine_name, message)

def densification(cell, t_steps_per_day):
    """

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.

    t_steps_per_day : int

    Returns
    -------

    """
    #     TODO - not important now, but implement for later.
    # change per timestep
    dt = 1 / (8 * t_steps_per_day * 365)
    # acceleration due to gravity [m s^-2]
    g = 9.81
    # values used in Arthern et al.
    ec = 60
    eg = 42.4
    # gas constant
    R = 8.3144598
    T_av = 264.56010894609415
    e = np.exp(-ec / (R * cell["firn_temperature"][0]) + eg / (R * T_av))
    cell["rho"] = (
        cell["Sfrac"] * cell["rho_ice"] + cell["Lfrac"] * cell["rho_water"]
    )
    # total annual accumulation - TODO taken direct from MATLAB, need to calc
    b = 0.4953 * (1000 / 350)
    for i in range(cell["vert_grid"]):
        if cell["rho"][i] > cell["rho_ice"]:
            cell["rho"][i] = cell["rho_ice"]
        if cell["rho"][i] < 550:
            c = 0.07
        else:
            c = 0.03
        d_rho = c * b * g * (cell["rho_ice"] - cell["rho"][i]) * e * dt
        cell["rho"][i] = cell["rho"][i] + d_rho
        cell["Sfrac"][i] = cell["rho"][i] / cell["rho_ice"]
        if cell["Sfrac"][i] > 1:
            print("Snow accumulation has caused Sfrac to be > 1")
