"""
Module handling snow accumulation on the surface.
Uses conservative regridding to maintain mass balance when adding
new layers to the top of the firn column.
"""

import numpy as np
from monarchs.core import utils
from monarchs.physics.regrid_column import conservative_regrid
from monarchs.physics.percolation import calc_saturation
from monarchs.core.error_handling import generic_error
from monarchs.physics.constants import rho_ice, rho_water

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

    if snow_depth <= 0:
        return

    # handle lids and lakes (no regridding of firn)
    if cell["lid"]:
        # add to lid depth
        cell["lid_depth"] += snow_depth * snow_rho / rho_ice
        cell["lid_snow_depth"] += snow_depth  # just a tracker

        # currently this branch doesnt change anything, the flag is not used
        # but could be used to track ice lid albedo with snow on top
        if cell["lid_snow_depth"] > 0.01 and not cell["snow_on_lid"]:
            # more than 1 cm of snow on lid
            cell["snow_on_lid"] = 1
        return

    # TODO - would snow sit on top and start forming a "virtual lid"? if conditions are
    # TODO - cold enough for snow it might not melt right away..
    # elif cell['lake']:
    #     # add to virtual lid depth - will melt instantly if it is too warm to sustain
    #     # cell["lake_depth"] += snow_depth * snow_rho / rho_water
    #     cell["v_lid_depth"] += snow_depth * snow_rho / rho_ice
    #     return
    elif cell["lake"] and not cell["lid"]:
        # add to lake depth
        cell["lake_depth"] += snow_depth * snow_rho / rho_water
        # modify temperature
        return

    # dry firn/ice lens
    # calculate mass
    current_mass = utils.calc_mass_sum(cell)
    added_mass = snow_depth * snow_rho
    expected_new_mass = current_mass + added_mass

    # sanitize input temperature
    if snow_T > 273.15:
        snow_T = 273.15

    nz = int(cell["vert_grid"])
    old_depth = float(cell["firn_depth"])

    # old edges start at 0 - shift them down by 'snow_depth'.
    old_edges = np.linspace(0, old_depth, nz + 1)
    shifted_old_edges = old_edges + snow_depth
    source_edges = np.concatenate((np.array([0.0]), shifted_old_edges))

    snow_sfrac = snow_rho / rho_ice
    # ensure Sfrac doesn't exceed 1 (e.g. if input rho > 917)
    if snow_sfrac > 1.0:
        snow_sfrac = 1.0

    # combine solid fraction for both snow and old firn
    # assume fresh snow is dry
    source_Sfrac = np.concatenate((np.array([snow_sfrac]), cell["Sfrac"]))
    source_Lfrac = np.concatenate((np.array([0.0]), cell["Lfrac"]))

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

    # regridding can cause sfrac + lfrac to exceed 1 - in this case
    # run saturation calculation to fix in the first instance.
    oversaturated_indices = np.where((cell["Sfrac"] + cell["Lfrac"]) > 1.0)[0]
    if len(oversaturated_indices) > 0:
        for idx in oversaturated_indices:
            calc_saturation(cell, idx, end=True)

    # clip Sfrac and Lfrac if regridding causes them to exceed physical limits
    # and this isnt fixed by the saturation calculation
    cell["Sfrac"] = np.clip(cell["Sfrac"], 0, 1)
    # don't clip top layer as this can be saturated
    cell["Lfrac"][1:] = np.clip(cell["Lfrac"][1:], 0, 1)

    # mass checks - will fail if there is an error that isnt fixed by
    # either of the above
    final_mass = utils.calc_mass_sum(cell)
    tol = max(1e-7, 1e-10 * expected_new_mass)

    if abs(final_mass - expected_new_mass) > tol:
        message = (
            f"Mass conservation failed in snowfall.\n" "Expected: ",
            float(expected_new_mass),
            "Actual: ",
            float(final_mass),
            "Diff:  ",
            float(final_mass - expected_new_mass),
        )
        generic_error(cell, routine_name, message)


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
    cell["rho"] = cell["Sfrac"] * rho_ice + cell["Lfrac"] * rho_water
    # total annual accumulation - TODO taken direct from MATLAB, need to calc
    b = 0.4953 * (1000 / 350)
    for i in range(cell["vert_grid"]):
        if cell["rho"][i] > rho_ice:
            cell["rho"][i] = rho_ice
        if cell["rho"][i] < 550:
            c = 0.07
        else:
            c = 0.03
        d_rho = c * b * g * (rho_ice - cell["rho"][i]) * e * dt
        cell["rho"][i] = cell["rho"][i] + d_rho
        cell["Sfrac"][i] = cell["rho"][i] / rho_ice
        if cell["Sfrac"][i] > 1:
            print("Snow accumulation has caused Sfrac to be > 1")
