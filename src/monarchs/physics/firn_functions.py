"""
Module containing functions relating to the firn column. Some physics is contained in percolation.py.
"""

import numpy as np
from monarchs.physics import percolation_functions
from monarchs.physics import surface_fluxes
from monarchs.physics import solver
from monarchs.core import utils


def firn_column(
    cell,
    dt,
    dz,
    LW_in,
    SW_in,
    T_air,
    p_air,
    T_dp,
    wind,
    toggle_dict,
    prescribed_height_change=False,
):
    """
    Perform the various processes applied to the firn each timestep where we don't have exposed water at the surface.

    The logic works as follows:
    Solve heat equation for firn

    If surface temperature is above melting point of water
    (i,e melting takes place):

    - Determine the height change as a result of the melting
    - Regrid everything to take account for this deformation
    - Re-solve heat equation, now with fixed surface temperature
    - Calculate the amount of water added to the surface as a result of melt
    - Percolate that water down, taking into account any lids or lakes that may have formed.

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
    LW_in : float
        Downwelling longwave radiation at the current timestep. [W m^-2]
    SW_in : float
        Downwelling shortwave radiation at the current timestep. [W m^-2]
    T_air : float
        Surface air temperature at the current timestep. [K]
    p_air : float
        Surface air pressure at the current timestep. [Pa]
    T_dp : float
        Dewpoint temperature of the air at the surface at the current timestep. [K]
    wind : float
        Wind speed at the surface at the current timestep. [m s^-1]
    toggle_dict : dict
        Dictionary containing some switches that affect the running of the model.

    prescribed_height_change : float, optional
        For testing purposes, it can be useful to set a prescribed height change to force the firn to lose height
        regardless of the meteorological conditions, with the corresponding increase in water. [m]

    Returns
    -------
    None (amends cell inplace)
    """
    original_mass = utils.calc_mass_sum(cell)
    percolation_toggle = toggle_dict["percolation_toggle"]
    perc_time_toggle = toggle_dict["perc_time_toggle"]
    heateqn_solver = 'hybr'
    x = cell["firn_temperature"]
    #x = np.clip(x, 0, 273.15)
    args = [cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind]
    root, fvec, success, info = solver.firn_heateqn_solver(
        x, args, fixed_sfc=False, solver_method=heateqn_solver
    )
    # print(f'Root[0] = {root[0]}')
    root0 = root[0]
    if root[0] > 273.15:
        cell["meltflag"][0] = 1
        cell["melt"] = True
        height_change = calc_height_change(
            cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind, root[0]
        )

        if prescribed_height_change is not False:
            height_change = 0.05
        if np.isnan(height_change):
            raise ValueError("Height change is NaN")

        dz = cell["firn_depth"] / cell["vert_grid"]
        args = cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind
        root, fvec, success_fixedsfc, info = solver.firn_heateqn_solver(
            x, args, fixed_sfc=True, solver_method=heateqn_solver
        )

        if success_fixedsfc:
            cell["firn_temperature"] = root
        regrid_after_melt(cell, height_change)

    elif success:
        cell["firn_temperature"] = root
        cell["melt"] = False
    else:
        pass

    if percolation_toggle:
        percolation_functions.percolation(cell, dt, perc_time_toggle=perc_time_toggle)

    cell["rho"] = cell["Sfrac"] * cell["rho_ice"] + cell["Lfrac"] * cell["rho_water"]
    new_mass = utils.calc_mass_sum(cell)

    assert abs(original_mass - new_mass) < 1.5 * 10**-7
    return root0

def regrid_after_melt(cell, height_change, lake=False):
    """
    After melting occurs, subtract the amount of melting from the firn height, convert it into meltwater,
    and interpolate the entire column to the new vertical profile accounting for this height change.
    This meltwater is either converted into surface liquid water fraction, or if there is a lake, into lake height.


    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    height_change : float
        Change in the firn height as a result of melting. [m]
    lake : bool, optional
        Flag to determine whether a lake is present or not. This is contained here so that we can re-use the bulk
        of this algorithm, but with some small changes to reflect the different situation that occurs when a lake is
        present.

    Returns
    -------
    None
    """
    original_mass = utils.calc_mass_sum(cell)
    dz_old = cell["firn_depth"] / cell["vert_grid"]
    old_firn_depth = cell["firn_depth"] + 0
    cell["firn_depth"] -= height_change
    bs = cell["Sfrac"]
    dz_new = cell["firn_depth"] / cell["vert_grid"]
    scale = dz_old / (dz_old - height_change)
    cell["Lfrac"][0] = cell["Lfrac"][0] * scale
    meltwater = (
        height_change
        * (cell["rho_ice"] / cell["rho_water"])
        * cell["Sfrac"][0]
        / dz_new
    )
    sfrac_hold = np.zeros(np.shape(cell["Sfrac"]))
    lfrac_hold = np.zeros(np.shape(cell["Lfrac"]))
    T_hold = np.zeros(np.shape(cell["firn_temperature"]))

    if np.isnan(cell["firn_temperature"]).any():
        raise ValueError("NaN in firn temperature before regridding")

    for i in range(len(cell["Sfrac"]) - 1):
        if height_change > (i + 1) * dz_old:
            print(
                "Whole layer melted - column = ",
                cell["column"],
                "row = ",
                cell["row"],
                "layer = ",
                i,
            )
            print("height change = ", height_change, "dz_old = ", dz_old)
            meltwater += cell["Lfrac"][i] * dz_old
            weight_1 = 0
        else:
            weight_1 = (
                cell["firn_depth"] - i * dz_new - (old_firn_depth - (i + 1) * dz_old)
            )
        weight_2 = (
            old_firn_depth - (i + 1) * dz_old - (cell["firn_depth"] - (i + 1) * dz_new)
        )

        if weight_1 < 0:
            if weight_1 > -1e-09:
                weight_1 = 0
            else:
                raise ValueError(
                    "Regridding weight 1 is negative and exceeds tolerance"
                )

        if weight_2 < 0:
            if weight_2 > -1e-09:
                weight_2 = 0
            else:
                raise ValueError(
                    "Regridding weight 2 is negative and exceeds tolerance"
                )

        lfrac_hold[i] = (
            cell["Lfrac"][i] * weight_1 + cell["Lfrac"][i + 1] * weight_2
        ) / (weight_1 + weight_2)
        sfrac_hold[i] = (
            cell["Sfrac"][i] * weight_1 + cell["Sfrac"][i + 1] * weight_2
        ) / (weight_1 + weight_2)

        T_hold[i] = (
            cell["firn_temperature"][i] * weight_1
            + cell["firn_temperature"][i + 1] * weight_2
        ) / (weight_1 + weight_2)

    lfrac_hold[-1] = cell["Lfrac"][-1]
    sfrac_hold[-1] = cell["Sfrac"][-1]
    T_hold[-1] = cell["firn_temperature"][-1]
    cell["Sfrac"] = sfrac_hold
    cell["Lfrac"] = lfrac_hold
    cell["firn_temperature"] = T_hold

    if np.isnan(T_hold).any():
        print(T_hold)
        print(cell["firn_temperature"])
        print(cell["column"])
        print(cell["row"])
        raise ValueError("NaN in firn temperature after regridding")

    cell["daily_melt"] += meltwater
    cell["Lfrac"][0] += meltwater
    if lake:
        if cell["Lfrac"][0] + cell["Sfrac"][0] > 1:
            excess_water = cell["Lfrac"][0] + cell["Sfrac"][0] - 1
            cell["lake_depth"] += excess_water * (
                cell["firn_depth"] / cell["vert_grid"]
            )
            cell["Lfrac"][0] = 1 - cell["Sfrac"][0]
        assert abs(utils.calc_mass_sum(cell) - original_mass) < 1.5 * 10**-7

    if np.isnan(cell["firn_temperature"]).any():
        print(cell["firn_temperature"])
        raise ValueError("NaN in firn temperature after regridding")

    if np.any(cell["Sfrac"] > 1.00000000001):
        where = np.where(cell["Sfrac"] > 1)
        print(where[0])
        print("Old Sfrac = ", bs[where])
        print("Sfrac = ", cell["Sfrac"][where])
        print("x = ", cell["column"], "y = ", cell["row"])
        print("height change = ", height_change)
        print("dz old = ", dz_old)
        print("firn depth = ", cell["firn_depth"])
        raise ValueError("Sfrac > 1 in firn regridding")

    cell["vertical_profile"] = np.linspace(0, cell["firn_depth"], cell["vert_grid"])

    assert abs(utils.calc_mass_sum(cell) - original_mass) < 1.5 * 10**-7


def calc_height_change(cell, timestep, LW_in, SW_in, T_air, p_air, T_dp, wind, surf_T):
    """
    Determine the amount of firn height change that arises due to melting.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    timestep : float
        Number of seconds in each timestep. [s]
    LW_in : float
        Downwelling longwave radiation at the current timestep. [W m^-2]
    SW_in : float
        Downwelling shortwave radiation at the current timestep. [W m^-2]
    T_air : float
        Surface air temperature at the current timestep. [K]
    p_air : float
        Surface air pressure at the current timestep. [Pa]
    T_dp : float
        Dewpoint temperature of the air at the surface at the current timestep. [K]
    wind : float
        Wind speed at the surface at the current timestep. [m s^-1]
    surf_T : float
        Calculated surface temperature of the firn from the initial (non-fixed surface) implementation of the heat
        equation. [K]

    Returns
    -------

    """
    epsilon = 0.98
    sigma = 5.670373 * 10**-8
    dz = cell["firn_depth"] / cell["vert_grid"]
    L_fus = 334000
    if cell["firn_temperature"][0] > 273.14999999 and cell["firn_temperature"][0] < 273.151:
        cell["firn_temperature"][0] = 273.15
    if cell["firn_temperature"][1] > 273.14999999 and cell["firn_temperature"][1] < 273.151:
        cell["firn_temperature"][1] = 273.15
    k_sfc = (
        1000 * 2.24 * 10**-3
        + 5.975
        * 10**-6
        * (273.15 - (cell["firn_temperature"][0] + cell["firn_temperature"][1]) / 2)
        ** 1.156
    )
    Q = surface_fluxes.sfc_flux(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["lake_depth"],
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
        surf_T,
    )
    dHdt = (
        timestep
        * (
            Q
            - epsilon * sigma * cell["firn_temperature"][0] ** 4
            - k_sfc * ((cell["firn_temperature"][0] - cell["firn_temperature"][1]) / dz)
        )
        / (cell["rho_ice"] * (cell["Sfrac"][0] * L_fus))
    )
    if 0 > dHdt > -0.01:
        dHdt = 0
    elif dHdt < -0.01:
        raise ValueError(
            "Height change during melt is negative, and outside the bounds of a numerical error"
        )
    elif np.isnan(dHdt):
        pass
        print("...")
        raise ValueError("Height change during melt is NaN")
    return dHdt


def interp_nb(x_vals, x, y):
    """
    Wrapper function for np.interp. This function exists purely so that an alternative interpolation algorithm can be
    used throughout the code without needing to change every instance. Named interp_nb as this function also has
    Numba compatibility in its default form.

    Parameters
    ----------
    x_vals : array_like
        New coordinates that we want to interpolate our input y values onto.
    x : array_like
        Original coordinates of our y values.
    y : array_like
        Values we want to interpolate.

    Returns
    -------
    res : array_like
        values from y, interpolated onto our new grid of x_vals
    """
    res = np.interp(x_vals, x, y)
    return res
