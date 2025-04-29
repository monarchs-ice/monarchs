"""
Created on Tue May  9 17:04:04 2023

@author: sglsb6
"""

import numpy as np
from monarchs.core.utils import calc_mass_sum


def snowfall(cell, snow_depth, snow_rho, snow_T):
    """
    Adds snowfall to surface of model and regrids model to incorporate it. This snow is added to the firn or lake
    depth, depending on the current state of the IceShelf (cell).
    Called in <timestep_loop>.
    TODO - what happens if we have a lid? Gets added to top and we just regrid as normal.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object to add snow onto.
    snow_depth : float
        Depth of the snow, read in from the meteorological data input [m]
    snow_rho : float
        Snow density, either read in from the meteorological data input, or assumed 300 [kg m^-3]
    snow_T : float
        Snow temperature, either read in from meterological data input, or assumed 260 [K]

    Returns
    -------
    None.
    """
    if snow_depth != 0:
        if cell["lake"]:
            cell["lake_depth"] += snow_depth * snow_rho / cell["rho_water"]
        else:
            original_mass = calc_mass_sum(cell)
            orig_liquid = np.sum(
                cell["Lfrac"]
                * cell["rho_water"]
                * (cell["firn_depth"] / cell["vert_grid"])
            )
            orig_solid = np.sum(
                cell["Sfrac"]
                * cell["rho_ice"]
                * (cell["firn_depth"] / cell["vert_grid"])
            )
            old_sfrac = np.copy(cell["Sfrac"])
            old_sfrac_0 = cell["Sfrac"][0]
            if snow_T > 273.15:
                snow_T = 273.149
            dz_old = cell["firn_depth"] / cell["vert_grid"]
            old_firn_depth = cell["firn_depth"] + 0
            cell["firn_depth"] += snow_depth
            dz_new = cell["firn_depth"] / cell["vert_grid"]
            sfrac_hold = np.zeros(np.shape(cell["Sfrac"]))
            lfrac_hold = np.zeros(np.shape(cell["Lfrac"]))
            T_hold = np.zeros(np.shape(cell["firn_temperature"]))
            weight1 = dz_old * (
                cell["Sfrac"][0] * cell["rho_ice"]
                + cell["Lfrac"][0] * cell["rho_water"]
            )
            weight2 = snow_depth * snow_rho
            cell["firn_temperature"][0] = (
                weight1 * cell["firn_temperature"][0] + weight2 * snow_T
            ) / (weight1 + weight2)
            weight1 = snow_depth
            weight2 = dz_old
            cell["Sfrac"][0] = (
                weight1 * (snow_rho / cell["rho_ice"]) + weight2 * cell["Sfrac"][0]
            ) / (weight1 + weight2)
            if np.any(cell["Lfrac"] < -0.001):
                print(np.where(lfrac_hold < -0.001))
                raise ValueError("Lfrac before snow < 0")
            cell["Lfrac"][0] = (weight1 * 0 + weight2 * cell["Lfrac"][0]) / (
                weight1 + weight2
            )
            if np.any(cell["Lfrac"] < -0.001):
                print(np.where(lfrac_hold < -0.001))
                print(cell["Lfrac"])
                print(lfrac_hold)
                raise ValueError("Lfrac before hold < 0")
            if np.any(cell["Sfrac"] < -0.001):
                print(np.where(cell["Sfrac"] < -0.001))
                raise ValueError("Sfrac before snow > 0")
            if np.any(cell["Sfrac"] > 1.001):
                print(np.where(cell["Sfrac"] > 1.001))
                raise ValueError("Sfrac before snow > 1")
            for i in range(1, len(cell["Sfrac"])):
                weight_1 = (
                    cell["firn_depth"] - i * dz_new - (old_firn_depth - i * dz_old)
                )
                weight_2 = (
                    old_firn_depth
                    - i * dz_old
                    - (cell["firn_depth"] - (i + 1) * dz_new)
                )
                lfrac_hold[i] = (
                    cell["Lfrac"][i - 1] * weight_1 + cell["Lfrac"][i] * weight_2
                ) / (weight_1 + weight_2)
                sfrac_hold[i] = (
                    cell["Sfrac"][i - 1] * weight_1 + cell["Sfrac"][i] * weight_2
                ) / (weight_1 + weight_2)
                T_hold[i] = (
                    cell["firn_temperature"][i - 1] * weight_1
                    + cell["firn_temperature"][i] * weight_2
                ) / (weight_1 + weight_2)
            lfrac_hold[0] = cell["Lfrac"][0]
            sfrac_hold[0] = cell["Sfrac"][0]
            T_hold[0] = cell["firn_temperature"][0]
            if np.any(lfrac_hold < -0.001):
                w = np.where(lfrac_hold < -0.001)
                print(w[0][0])
                print("New Lfrac = ", lfrac_hold[: w[0][0] + 2])
                print("Old Lfrac = ", cell["Lfrac"][: w[0][0] + 2])
                raise ValueError("Lfrac hold > 0")
            if np.any(sfrac_hold < -0.001):
                w = np.where(sfrac_hold < 0)
                print(w[0][0])
                print("New Sfrac = ", sfrac_hold[: w[0][0] + 2])
                print("Old Sfrac = ", cell["Sfrac"][: w[0][0] + 2])
                raise ValueError("Sfrac hold > 0")
            if np.any(sfrac_hold > 1.001):
                print(np.where(sfrac_hold > 1))
                print(sfrac_hold)
                raise ValueError("Sfrac before snow > 1")
            cell["Sfrac"] = sfrac_hold
            cell["Lfrac"] = lfrac_hold
            cell["firn_temperature"] = T_hold
            assert (
                abs(calc_mass_sum(cell) - (original_mass + snow_depth * snow_rho))
                < 1.5 * 10**-7
            )


def densification(cell, t_steps_per_day):
    """
    TODO - not important now, but implement for later.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf

    t_steps_per_day : int

    Returns
    -------

    """
    dt = 1 / (8 * t_steps_per_day * 365)
    g = 9.81
    ec = 60
    eg = 42.4
    R = 8.3144598
    T_av = 264.56010894609415
    e = np.exp(-ec / (R * cell["firn_temperature"][0]) + eg / (R * T_av))
    cell["rho"] = cell["Sfrac"] * cell["rho_ice"] + cell["Lfrac"] * cell["rho_water"]
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
