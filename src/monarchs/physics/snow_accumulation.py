""" """

# TODO - module-level docstring, densification
import numpy as np
from monarchs.core import utils

MODULE_NAME = "monarchs.physics.snow_accumulation"

def snowfall(cell, snow_depth, snow_rho, snow_T):
    """
    Adds snowfall to surface of model and regrids model to incorporate it.
    This snow is added to the firn or lake depth, depending on the current
    state of the cell.
    Called in <timestep_loop>.


    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    snow_depth : float
        Depth of the snow, read in from the meteorological data input [m]
    snow_rho : float
        Snow density, either read in from the meteorological data input,
        or assumed 300 [kg m^-3]
    snow_T : float
        Snow temperature, either read in from meterological data input,
        or assumed 273.15 [K]

    Returns
    -------
    None.
    """
    #     TODO - what happens if we have a lid? Gets added to top and we just
    #      regrid as normal.
    # TODO - need to make sure we use actual measured snow T values
    routine_name = f"{MODULE_NAME}.snowfall"
    if snow_depth != 0:
        if cell["lid"]:
            # if we have a lid, add snow to the top of the lid
            # TODO - actually need to regrid, but just do this to test for now.
            cell["lid_depth"] += snow_depth * snow_rho / cell["rho_water"]
        elif (
            cell["lake"] and not cell["lid"]
        ):  # if lake - add it to lake depth not firn
            cell["lake_depth"] += snow_depth * snow_rho / cell["rho_water"]
        elif not cell["lid"]:
            original_mass = utils.calc_mass_sum(cell)

            # Set temperature just below freezing if it is somehow above 0
            # degrees, which it shouldn't be
            # if we don't do this, we can run into errors when solving the
            # heat equation later
            if snow_T > 273.15:
                snow_T = 273.149
            # track pre-snow dz and firn depth
            dz_old = cell["firn_depth"] / cell["vert_grid"]
            old_firn_depth = cell["firn_depth"] + 0

            # add height to the firn
            cell["firn_depth"] += snow_depth
            dz_new = cell["firn_depth"] / cell["vert_grid"]
            # create a new layer for the firn depth

            # First interpolate the existing profiles
            sfrac_hold = np.zeros(np.shape(cell["Sfrac"]))
            lfrac_hold = np.zeros(np.shape(cell["Lfrac"]))
            T_hold = np.zeros(np.shape(cell["firn_temperature"]))
            # Scale temperature in the top layer to account for the fact
            # that we have added snow
            weight1 = dz_old * (
                cell["Sfrac"][0] * cell["rho_ice"]
                + cell["Lfrac"][0] * cell["rho_water"]
            )
            weight2 = snow_depth * snow_rho
            cell["firn_temperature"][0] = (
                weight1 * cell["firn_temperature"][0] + weight2 * snow_T
            ) / (weight1 + weight2)
            # Scale Sfrac and Lfrac in the top layer also
            weight1 = snow_depth
            weight2 = dz_old

            cell["Sfrac"][0] = (
                weight1 * (snow_rho / cell["rho_ice"])
                + weight2 * cell["Sfrac"][0]
            ) / (weight1 + weight2)

            # error handling
            if np.any(cell["Lfrac"] < -0.001):
                message = "Lfrac before snow < 0"
                message += f"{np.where(cell["Lfrac"] < -0.001)}"
                message += f"{cell["Lfrac"]}"
                utils.generic_error(cell, routine_name, message)

            cell["Lfrac"][0] = (weight1 * 0 + weight2 * cell["Lfrac"][0]) / (
                weight1 + weight2
            )
            if np.any(cell["Lfrac"] < -0.001):
                message = "Lfrac before hold < 0"
                message += f"{np.where(cell["Lfrac"] < -0.001)}"
                message += f"{cell["Lfrac"]}"
                message += f"{lfrac_hold}"
                utils.generic_error(cell, routine_name, message)
            if np.any(cell["Sfrac"] < -0.001):
                message = ("Sfrac before snow > 0")
                message += f"{np.where(cell["Sfrac"] < -0.001)}"
                message += f"{cell["Sfrac"]}"
                utils.generic_error(cell, routine_name, message)
            if np.any(cell["Sfrac"] > 1.001):
                message = "Sfrac before snow > 1"
                message += f"{np.where(cell["Sfrac"] < -0.001)}"
                message += f"{cell["Sfrac"]}"
                utils.generic_error(cell, routine_name, message)


            # Calculate the new profile as a weighted average of the proportion
            # of the new cell that was made up by the old
            # cells.
            for i in range(1, len(cell["Sfrac"])):
                weight_1 = (
                    cell["firn_depth"]
                    - i * dz_new
                    - (old_firn_depth - i * dz_old)
                )  # new top of upper layer - bottom of old upper layer
                weight_2 = (
                    old_firn_depth
                    - i * dz_old
                    - (cell["firn_depth"] - (i + 1) * dz_new)
                )  # old bottom of upper layer - new bottom of upper layer
                lfrac_hold[i] = (
                    cell["Lfrac"][i - 1] * weight_1
                    + cell["Lfrac"][i] * weight_2
                ) / (weight_1 + weight_2)
                sfrac_hold[i] = (
                    cell["Sfrac"][i - 1] * weight_1
                    + cell["Sfrac"][i] * weight_2
                ) / (weight_1 + weight_2)
                T_hold[i] = (
                    cell["firn_temperature"][i - 1] * weight_1
                    + cell["firn_temperature"][i] * weight_2
                ) / (weight_1 + weight_2)
            lfrac_hold[0] = cell["Lfrac"][0]
            sfrac_hold[0] = cell["Sfrac"][0]
            T_hold[0] = cell["firn_temperature"][0]
            # further error handling
            if np.any(lfrac_hold < -0.001):
                w = np.where(lfrac_hold < -0.001)
                message = "Lfrac hold < 0"
                message += f"{w[0][0]}"
                message += ("New Lfrac = ", lfrac_hold[: w[0][0]
                + 2])
                message += ("Old Lfrac = ", cell["Lfrac"][: w[0][
                0] + 2])
                utils.generic_error(cell, routine_name, message)

            if np.any(sfrac_hold < -0.001):
                w = np.where(sfrac_hold < 0)
                message = "Sfrac hold < 0"
                message += f"{w[0][0]}"
                message += f"New Sfrac = {sfrac_hold[: w[0][0] + 2]}"
                message += f"Old Sfrac = {cell["Sfrac"][: w[0][0] + 2]}"
                utils.generic_error(cell, routine_name, message)

            if np.any(sfrac_hold > 1.001):
                message = "Sfrac hold > 1"
                message += f"{np.where(sfrac_hold > 1.001)}"
                message += f"{sfrac_hold}"
                utils.generic_error(cell, routine_name, message)

            cell["Sfrac"] = sfrac_hold
            cell["Lfrac"] = lfrac_hold
            cell["firn_temperature"] = T_hold
            new_mass = utils.calc_mass_sum(cell)
            utils.check_for_mass_conservation(cell, original_mass, new_mass,
                                                routine_name)


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
