"""
Module containing functions relating to the percolation of water through the firn column.
"""

import numpy as np


def calc_solid_mass(cell):
    """Calculate the mass of the solid part of the column."""
    return np.sum(
        cell["Sfrac"] * cell["rho_ice"] * (cell["firn_depth"] / cell["vert_grid"])
    )


def calc_liquid_mass(cell):
    """Calculate the mass of the liquid part of the column."""
    return np.sum(
        cell["Lfrac"] * cell["rho_water"] * (cell["firn_depth"] / cell["vert_grid"])
    )


def percolation(cell, timestep, lateral_refreeze_flag=False, perc_time_toggle=True):
    """
    Main function to handle percolation of water within the firn column.
    This percolation is performed from the bottom up, i.e. we iterate from the end of the
    firn column (i.e. the deepest point). We find the lowest point with meltwater, then
    iterate back toward the end of the firn column to move it down. We then continue the
    search upward, repeating the same process as we go. At each point during the
    percolation step, some water is refrozen due to the firn temperature being sub-zero,
    and some water is left behind due to capillary effects.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object containing the information on the gridcell we are operating on.
    timestep : int
        Amount of time in seconds for each timestep. [s]
    lateral_refreeze_flag : bool, optional
        We can run the percolation at the end of a lateral movement step. If we do this, we don't
        want to trigger the refreezing as this has already occurred in the single-column calculations,
        and we would get more refreezing than expected.
        This flag controls this behaviour - it should be False unless being called from
        <lateral_functions.move_water>.
        Default False.
    perc_time_toggle : bool, optional
        Run with the calculation of the percolation time if True, or else it can be disabled for
        testing (in which case water can percolate freely during the whole timestep).
        Default True. Defined in model setup script.

    Returns
    -------
    None (amends Cell inplace)
    """
    for point in range(0, len(cell["firn_temperature"])):
        v_lev = len(cell["firn_temperature"]) - (point + 1)
        if cell["meltflag"][v_lev] and cell["Lfrac"][v_lev] != 0:
            time_remaining = timestep
            while time_remaining > 0:
                if not lateral_refreeze_flag:
                    calc_refreezing(cell, v_lev)
                if cell["Sfrac"][v_lev] * cell["rho_ice"] > cell["pore_closure"]:
                    cell["ice_lens"] = True
                    cell["saturation"][v_lev] = True
                    if v_lev < cell["ice_lens_depth"]:
                        cell["ice_lens_depth"] = v_lev
                    calc_saturation(cell, v_lev)
                    time_remaining = 0
                elif cell["saturation"][v_lev] == 1 or cell["ice_lens_depth"] == v_lev:
                    calc_saturation(cell, v_lev)
                    time_remaining = 0
                elif cell["Lfrac"][v_lev] > 0:
                    if perc_time_toggle:
                        p_time = perc_time(cell, v_lev)
                    else:
                        p_time = 0
                    time_remaining = time_remaining - p_time
                    if time_remaining > 0:
                        cell["meltflag"][v_lev] = 0
                        capillary_remain = capillary(cell, v_lev)
                        if capillary_remain < cell["Lfrac"][v_lev]:
                            if v_lev == cell["vert_grid"] - 2:
                                time_remaining = 0
                            cell["Lfrac"][v_lev + 1] = (
                                cell["Lfrac"][v_lev + 1]
                                + cell["Lfrac"][v_lev]
                                - capillary_remain
                            )
                            cell["Lfrac"][v_lev] = capillary_remain
                            cell["meltflag"][v_lev] = 0
                            v_lev += 1
                            cell["meltflag"][v_lev] = 1

                        else:
                            for i in np.arange(v_lev + 1)[::-1]:
                                calc_saturation(cell, i, end=True)
                            time_remaining = 0
                            break
                    else:
                        for i in np.arange(v_lev + 1)[::-1]:
                            calc_saturation(cell, i, end=True)
                else:
                    cell["meltflag"][v_lev] = 0
                    time_remaining = 0
                    for i in np.arange(v_lev + 1)[::-1]:
                        calc_saturation(cell, i, end=True)


def calc_refreezing(cell, v_lev):
    """
    Calculate refreezing of water within the firn column at a specified layer v_lev.
    This refreezing changes Sfrac into Lfrac, accounting for expansion due to the different
    densities.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object containing the information on the gridcell we are operating on.
    v_lev : int
        Vertical level at which to calculate the refreezing of the firn.

    Returns
    -------
    None (amends cell inplace)

    Raises
    ------
    ValueError
        If Lfrac is less than 1, then throw ValueError as model has entered an unphysical
        state.

    """
    T_change_max = 273.15 - cell["firn_temperature"][v_lev]
    cp = 7.16 * cell["firn_temperature"][v_lev] + 138
    excess_water = 0
    T_change_all = (
        cell["Lfrac"][v_lev]
        * cell["L_ice"]
        * cell["rho_water"]
        / (cell["rho_ice"] * cp * cell["Sfrac"][v_lev])
    )
    Vol_Rfrz_Max = (
        (1 - cell["Sfrac"][v_lev])
        * (cell["firn_depth"] / cell["vert_grid"])
        / (cell["rho_water"] / cell["rho_ice"])
    )
    if Vol_Rfrz_Max < cell["Lfrac"][v_lev] * (cell["firn_depth"] / cell["vert_grid"]):
        excess_water = (
            cell["Lfrac"][v_lev] * (cell["firn_depth"] / cell["vert_grid"])
            - Vol_Rfrz_Max
        )
        cell["Lfrac"][v_lev] = Vol_Rfrz_Max / (cell["firn_depth"] / cell["vert_grid"])
    if T_change_all >= T_change_max:
        Vol_Change = (
            cell["rho_ice"]
            * cp
            * cell["Sfrac"][v_lev]
            * T_change_max
            * (cell["firn_depth"] / cell["vert_grid"])
            / (cell["L_ice"] * cell["rho_water"])
        )
        if Vol_Change > Vol_Rfrz_Max:
            Vol_Change = Vol_Rfrz_Max
        cell["firn_temperature"][v_lev] = 273.15
        if cell["Lfrac"][v_lev] - Vol_Change < 0:
            Vol_Change = cell["Lfrac"][v_lev] * (cell["firn_depth"] / cell["vert_grid"])
            cell["Lfrac"][v_lev] = 0
        else:
            cell["Lfrac"][v_lev] = cell["Lfrac"][v_lev] - Vol_Change / (
                cell["firn_depth"] / cell["vert_grid"]
            )
        if cell["Lfrac"][v_lev] < 0:
            raise ValueError("Lfrac < 0 in saturation Sfrac > 1 calculation")
        cell["Sfrac"][v_lev] = cell["Sfrac"][v_lev] + Vol_Change * (
            cell["rho_water"] / cell["rho_ice"]
        ) / (cell["firn_depth"] / cell["vert_grid"])
    else:
        cell["Sfrac"][v_lev] = cell["Sfrac"][v_lev] + cell["Lfrac"][v_lev] * (
            cell["rho_water"] / cell["rho_ice"]
        )
        cell["firn_temperature"][v_lev] = cell["firn_temperature"][v_lev] + T_change_all
        cell["Lfrac"][v_lev] = 0
    if cell["Lfrac"][v_lev] < 0:
        raise ValueError("Lfrac < 0 in saturation Sfrac > 1 calculation")
    cell["Lfrac"][v_lev] = cell["Lfrac"][v_lev] + excess_water / (
        cell["firn_depth"] / cell["vert_grid"]
    )


def calc_saturation(cell, v_lev_in, end=False):
    """
    Determine whether the firn is saturated, calculating upwards from a vertical level
    specified by v_lev_in. If the firn is saturated, then water can no longer percolate
    downward, and we need to set the IceShelf attribute saturation to True at the
    relevant vertical level. Any water that is left over is then moved upwards to the
    next vertical layer, and the process repeats until either all water is accounted for
    (i.e. can remain in the current layer), or we reach the surface (at which point we
    have exposed water and a lake will begin to form).
    This function can also be called to ensure that water is simply moved to the correct place
    due to changes from other effects (e.g. refreezing), without determining whether it is
    saturated. The variable end determines whether this is the case.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object containing the information on the gridcell we are operating on.
    v_lev_in : int
        Vertical level from which to determine whether the cell is saturated.
    end : bool, optional
        Flag that determines whether the function is called at the end of the percolation
        step, or not. If it is (i.e. the flag is True), then instead of saying the firn is
        saturated, we say that there is meltwater present. This allows the water to percolate
        again at the next timestep, and allows us to reuse most of the same logic for this
        case where there may be water that can percolate again later.
        Default False.

    Returns
    -------
    None (amends cell inplace)

    Raises
    ------
    ValueError
        If lake depth goes negative, model is in an unphysical state so we throw an error.
    """
    v_lev = int(v_lev_in)
    Lfrac_max = 1 - cell["Sfrac"][v_lev]
    if Lfrac_max < 0:
        Lfrac_max = 0
    if cell["Lfrac"][v_lev] > Lfrac_max:
        excess_water = cell["Lfrac"][v_lev] - Lfrac_max
        cell["Lfrac"][v_lev] = Lfrac_max
        if not end:
            cell["saturation"][v_lev] = 1
        elif end:
            cell["meltflag"][v_lev] = 1
            print('Setting meltflag to 1 - current cell')

        if excess_water > 0:
            while v_lev > 0:
                cell["Lfrac"][v_lev] = cell["Lfrac"][v_lev] + excess_water
                Lfrac_max = 1 - cell["Sfrac"][v_lev]
                if cell["Lfrac"][v_lev] > Lfrac_max:
                    excess_water = cell["Lfrac"][v_lev] - Lfrac_max
                    cell["Lfrac"][v_lev] = Lfrac_max
                    if not end:
                        cell["saturation"][v_lev] = 1

                    else:
                        cell["meltflag"][v_lev] = 1
                        print('Setting meltflag to 1 - going upward')

                    v_lev = v_lev - 1
                else:
                    break
            if v_lev <= 0:
                cell["Lfrac"][0] = cell["Lfrac"][0] + excess_water
                Lfrac_max = 1 - cell["Sfrac"][0]
                if cell["Lfrac"][0] > Lfrac_max:
                    cell["exposed_water"] = True
                    cell["saturation"][0] = True
                    excess_water = cell["Lfrac"][0] - Lfrac_max
                    cell["Lfrac"][0] = Lfrac_max
                    cell["lake_depth"] += excess_water * (
                        cell["firn_depth"] / cell["vert_grid"]
                    )
                    if cell["lake"] and cell["lake_depth"] < 0:
                        raise ValueError("Lake depth is negative - problem...")
    elif cell["Lfrac"][v_lev] < Lfrac_max:
        cell["saturation"][v_lev] = 0


def perc_time(cell, v_lev):
    """
    Calculate the time it takes for water to percolate through the firn at vertical level v_lev.
    This will determine how long the model has in practice to percolate water down through
    the current vertical level. This is calculated as the height of the vertical level
    divided by the speed of the flow.
    # TODO - possibly add a reference and a DOI here.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object containing the information on the gridcell we are operating on.
    v_lev : int
        Vertical level of cell that we are looking at to determine the available percolation
        time for.

    Returns
    -------
    perc_time : float
        Amount of time that the water has left to percolate down the firn column. [s]
    """
    if cell["Lfrac"][v_lev] < 1e-10:
        p_time = 0
        return p_time
    delta = 0.001
    rho_s_star = cell["Sfrac"][v_lev] * cell["rho_ice"] / cell["rho_water"]
    perm_s = 0.077 * delta**2 * np.exp(-7.8 * rho_s_star)
    eta = 0.001787
    delta_p = cell["rho_water"] / (
        cell["firn_depth"] / cell["vert_grid"] / cell["Lfrac"][v_lev]
    )
    u = -perm_s / eta * delta_p
    p_time = -(cell["firn_depth"] / cell["vert_grid"]) / u
    return p_time


def capillary(cell, v_lev):
    """
    Calculate the amount of water left behind due to capillary effects.
    For more information, see Ligtenberg et al. (2011), The Cryosphere, doi:10.5194/tc-5-809-2011.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object containing the information on the gridcell we are operating on.
    v_lev : int
        Vertical level of cell that we are looking at to determine the capillary effect for.

    Returns
    -------
    capillary_remain : float
        Amount of water (in units of liquid fraction) that is left in the cell.
    """
    capillary_remain = 0.05 * (1 - cell["Sfrac"][v_lev])
    return capillary_remain
