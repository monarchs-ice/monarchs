import numpy as np
from monarchs.physics import surface_fluxes, solver
from monarchs.core import utils


def lid_development(cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind):
    """
    Once a permanent lid forms, it can refreeze the lake below. This function
    calculates this refreezing, as well as the surface energy balance and heat
    transfer through the lid, and adjusts the lid depth and lake depth
    accordingly. Once the lid refreezes the entire lake below, then we need to
    combine the frozen lid and the firn to make one singular profile.
    The model only enters this state once the lid depth exceeds 0.1 m. Before
    this, the lid is evolved using virtual_lid.

    Called in timestep_loop.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    dt : int
        Number of seconds in the timestep, very like 3600 (i.e. 1h) [s]
    LW_in : float
        Incoming longwave radiation. [W m^-2].
    SW_in : float
        Incoming shortwave (solar) radiation. [W m^-2].
    T_air : float
        Surface-layer air temperature. [K].
    p_air : float
        Surface-layer air pressure. [hPa].
    T_dp : float
        Dew-point temperature at the surface. [K]
    wind : float
        Wind speed. [m s^-1].
    Returns
    -------
    None (amends cell inplace)
    """

    original_mass = utils.calc_mass_sum(cell)

    # Calculate the surface flux, used later for the surface energy balance
    x = cell["lid_temperature"]
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
        x[0],
    )
    # If this is the first time we have a lid for the current state,
    # then initialise its temperature profile
    initialise_lid(cell, Q)
    # Assume no water present in lid or snow above it
    cp_ice, k_ice = calc_k_and_cp(cell)
    Sfrac_lid = cell["rho_lid"] / cell["rho_ice"]
    # needs to be an array for Numba compatibility
    k_lid_seb = Sfrac_lid[0] * k_ice[0] + (1 - Sfrac_lid[0]) * cell["k_air"]

    # Determine if the lid is melting or freezing at the surface. We adjust
    # cell["lid_melt_count"] accordingly. If this is above a certain threshold,
    # then we have too much melt at the surface, and the lid and firn are later
    # combined into one profile (see `combine_lid_firn` and `timestep.py`).
    if cell["lid_temperature"][0] < 273.15:  # frozen
        # Decrement lid_melt_count if the lid is freezing.
        k_lid = surface_freezing(cell)
    else:  # melting
        # if lid_temperature_sfc >= 273.15 i.e. melting.
        # Melting on the lid is not taken into account here. However the level
        # of melting that would take place is monitored to check the
        # assumption that this is not important. We iterate lid_melt_count
        # here to track how many timesteps the lid has melted.
        k_lid = surface_melt(cell, dt, Q)

    # Solve surface energy balance for the lid.
    args = np.array(
        [
            Q,
            k_lid_seb,
            cell["lid_depth"],
            cell["vert_grid_lid"],
            cell["lid_temperature"][0],
        ]
    )
    x = np.array([float(T_air)])
    cell["lid_temperature"][0] = solver.lid_seb_solver(x, args)[0][0]
    cell["lid_temperature"] = np.clip(cell["lid_temperature"], 0, 273.15)

    # Adjust lid and lake heights. The lid can only ever grow, not shrink.
    # This is because a) basal melt is not considered, and b) any surface
    # melt complicates the system greatly, as we end up with lakes on top
    # of lids on top of lakes. This is an area for future development.
    # As mentioned above, if we have too much surface melt, then we reset
    # the profile to a firn column only.
    adjust_lid_height(cell, dt)
    new_mass = utils.calc_mass_sum(cell)
    assert abs(new_mass - original_mass) < 1.5 * 10**-7

    x = cell["lid_temperature"]
    dz = cell["lid_depth"] / cell["vert_grid_lid"]
    args = (
        cell,
        dt,
        dz,
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
        Sfrac_lid,
        k_lid,
    )

    # Solve the heat equation for the lid
    # Force lake-lid boundary temperature to 273.15 before and after.
    cell["lid_temperature"][-1] = 273.15
    cell["lid_temperature"] = solver.lid_heateqn_solver(x, args)[0]
    cell["lid_temperature"] = np.clip(cell["lid_temperature"], 0, 273.15)
    cell["lid_temperature"][-1] = 273.15

    # If the lid has shrunk to < 10cm, then revert it back to a virtual lid.
    if cell["lid_depth"] < 0.1:
        cell["v_lid_depth"] = cell["lid_depth"]
        cell["lid_depth"] = 0
        cell["lid"] = False
        cell["v_lid"] = True
        print("Reverting true lid back to a virtual lid")
    new_mass = utils.calc_mass_sum(cell)
    assert abs(new_mass - original_mass) < 1.5 * 10**-7


def surface_freezing(cell):
    """
    Determine the thermal conductivity of the lid when the surface temperature
    is below freezing. If the model is in this state, then we decrement
    lid_melt_count, so that we do not call combine_lid_firn prematurely
    (e.g. if there are a few timesteps of melt followed by many of freezing).

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    Returns
    -------
    k_lid - float - thermal conductivity of the lid [W m^-1 K^-1]
    """
    k_lid = 1000 * (
        2.24 * 10**-3
        + 5.975 * 10**-6 * (273.15 - cell["lid_temperature"][0]) ** 1.156
    )
    if cell["lid_melt_count"] > 0:
        print(
            "Decrementing lid melt count as lid is frozen, count = ",
            cell["lid_melt_count"],
        )
        # decrement the melt count if the lid is frozen
        cell["lid_melt_count"] -= 1
    # ensure it doesn't go below 0
    if cell["lid_melt_count"] < 0:
        cell["lid_melt_count"] = 0
    return k_lid


def surface_melt(cell, dt, Q):
    """
    Determine the amount of melting that occurs at the surface of the frozen
    lid when the surface temperature is above freezing.
    Also track the amount of times that the lid has melted in the variable
    lid_melt_count.

    Called in lid_development.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    dt : int
        Number of seconds in the timestep, very likely 3600 (i.e. 1h) [s]
    Q : float
        Surface energy flux, taken from surface_fluxes.sfc_flux. [W m^-2]
    Returns
    -------
    k_lid - float - thermal conductivity of the lid [W m^-1 K^-1]
    """
    original_mass = utils.calc_mass_sum(cell)
    # Switch to determine if this is being run as part of the initial lid
    # formation - if so then don't iterate the melt (but the rest is the same)
    cell["lid_melt_count"] += 1
    k_lid = 1000 * (
        1.017 * 10**-4 + 1.695 * 10**-6 * cell["lid_temperature"][0]
    )
    kdTdz = (
        (cell["lid_temperature"][0] - 273.15)
        * abs(k_lid)
        / (
            cell["lid_depth"] / (cell["vert_grid_lid"] / 2)
            + cell["lid_sfc_melt"]
        )
    )
    cell["lid_sfc_melt"] += (
        (Q - kdTdz) / (cell["L_ice"] * cell["rho_ice"])
    ) * dt
    cell["lid_temperature"][0] = 273.15

    # TODO - Currently we just *track* the amount of lid surface melt.
    # TODO - But in reality, we don't actually do anything with this water,
    # TODO - and the lid doesn't actually decrease in size.
    # TODO - what is to be done about this? It could get v messy if it can
    # TODO - melt/refreeze infinitely.
    # TODO - track it for now, but don't add it as melt.
    # TODO - This is long-term work for climate modelling.

    new_mass = utils.calc_mass_sum(cell)
    assert abs(new_mass - original_mass) < 1.5 * 10**-7
    return k_lid


def calc_k_and_cp(cell):
    """
    Calculate the thermal conductivity and specific heat capacity of the
     ice lid based on its temperature profile.
    Parameters
    ----------
    cell

    Returns
    -------

    """
    # initialise some arrays
    cp_ice = np.zeros(cell["vert_grid_lid"])
    k_ice = np.zeros(cell["vert_grid_lid"])
    for i in np.arange(0, cell["vert_grid_lid"]):
        if cell["lid_temperature"][i] > 273.15:
            cp_ice[i] = 4186.8
            k_ice[i] = 1000 * (
                1.017 * 10**-4 + 1.695 * 10**-6 * cell["lid_temperature"][i]
            )
        else:
            # Alexiades & Solomon pg. 8
            cp_ice[i] = 1000 * (
                7.16 * 10**-3 * cell["lid_temperature"][i] + 0.138
            )
            k_ice[i] = 1000 * (
                2.24 * 10**-3
                + 5.975
                * 10**-6
                * (273.15 - cell["lid_temperature"][i]) ** 1.156
            )
    return k_ice, cp_ice


def adjust_lid_height(cell, dt):
    """
    Adjust the lid and lake heights based on the temperature gradient at the
    lake-lid boundary. Called in lid_development.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    dt : int
        Number of seconds in the timestep, very likely 3600 (i.e. 1h) [s]

    Returns
    -------
    None (amends cell inplace)
    """
    # Adjust lake and lid heights. We "measure" from the surface of the lid
    # down to the centre of the lake.

    # MATLAB code version - calculating gradient from entire lid
    kdTdz = (
        (
            cell["lake_temperature"][int(cell["vert_grid_lake"] / 2)]
            - cell["lid_temperature"][0]
        )
        * abs(cell["k_water"])
        / (
            cell["lake_depth"] / (cell["vert_grid_lake"] / 2)
            + cell["lid_depth"]
        )
    )
    # Other version - calculating the gradient from the local cells
    # kdTdz = -(
    #     (-cell["lake_temperature"][2] + cell["lid_temperature"][-2])
    #     * abs(cell["k_water"])
    #     / (
    #         cell["lake_depth"] / cell["vert_grid_lake"]
    #         + cell["lid_depth"] / cell["vert_grid_lid"]
    #     )
    # )
    # coordinate system defined going downward. dT/dz is therefore positive.
    # -kdTdz is the heat flux, positive downwards. So if kdTdz is positive,
    # then the lake is losing heat, leading to freezing (as -kdTdz is therefore
    # negative, i.e. going upward). So we don't have a - sign as we do in the
    # lake case, as we are freezing rather than melting - the boundary shifts
    # down in response to a positive temperature gradient, whereas for a lake
    # above firn a negative temperature gradient leads to melting and a
    # downward shift of the boundary.
    # Flux term - energy going from the lake into the lid - leads to freezing
    new_boundary_change = kdTdz / (cell["L_ice"] * cell["rho_ice"]) * dt
    if abs(new_boundary_change) > 0:
        # if the lake would freeze completely
        if cell["lake_depth"] - new_boundary_change < 0:
            new_boundary_change = cell["lake_depth"] * (
                cell["rho_water"] / cell["rho_ice"]
            )
        # boundary change is therefore +ve if lid is growing
        # (boundary goes deeper into column)
        cell["lake_depth"] -= (
            new_boundary_change * cell["rho_ice"] / cell["rho_water"]
        )
        cell["lid_depth"] += new_boundary_change
    cell["lid_boundary_change"] += new_boundary_change


def initialise_lid(cell, Q):
    """
    Check to determine if a lid has been formed previously - if not, then
    initialise the lid by calculating its surface energy balance

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    Q : float
        Surface energy flux, taken from surface_fluxes.sfc_flux. [W m^-2]

    Returns
    -------
    None (amends cell inplace)
    """

    if not cell["has_had_lid"]:
        cell["has_had_lid"] = True
        k_lid = 1000 * (
            2.24 * 10**-3
            + 5.975 * 10**-6 * (273.15 - cell["lid_temperature"][0]) ** 1.156
        )
        # Surface energy balance for the lid
        x = np.array([cell["lid_temperature"][0]])
        args = np.array(
            [Q, k_lid, cell["lid_depth"], cell["vert_grid_lid"], 273.15]
        )
        # need the [0][0] as we want the first
        # element of the output array for initialisation
        cell["lid_temperature"][0] = solver.lid_seb_solver(x, args)[0][0]
        # assume it is linear from surface to bottom of lid at 0C
        cell["lid_temperature"] = np.linspace(
            cell["lid_temperature"][0], 273.15, cell["vert_grid_lid"]
        )


def interpolate_profiles(cell, new_depth_grid, old_depth_grid):
    """
    Interpolate model variables when combining the lid and firn profiles.
    The variables being regridded have the lid and firn profiles concatenated
    within combine_lid_firn prior to this function call.
    This could also be reused in other places to perform the same job when
    melting or freezing takes place.

    Called in combine_lid_firn.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on..
    new_depth_grid : array_like, float, dimension(cell.vert_grid)
        New vertical profile to be interpolated to. This spans the bottom of
        the firn to the new top extent,
        i.e. the old firn depth + the old lid depth.
    old_depth_grid : array_like, float,
            dimension(cell.vert_grid + cell.vert_grid_lid)
        Vertical profile of the combined lid and firn before regridding takes
        place. This maps the relevant variables for the old lid profile and old
        firn profile, and concatenates one on top of the other.

    Returns
    -------
    None (amends cell inplace)
    """
    cell["firn_temperature"] = np.interp(
        new_depth_grid, old_depth_grid, cell["firn_temperature"]
    )
    cell["rho"] = np.interp(new_depth_grid, old_depth_grid, cell["rho"])
    cell["Lfrac"] = np.interp(new_depth_grid, old_depth_grid, cell["Lfrac"])
    cell["Sfrac"] = np.interp(new_depth_grid, old_depth_grid, cell["Sfrac"])
    new_saturation = np.zeros(cell["vert_grid"])
    np.round(
        np.interp(new_depth_grid, old_depth_grid, cell["saturation"]),
        0,
        new_saturation,
    )
    cell["saturation"] = new_saturation
    new_meltflag = np.zeros(cell["vert_grid"])
    np.round(
        np.interp(new_depth_grid, old_depth_grid, cell["meltflag"]),
        0,
        new_meltflag,
    )
    cell["meltflag"] = new_meltflag
