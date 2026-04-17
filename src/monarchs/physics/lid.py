import numpy as np
from monarchs.physics import surface_fluxes, solver
from monarchs.core import utils
from monarchs.core.error_handling import check_for_mass_conservation
from monarchs.physics.constants import (
    L_ice,
    rho_ice,
    rho_water,
    emissivity,
    stefan_boltzmann,
)

MODULE_NAME = "monarchs.physics.lid"


def lid_development(
    cell, dt, met_data, Fu
):
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
    lw_in : float
        Incoming longwave radiation. [W m^-2].
    sw_in : float
        Incoming shortwave (solar) radiation. [W m^-2].
    air_temp : float
        Surface-layer air temperature. [K].
    p_air : float
        Surface-layer air pressure. [hPa].
    dew_point_temperature : float
        Dew-point temperature at the surface. [K]
    wind : float
        Wind speed. [m s^-1].
    Returns
    -------
    None (amends cell inplace)
    """
    routine_name = f"{MODULE_NAME}.lid_development"

    if np.isnan(cell["lid_depth"]):
        print("Error - start of timestep, lid depth is NaN")
        cell["error_flag"] = 1
    original_mass = utils.calc_mass_sum(cell)
    if cell["lid_temperature"][0] > 273.15:
        cell["lid_temperature"][0] = 273.15
    # clamp temperature change to avoid extreme conductivities if
    # numerical noise causes lid temp to go above 273.15 K
    temp_diff = max(0.0, 273.15 - cell["lid_temperature"][0])
    k_lid_seb = 1000 * (
        2.24 * 10 ** -3 + 5.975 * 10 ** -6 * (temp_diff) ** 1.156
    )
    # If this is the first time we have a lid for the current state,
    # then initialise its temperature profile
    if not cell["has_had_lid"]:
        initialise_lid(cell, met_data, dt, 0, k_lid_seb)

    # Assume no water present in lid or snow above it
    k_ice, _ = calc_k_and_cp(cell)
    # Determine if the lid is melting or freezing at the surface. We adjust
    # cell["lid_melt_count"] accordingly. If this is above a certain threshold,
    # then we have too much melt at the surface, and the lid and firn are later
    # combined into one profile (see `combine_lid_firn` and `timestep.py`).
    dz = cell["lid_depth"] / cell["vert_grid_lid"]

    # Update cell albedo
    cell["albedo"] = surface_fluxes.sfc_albedo(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["v_lid"],
        cell["lake_depth"],
        cell["snow_on_lid"]
    )
    # Solve the heat equation for the lid (including surface energy balance)
    # Force lake-lid boundary temperature to 273.15 before and after.
    cell["lid_temperature"][-1] = 273.15
    cell["lid_temperature"], ierr, success, info = solver.lid_heateqn_solver(
        cell, met_data, dt, dz,
    )
    cell["lid_temperature"] = np.clip(cell["lid_temperature"], 0, 273.15)
    cell["lid_temperature"][-1] = 273.15

    x = cell["lid_temperature"]
    # Update cell albedo
    cell["albedo"] = surface_fluxes.sfc_albedo(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["v_lid"],
        cell["lake_depth"],
        cell["snow_on_lid"]
    )
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
        x[0],
    )
    if cell["lid_temperature"][0] < 273.15: #and cell["lid_sfc_melt"] > 0:  # frozen
        # Decrement lid_melt_count if the lid is freezing.
        surface_freezing(cell, dt, Q)
    else:  # melting
        # if lid_temperature_sfc >= 273.15 i.e. melting.
        # Melting on the lid is not taken into account here. However the level
        # of melting that would take place is monitored to check the
        # assumption that this is not important. We iterate lid_melt_count
        # here to track how many timesteps the lid has melted.
        surface_melt(cell, dt, Q)

    # Adjust lid and lake heights. The lid can only ever grow, not shrink.
    # This is because a) basal melt is not considered, and b) any surface
    # melt complicates the system greatly, as we end up with lakes on top
    # of lids on top of lakes. This is an area for future development.
    # As mentioned above, if we have too much surface melt, then we reset
    # the profile to a firn column only.
    adjust_lid_height(cell, dt, Fu, k_ice)
    new_mass = utils.calc_mass_sum(cell)
    check_for_mass_conservation(cell, original_mass, new_mass, routine_name)
    if np.any(np.isnan(k_ice)):
        print("ERROR: monarchs.physics.lid.lid_development: k_lid = NaN")
        cell["error_flag"] = 1
    # If the lid has shrunk to < 10cm, then revert it back to a virtual lid.
    if cell["lid_depth"] < 0.1:
        cell["v_lid_depth"] = cell["lid_depth"]
        cell["lid_depth"] = 0
        cell["lid"] = False
        cell["v_lid"] = True

    new_mass = utils.calc_mass_sum(cell)
    check_for_mass_conservation(cell, original_mass, new_mass, routine_name)
    if np.isnan(cell["lid_depth"]):
        print("Error (end of timestep) - lid depth is NaN")
        cell["error_flag"] = 1


def surface_freezing(cell, dt, Q):
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
    # calculate the amount of freezing at the surface
    k_lid, _ = calc_k_and_cp(cell)
    kdTdz = (
        (cell["lid_temperature"][0] - cell["lid_temperature"][1])
        * abs(k_lid[0])
        / (
            cell["lid_depth"] / (cell["vert_grid_lid"])
        )
    )
    freezing_amount = (Q - kdTdz - (emissivity * stefan_boltzmann * cell["lid_temperature"][0] ** 4)) / (L_ice * rho_ice) * dt
    cell["lid_sfc_melt"] -= min(cell["lid_sfc_melt"], np.abs(freezing_amount))

    if cell["lid_sfc_melt"] < 0.1:
        cell["lid_melt_count"] -= 1
    if cell["lid_sfc_melt"] < 0:
        cell["lid_sfc_melt"] = 0.0
    if cell["lid_melt_count"] < 0:
        cell["lid_melt_count"] = 0


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
    routine_name = f"{MODULE_NAME}.surface_melt"
    original_mass = utils.calc_mass_sum(cell)
    k_lid, _ = calc_k_and_cp(cell)
    kdTdz = (
        (cell["lid_temperature"][0] - cell["lid_temperature"][1])
        * abs(k_lid[0])
        / (
            cell["lid_depth"] / (cell["vert_grid_lid"])
        )
    )
    lid_melt_this_timestep = (Q - kdTdz - (emissivity * stefan_boltzmann * cell["lid_temperature"][0] ** 4)) * dt / (L_ice * rho_ice)
    cell["lid_sfc_melt"] += lid_melt_this_timestep
    cell["lid_temperature"][0] = 273.15

    # TODO - Currently we just *track* the amount of lid surface melt.
    #      - But in reality, we don't actually do anything with this water,
    #      - and the lid doesn't actually decrease in size.
    #      - what is to be done about this? It could get v messy if it can
    #      - melt/refreeze infinitely.
    #      - track it for now, but don't add it as melt.
    #      - This is long-term work for climate modelling.

    new_mass = utils.calc_mass_sum(cell)
    check_for_mass_conservation(cell, original_mass, new_mass, routine_name)
    # melting lid surface removes snow, so albedo reduces
    cell["lid_snow_depth"] -= min(cell["lid_snow_depth"], lid_melt_this_timestep * (rho_ice / 350))
    if cell["snow_on_lid"] == 1:
        cell["snow_on_lid"] = 2  # now wet snow
    if cell["lid_snow_depth"] <= 0:
        cell["snow_on_lid"] = 0
    if cell["lid_sfc_melt"] > 0.02:  # 2 cm deep
        cell["lid_melt_count"] += 1  # force it above the timestep.py threshold


def calc_k_and_cp(cell):
    """
    Calculate the thermal conductivity and specific heat capacity of the
     ice lid based on its temperature profile.
    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    Returns
    -------
    None.
    """
    # initialise some arrays
    cp_ice = np.zeros(cell["vert_grid_lid"])
    k_ice = np.zeros(cell["vert_grid_lid"])
    for i in np.arange(0, cell["vert_grid_lid"]):
        if cell["lid_temperature"][i] > 273.15:
            cp_ice[i] = 4186.8
            k_ice[i] = 1000 * (
                1.017 * 10 ** -4
                + 1.695 * 10 ** -6 * cell["lid_temperature"][i]
            )
        else:
            # Alexiades & Solomon pg. 8
            cp_ice[i] = 1000 * (
                7.16 * 10 ** -3 * cell["lid_temperature"][i] + 0.138
            )
            k_ice[i] = 1000 * (
                2.24 * 10 ** -3
                + 5.975
                * 10 ** -6
                * (273.15 - cell["lid_temperature"][i]) ** 1.156
            )
    return k_ice, cp_ice


def adjust_lid_height(cell, dt, Fu, k_ice):
    """
    Adjust the lid and lake heights based on the temperature gradient at the
    lake-lid boundary. Called in lid_development.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    dt : int
        Number of seconds in the timestep, very likely 3600 (i.e. 1h) [s]
    Fu: float
        Heat flux from the lake into the lid [W m^-2]
    k_ice: array_like, float, dimension(cell.vert_grid_lid)
        Thermal conductivity of the ice lid [W m^-1 K^-1]

    Returns
    -------
    None (amends cell inplace)
    """
    # Adjust lake and lid heights. We "measure" from the surface of the lid
    # down to the centre of the lake (fluxes +ve downwards)
    dz_i = cell["lid_depth"] / cell["vert_grid_lid"]
    T_int = 273.15
    T_above = cell["lid_temperature"][-2]
    q_ice = k_ice[-1] * (T_above - T_int) / dz_i  # positive downwards
    if np.isnan(Fu):
        Fu = 0.0
    # net fluxes at interface. fluxes are positive downwards.
    # which means we want flux down into interface (positive)
    # minus (- Fu) as Fu is positive *downwards* also
    q_net = q_ice - Fu  # Fu positive downward by convention
    if np.isnan(q_ice):
        print("ERROR: monarchs.physics.lid.adjust_lid_height: Q ice = np.nan")
        cell["error_flag"] = 1
    # - sign as freezing - flux is upwards (negative) when freezing,
    # but want dh to be positive when lid grows
    dh = - (q_net / (rho_ice * L_ice)) * dt
    if dh <= 0:
        cell["lid_boundary_change"] += dh
        cell["lake_boundary_change"] -= dh
        return

    # Limit so lake doesn't go negative
    max_dh = cell["lake_depth"] * (rho_water / rho_ice)
    dh = min(dh, max_dh)
    cell["lid_depth"] += dh
    cell["lake_depth"] -= dh * (rho_ice / rho_water)
    cell["lid_boundary_change"] += dh
    cell["lake_boundary_change"] -= dh


def initialise_lid(cell, met_data, dt, dz, k_lid):
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

    cell["has_had_lid"] = True
    # Surface energy balance for the lid
    x = np.array([cell["lid_temperature"][0]])

    # need the [0][0] as we want the first
    # element of the output array for initialisation
    cell["lid_temperature"][0] = solver.lid_seb_solver(cell, met_data, dt, dz, k_lid
    )[0][0]
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
