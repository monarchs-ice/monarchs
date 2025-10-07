import numpy as np
from monarchs.physics import surface_fluxes
from monarchs.physics import solver
from monarchs.physics import percolation_functions
from monarchs.core import utils


def virtual_lid(cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind):
    """
    When a lake undergoes freezing from the top due to the surface conditions, a lid forms. However, the depth of this
    lid can oscillate significantly during the initial stages of formation, including disappearing entirely. To
    model this, we assume that the model is in a "virtual lid" state while the lid depth is less than 0.1 m.
    This function tracks the development of this virtual lid, and switches the model into a non-lid state if it
    melts entirely, or into a true lid state if the lid depth exceeds the 0.1 m depth threshold.

    This virtual lid, unlike the "true" lid, has a single value for its temperature (cell.v_lid_temperature), rather
    than a vertical profile. The function handles freezing and melting of this virtual lid from the top and bottom.

    As with many other functions, this operates on a single column.

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
    x = np.array([cell["virtual_lid_temperature"]])

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
    k_v_lid = 1000 * (
        2.24 * 10**-3
        + 5.975 * 10**-6 * (273.15 - cell["virtual_lid_temperature"]) ** 1.156
    )

    # Surface energy balance calculation to determine the virtual lid temperature
    args = np.array(
        [Q, k_v_lid, cell["lake_depth"], cell["vert_grid_lake"], cell["v_lid_depth"]]
    )
    args = np.append(args, cell["lake_temperature"])

    # want only root, not fvec etc, and root is an array of one element so extract out first elem
    cell["virtual_lid_temperature"] = solver.lid_seb_solver(x, args, v_lid=True)[0][0]
    #print('Virtual lid temperature after seb = ', cell["virtual_lid_temperature"])
    # JE TODO - are we missing Fu from here also?
    kdTdz = (
        (cell["virtual_lid_temperature"] - cell["lake_temperature"][1])
        * abs(cell["k_water"])
        / (cell["lake_depth"] / (cell["vert_grid_lake"] / 2) + cell["v_lid_depth"])
    )

    new_boundary_change = kdTdz / (cell["L_ice"] * cell["rho_ice"]) * dt

    if cell["virtual_lid_temperature"] < 273.15:  # further freezing of the virtual lid
        if new_boundary_change < 0:
            if (new_boundary_change * cell["rho_ice"] / cell["rho_water"]) < cell["lake_depth"]:
                old_depth_grid = np.linspace(
                    0, cell["lake_depth"], cell["vert_grid_lake"]
                )
                # now reduce the size of the lake due to freezing
                cell["lake_depth"] += new_boundary_change * (
                    cell["rho_ice"] / cell["rho_water"]
                )

                # new grid - same # of vertical points, but with a reduced value
                new_depth_grid = np.linspace(
                    0, cell["lake_depth"], cell["vert_grid_lake"]
                )
                cell["lake_temperature"] = np.interp(
                    new_depth_grid, old_depth_grid, cell["lake_temperature"]
                )
                cell["v_lid_depth"] -= new_boundary_change
                cell["lake_temperature"][-1] = 273.15
            else:  # whole lake freezes
                cell["v_lid_depth"] += cell["lake_depth"] * (
                    cell["rho_water"] / cell["rho_ice"]
                )
                orig_lake_depth = cell["lake_depth"] + 0
                cell["lake_depth"] = 0
                cell["lake"] = False
    else:  # melting of the virtual lid
        cell["virtual_lid_temperature"] = 273.15
        k_im_lid = 1000 * (
            1.017 * 10**-4 + 1.695 * 10**-6 * cell["virtual_lid_temperature"]
        )
        kdTdz = (
            (cell["virtual_lid_temperature"] - 273.15)
            * abs(k_im_lid)
            / (cell["lake_depth"] / (cell["vert_grid_lake"] / 2 + cell["v_lid_depth"]))
        )
        new_boundary_change = (Q - kdTdz) / (cell["L_ice"] * cell["rho_ice"]) * dt


        if new_boundary_change > 0:
            if new_boundary_change > cell["v_lid_depth"]:  # whole virtual lid melts
                cell["lake_depth"] += (
                    cell["v_lid_depth"] * cell["rho_ice"] / cell["rho_water"]
                )
                cell["v_lid_depth"] = 0
                cell["total_melt"] = cell["total_melt"] + cell["v_lid_depth"]
            else:   # some of the lid melts
                cell["lake_depth"] = (
                    cell["lake_depth"]
                    + new_boundary_change * cell["rho_ice"] / cell["rho_water"]
                )
                cell["v_lid_depth"] = cell["v_lid_depth"] - new_boundary_change
                cell["total_melt"] = cell["total_melt"] + new_boundary_change
        # check if we still have a virtual lid
        if cell["v_lid_depth"] <= 0 and cell['v_lid']:
            print('Virtual lid no longer present')
            cell["v_lid"] = False
            cell["v_lid_depth"] = 0

    # If the lid is greater than 10 cm - now have a permanent lid.
    if cell["v_lid_depth"] > 0.1 or cell["lake_depth"] == 0:
        cell["lid"] = True
        cell["v_lid"] = False
        cell["lid_depth"] = cell["v_lid_depth"]
        cell["v_lid_depth"] = 0


    if cell["v_lid_depth"] <= 0 and cell["lid_depth"] <= 0:
        cell["lid"] = False
        cell["v_lid"] = False

    # check we've not gained or lost too much mass
    new_mass = utils.calc_mass_sum(cell)
    try:
        assert abs(new_mass - original_mass) < 1.5 * 10**-7
    except Exception:
        print("Original mass = ", original_mass)
        print("New mass = ", new_mass)
        print("Difference = ", abs(new_mass - original_mass))
        raise Exception


def calc_surface_melt(cell, dt, Q):
    """
    Determine the amount of melting that occurs at the surface of the frozen lid when the surface temperature
    is above freezing. Also track the amount of times that the lid has melted in the variable lid_melt_count.

    Called in lid_development.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    dt : int
        Number of seconds in the timestep, very like 3600 (i.e. 1h) [s]
    Q : float
        Surface energy flux, taken from surface_fluxes.sfc_flux. [W m^-2]
    Returns
    -------
    None (amends cell inplace)
    """
    original_mass = utils.calc_mass_sum(cell)
    # Switch to determine if this is being run as part of the initial lid
    # formation - if so then don't iterate the melt (but the rest is the same)
    cell["lid_melt_count"] += 1
    k_lid = 1000 * (1.017 * 10**-4 + 1.695 * 10**-6 * cell["lid_temperature"][0])
    kdTdz = (
        (cell["lid_temperature"][0] - 273.15)
        * abs(k_lid)
        / (cell["lid_depth"] / (cell["vert_grid_lid"] / 2) + cell["lid_sfc_melt"])
    )
    cell["lid_sfc_melt"] += ((Q - kdTdz) / (cell["L_ice"] * cell["rho_ice"])) * dt
    cell["lid_temperature"][0] = 273.15

    # TODO - Currently we just *track* the amount of lid surface melt. But in reality, we don't actually
    # TODO - do anything with this water, and the lid doesn't actually decrease in size.
    # TODO - what is to be done about this? It could get v messy if it can melt/refreeze infinitely
    # TODO - track it for now, but don't add it as melt. This is long-term work for climate modelling.

    new_mass = utils.calc_mass_sum(cell)
    assert abs(new_mass - original_mass) < 1.5 * 10**-7


def adjust_lid_height(cell, dt):
    # Adjust lake and lid heights. We "measure" from the surface of the lid down to the centre of the lake.

    # MATLAB code version - calculating gradient from entire lid
    kdTdz = (
        (cell["lake_temperature"][int(cell["vert_grid_lake"]/2)] - cell["lid_temperature"][0])
        * abs(cell["k_water"])
        / (cell["lake_depth"] / (cell["vert_grid_lake"] / 2) + cell["lid_depth"])
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
    # coordinate system defined going downward. dT/dz is therefore positive. -kdTdz is the heat flux,
    # positive downwards. So if kdTdz is positive, then the lake is losing heat, leading to freezing
    # (as -kdTdz is therefore negative, i.e. going upward). So we don't have a - sign as we do in
    # the lake case, as we are freezing rather than melting - the boundary shifts down in response
    # to a positive temperature gradient, whereas for a lake above firn a negative temperature
    # gradient leads to melting and a downward shift of the boundary.
    # Flux term - energy going from the lake into the lid - leads to freezing
    new_boundary_change = kdTdz / (cell["L_ice"] * cell["rho_ice"]) * dt
    if abs(new_boundary_change) > 0:
        if cell["lake_depth"] - new_boundary_change < 0:  # if the lake would freeze completely
            new_boundary_change = cell["lake_depth"] * (
                cell["rho_water"] / cell["rho_ice"]
            )
        # boundary change is therefore +ve if lid is growing (boundary goes deeper into column)
        cell["lake_depth"] -= new_boundary_change * cell["rho_ice"] / cell["rho_water"]
        cell["lid_depth"] += new_boundary_change
    cell["lid_boundary_change"] += new_boundary_change

def initialise_lid(cell, Q):
    # Check to determine if a lid has been formed previously - if not, then initialise the lid
    # by calculating its surface energy balance
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


def lid_development(cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind, Fu):
    """
    Once a permanent lid forms, it can refreeze the lake below. This function calculates this refreezing, as well as the
    surface energy balance and heat transfer through the lid, and adjusts the lid depth and lake depth accordingly.
    Once the lid refreezes the entire lake below, then we need to combine the frozen lid and the firn to make one
    singular profile.
    The model only enters this state once the lid depth exceeds 0.1 m. Before this, the lid is evolved using
    virtual_lid.

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
    # initialise some arrays
    cp_ice = np.zeros(cell["vert_grid_lid"])
    k_ice = np.zeros(cell["vert_grid_lid"])
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

    initialise_lid(cell, Q)
    # Assume no water present in lid or snow above it
    for i in np.arange(0, cell["vert_grid_lid"]):
        if cell["lid_temperature"][i] > 273.15:
            cp_ice[i] = 4186.8
            k_ice[i] = 1000 * (
                1.017 * 10**-4 + 1.695 * 10**-6 * cell["lid_temperature"][i]
            )
        else:
            cp_ice[i] = 1000 * (7.16 * 10**-3 * cell["lid_temperature"][i] + 0.138)   # Alexiades & Solomon pg. 8
            k_ice[i] = 1000 * (
                2.24 * 10**-3
                + 5.975 * 10**-6 * (273.15 - cell["lid_temperature"][i]) ** 1.156
            )
    Sfrac_lid = cell["rho_lid"] / cell["rho_ice"]
    k_lid = Sfrac_lid[0] * k_ice[0] + (1 - Sfrac_lid[0]) * cell["k_air"]
    # arguments for later call to sfc_energy_lid
    if cell["lid_temperature"][0] < 273.15:  # frozen
        k_lid = 1000 * (
            2.24 * 10**-3
            + 5.975 * 10**-6 * (273.15 - cell["lid_temperature"][0]) ** 1.156
        )
        if cell["lid_melt_count"] > 0:
            print('Decrementing lid melt count as lid is frozen, count = ', cell["lid_melt_count"])
            cell["lid_melt_count"] -= 1  # decrement the melt count if the lid is frozen
        # ensure it doesn't go below 0
        if cell["lid_melt_count"] < 0:
            cell["lid_melt_count"] = 0

    else:  # melting
        # if lid_temperature_sfc >= 273.15 i.e. melting
        # Melting on the lid is not taken into account here.
        # However the level of melting that would take place
        # is monitored to check the assumption that this
        # is not important
        calc_surface_melt(cell, dt, Q)

    # Solve surface energy balance for the lid.
    args = np.array(
        [Q, k_lid, cell["lid_depth"], cell["vert_grid_lid"], cell["lid_temperature"][0]]
    )
    x = np.array([float(T_air)])
    cell["lid_temperature"][0] = solver.lid_seb_solver(x, args)[0][0]
    cell["lid_temperature"] = np.clip(cell["lid_temperature"], 0, 273.15)

    # Adjust lid and lake heights
    adjust_lid_height(cell, dt)
    new_mass = utils.calc_mass_sum(cell)
    assert abs(new_mass - original_mass) < 1.5 * 10**-7

    x = cell["lid_temperature"]
    dz = cell["lid_depth"] / cell["vert_grid_lid"]
    args = (cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind, Sfrac_lid, k_lid)

    # Solve the heat equation for the lid
    # Force lake-lid boundary temperature to 273.15 before and after.
    cell["lid_temperature"][-1] = 273.15
    cell["lid_temperature"] = solver.lid_heateqn_solver(x, args)[0]
    cell["lid_temperature"] = np.clip(cell["lid_temperature"], 0, 273.15)
    cell["lid_temperature"][-1] = 273.15

    # If the lid has shrunk to below 10cm, then revert it back to a virtual lid.
    if cell["lid_depth"] < 0.1:
        cell["v_lid_depth"] = cell["lid_depth"]
        cell["lid_depth"] = 0
        cell["lid"] = False
        cell["v_lid"] = True
        print('Reverting true lid back to a virtual lid')
    new_mass = utils.calc_mass_sum(cell)
    assert abs(new_mass - original_mass) < 1.5 * 10**-7


def interpolate_profiles(cell, new_depth_grid, old_depth_grid):
    """
    Interpolate various model variables when combining the lid and firn profiles.
    The variables being regridded have the lid and firn profiles concatenated within combine_lid_firn prior to
    this function call.
    This could also be reused in other places to perform the same job when melting or freezing takes place.

    Called in combine_lid_firn.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on..
    new_depth_grid : array_like, float, dimension(cell.vert_grid)
        New vertical profile to be interpolated to. This spans the bottom of the firn to the new top extent,
        i.e. the old firn depth + the old lid depth.
    old_depth_grid : array_like, float, dimension(cell.vert_grid + cell.vert_grid_lid)
        Vertical profile of the combined lid and firn before regridding takes place. This maps the relevant
        variables for the old lid profile and old firn profile, and concatenates one on top of the other.

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
        np.interp(new_depth_grid, old_depth_grid, cell["saturation"]), 0, new_saturation
    )
    cell["saturation"] = new_saturation
    new_meltflag = np.zeros(cell["vert_grid"])
    np.round(
        np.interp(new_depth_grid, old_depth_grid, cell["meltflag"]), 0, new_meltflag
    )
    cell["meltflag"] = new_meltflag


def combine_lid_firn(cell):
    """
    Combines the lid and firn profiles into a single column when the lake is completely frozen.
    This avoids overwriting fixed-length arrays by creating new arrays for the combined profiles.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.

    Returns
    -------
    None (amends cell inplace)
    """
    original_mass = utils.calc_mass_sum(cell)
    print(
        f"Combining lid and firn to create one profile..., column = {cell['column']}, row = {cell['row']}"
    )
    cell['lid_depth'] = cell['v_lid_depth'] + cell['lid_depth'] + cell['lake_depth'] * cell['rho_water'] / cell['rho_ice']

    # Create new arrays for the combined profiles
    new_vert_grid = cell["vert_grid"]
    new_depth_grid = np.linspace(
        0, cell["firn_depth"] + cell["lid_depth"], new_vert_grid
    )

    # If we have a lake, and we are combining the lid and lake profiles, we need to ensure that the
    # density of the lake is added to that of the lid, accounting for the depths.
    rho_lake = cell['rho_water'] * np.ones(cell['vert_grid_lake'])

    # Add lake depth to lid depth. If the lake has frozen, then this will ensure that mass is conserved.
    # We need to interpolate in such a way that we conserve mass.
    # Interpolate lid and firn properties to the new depth grid.
    # Three separate arrays - top to bottom of lid, bottom of lid to bottom of lake,
    # bottom of lake to bottom of firn.
    old_depth_grid = np.concatenate((
        np.linspace(0, cell["lid_depth"], cell["vert_grid_lid"]),
        np.linspace(cell["lid_depth"], cell["lake_depth"] + cell["lid_depth"], cell["vert_grid_lake"]),
        np.linspace(
            cell["lid_depth"] + cell["lake_depth"], cell["firn_depth"] +
                cell["lid_depth"] + cell['lake_depth'], cell["vert_grid"]
        )
    ))

    new_firn_temperature = np.interp(
        new_depth_grid,
        old_depth_grid,
        np.concatenate((cell["lid_temperature"],  cell["lake_temperature"], cell["firn_temperature"])),
    )
    new_rho = np.interp(
        new_depth_grid, old_depth_grid, np.concatenate((cell["rho_lid"],
                                                        np.ones(cell["vert_grid_lake"]) * cell["rho_water"],
                                                        cell["rho"]))
    )

    sfrac_new = mass_conserving_profile(cell, var='Sfrac')
    lfrac_new = mass_conserving_profile(cell, var='Lfrac')

    # Determine which parts of the new profile are saturated.
    percolation_functions.calc_saturation(cell, cell['vert_grid'] - 1)
    new_meltflag = np.zeros(cell['vert_grid'])  # no meltwater present in the new profile

    # Update cell properties with the new combined profiles
    cell["firn_temperature"] = new_firn_temperature
    cell["rho"] = new_rho
    cell["Lfrac"] = lfrac_new
    cell["Sfrac"] = sfrac_new
    cell["meltflag"] = new_meltflag
    cell["firn_depth"] += cell["lid_depth"]
    cell["vertical_profile"] = new_depth_grid
    # Recalculate saturation based on new Sfrac and Lfrac
    percolation_functions.calc_saturation(cell, cell['vert_grid'] - 1)
    # Reset lake/lid-related properties
    cell["lid_depth"] = 0
    cell["v_lid"] = False
    cell["lid"] = False
    cell["lake"] = False

    cell["ice_lens"] = True
    cell["ice_lens_depth"] = 0
    cell["exposed_water"] = False
    cell["v_lid_depth"] = 0
    cell["has_had_lid"] = False
    cell["water"][0] = 0
    cell["melt"] = False
    cell["reset_combine"] = True
    cell["virtual_lid_temperature"] = 273.15
    cell["lid_temperature"] = np.ones(cell["vert_grid_lid"]) * 273.15
    cell["lake_temperature"] = np.ones(cell["vert_grid_lake"]) * 273.15
    cell["lid_melt_count"] = 0
    cell["lake_depth"] = 0.0
    cell["lid_sfc_melt"] = 0.0
    print('Saturation at point 0 = ', cell['saturation'][0])
    print('Lfrac at point 0 = ', cell['Lfrac'][0])
    print('Sfrac at point 0 = ', cell['Sfrac'][0])
    print('Lake depth = ', cell['lake_depth'])
    # Validate mass conservation
    new_mass = utils.calc_mass_sum(cell)
    try:
        assert abs(new_mass - original_mass) < original_mass / 1000
    except Exception:
        print(f"new mass = {new_mass}, original mass = {original_mass}")
        raise Exception
    pass


def mass_conserving_profile(cell, var='Sfrac'):


    lid_dz = np.full(cell['vert_grid_lid'], cell['lid_depth'] / cell['vert_grid_lid'])
    if var == 'Sfrac':
        var_lid = np.ones(cell['vert_grid_lid'])
        rho = cell['rho_ice']

    else:
        var_lid = np.zeros(cell['vert_grid_lid'])
        rho = cell['rho_water']

    column_dz = np.full(cell['vert_grid'], cell['firn_depth'] / cell['vert_grid'])
    var_column = cell[var]

    # Combine into full profile
    dz_full = np.concatenate((lid_dz, column_dz))
    sfrac_full = np.concatenate((var_lid, var_column))

    # Depth edges of full profile (top at 0)
    z_edges_full = np.concatenate((np.array([0]), np.cumsum(dz_full)))

    # Total depth
    total_depth = np.sum(dz_full)

    num_layers_new = cell['vert_grid']
    dz_new = np.full(num_layers_new, total_depth / num_layers_new)
    z_edges_new = np.linspace(0, total_depth, num_layers_new + 1)

    # Solid mass per layer in old grid
    mass_old = sfrac_full * dz_full * rho

    # Create mass function to integrate
    mass_profile = np.zeros_like(z_edges_full)
    mass_profile[1:] = np.cumsum(mass_old)

    # Interpolate cumulative mass to new layer edges
    mass_interp_edges = np.interp(z_edges_new, z_edges_full, mass_profile)

    # New solid mass per layer
    mass_new = np.diff(mass_interp_edges)

    # Recover new solid fraction
    var_new = mass_new / (dz_new * rho)

    return np.clip(var_new, 0, 1)

