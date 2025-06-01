import numpy as np
from monarchs.physics.surface_fluxes import sfc_flux
from monarchs.physics import solver
from monarchs.core.utils import calc_mass_sum


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
    cell : core.iceshelf_class.IceShelf
        Instance of the IceShelf that we are tracking the virtual lid for.
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
    original_mass = calc_mass_sum(cell)
    x = np.array([cell["virtual_lid_temperature"]])

    Q = sfc_flux(
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
    args = np.array(
        [Q, k_v_lid, cell["lake_depth"], cell["vert_grid_lake"], cell["v_lid_depth"]]
    )
    args = np.append(args, cell["lake_temperature"])

    # want only root, not fvec etc, and root is an array of one element so extract out first elem
    cell["virtual_lid_temperature"] = solver.lid_seb_solver(x, args, v_lid=True)[0][0]
    #print('Virtual lid temperature after seb = ', cell["virtual_lid_temperature"])
    kdTdz = (
        (-cell["lake_temperature"][1] + cell["virtual_lid_temperature"])
        * abs(cell["k_water"])
        / (cell["lake_depth"] / (cell["vert_grid_lake"] / 2) + cell["v_lid_depth"])
    )

    new_boundary_change = -kdTdz / (cell["L_ice"] * cell["rho_ice"]) * dt
    # print('v_lid_temperature = ', cell["virtual_lid_temperature"])
    # print('New boundary change = ', new_boundary_change)
    # print('Lake depth = ', cell["lake_depth"])
    # print('Virtual lid depth = ', cell["v_lid_depth"])
    if cell["virtual_lid_temperature"] < 273.15:
        if new_boundary_change > 0:
            if new_boundary_change < cell["lake_depth"]:
                old_depth_grid = np.linspace(
                    0, cell["lake_depth"], cell["vert_grid_lake"]
                )
                cell["lake_depth"] -= new_boundary_change * (
                    cell["rho_ice"] / cell["rho_water"]
                )
                new_depth_grid = np.linspace(
                    0, cell["lake_depth"], cell["vert_grid_lake"]
                )
                cell["lake_temperature"] = np.interp(
                    new_depth_grid, old_depth_grid, cell["lake_temperature"]
                )
                cell["v_lid_depth"] += new_boundary_change
            else:
                cell["v_lid_depth"] += cell["lake_depth"] * (
                    cell["rho_water"] / cell["rho_ice"]
                )
                orig_lake_depth = cell["lake_depth"] + 0
                cell["lake_depth"] = 0
                cell["lake"] = False
    else:
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
            if new_boundary_change > cell["v_lid_depth"]:
                cell["lake_depth"] += (
                    cell["v_lid_depth"] * cell["rho_ice"] / cell["rho_water"]
                )
                cell["v_lid_depth"] = 0
                cell["total_melt"] = cell["total_melt"] + cell["v_lid_depth"]
            else:
                cell["lake_depth"] = (
                    cell["lake_depth"]
                    + new_boundary_change * cell["rho_ice"] / cell["rho_water"]
                )
                cell["v_lid_depth"] = cell["v_lid_depth"] - new_boundary_change
                cell["total_melt"] = cell["total_melt"] + new_boundary_change

        if cell["v_lid_depth"] <= 0:
            cell["v_lid"] = True
            cell["v_lid_depth"] = 0
    if cell["v_lid_depth"] > 0.1 or cell["lake_depth"] == 0:
        cell["lid"] = True
        cell["v_lid"] = False
        cell["lid_depth"] = cell["v_lid_depth"]
        cell["v_lid_depth"] = 0
    if cell["v_lid_depth"] <= 0 and cell["lid_depth"] <= 0:
        cell["lid"] = False
        cell["v_lid"] = False
    new_mass = calc_mass_sum(cell)
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
    cell : core.iceshelf_class.IceShelf
        Instance of the IceShelf that we are calculating the lid surface melt of.
    dt : int
        Number of seconds in the timestep, very like 3600 (i.e. 1h) [s]
    Q : float
        Surface energy flux, taken from surface_fluxes.sfc_flux. [W m^-2]
    Returns
    -------
    None (amends cell inplace)
    """
    original_mass = calc_mass_sum(cell)
    cell["lid_melt_count"] += 1
    k_lid = 1000 * (1.017 * 10**-4 + 1.695 * 10**-6 * cell["lid_temperature"][0])
    kdTdz = (
        (cell["lid_temperature"][0] - 273.15)
        * abs(k_lid)
        / (cell["lid_depth"] / (cell["vert_grid_lid"] / 2) + cell["lid_sfc_melt"])
    )
    cell["lid_sfc_melt"] += (Q - kdTdz) / (cell["L_ice"] * cell["rho_ice"]) * dt
    cell["lid_temperature"][0] = 273.15
    new_mass = calc_mass_sum(cell)
    assert abs(new_mass - original_mass) < 1.5 * 10**-7


def lid_development(cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind):
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
    cell : core.iceshelf_class.IceShelf
        Instance of the IceShelf that we are evolving the lid for.
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
    original_mass = calc_mass_sum(cell)
    x = cell["lid_temperature"]
    Q = sfc_flux(
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
    if not cell["has_had_lid"]:
        cell["has_had_lid"] = True
        if cell["lid_temperature"][-2] == 0:
            calc_surface_melt(cell, dt, Q)
            cell["lid_melt_count"] -= 1
        else:
            k_lid = 1000 * (
                2.24 * 10**-3
                + 5.975 * 10**-6 * (273.15 - cell["lid_temperature"][0]) ** 1.156
            )
            x = np.array([cell["lid_temperature"][0]])
            args = np.array(
                [Q, k_lid, cell["lid_depth"], cell["vert_grid_lid"], 273.15]
            )
            cell["lid_temperature"][0] = solver.lid_seb_solver(x, args)[0][0]
        cell["lid_temperature"] = np.linspace(
            cell["lid_temperature"][0], 273.15, cell["vert_grid_lid"]
        )
    cp_ice = np.zeros(cell["vert_grid_lid"])
    k_ice = np.zeros(cell["vert_grid_lid"])
    for i in np.arange(0, cell["vert_grid_lid"]):
        if cell["lid_temperature"][i] > 273.15:
            cp_ice[i] = 4186.8
            k_ice[i] = 1000 * (
                1.017 * 10**-4 + 1.695 * 10**-6 * cell["lid_temperature"][i]
            )
        else:
            cp_ice[i] = 1000 * (7.16 * 10**-3 * cell["lid_temperature"][i] + 0.138)
            k_ice[i] = k_ice[i] = 1000 * (
                2.24 * 10**-3
                + 5.975 * 10**-6 * (273.15 - cell["lid_temperature"][i]) ** 1.156
            )
    Sfrac_lid = cell["rho_lid"] / cell["rho_ice"]
    k_lid = Sfrac_lid[0] * k_ice[0] + (1 - Sfrac_lid[0]) * cell["k_air"]
    if cell["lake_depth"] < 0:
        cell["lake_depth"] = 0
    if cell["lid_temperature"][0] < 273.15:
        k_lid = 1000 * (
            2.24 * 10**-3
            + 5.975 * 10**-6 * (273.15 - cell["lid_temperature"][0]) ** 1.156
        )
    else:
        calc_surface_melt(cell, dt, Q)
    for i in range(len(cell["lid_temperature"])):
        if cell["lid_temperature"][i] > 273.15:
            cell["lid_temperature"][i] = 273.15
    args = np.array(
        [Q, k_lid, cell["lid_depth"], cell["vert_grid_lid"], cell["lid_temperature"][0]]
    )
    x = np.array([float(T_air)])

    cell["lid_temperature"][0] = solver.lid_seb_solver(x, args)[0][0]

    if cell["lid_temperature"][0] > 273.15:
        calc_surface_melt(cell, dt, Q)

    cell["lid_temperature"] = np.clip(cell["lid_temperature"], 0, 273.15)
    kdTdz = (
        (-cell["lake_temperature"][2] + cell["lid_temperature"][-2])
        * abs(cell["k_water"])
        / (
            cell["lake_depth"] / cell["vert_grid_lake"]
            + cell["lid_depth"] / cell["vert_grid_lid"]
        )
    )
    new_boundary_change = -kdTdz / (cell["L_ice"] * cell["rho_ice"]) * dt
    if abs(new_boundary_change) > 0:
        if cell["lake_depth"] - new_boundary_change < 0:
            new_boundary_change = cell["lake_depth"] * (
                cell["rho_water"] / cell["rho_ice"]
            )
        cell["lake_depth"] -= new_boundary_change * cell["rho_ice"] / cell["rho_water"]
        cell["lid_depth"] += new_boundary_change
    new_mass = calc_mass_sum(cell)
    assert abs(new_mass - original_mass) < 1.5 * 10**-7
    x = cell["lid_temperature"]
    dz = cell["lid_depth"] / cell["vert_grid_lid"]
    args = (cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind, Sfrac_lid, k_lid)
    cell["lid_temperature"][-1] = 273.15
    cell["lid_temperature"] = solver.lid_heateqn_solver(x, args)[0]
    cell["lid_temperature"] = np.clip(cell["lid_temperature"], 0, 273.15)
    if cell["lid_depth"] < 0.1:
        cell["v_lid_depth"] = cell["lid_depth"]
        cell["lid_depth"] = 0
        cell["lid"] = False
        cell["has_had_lid"] = False
        cell["v_lid"] = True
    new_mass = calc_mass_sum(cell)
    assert abs(new_mass - original_mass) < 1.5 * 10**-7


def interpolate_profiles(cell, new_depth_grid, old_depth_grid):
    """
    Interpolate various IceShelf variables when combining the lid and firn profiles.
    The variables being regridded have the lid and firn profiles concatenated within combine_lid_firn prior to
    this function call.
    This could also be reused in other places to perform the same job when melting or freezing takes place.

    Called in combine_lid_firn.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        Instance of IceShelf within the model grid that is to have the lid and firn profiles combined.
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
    cell : core.iceshelf_class.IceShelf
        Instance of IceShelf within the model grid that is to have the lid and firn profiles combined.

    Returns
    -------
    None (amends cell inplace)
    """
    old_sfrac = np.copy(cell["Sfrac"])
    old_lfrac = np.copy(cell["Lfrac"])
    original_mass = calc_mass_sum(cell)
    # print(
    #     f"Combining lid and firn to create one profile..., column = {cell['column']}, row = {cell['row']}"
    # )
    cell['lid_depth'] = cell['v_lid_depth'] + cell['lid_depth'] + cell['lake_depth'] * cell['rho_water'] / cell['rho_ice']

    # Create new arrays for the combined profiles
    new_vert_grid = cell["vert_grid"]
    new_depth_grid = np.linspace(
        0, cell["firn_depth"] + cell["lid_depth"], new_vert_grid
    )

    # Add lake depth to lid depth. If the lake has frozen, then this will ensure that mass is conserved.
    # We need to interpolate in such a way that we conserve mass.


    # Interpolate lid and firn properties to the new depth grid
    old_depth_grid = np.append(
        np.linspace(0, cell["lid_depth"], cell["vert_grid_lid"]),
        np.linspace(
            cell["lid_depth"], cell["firn_depth"] + cell["lid_depth"] + cell['lake_depth'], cell["vert_grid"]
        ),
    )

    new_firn_temperature = np.interp(
        new_depth_grid,
        old_depth_grid,
        np.append(cell["lid_temperature"],  cell["firn_temperature"]),
    )
    new_rho = np.interp(
        new_depth_grid, old_depth_grid, np.append(cell["rho_lid"], cell["rho"])
    )

    column_dz = cell['vertical_profile']
    lid_dz = np.linspace(0, cell['lid_depth'], cell['vert_grid_lid'])
    dz_full = np.concatenate((lid_dz, column_dz))

    sfrac_new = mass_conserving_profile(cell, var='Sfrac')
    lfrac_new = mass_conserving_profile(cell, var='Lfrac')


    new_saturation = np.round(
        np.interp(
            new_depth_grid,
            old_depth_grid,
            np.append(np.ones(cell["vert_grid_lid"]), cell["saturation"]),
        )
    )
    new_meltflag = np.round(
        np.interp(
            new_depth_grid,
            old_depth_grid,
            np.append(np.zeros(cell["vert_grid_lid"]), cell["meltflag"]),
        )
    )


    # Update cell properties with the new combined profiles
    cell["firn_temperature"] = new_firn_temperature
    cell["rho"] = new_rho
    cell["Lfrac"] = lfrac_new
    cell["Sfrac"] = sfrac_new
    cell["saturation"] = new_saturation
    cell["meltflag"] = new_meltflag
    cell["firn_depth"] += cell["lid_depth"]
    cell["vertical_profile"] = new_depth_grid

    # Reset lake/lid-related properties
    cell["lid_depth"] = 0
    cell["v_lid"] = False
    cell["lid"] = False
    cell["lake"] = False
    cell["lake_depth"] = 0.0
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

    # Validate mass conservation
    new_mass = calc_mass_sum(cell)
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
    z_edges_full = np.concatenate(([0], np.cumsum(dz_full)))
    z_centers_full = 0.5 * (z_edges_full[:-1] + z_edges_full[1:])

    # Total depth
    total_depth = np.sum(dz_full)

    num_layers_new = cell['vert_grid']
    dz_new = np.full(num_layers_new, total_depth / num_layers_new)
    z_edges_new = np.linspace(0, total_depth, num_layers_new + 1)
    z_centers_new = 0.5 * (z_edges_new[:-1] + z_edges_new[1:])

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

    mass_initial = np.sum(mass_old)
    mass_final = np.sum(var_new * dz_new * rho)


    # print(f'Solid mass from MONARCHS: {mon_mass:.3f} kg/m²')
    # print(f"Initial solid mass: {mass_initial:.3f} kg/m²")
    # print(f"Final solid mass:   {mass_final:.3f} kg/m²")
    # print(f"Difference:         {mass_final - mass_initial:.3e} kg/m²")

    return np.clip(var_new, 0, 1)

