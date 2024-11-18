import numpy as np
from monarchs.physics.surface_fluxes import sfc_flux
from monarchs.physics import solver

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


    x = np.array(
        [cell.virtual_lid_temperature]
    )  # initial guess. Set to array as sfc_flux expects one
    Q = sfc_flux(
        cell.melt,
        cell.exposed_water,
        cell.lid,
        cell.lake,
        cell.lake_depth,
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
        + 5.975 * 10**-6 * (273.15 - cell.virtual_lid_temperature) ** 1.156
    )

    # Surface energy balance calculation to determine the virtual lid temperature
    args = np.array(
        [Q, k_v_lid, cell.lake_depth, cell.vert_grid_lake, cell.v_lid_depth]
    )
    args = np.append(args, cell.lake_temperature)

    cell.virtual_lid_temperature = solver.lid_seb_solver(x, args, v_lid=True)[0][0]
    # two zeros since output is an array and we want a single value

    kdTdz = (
        (-cell.lake_temperature[1] + cell.virtual_lid_temperature)
        * abs(cell.k_water)
        / (cell.lake_depth / (cell.vert_grid_lake / 2) + cell.v_lid_depth)
    )

    # print('kdTdz = ', kdTdz, 'Tv_lid = ', cell.virtual_lid_temperature)
    new_boundary_change = -kdTdz / (cell.L_ice * cell.rho_ice) * dt
    # print('New boundary change = ', new_boundary_change)

    if cell.virtual_lid_temperature < 273.15:  # further freezing of the virtual lid
        # cell.log = cell.log + 'Further freezing of virtual lid...\n'
        if new_boundary_change > 0:
            cell.v_lid_depth += new_boundary_change

            if new_boundary_change < cell.lake_depth:
                # print('nbc = ', new_boundary_change)
                old_depth_grid = np.linspace(0, cell.lake_depth, cell.vert_grid_lake)
                # now reduce the size of the lake due to freezing
                cell.lake_depth -= new_boundary_change * (cell.rho_ice / cell.rho_water)
                # print('lake depth = ', cell.lake_depth)

                # new grid - same # of vertical points, but with a reduced
                # value
                new_depth_grid = np.linspace(0, cell.lake_depth, cell.vert_grid_lake)
                # print('lake_temperature before freezing = ', cell.lake_temperature)
                cell.lake_temperature = np.interp(
                    new_depth_grid, old_depth_grid, cell.lake_temperature
                )
                # print('lake_temperature after freezing = ', cell.lake_temperature)
                # print('Lake depth after freezing = ', cell.lake_depth)

            else:  # whole lake freezes
                cell.lake_depth = 0
                cell.lake = False

    else:  # melting of the virtual lid
        cell.virtual_lid_temperature = 273.15
        k_im_lid = 1000 * (
            1.017 * 10**-4 + 1.695 * 10**-6 * cell.virtual_lid_temperature
        )

        kdTdz = (
            (cell.virtual_lid_temperature - 273.15)
            * abs(k_im_lid)
            / (cell.lake_depth / ((cell.vert_grid_lake / 2) + cell.v_lid_depth))
        )

        # JE - I think this is the change from v_lid_depth to now
        new_boundary_change = ((Q - kdTdz) / (cell.L_ice * cell.rho_ice)) * dt

        if new_boundary_change > 0:
            if new_boundary_change > cell.v_lid_depth:  # whole virtual lid melts
                cell.lake_depth += cell.v_lid_depth * cell.rho_ice / cell.rho_water
                cell.v_lid_depth = 0
                cell.total_melt = cell.total_melt + cell.v_lid_depth

            else:  # some of the lid melts
                cell.lake_depth = cell.lake_depth + (
                    new_boundary_change * cell.rho_ice / cell.rho_water
                )
                cell.v_lid_depth = cell.v_lid_depth - new_boundary_change

                cell.total_melt = cell.total_melt + new_boundary_change

        if cell.v_lid_depth <= 0:
            cell.v_lid = True
            cell.v_lid_depth = 0
        # end
    # end

    # If the lid is greater than 10 cm - now have a permanent lid.
    if cell.v_lid_depth > 0.1 or cell.lake_depth == 0:
        cell.lid = True
        cell.v_lid = False
        cell.lid_depth = cell.v_lid_depth
        cell.v_lid_depth = 0

    if cell.v_lid_depth <= 0 and cell.lid_depth <= 0:
        cell.lid = False
        cell.v_lid = False
        # cell.log = cell.log + 'Virtual lid has completely melted \n'


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
    # Switch to determine if this is being run as part of the initial lid
    # formation - if so then don't iterate the melt (but the rest is the same)
    cell.lid_melt_count += 1
    k_lid = 1000 * (1.017 * 10 ** (-4) + 1.695 * 10 ** (-6) * cell.lid_temperature[0])
    kdTdz = ((cell.lid_temperature[0] - 273.15) * abs(k_lid)) / (
        cell.lid_depth / (cell.vert_grid_lid / 2) + cell.lid_sfc_melt
    )

    cell.lid_sfc_melt += ((Q - kdTdz) / (cell.L_ice * cell.rho_ice)) * dt
    cell.lid_temperature[0] = 273.15
    # TODO - Currently we just *track* the amount of lid surface melt. But in reality, we don't actually
    # TODO - do anything with this water, and the lid doesn't actually decrease in size.
    # TODO - what is to be done about this? It could get v messy if it can melt/refreeze infinitely
    # TODO - track it for now, but don't add it as melt. This is long-term work for climate modelling.
    # cell.lid_depth -= cell.lid_sfc_melt * (cell.lid_depth/cell.vert_grid_lid)
    # cell.lake_depth += cell.lid_sfc_melt * (cell.lid_depth/cell.vert_grid_lid) * (cell.rho_ice/cell.rho_water)


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



    x = cell.lid_temperature
    Q = sfc_flux(
        cell.melt,
        cell.exposed_water,
        cell.lid,
        cell.lake,
        cell.lake_depth,
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
        x[0],
    )

    # Check to determine if a lid has been formed previously - if not, then initialise the lid
    if not cell.has_had_lid:
        cell.has_had_lid = True

        if cell.lid_temperature[-2] == 0:
            calc_surface_melt(cell, dt, Q)
            cell.lid_melt_count -= 1

        else:
            k_lid = 1000 * (
                2.24 * 10 ** (-3)
                + 5.975 * 10 ** (-6) * (273.15 - cell.lid_temperature[0]) ** 1.156
            )
            # Surface energy balance for the lid
            x = np.array([cell.lid_temperature[0]])
            args = np.array([Q, k_lid, cell.lid_depth, cell.vert_grid_lid, 273.15])
            cell.lid_temperature[0] = solver.lid_seb_solver(x, args)[0][
                0
            ]  # as output is an array
            # need the[0]

        cell.lid_temperature = np.linspace(
            cell.lid_temperature[0], 273.15, cell.vert_grid_lid
        )

    # Initialise variables cp_ice and k_ice
    cp_ice = np.zeros(cell.vert_grid_lid)
    k_ice = np.zeros(cell.vert_grid_lid)
    # Assume no water present in lid or snow above it
    for i in np.arange(0, cell.vert_grid_lid):
        if cell.lid_temperature[i] > 273.15:
            cp_ice[i] = 4186.8
            k_ice[i] = 1000 * (
                1.017 * 10 ** (-4) + 1.695 * 10 ** (-6) * cell.lid_temperature[i]
            )

        else:
            cp_ice[i] = 1000 * (
                7.16 * 10 ** (-3) * cell.lid_temperature[i] + 0.138
            )  # Alexiades & Solomon pg. 8
            k_ice[i] = k_ice[i] = 1000 * (
                2.24 * 10 ** (-3)
                + 5.975 * 10 ** (-6) * (273.15 - cell.lid_temperature[i]) ** 1.156
            )

    Sfrac_lid = cell.rho_lid / cell.rho_ice
    k_lid = Sfrac_lid[0] * k_ice[0] + (1 - Sfrac_lid[0]) * cell.k_air

    # arguments for later call to sfc_energy_lid
    if cell.lake_depth < 0:  # fix errors where lake depth is negative
        cell.lake_depth = 0

    if cell.lid_temperature[0] < 273.15:  # frozen
        k_lid = 1000 * (
            2.24 * 10 ** (-3)
            + 5.975 * 10 ** (-6) * (273.15 - cell.lid_temperature[0]) ** 1.156
        )

    else:  # if lid_temperature_sfc >= 273.15 i.e. melting
        calc_surface_melt(cell, dt, Q)

    for i in range(len(cell.lid_temperature)):
        if cell.lid_temperature[i] > 273.15:
            cell.lid_temperature[i] = 273.15

    # print('k_lid = ', k_lid)

    args = np.array(
        [Q, k_lid, cell.lid_depth, cell.vert_grid_lid, cell.lid_temperature[0]]
    )

    x = np.array([float(T_air)])
    cell.lid_temperature[0] = solver.lid_seb_solver(x, args)[0][0]

    # print(cell.lid_temperature[0])
    if cell.lid_temperature[0] > 273.15:
        # Melting on the lid is not taken into account here.
        # However the level of melting that would take place
        # is monitored to check the assumption that this
        # is not important
        calc_surface_melt(cell, dt, Q)

    cell.lid_temperature = np.clip(cell.lid_temperature, 0, 273.15)

    # # Adjust lake and lid heights -

    kdTdz = (
        (-cell.lake_temperature[2] + cell.lid_temperature[-2])
        * abs(cell.k_water)
        / (
            cell.lake_depth / (cell.vert_grid_lake)
            + (cell.lid_depth / cell.vert_grid_lid)
        )
    )

    new_boundary_change = (-kdTdz) / (cell.L_ice * cell.rho_ice) * dt

    if abs(new_boundary_change) > 0:
        cell.lid_depth += new_boundary_change
        cell.lake_depth -= new_boundary_change * cell.rho_ice / cell.rho_water
        if cell.lake_depth < 0:
            # cell.log = cell.log + 'Lake depth went negative with kdTdz... \n'
            cell.lake_depth = 0

    x = cell.lid_temperature
    dz = (
        cell.lid_depth / cell.vert_grid_lid
    )  # height of a single vertical cell within the lid [m]

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
    # Force boundary temperature to 273.15 - else can get error
    cell.lid_temperature[-1] = 273.15
    cell.lid_temperature = solver.lid_heateqn_solver(x, args)[0]  # 0 to just get root
    cell.lid_temperature = np.clip(cell.lid_temperature, 0, 273.15)

    if cell.lid_depth < 0.1:
        cell.v_lid_depth = cell.lid_depth
        cell.lid_depth = 0
        cell.lid = False
        cell.has_had_lid = False
        cell.v_lid = True
        # cell.log = cell.log + 'Lid melted - reverting to virtual lid \n'


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

    cell.firn_temperature = np.interp(
        new_depth_grid, old_depth_grid, cell.firn_temperature
    )
    cell.rho = np.interp(new_depth_grid, old_depth_grid, cell.rho)
    cell.Lfrac = np.interp(new_depth_grid, old_depth_grid, cell.Lfrac)
    cell.Sfrac = np.interp(new_depth_grid, old_depth_grid, cell.Sfrac)
    # meltflag, saturation next.
    new_saturation = np.zeros(cell.vert_grid)

    np.round(
        np.interp(new_depth_grid, old_depth_grid, cell.saturation),
        0,
        new_saturation,
    )
    cell.saturation = new_saturation

    new_meltflag = np.zeros(cell.vert_grid)
    np.round(
        np.interp(new_depth_grid, old_depth_grid, cell.meltflag),
        0,
        new_meltflag,
    )
    cell.meltflag = new_meltflag


def combine_lid_firn(cell):
    """
    When a lid completely freezes the underlying lake from above, we no longer have a meaningful difference between
    the frozen lid and the firn column underneath. This function triggers when the lake depth hits zero, or if
    the lake is completely frozen but the lake depth hasn't updated (i.e. temperature goes below 273.15 everywhere).
    The profiles of the lake and the lid are combined, with some assumptions about the solid fraction of the
    frozen lid, and some of the boolean flags e.g. saturation being set False. The firn depth is updated to be
    equal to the sum of the firn and lid depth, and everything is interpolated to the new vertical profile, with
    cell.vert_grid points.

    Called in timestep_loop.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        Instance of IceShelf within the model grid that is to have the lid and firn profiles combined.

    Returns
    -------
    None (amends cell inplace)
    """

    # cell.log = cell.log + 'Combining lid and firn to create one profile...'
    print(
        f"Combining lid and firn to create one profile..., x = {cell.x}, y = {cell.y}"
    )
    # if cell.lake_depth <= 0: # whole lake is frozen
    Lfrac_lid = np.zeros(cell.vert_grid_lid)
    Sfrac_lid = (
        np.ones(cell.vert_grid_lid) * 0.999
    )  # 0.999 so we don't get numerical errors

    # combine profiles of firn and refrozen lake
    cell.Lfrac = np.append(Lfrac_lid, cell.Lfrac)
    cell.rho = np.append(cell.rho_lid, cell.rho)
    cell.firn_temperature = np.append(cell.lid_temperature, cell.firn_temperature)
    cell.Sfrac = np.append(Sfrac_lid, cell.Sfrac)
    cell.lid_depth += cell.lake_depth * (cell.rho_water / cell.rho_ice)
    saturated_lid = np.ones(cell.vert_grid_lid)
    cell.saturation = np.append(saturated_lid, cell.saturation)
    lid_meltflag = np.zeros(cell.vert_grid_lid)
    cell.meltflag = np.append(lid_meltflag, cell.meltflag)

    # Now we need to interpolate this new profile on the grid (vert_grid_lake + vert_grid)
    # to a profile on just vert_grid. We need to therefore create a coordinate
    # representing depth in m from the surface, with vert_grid + vert_grid_lake
    # points in total.

    dz_lid = cell.lid_depth / cell.vert_grid_lid
    dz_firn = cell.firn_depth / cell.vert_grid
    old_depth_grid = np.append(
        dz_lid * np.arange(cell.vert_grid_lid),  #
        dz_lid * cell.vert_grid_lid + (dz_firn * np.arange(cell.vert_grid)),
    )

    # New grid - with just vert_grid points in total.
    new_depth_grid = np.linspace(old_depth_grid[0], old_depth_grid[-1], cell.vert_grid)
    interpolate_profiles(cell, new_depth_grid, old_depth_grid)

    # New total profile depth
    cell.firn_depth += cell.lid_depth
    cell.vertical_profile = np.linspace(0, cell.firn_depth, cell.vert_grid)
    cell.Sfrac = np.clip(cell.Sfrac, 0, 1)
    cell.lid_depth = 0
    cell.v_lid = False
    cell.lid = False
    cell.lake = False
    cell.lake_depth = 0
    cell.ice_lens = False
    cell.ice_lens_depth = 999
    cell.exposed_water = False
    cell.v_lid_depth = 0
    cell.has_had_lid = False
    cell.water[0] = 0
    cell.melt = False
    cell.reset_combine = True
    cell.saturation = np.zeros(cell.vert_grid)
    cell.virtual_lid_temperature = 273.15
    cell.lid_temperature = np.ones(cell.vert_grid_lid) * 273.15
    cell.lake_temperature = np.ones(cell.vert_grid_lake) * 273.15


# Conditional jitting of functions if Numba optimisation is required
