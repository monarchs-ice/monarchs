import numpy as np
from monarchs.physics.firn_functions import regrid_after_melt
from monarchs.physics.surface_fluxes import sfc_flux, sfc_albedo
from monarchs.physics import solver
from monarchs.core.utils import calc_mass_sum
# conditional import - use Numba version of solvers if necessary



def sfc_energy_lake(J, Q, cell):
    """
    Calculate the surface energy balance for the lake, after it has already formed.
    Called by lake_solver, which is in turn called by either lake_formation, or lake_development.

    Parameters
    ----------
    J : float
        Turbulent heat flux factor, equal to 1.907 E-5. [m s^-1 K^-(1/3)]
        See Buzzard (2017), pp. 43 for details.
    Q : float
        Surface energy flux, calculated via surface_fluxes.sfc_flux.  [W m^-2]
    cell : core.iceshelf_class.IceShelf
        IceShelf object we wish to determine the lake properties for.
    Returns
    -------
    lake_surf_temp : float
        Surface temperature of the lake. [K]
    """
    # This is solved for x to give a temp at top of lake,
    # then turbulent mixing can occur, then lake dev
    # lake well mixed - boundary and core temp so only boundaries have a diff
    # temperature.

    # lid_functions.virtual_lid and lake_functions.lake_development
    x = np.array(cell.lake_temperature[int(cell.vert_grid_lake / 2)])

    args = np.array([J, Q, cell.vert_grid_lake])
    # print(args)
    # cell.lake_temperature has more than 1 value, so need to append to end to avoid using
    # dtype = object
    args = np.append(args, cell.lake_temperature)

    lake_surf_temp = solver.lake_solver(x, args)[
        0
    ]  # output of the function is root, fvec, success,info; we just want root

    return lake_surf_temp


def sfc_energy_lake_formation(T_air, Q, k, cell):
    """
    Calculate the surface energy balance for the lake, during the formation step.
    Called by lake_solver, which is in turn called by either lake_formation, or lake_development.

    Parameters
    ----------
    T_air : float
        Surface air temperature. [K]
    Q : float
        Surface energy balance, as calculated by surface_fluxes.sfc_flx
    k : array_like, float, dimension(cell.vert_grid)
        Thermal conductivity of the firn column, as obtained by an Sfrac/Lfrac/air fraction weighted calculation
        using k_ice, k_water and k_air respectively.
        We only use the first element of this, i.e. the surface value. [W m^-1 K^-1]
    cell : core.iceshelf_class.IceShelf
        IceShelf object we wish to determine the lake properties for.

    Returns
    -------
    old_surf_temp : float
        Surface temperature of the lake. [K]
    """
    x = np.array([float(T_air)])
    # k is a 1D array - hence need the [0] else Numba doesn't like it
    # args = np.array([cell.firn_depth, float(cell.vert_grid), Q])
    # cell.firn_temperaturehas more than 1 value, so need to append to end to avoid using
    # dtype = object
    args = np.array(
        [cell.firn_depth, cell.vert_grid, Q, k[0], cell.firn_temperature[1]]
    )

    # output of the function is root, fvec, success,info; we
    # just want root
    old_surf_temp = solver.lake_solver(x, args, formation=True)[0]

    return old_surf_temp


def turbulent_mixing(cell, SW_in, dt):
    """
    The lake has a temperature profile governed by its boundary conditions - 0 degrees at the firn-lake boundary,
    and driven by the surface energy balance at the surface. The lake is turbulent, meaning there is a significant
    amount of mixing, causing the temperature profile to even out.
    Called by lake_development.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object we wish to determine the lake properties for.
    SW_in : float
        Downwelling shortwave radiation. [W m^-2]
    dt : int
        Number of seconds in the current timestep. [s]
    Returns
    -------
    None (amends cell inplace).
    """

    # TODO - this function is unfortunately rather slow. The loop was necessary to stop the solution
    # TODO - from being unstable. I am looking into ways to increase performance for this further.

    # Calculates the lake temperature profile once turbulent mixing has occurred.
    tau = 0.025
    J = 0.1 * ((9.8 * 5 * 10 ** (-5) * (1.19 * 10 ** (-7)) ** 2) / (10 ** (-6))) ** (
        1 / 3
    )

    albedo = sfc_albedo(cell.melt, cell.exposed_water, cell.lid, cell.lake, cell.lake_depth)
    # Beer's Law but (1 - alpha) taken out as already calculated with incoming SW
    Int = ((1 - albedo) * SW_in * np.exp(-tau * cell.lake_depth)) - (
        (1 - albedo) * SW_in * np.exp(-tau * 0)  # top of lake
    )
    dt_scaling = 2  # factor by which you want to scale the temporal resolution of this calculation
    # it is very slow (taking up to half of the overall model runtime when not using Numba).
    # Increasing this value up to the max value (dt) will make the model run faster, but you
    # increase the likelihood of numerical instability

    for i in range(int(dt / dt_scaling)):
        T_core = cell.lake_temperature[int(cell.vert_grid_lake / 2)]
        # Constant used in 4/3rds law

        # Flux at upper boundary
        Fu = (
            np.sign(T_core - cell.lake_temperature[0])
            * 1000
            * 4181
            * J
            * abs(T_core - cell.lake_temperature[0]) ** (4 / 3)
        )

        # Flux at lower boundary
        Fl = (
            np.sign(T_core - 273.15) * 1000 * 4181 * J * abs(T_core - 273.15) ** (4 / 3)
        )

        temp_change = (-Fl - Fu - Int) / (
            1000 * 4181 * cell.lake_depth
        )  # Temperature change in lake core

        T_core = T_core + temp_change * dt_scaling  # * dt

        # start at 1 as want layers below surface, end at len-1 to avoid lower boundary
        indices = np.arange(1, cell.vert_grid_lake - 1)
        cell.lake_temperature[indices] = T_core
        # to_fix = np.where(cell.lake_temperature < 273.15)
        # cell.lake_temperature[to_fix] = 273.15


def lake_formation(cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind):
    """
    Generate a lake, and track its evolution until we reach the point where it can evolve freely according to
    lake_development, when it goes about 10 cm deep.
    Called in timestep_loop.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object we wish to determine the lake properties for.
    dt : int
        Number of seconds in the timestep, most likely 3600 (i.e. 1 hour) [s]
    LW_in : float
        Downwelling longwave radiation at the surface. [W m^-2]
    SW_in : float
        Downwelling shortwave radiation at the surface [W m^-2]
    T_air : float
        Surface air temperature. [K]
    p_air : float
        Surface air pressure. [Pa]
    T_dp : float
        Dewpoint temperature of the air at the surface. [K]
    wind : float
        Wind speed at the surface. [m s^-1]
    Returns
    -------
    None (amends cell inplace).
    """

    original_mass = calc_mass_sum(cell)
    dz = cell.firn_depth / cell.vert_grid
    cp_ice = np.zeros(cell.vert_grid)
    k_ice = np.zeros(cell.vert_grid)
    air = np.zeros(cell.vert_grid)

    for i in np.arange(0, cell.vert_grid):
        if cell.firn_temperature[i] > 273.15:
            cp_ice[i] = 4186.8
            k_ice[i] = 1000 * (
                1.017 * 10 ** (-4) + 1.695 * 10 ** (-6) * cell.firn_temperature[i]
            )
        else:
            cp_ice[i] = 1000 * (
                7.16 * 10 ** (-3) * cell.firn_temperature[i] + 0.138
            )  # Alexiades & Solomon pg. 8
            k_ice[i] = 1000 * (
                2.24 * 10 ** (-3)
                + 5.975 * 10 ** (-6) * (273.15 - cell.firn_temperature[i]) ** 1.156
            )

        air[i] = 1 - cell.Sfrac[i] - cell.Lfrac[i]

    k = cell.Sfrac * k_ice + air * cell.k_air + cell.Lfrac * cell.k_water
    x = cell.lake_temperature
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

    old_T_sfc = sfc_energy_lake_formation(T_air, Q, k, cell)
    new_mass = calc_mass_sum(cell)
    assert abs(original_mass - new_mass) < (1.5 * 10**-7)
    if old_T_sfc >= 273.15 and Q > 0:  # melting occurring at the surface
        kdTdz = (
            (cell.firn_temperature[0] - cell.firn_temperature[1])
            * abs(k[0])
            / ((cell.firn_depth / cell.vert_grid))
        )
        # change in firn height due to melting
        dHdt = ((Q - kdTdz) / (cell.Sfrac[0] * cell.L_ice * cell.rho_ice)) * dt

        if dHdt < 0:
            # cell.log = cell.log + "Error in surface temperature in lake formation \n"
            raise ValueError("Error in surface temperature in lake formation \n")

        cell.melt_hours += 1
        regrid_after_melt(
            cell, dHdt, lake=True
        )  # we reduce the firn height and add to the lake depth here
        new_mass = calc_mass_sum(cell)
        assert abs(original_mass - new_mass) < (1.5 * 10**-7)
    # else:  # JE TODO - I think this might refreeze the lake somewhat if it hasn't fully formed yet? Otherwise the model
    #     # TODO - can get sort of stuck in a state where a lake hasn't fully formed, but is incapable of refreezing as
    #     # TODO - a virtual lid can't form.
    #     kdTdz = (
    #         (cell.firn_temperature[0] - cell.firn_temperature[1])
    #         * abs(k[0])
    #         / ((cell.firn_depth / cell.vert_grid))
    #     )
    #     dHdt = kdTdz / (cell.L_ice * cell.rho_ice) * dt
    #     orig_lake_depth = cell.lake_depth + 0
    #     orig_firn_depth = cell.firn_depth + 0
    #     cell.lake_depth -= dHdt
    #     cell.firn_depth += dHdt * (cell.rho_water/cell.rho_ice)
    #     cell.lake_depth = max(cell.lake_depth, 0)
    #     new_mass = calc_mass_sum(cell)
    #     try:
    #         assert abs(original_mass - new_mass) < (1.5 * 10**-7)
    #     except AssertionError:
    #         print('Row = ', cell.row, 'Column = ', cell.column, 'Timestep = ', cell.t_step)
    #         print(f'original_mass = {original_mass}, new_mass = {new_mass}')
    #         print('New lake depth = ', cell.lake_depth)
    #         print('Orig lake depth = ', orig_lake_depth)
    #         print('New firn depth = ', cell.firn_depth)
    #         print('Orig firn depth = ', orig_firn_depth)
    # if abs(dHdt) > 0:
    #     dz = cell.firn_depth / cell.vert_grid

    # for i in np.arange(0, cell.vert_grid):
    #     if cell.firn_temperature[i] > 273.15:
    #         cp_ice[i] = 4186.8
    #         k_ice[i] = 1000 * (
    #             1.017 * 10 ** (-4) + 1.695 * 10 ** (-6) * cell.firn_temperature[i]
    #         )
    #     else:
    #         cp_ice[i] = 1000 * (
    #             7.16 * 10 ** (-3) * cell.firn_temperature[i] + 0.138
    #         )  # Alexiades & Solomon pg. 8
    #         k_ice[i] = 1000 * (
    #             2.24 * 10 ** (-3)
    #             + 5.975 * 10 ** (-6) * (273.15 - cell.firn_temperature[i]) ** 1.156
    #         )

        air[i] = 1 - cell.Sfrac[i] - cell.Lfrac[i]

    cell.vertical_profile = np.linspace(0, cell.firn_depth, cell.vert_grid)

    x = cell.firn_temperature
    args = (cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind)

    root, fvec, success, info = solver.firn_heateqn_solver(x, args, fixed_sfc=True)
    if success:
        cell.firn_temperature = root

    if cell.lake_depth >= 0.1:
        cell.lake = True
    new_mass = calc_mass_sum(cell)
    assert abs(original_mass - new_mass) < (1.5 * 10**-7)

def lake_development(cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind):
    """
    Once a lake of at least 10 cm deep is present this function calculates
    its evolution through a Stefan problem calculation of the lake-ice boundary.
    Called by timestep_loop.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object we wish to determine the lake properties for.
    dt : int
        Number of seconds in the timestep, most likely 3600 (i.e. 1 hour) [s]
    LW_in : float
        Downwelling longwave radiation at the surface. [W m^-2]
    SW_in : float
        Downwelling shortwave radiation at the surface [W m^-2]
    T_air : float
        Surface air temperature. [K]
    p_air : float
        Surface air pressure. [Pa]
    T_dp : float
        Dewpoint temperature of the air at the surface. [K]
    wind : float
        Wind speed at the surface. [m s^-1]

    Returns
    -------

    """
    # Constant used in lake temperature calculations
    J = 0.1 * ((9.8 * 5 * 10 ** (-5) * (1.19 * 10 ** (-7)) ** 2) / (10 ** (-6))) ** (
        1 / 3
    )
    original_mass = calc_mass_sum(cell)
    if not cell.v_lid and not cell.lid:
        x = cell.lake_temperature
        # If no refrozen lid is present then the temperature of the top of the lake
        # is calculated using the surface energy balance
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

        # Calculate cell.lake_temperature and lid_temperature
        cell.lake_temperature[0] = sfc_energy_lake(
            J, Q, cell
        )
        # cell.log = cell.log +  f"lake_temperature at surface = {cell.lake_temperature[0]}\n"
        # when 10 cm - switch to a real lid - think this is done in lid_functions.virtual_lid
        if cell.lake_temperature[0] < 273.15:  # freezing will begin from top of lake
            # print("Lake freezing...")
            cell.lid_temperature[:] = cell.lake_temperature[0]
            # print("Set cell.lid_temperature to ", cell.lid_temperature)
            cell.lake_temperature[0] = 273.15
            cell.v_lid = True
            # cell.log = cell.log + "Virtual lid creation due to freezing temperatures at the surface \n"
        if cell.lake_temperature[0] > 300:
            print(f'lake_temp = {cell.lake_temperature[0]}')
            raise ValueError('Lake too warm!')
    # Calculate cp and k for firn below lake (note:orig also calculated if
    # T > 273.15 but shouldn't be possible here)
    # TODO - can we remove the "if above 273.15" clause here? If so, then easy to make a bit faster
    # by using pure Numpy operations rather than loops.
    k_ice = np.zeros(cell.vert_grid)
    for i in np.arange(0, cell.vert_grid):
        if cell.firn_temperature[i] > 273.15:
            k_ice[i] = 1000 * (
                1.017 * 10 ** (-4) + 1.695 * 10 ** (-6) * cell.firn_temperature[i]
            )

            if cell.firn_temperature[i] > 273.15000001:
                raise ValueError('Firn temperature > 273.15')
        else:
            k_ice[i] = 1000 * (
                2.24 * 10 ** (-3)
                + 5.975 * 10 ** (-6) * (273.15 - cell.firn_temperature[i]) ** 1.156
            )

    # Lake is assumed turbulent as over 10cm.
    turbulent_mixing(cell, SW_in, dt)

    air = 1 - (cell.Sfrac + cell.Lfrac)
    k = cell.Sfrac * k_ice + air * cell.k_air + cell.Lfrac * cell.k_water

    if (
        cell.lake_temperature[-2] > 273.15
    ):  # If lake is above freezing it will begin to melt the firn below it
        if cell.firn_depth > 0:  # but only if the firn isn't completely melted.
            kdTdz = (
                (273.15 - cell.lake_temperature[-2])
                * abs(k[0])
                / (2 * (cell.firn_depth / cell.vert_grid))
            )

            # change height of firn due to the melting
            boundary_change = -kdTdz / (cell.Sfrac[0] * cell.L_ice * cell.rho_ice) * dt
            # cell.log = cell.log + 'Regridding...\n'

            # Regrid the firn column to account for the change in boundary (which is
            # subtracted from the firn and added to the lake in this subroutine)
            regrid_after_melt(cell, boundary_change, lake=True)

        else:
            raise ValueError("Firn has all completely melted")


    # Record changes to lake height
    # depth_increase = boundary_change * (cell.rho_ice / cell.rho_water)

    # Increase lake depth and regrid lake to account for depth increase
    old_depth_grid = np.linspace(0, cell.lake_depth, cell.vert_grid_lake)

    if cell.lake_depth > 10:
        print("Lake depth = ", cell.lake_depth)
        print("column = ", cell.column)
        print("row = ", cell.row)

    if cell.lake_depth > 0:
        new_depth_grid = np.linspace(0, cell.lake_depth, cell.vert_grid_lake)
    else:
        print("Lake depth less than zero - problem")
        print("Lake depth = ", cell.lake_depth, "column = ", cell.column, "row = ", cell.row)
        return


    cell.lake_temperature = np.interp(
        new_depth_grid, old_depth_grid, cell.lake_temperature
    )

    # Solve the heat equation to calculate the firn temperature given the
    # new grid of firn depth.
    # TODO - Possibly add variable LFBoundary (calculates/tracks basal melt from lake)

    x = cell.firn_temperature
    dz = cell.firn_depth / cell.vert_grid

    args = (cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind)

    root, fvec, success, info = solver.firn_heateqn_solver(x, args, fixed_sfc=True)
    if success:
        cell.firn_temperature = root


    # if there is a virtual lid then the top layer needs
    # to be set to the freezing temperature as it is an
    # ice-water boundary
    if cell.v_lid:
        cell.lake_temperature[0] = 273.15

    cell.lake_temperature[-1] = 273.15  # bottom of lake is ice-lake boundary so set
    # to the freezing temperature
    new_mass = calc_mass_sum(cell)
    assert abs(original_mass - new_mass) < (1.5 * 10**-7)
