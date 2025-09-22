import numpy as np
from monarchs.physics import firn_functions
from monarchs.physics import surface_fluxes
from monarchs.physics import solver
from monarchs.physics import snow_accumulation
from monarchs.physics import percolation_functions
from monarchs.core import utils


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
    cell : numpy structured array
        Element of the model grid we are operating on.
    Returns
    -------
    lake_surf_temp : float
        Surface temperature of the lake. [K]
    """

    # This is solved for x to give a temp at top of lake,
    # then turbulent mixing can occur, then lake dev
    # lake well mixed - boundary and core temp so only boundaries have a diff
    # temperature.
    x = np.array([cell["lake_temperature"][int(cell["vert_grid_lake"] / 2)]])
    args = np.array([J, Q, cell["vert_grid_lake"]])
    args = np.append(args, cell["lake_temperature"])
    lake_surf_temp = solver.lake_solver(x, args)[0][0]
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
    cell : numpy structured array
        Element of the model grid we are operating on.

    Returns
    -------
    old_surf_temp : float
        Surface temperature of the lake. [K]
    """
    x = np.array([float(T_air)])
    # k is a 1D array - hence need the [0] else Numba doesn't like it
    # args = np.array([cell.firn_depth, float(cell.vert_grid), Q])
    args = np.array(
        [cell["firn_depth"], cell["vert_grid"], Q, k[0], cell["firn_temperature"][1]]
    )
    old_surf_temp = solver.lake_solver(x, args, formation=True)[0][0]
    return old_surf_temp


def turbulent_mixing(cell, SW_in, dt):
    """
    The lake has a temperature profile governed by its boundary conditions - 0 degrees at the firn-lake boundary,
    and driven by the surface energy balance at the surface. The lake is turbulent, meaning there is a significant
    amount of mixing, causing the temperature profile to even out.
    Called by lake_development.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    SW_in : float
        Downwelling shortwave radiation. [W m^-2]
    dt : int
        Number of seconds in the current timestep. [s]
    Returns
    -------
    None (amends cell inplace).
    """

    tau = 0.025
    J = 0.1 * (9.8 * 5 * 10**-5 * (1.19 * 10**-7) ** 2 / 10**-6) ** (1 / 3)
    albedo = surface_fluxes.sfc_albedo(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["lake_depth"],
    )

    # Beer's Law but (1-alpha) taken out as already calculated with incoming SW
    Int = (1 - albedo) * SW_in * np.exp(-tau * cell["lake_depth"]) - \
          (1 - albedo) * SW_in * np.exp(-tau * 0)
    # factor by which you want to scale the temporal resolution of this calculation
    # it is very slow (taking up to half of the overall model runtime when not using Numba).
    # Increasing this value up to the max value (dt) will make the model run faster, but you
    # increase the likelihood of numerical instability
    dt_scaling = 20
    for i in range(int(dt / dt_scaling)):
        T_core = cell["lake_temperature"][int(cell["vert_grid_lake"] / 2)]
        # Flux at upper boundary
        Fu = (
            np.sign(T_core - cell["lake_temperature"][0])
            * 1000
            * 4181
            * J
            * abs(T_core - cell["lake_temperature"][0]) ** (4 / 3)
        )
        # Flux at lower boundary
        Fl = (
            np.sign(T_core - 273.15) * 1000 * 4181 * J * abs(T_core - 273.15) ** (4 / 3)
        )
        temp_change = (-Fl - Fu - Int) / (1000 * 4181 * cell["lake_depth"])
        T_core = T_core + temp_change * dt_scaling
        # start at 1 as want layers below surface, end at len-1 to avoid lower boundary
        indices = np.arange(1, cell["vert_grid_lake"] - 1)
        cell["lake_temperature"][indices] = T_core
    # calculate Fl and Fu again one last time - Fu is used in lid development.
    Fl = np.sign(T_core - 273.15) * 1000 * 4181 * J * abs(T_core - 273.15) ** (4 / 3)
    Fu = (np.sign(T_core - cell["lake_temperature"][0]) * 1000 * 4181 * J *
          abs(T_core - cell["lake_temperature"][0]) ** (4 / 3))
    return Fl, Fu

def lake_formation(cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind):
    """
    Generate a lake, and track its evolution until we reach the point where it can evolve freely according to
    lake_development, when it goes about 10 cm deep.
    Called in timestep_loop.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
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
    original_mass = utils.calc_mass_sum(cell)
    dz = cell["firn_depth"] / cell["vert_grid"]
    cp_ice = np.zeros(cell["vert_grid"])
    k_ice = np.zeros(cell["vert_grid"])
    air = np.zeros(cell["vert_grid"])
    for i in np.arange(0, cell["vert_grid"]):
        if cell["firn_temperature"][i] > 273.15:
            cp_ice[i] = 4186.8
            k_ice[i] = 1000 * (
                1.017 * 10**-4 + 1.695 * 10**-6 * cell["firn_temperature"][i]
            )
        else:
            cp_ice[i] = 1000 * (7.16 * 10**-3 * cell["firn_temperature"][i] + 0.138) # Alexiades & Solomon pg. 8
            k_ice[i] = 1000 * (
                2.24 * 10**-3
                + 5.975 * 10**-6 * (273.15 - cell["firn_temperature"][i]) ** 1.156
            )
        air[i] = 1 - cell["Sfrac"][i] - cell["Lfrac"][i]
    k = cell["Sfrac"] * k_ice + air * cell["k_air"] + cell["Lfrac"] * cell["k_water"]
    x = cell["lake_temperature"]
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

    old_T_sfc = sfc_energy_lake_formation(T_air, Q, k, cell)
    # Check for conservation of mass
    new_mass = utils.calc_mass_sum(cell)
    assert abs(original_mass - new_mass) < 1.5 * 10**-7
    if old_T_sfc >= 273.15 and Q > 0:  # melting occurring at the surface
        kdTdz = (
            (cell["firn_temperature"][0] - cell["firn_temperature"][1])
            * abs(k[0])
            / (cell["firn_depth"] / cell["vert_grid"])
        )
        # change in firn height due to melting
        dHdt = (Q - kdTdz) / (cell["Sfrac"][0] * cell["L_ice"] * cell["rho_ice"]) * dt
        if dHdt < 0:
            raise ValueError("Error in surface temperature in lake formation \n")
        cell["melt_hours"] += 1

        # we reduce the firn height and add to the lake depth here
        firn_functions.regrid_after_melt(cell, dHdt, lake=True)
        # Set end=False since we only care about the top cell, and in this case we want to put this water into
        # the lake.
        percolation_functions.calc_saturation(cell, 0)

        air[i] = 1 - cell["Sfrac"][i] - cell["Lfrac"][i]

    # If we have 48h of no melt and the surface temp is below freezing then we refreeze the exposed water if
    # it is less than 10cm deep
    else:
        cell["exposed_water_refreeze_counter"] += 1
        if cell["exposed_water_refreeze_counter"] > 48 and cell["lake_depth"] < 0.1:
            print('Refreezing tiny lake')
            cell["exposed_water"] = False
            cell["exposed_water_refreeze_counter"] = 0
            dHdt = cell["lake_depth"] * cell["rho_water"] / cell["rho_ice"]
            cell["lake_depth"] = 0
            cell["firn_depth"] += dHdt
            cell["Sfrac"][0] += cell["Lfrac"][0] * cell["rho_water"] / cell["rho_ice"]  # freeze all water in top layer
            cell["Lfrac"][0] = 0
            # expansion of this water can cause Sfrac to be > 1, but the volume will be
            # so small that it should not matter.
            if cell["Sfrac"][0] > 1:
                cell["Sfrac"][0] = 1
                print('Sfrac > 1 in exposed water refreeze')


            # We can use our snowfall algorithm here as it effectively does the same thing (adds to the top
            # of the firn), with a density instead of 917 (density of ice)
            # snow_accumulation.snowfall(cell, -dHdt, 917, 273.15)

    cell["vertical_profile"] = np.linspace(0, cell["firn_depth"], cell["vert_grid"])
    x = cell["firn_temperature"]
    args = cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind
    root, fvec, success, info = solver.firn_heateqn_solver(
        x, args, fixed_sfc=True, solver_method='hybr'
    )
    if success:
        cell["firn_temperature"] = root
    if cell["lake_depth"] >= 0.1:
        cell["lake"] = True
    new_mass = utils.calc_mass_sum(cell)


def lake_development(cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind):
    """
    Once a lake of at least 10 cm deep is present this function calculates
    its evolution through a Stefan problem calculation of the lake-ice boundary.
    Called by timestep_loop.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
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
    J = 0.1 * (9.8 * 5 * 10**-5 * (1.19 * 10**-7) ** 2 / 10**-6) ** (1 / 3)
    original_mass = utils.calc_mass_sum(cell)
    if not cell["v_lid"] and not cell["lid"]:
        x = cell["lake_temperature"]
        # If no refrozen lid is present then the temperature of the top of the lake
        # is calculated using the surface energy balance
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
        # Calculate cell.lake_temperature and lid_temperature
        cell["lake_temperature"][0] = sfc_energy_lake(J, Q, cell)
        # when 10 cm - switch to a real lid
        if cell["lake_temperature"][0] < 273.15:

            cell["lid_temperature"][:] = cell["lake_temperature"][0]
            cell["lake_temperature"][0] = 273.15
            cell["v_lid"] = True

        if cell["lake_temperature"][0] > 300:
            print(f"lake_temp = {cell['lake_temperature'][0]}")
            raise ValueError("Lake too warm!")
    # Calculate cp and k for firn below lake (note:orig also calculated if
    # T > 273.15 but shouldn't be possible here)
    k_ice = np.zeros(cell["vert_grid"])
    for i in np.arange(0, cell["vert_grid"]):
        if cell["firn_temperature"][i] > 273.15:
            k_ice[i] = 1000 * (
                1.017 * 10**-4 + 1.695 * 10**-6 * cell["firn_temperature"][i]
            )
            if cell["firn_temperature"][i] > 273.15000001:
                raise ValueError("Firn temperature > 273.15")
        else:
            k_ice[i] = 1000 * (
                2.24 * 10**-3
                + 5.975 * 10**-6 * (273.15 - cell["firn_temperature"][i]) ** 1.156
            )

    # Lake is assumed turbulent as over 10cm.
    Fl, Fu = turbulent_mixing(cell, SW_in, dt)
    air = 1 - (cell["Sfrac"] + cell["Lfrac"])
    k = cell["Sfrac"] * k_ice + air * cell["k_air"] + cell["Lfrac"] * cell["k_water"]
    if cell["lake_temperature"][-2] > 273.15:  # If lake is above freezing it will begin to melt the firn below it
        if cell["firn_depth"] > 0:  # but only if the firn isn't completely melted.
            # coordinate system is defined in depth terms. Therefore, if the firn is colder than the lake,
            # the gradient is negative, so heat flows in the positive direction (downwards).
            kdTdz = (
                (273.15 - cell['lake_temperature'][-2])
                * abs(k[0])
                / (2 * (cell["lake_depth"] / cell["vert_grid_lake"]))
                # / (2 * (cell["firn_depth"] / cell["vert_grid"]))
            )
            # -ve kdTdz = +ve boundary_change, as energy flows from the lake into the firn, causing
            # it to melt. This motivates the - sign in front of kdTdz since the response to a positive
            # kdTdz is melting (i.e. boundary shifts down/positive boundary_change). The lid case
            # will have a reversed sign, since in that case the response to a negative kdTdz is
            # freezing (which in turn shifts the boundary down).

            boundary_change = (
                    # (kdTdz + Fl) / (cell["Sfrac"][0] * cell["L_ice"] * cell["rho_ice"]) * dt
                    -(kdTdz) / (cell["Sfrac"][0] * cell["L_ice"] * cell["rho_ice"]) * dt
            )
            cell["lake_boundary_change"] += boundary_change
            # Regrid the firn column to account for the change in boundary (which is
            # subtracted from the firn and added to the lake in this subroutine)
            # print('Lake melting firn, boundary change = ', boundary_change)
            # print('Fl = ', Fl, 'kdTdz = ', kdTdz)

            firn_functions.regrid_after_melt(cell, boundary_change, lake=True)
            # print('New firn depth = ', cell['firn_depth'])
            # Set end=False since we only care about the top cell, and in this case we want to put this water into
            # the lake.
            percolation_functions.calc_saturation(cell, 0)

        else:
            raise ValueError("Firn has all completely melted")

    # Increase lake depth and regrid lake to account for depth increase
    old_depth_grid = np.linspace(0, cell["lake_depth"], cell["vert_grid_lake"])

    if cell["lake_depth"] > 0:
        new_depth_grid = np.linspace(0, cell["lake_depth"], cell["vert_grid_lake"])
    else:
        print("Lake depth less than zero - problem")
        print(
            "Lake depth = ",
            cell["lake_depth"],
            "column = ",
            cell["column"],
            "row = ",
            cell["row"],
        )
        return

    cell["lake_temperature"] = np.interp(
        new_depth_grid, old_depth_grid, cell["lake_temperature"]
    )

    # Solve the heat equation to calculate the firn temperature given the
    # new grid of firn depth.
    x = cell["firn_temperature"]
    dz = cell["firn_depth"] / cell["vert_grid"]
    args = cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind
    root, fvec, success, info = solver.firn_heateqn_solver(
        x, args, fixed_sfc=True, solver_method='hybr'
    )
    if success:
        cell["firn_temperature"] = root

    # if there is a virtual lid then the top layer needs
    # to be set to the freezing temperature as it is an
    # ice-water boundary
    if cell["v_lid"]:
        cell["lake_temperature"][0] = 273.15

    cell["lake_temperature"][-1] = 273.15

    # mass conservation testing
    new_mass = utils.calc_mass_sum(cell)
    assert abs(original_mass - new_mass) < 1.5 * 10**-7
    return Fu  # used in lid development later