""" """

# TODO - module level docstring, split/refactor lake_formation and
#      - lake_development if possible
import numpy as np
from monarchs.physics import regrid_column, surface_fluxes, solver, percolation
from monarchs.core import utils
from monarchs.core.error_handling import (
    check_for_mass_conservation,
    generic_error,
)

MODULE_NAME = "monarchs.physics.lake"


def sfc_energy_lake(
    cell, lw_in, sw_in, air_temp, p_air, dew_point_temperature, wind
):
    """
    Calculate the surface energy balance for the lake, after it has already
    formed.
    Called by lake_solver, which is in turn called by either lake_formation,
    or lake_development.

    Parameters
    ----------
    J : float
        Turbulent heat flux factor, equal to 1.907 E-5. [m s^-1 K^-(1/3)]
        See Buzzard (2017), pp. 43 for details.
    Q : float
        Surface energy flux, calculated via surface_fluxes.sfc_flux.  [W m^-2]
    cell : numpy structured array
        Element of the model grid we are operating on.
    lw_in:  float
        Downwelling longwave radiation at the surface. [W m^-2]
    sw_in : float
        Downwelling shortwave radiation at the surface [W m^-2]
    air_temp : float
        Surface air temperature. [K]
    p_air : float
        Surface air pressure. [Pa]
    dew_point_temperature : float
        Dewpoint temperature of the air at the surface. [K]
    wind : float
        Wind speed at the surface. [m s^-1]

    Returns
    -------
    lake_surf_temp : float
        Surface temperature of the lake. [K]
    """

    # This is solved for x to give a temp at top of lake,
    # then turbulent mixing can occur, then lake dev
    # lake well mixed - boundary and core temp so only boundaries have a diff
    # temperature.
    x = np.array([cell["lake_temperature"][0]])
    args = np.array(
        [
            cell["vert_grid_lake"],
            cell["melt"],
            cell["exposed_water"],
            cell["lid"],
            cell["lake"],
            cell["lake_depth"],
            lw_in,
            sw_in,
            air_temp,
            p_air,
            dew_point_temperature,
            wind,
        ]
    )
    args = np.append(args, cell["lake_temperature"])
    lake_surf_temp = solver.lake_solver(x, args)[0][0]

    return lake_surf_temp


def sfc_energy_lake_formation(air_temp, Q, k, cell):
    """
    Calculate the surface energy balance for the lake, during the formation
    step.
    Called by lake_solver, which is in turn called by either lake_formation,
    or lake_development.

    Parameters
    ----------
    air_temp : float
        Surface air temperature. [K]
    Q : float
        Surface energy balance, as calculated by surface_fluxes.sfc_flx
    k : array_like, float, dimension(cell.vert_grid)
        Thermal conductivity of the firn column, as obtained by an
        Sfrac/Lfrac/air fraction weighted calculation using k_ice, k_water and
        k_air respectively. We only use the first element of this,
        i.e. the surface value. [W m^-1 K^-1]
    cell : numpy structured array
        Element of the model grid we are operating on.

    Returns
    -------
    old_surf_temp : float
        Surface temperature of the lake. [K]
    """
    x = np.array([float(air_temp)])
    # k is a 1D array - hence need the [0] else Numba doesn't like it
    # args = np.array([cell.firn_depth, float(cell.vert_grid), Q])
    args = np.array(
        [
            cell["firn_depth"],
            cell["vert_grid"],
            Q,
            k[0],
            cell["firn_temperature"][1],
        ]
    )
    old_surf_temp = solver.lake_solver(x, args, formation=True)[0][0]
    return old_surf_temp


def freeze_pre_lake(cell):
    """ """
    # TODO - this needs to properly regrid the column
    #      - since we can have tiny lakes freezing after
    #      - a lid is combined into the firn (meaning
    #      - that we have Sfrac = 1 at the surface)
    cell["exposed_water"] = False
    cell["exposed_water_refreeze_counter"] = 0
    dHdt = cell["lake_depth"] * cell["rho_water"] / cell["rho_ice"]
    cell["lake_depth"] = 0
    cell["firn_depth"] += dHdt
    cell["Sfrac"][0] += (
        cell["Lfrac"][0] * cell["rho_water"] / cell["rho_ice"]
    )  # freeze all water in top layer
    cell["Lfrac"][0] = 0
    # expansion of this water can cause Sfrac to be > 1, but the volume
    # will be so small that it should not matter.
    if cell["Sfrac"][0] > 1:
        cell["Sfrac"][0] = 1


def radiative_transfer(cell, sw_in):
    albedo = surface_fluxes.sfc_albedo(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["lake_depth"],
    )
    """
    # JE - I'm thinking about the solar radiation in the lake.
    # I'm a bit concerned about how it was handled before. The assumption is
    # that we use surface_fluxes to calculate Q. But this assumes that all of
    # the shortwave radiation is absorbed at the surface. This is *basically*
    # true for firn. But for a lake, we have to consider the spectral absorption
    # coefficient. For SWIR and NIR, most of the radiation will be absorbed near
    # the surface (absorption coefficient ~1-10 m^-1).
    # But for visible/UV, the absorption coefficient is much lower (~0.1 m^-1).
    # I think this means that in order to budget energy correctly, we need
    # data on the absorption spectrum of water. We know that heuristically the
    # incoming radiation is ~50% in the UV/Vis, ~20% in the NIR and ~30% in the
    # SWIR.
    # The datasets I am using for this heuristic are:
    # https://omlc.org/spectra/water/abs/buiteveld94.html
    # L.Kou, D.Labrie, P.Chylek, "Refractive indices of water and ice in
    # the 0.65-2.5 µm spectral range,"
    # Appl.Opt., 32, 3531 - 3540(1993).  #
    # https://omlc.org/spectra/water/data/kou93b.txt
    # for the SWIR/NIR, and
    # R. M. Pope, E. S. Fry, "Absorption spectrum (380-700 nm) of pure water. II.
    # Integrating cavity measurements," Appl. Opt.,36, 8710-8723, (1997).
    # https://omlc.org/spectra/water/data/pope97.txt
    # for the vis.
    # Looking at the datasets, some conservative suggested values are:
    # UV/Vis (380-700 nm): k ~ 0.025 - 0.1 m^-1
    # NIR (700-1000 nm): k ~ 5-10 m^-1
    # SWIR (1000-2500 nm): k ~ 500 m^-1
    # this accounts for the median values weighted by incoming radiation.
    # - n.b. these values are for clear water. We probably have some level
    # of turbidity? Which will increase the absorption coefficients a bit.
    # Using these values, we can estimate the depth penetration of the
    # different bands. Assuming a surface layer of a couple cm:
    # SWIR - almost everything is absorbed extremely quickly
    # NIR - Optical depth of 1 at ~10-20 cm^-1. So most absorbed in top few
    # cm.
    # Vis - Optical depth of 1 at ~10s of m. So basically all of it penetrates
    # into the lake.
    # So we need to:
    # weight surface_fluxes by the amount we absorb specifically
    # at the surface. We can heuristically use ~0.4 as a fractional
    # absorption at the surface (i.e. 60% penetrates).
    # Then, get an estimated combined NIR+Vis absorption coefficient to use
    # to model the penetration of the rest of the radiation into the lake.
    # The rest of the radiation will penetrate all the way to the
    # firn-lake boundary. In this case, what happens?
    # I think that the boundary will basically absorb everything, minus some
    # fraction that will be reflected. I am assuming this will be based
    # on the saturated firn albedo (0.6). 
    """
    not_absorbed_frac = 0.6  # fraction of sw_in that penetrates lake surface
    tau_water = 0.1  # more realistic I think than 0.025
    sw_penetrating = (1 - albedo) * sw_in * not_absorbed_frac
    tau_ice = 1.5  # black ice extinction coefficient
    # rough value taking into account:
    # https://tc.copernicus.org/articles/15/1931/2021/#section3
    # Cooper, M.G., et al., 2021. Spectral attenuation coefficients from
    # measurements of light transmission in bare ice on the Greenland Ice Sheet.
    # The Cryosphere, 15(4), pp.1931-1953.

    if cell["lid"]:
        # Ice has roughly the same absorption coefficient in the SWIR/NIR
        # as water, so we can assume that "sw_penetrating" is the same. The
        # difference is that the optical depth of ice is much higher, so more
        # radiation will be absorbed throughout the ice lid. But since the lid
        # is optically thin compared to firn (and just thin in general),
        # we need to actually model the penetration of radiation into the ice
        I0_below_ice = (
            ((1.0 - albedo) * sw_in)
            * not_absorbed_frac
            * np.exp(-tau_ice * cell["lid_depth"])
        )
        lake_absorbed_solar = I0_below_ice
        radiation_at_bottom = 0
    else:
        # i0 - i0 exp(-tau_water*z)
        radiation_at_bottom = sw_penetrating * np.exp(
            -tau_water * cell["lake_depth"]
        )
        lake_absorbed_solar = sw_penetrating - radiation_at_bottom
    # We aren't quite done yet. We also have to consider the albedo of the
    # firn at the bottom of the lake. From surface_fluxes, the albedo
    # of saturated firn is 0.6. So accounting for this:
    saturated_firn_albedo = 0.6
    lake_reflected_radiation = radiation_at_bottom * saturated_firn_albedo
    radiation_at_bottom *= 1 - saturated_firn_albedo
    # and this reflected radiation will also be absorbed again in the lake.
    lake_absorbed_solar = lake_absorbed_solar + (
        lake_reflected_radiation
        * (1 - np.exp(-tau_water * cell["lake_depth"]))
    )
    # We don't especially care about the outgoing radiation as it isn't going
    # to be absorbed by the surface. This effectively becomes an additional
    # contribution to the albedo.
    # Is this true? It matters if we have a lid! But I think the assumption is
    # that most radiation will be absorbed by now...?
    return lake_absorbed_solar, radiation_at_bottom


def turbulent_mixing(cell, sw_in, dt, k):
    """
    The lake has a temperature profile governed by its boundary conditions
    - 0 degrees at the firn-lake boundary, and driven by the surface energy
    balance at the surface. The lake is turbulent, meaning there is a
    significant amount of mixing, causing the temperature profile to even out.
    Called by lake_development.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    sw_in : float
        Downwelling shortwave radiation. [W m^-2]
    dt : int
        Number of seconds in the current timestep. [s]
    Returns
    -------
    None (amends cell inplace).
    """
    # pylint: disable=invalid-name
    J = 0.1 * (9.8 * 5 * 10 ** -5 * (1.19 * 10 ** -7) ** 2 / 10 ** -6) ** (
        1 / 3
    )
    # pylint: enable=invalid-name
    # If lake is below 0.1 m deep then it is no longer turbulent
    if cell["lake_depth"] < 0.1:
        return 0, 0
    lake_absorbed_solar, radiation_at_bottom = radiative_transfer(cell, sw_in)
    # factor by which you want to scale the temporal resolution of this
    # calculation. it is very slow (taking up to half of the overall
    # model runtime when not using Numba).
    # Increasing this value up to the max value (dt) will make the
    # model run faster, but you increase the likelihood of
    # numerical instability
    dt_scaling = 1
    nsteps = int(dt / dt_scaling)

    dh = 0
    cap_reached = False
    for _ in range(nsteps):
        lake_core_temp = cell["lake_temperature"][
            int(cell["vert_grid_lake"] / 2)
        ]

        flux_upper = (
            np.sign(cell["lake_temperature"][0] - lake_core_temp)
            * 1000
            * 4181
            * J
            * abs(cell["lake_temperature"][0] - lake_core_temp) ** (4 / 3)
        )

        flux_lower = (
            np.sign(lake_core_temp - 273.15)
            * 1000
            * 4181
            * J
            * abs(lake_core_temp - 273.15) ** (4 / 3)
        )

        # temp change *from turbulent divergence* (you already do this)
        temp_change = (flux_upper - flux_lower + lake_absorbed_solar) / (
            1000 * 4181 * cell["lake_depth"]
        )

        lake_core_temp += temp_change * dt_scaling

        net_lower_flux_for_dh = flux_lower + radiation_at_bottom
        # record energy removed from lake by the bottom flux this substep
        # (flux_lower positive downward => energy leaving lake if positive)
        dh_change, cap_reached = calc_height_adjustment(
            cell, k, dt_scaling, net_lower_flux_for_dh
        )

        dh += dh_change
        if dh > (cell["firn_depth"] / cell["vert_grid"]):
            dh = cell["firn_depth"] / cell["vert_grid"]
        # apply mixed core temp to interior nodes
        indices = np.arange(1, cell["vert_grid_lake"] - 1)
        cell["lake_temperature"][indices] = lake_core_temp

    # return both the bottom flux (W/m^2) and the cumulative energy moved there (J/m^2)
    return flux_upper, dh


def lake_formation(
    cell, dt, lw_in, sw_in, air_temp, p_air, dew_point_temperature, wind
):
    """
    Generate a lake, and track its evolution until we reach the point where
    it can evolve freely according to lake_development, when it goes about
    10 cm deep.
    Called in timestep_loop.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    dt : int
        Number of seconds in the timestep, most likely 3600 (i.e. 1 hour) [s]
    lw_in : float
        Downwelling longwave radiation at the surface. [W m^-2]
    sw_in : float
        Downwelling shortwave radiation at the surface [W m^-2]
    air_temp : float
        Surface air temperature. [K]
    p_air : float
        Surface air pressure. [Pa]
    dew_point_temperature : float
        Dewpoint temperature of the air at the surface. [K]
    wind : float
        Wind speed at the surface. [m s^-1]
    Returns
    -------
    None (amends cell inplace).
    """
    routine_name = f"{MODULE_NAME}.lake_formation"
    if cell["lake_depth"] > 0.1:
        cell["lake"] = True
    if np.isnan(cell["lake_depth"]):
        print("Error (start of timestep) - lake depth is NaN")
        cell["error_flag"] = 1
    original_mass = utils.calc_mass_sum(cell)
    dz = cell["firn_depth"] / cell["vert_grid"]
    cp_ice = np.zeros(cell["vert_grid"])
    k_ice = np.zeros(cell["vert_grid"])
    air = np.zeros(cell["vert_grid"])
    for i in np.arange(0, cell["vert_grid"]):
        if cell["firn_temperature"][i] > 273.15:
            cp_ice[i] = 4186.8
            k_ice[i] = 1000 * (
                1.017 * 10 ** -4
                + 1.695 * 10 ** -6 * cell["firn_temperature"][i]
            )
        else:
            cp_ice[i] = 1000 * (
                7.16 * 10 ** -3 * cell["firn_temperature"][i] + 0.138
            )  # Alexiades & Solomon pg. 8
            k_ice[i] = 1000 * (
                2.24 * 10 ** -3
                + 5.975
                * 10 ** -6
                * (273.15 - cell["firn_temperature"][i]) ** 1.156
            )
        air[i] = 1 - cell["Sfrac"][i] - cell["Lfrac"][i]
    k = (
        cell["Sfrac"] * k_ice
        + air * cell["k_air"]
        + cell["Lfrac"] * cell["k_water"]
    )
    x = cell["firn_temperature"]
    args = (
        cell,
        dt,
        dz,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
    )

    root, _, success, _ = solver.solve_firn_heateqn(
        x, args, fixed_sfc=True, solver_method="hybr"
    )
    if success:
        cell["firn_temperature"] = root

    x = cell["lake_temperature"]
    Q = surface_fluxes.sfc_flux(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["lake_depth"],
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        x[0],
    )

    old_T_sfc = sfc_energy_lake_formation(air_temp, Q, k, cell)
    # Check for conservation of mass
    new_mass = utils.calc_mass_sum(cell)
    errflag = check_for_mass_conservation(
        cell, original_mass, new_mass, routine_name
    )

    if old_T_sfc > 273.15 and Q > 0:  # melting occurring at the surface
        kdTdz = (
            (cell["firn_temperature"][0] - cell["firn_temperature"][1])
            * abs(k[0])
            / (cell["firn_depth"] / cell["vert_grid"])
        )
        # change in firn height due to melting
        if cell["Sfrac"][0] < 0.1:
            dHdt = cell["firn_depth"] / cell["vert_grid"]
        else:
            dHdt = (
                (Q - kdTdz)
                / (cell["Sfrac"][0] * cell["L_ice"] * cell["rho_ice"])
                * dt
            )

        if dHdt < 0:
            cell["error_flag"] = True
            message = "Error in surface temperature in lake formation \n"
            message += f"\tdHdt = {dHdt}\n"
            generic_error(cell, routine_name, message)
            return cell

        cell["melt_hours"] += 1

        # we reduce the firn height and add to the lake depth here
        regrid_column.regrid_after_melt(cell, dHdt, lake=True)
        # Set end=False since we only care about the top cell, and in this case
        # we want to put this water into the lake.
        percolation.calc_saturation(cell, 0)

    # If we have 48h of no melt and the surface temp is below freezing then we
    # refreeze the exposed water if it is less than 10cm deep
    else:
        cell["exposed_water_refreeze_counter"] += 1
        if (
            cell["exposed_water_refreeze_counter"] > 48
            and cell["lake_depth"] < 0.1
        ):
            # freeze_pre_lake(cell)
            pass

    cell["vertical_profile"] = np.linspace(
        0, cell["firn_depth"], cell["vert_grid"]
    )
    if cell["lake_depth"] >= 0.1:
        cell["lake"] = True
    # Another round of mass conservation checks
    new_mass = utils.calc_mass_sum(cell)
    # end of timestep so not bothered about returning early if error
    check_for_mass_conservation(cell, original_mass, new_mass, routine_name)
    if np.isnan(cell["lake_depth"]):
        print("Error - lake depth is NaN (end of timestep)")
        cell["error_flag"] = 1


def lake_development(
    cell, dt, lw_in, sw_in, air_temp, p_air, dew_point_temperature, wind
):
    """
    Once a lake of at least 10 cm deep is present this function calculates
    its evolution through a Stefan problem calculation of the lake-ice
    boundary.
    Called by timestep_loop.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    dt : int
        Number of seconds in the timestep, most likely 3600 (i.e. 1 hour) [s]
    lw_in : float
        Downwelling longwave radiation at the surface. [W m^-2]
    sw_in : float
        Downwelling shortwave radiation at the surface [W m^-2]
    air_temp : float
        Surface air temperature. [K]
    p_air : float
        Surface air pressure. [Pa]
    dew_point_temperature : float
        Dewpoint temperature of the air at the surface. [K]
    wind : float
        Wind speed at the surface. [m s^-1]

    Returns
    -------

    """
    routine_name = f"{MODULE_NAME}.lake_development"
    original_mass = utils.calc_mass_sum(cell)
    if not cell["v_lid"] and not cell["lid"]:
        # Solve lake surface temperature
        sw_in_adjusted = 0.4 * sw_in  # assume 40% absorbed by surface layer(s)
        cell["lake_temperature"][0] = sfc_energy_lake(
            cell,
            lw_in,
            sw_in_adjusted,
            air_temp,
            p_air,
            dew_point_temperature,
            wind,
        )

        # If surface cooled below freezing, create virtual lid
        if cell["lake_temperature"][0] < 273.15:
            cell["lid_temperature"][:] = cell["lake_temperature"][0]
            cell["lake_temperature"][0] = 273.15
            cell["v_lid"] = True

        if cell["lake_temperature"][0] > 300:
            message = "Unrealistic lake surface temperature > 300K\n"
            message += f"\tT_sfc = {cell['lake_temperature'][0]}\n"
            generic_error(cell, routine_name, message)

    elif cell["lid"] or cell["v_lid"]:
        cell["lake_temperature"][0] = 273.15

    # Conductivity below lake
    k_ice = np.zeros(cell["vert_grid"])
    for i in np.arange(0, cell["vert_grid"]):
        if cell["firn_temperature"][i] > 273.15:
            k_ice[i] = 1000.0 * (
                1.017e-4 + 1.695e-6 * cell["firn_temperature"][i]
            )
        else:
            k_ice[i] = 1000.0 * (
                2.24e-3
                + 5.975e-6 * (273.15 - cell["firn_temperature"][i]) ** 1.156
            )
    air = 1.0 - (cell["Sfrac"] + cell["Lfrac"])
    k = (
        cell["Sfrac"] * k_ice
        + air * cell["k_air"]
        + cell["Lfrac"] * cell["k_water"]
    )
    # Lake is turbulent (>= 10 cm)
    # Compute dh using the actual fluxes over the timestep rather
    # than the instantaneous flux at the end of the timestep * dt
    """ Height change calculation is embedded in turbulent_mixing """
    Fu, boundary_change = turbulent_mixing(cell, sw_in, dt, k)

    # Regrid the firn column to account for the change in boundary
    # (which is subtracted from the firn and added to the lake in
    # this subroutine)
    regrid_column.regrid_after_melt(cell, boundary_change, lake=True)
    # Set end=False since we only care about the top cell, and in this
    # case we want to put this water into the lake.
    percolation.calc_saturation(cell, 0)

    # Bottom at freezing
    cell["lake_temperature"][-1] = 273.15

    new_mass = utils.calc_mass_sum(cell)
    check_for_mass_conservation(cell, original_mass, new_mass, routine_name)
    return Fu


def calc_height_adjustment(cell, k, dt_scaling, Fl):
    routine_name = f"{MODULE_NAME}.calc_height_adjustment"
    # If lake is above freezing it will begin to melt the firn below it
    boundary_change = 0
    cap_reached = False
    if cell["lake_temperature"][-2] > 273.15:
        # but only if the firn isn't completely melted.
        if cell["firn_depth"] > 0:
            # coordinate system is defined in depth terms. Therefore, if the
            # firn is colder than the lake, the gradient is positive, so heat
            # flows in the positive direction (downwards).
            # get avg_k from average of k from 0:2
            avg_k = np.mean(k[0:2])
            kdTdz = (
                (cell["firn_temperature"][0] - cell["firn_temperature"][1])
                * abs(avg_k)
                / ((cell["firn_depth"] / cell["vert_grid"]))
            )

            # First work out the maximum available amount of melt based
            # on the solid fraction of the top cell
            # Limit sfrac_top to a minimum of 1e-6 to avoid zero division error
            sfrac_top = max(1e-6, cell["Sfrac"][0])

            boundary_change_raw = (
                (Fl - kdTdz)
                / (sfrac_top * cell["L_ice"] * cell["rho_ice"])
                * dt_scaling
            )

            cap = cell["firn_depth"] / cell["vert_grid"]

            # Take the minimum by absolute value, but keep the
            # sign of boundary_change_raw
            if abs(boundary_change_raw) > cap:
                boundary_change = np.sign(boundary_change_raw) * cap
                cap_reached = True
            else:
                boundary_change = boundary_change_raw

            if kdTdz < 0:
                message = "Error in lake development kdTdz < 0\n"
                message += f"kdTdz = {kdTdz}\n"
                generic_error(cell, routine_name, message)

            cell["lake_boundary_change"] += boundary_change
            cell["firn_boundary_change"] -= boundary_change

        else:
            message = "Lake over completely melted firn\n"
            generic_error(cell, routine_name, message)

    return boundary_change, cap_reached
