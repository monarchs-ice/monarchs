""" """

# TODO - module level docstring, split/refactor lake_formation and
#      - lake_development if possible
import numpy as np
from monarchs.physics import surface_fluxes, percolation
from monarchs.core import utils
from monarchs.physics import regrid_column
from monarchs.core.error_handling import (
    check_for_mass_conservation,
    generic_error,
)
from monarchs.physics import solver
from monarchs.physics.constants import (
    L_ice,
    rho_ice,
    rho_water,
    k_air,
    k_water, sfc_absorbed_frac,
    tau_water,
    tau_ice
)

MODULE_NAME = "monarchs.physics.lake"



def freeze_pre_lake(cell):
    """
    Refreeze a shallow 'pre-lake' (exposed water film) into the firn column, conserving mass.

    Converts the entire lake water depth H_w to an equivalent ice thickness
    H_i = H_w * (rho_water / rho_ice), adds that thickness at the surface
    (pure ice), and removes the lake water. Uses the same regridding
    routine as other freezing events to keep all state arrays consistent.
    """
    # Clear exposed-water flags/counters up-front
    cell["exposed_water"] = False
    cell["exposed_water_refreeze_counter"] = 0

    # Nothing to do if the pre-lake is already zero
    H_w = float(max(0.0, cell["lake_depth"]))
    if H_w == 0.0:
        return

    # Equivalent ice thickness to add at the surface (mass conservation)
    H_i = H_w * (rho_water / rho_ice)

    # Remove all lake water and mark no lake
    cell["lake_depth"] = 0.0
    cell["lake"] = False

    # Add a solid-ice layer of thickness H_i on top of the firn column.
    # This routine handles firn_depth, temperature, Sfrac/Lfrac, rho, etc.
    regrid_column.regrid_after_freeze(cell, H_i)

    # Newly formed surface ice should be at the melting point (freshwater)
    cell["firn_temperature"][0] = 273.15

    # If any lid flags were left over from previous states, ensure they're off
    cell["v_lid"] = False
    cell["lid"]   = False


def radiative_transfer(cell, sw_in):

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
    # at the surface. We can heuristically use ~0.5 as a fractional
    # absorption at the surface (i.e. 50% penetrates).
    # Then, get an estimated combined NIR+Vis absorption coefficient to use
    # to model the penetration of the rest of the radiation into the lake.
    # The rest of the radiation will penetrate all the way to the
    # firn-lake boundary. In this case, what happens?
    # I think that the boundary will basically absorb everything, minus some
    # fraction that will be reflected. I am assuming this will be based
    # on the saturated firn albedo (0.6). 
    # We also need to consider that the lake water will be turbid, and 
    # not pure (it will be grainy meltwater). So we should use a higher
    # absorption coefficient than pure water.

    Section 4.1.2 of Leppäranta (2015): Freezing of Lakes and the Evolution of
    their Ice Cover explicitly mentions a factor of ~0.45-0.5 for the
    surface absorption fraction.
    """

    cell["albedo"] = surface_fluxes.sfc_albedo(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["v_lid"],
        cell["lake_depth"],
        cell["snow_on_lid"],
    )
    not_absorbed_frac = (1 - sfc_absorbed_frac)  # fraction of sw_in that penetrates lake surface
    sw_penetrating = (1 - cell["albedo"]) * sw_in * not_absorbed_frac
    #print('SW in = ', sw_in)
    #print('SW penetrating = ', sw_penetrating)
    if cell["lid"]:
        # Ice has roughly the same absorption coefficient in the SWIR/NIR
        # as water, so we can assume that "sw_penetrating" is the same. The
        # difference is that the optical depth of ice is much higher, so more
        # radiation will be absorbed throughout the ice lid. But since the lid
        # is optically thin compared to firn (and just thin in general),
        # we need to actually model the penetration of radiation into the ice

        sw_entering_lake = sw_penetrating * np.exp(-tau_ice * cell["lid_depth"])
        radiation_at_bottom = sw_entering_lake * np.exp(
            -tau_water * cell["lake_depth"])
        lake_absorbed_solar = sw_entering_lake - radiation_at_bottom

    else:
        if cell["v_lid"]:
            sw_penetrating = sw_penetrating * np.exp(-tau_ice * cell["v_lid_depth"])

        radiation_at_bottom = sw_penetrating * np.exp(
            -tau_water * cell["lake_depth"]
        )
        lake_absorbed_solar = sw_penetrating - radiation_at_bottom
    #print('Transmittance = ', np.exp(-tau_ice * cell["lake_depth"]))
    #print('Lake depth = ', cell["lake_depth"])

    # We aren't quite done yet. We also have to consider the albedo of the
    # firn at the bottom of the lake. From surface_fluxes, the albedo
    # of saturated firn is 0.6. So accounting for this:
    saturated_firn_albedo = 0.6
    lake_reflected_radiation = radiation_at_bottom * saturated_firn_albedo
    #print('Radiation reflected at bottom = ', lake_reflected_radiation)
    radiation_at_bottom *= (1 - saturated_firn_albedo)
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
    #print('Bottom radiation = ', radiation_at_bottom)
    #print('Lake absorbed solar = ', lake_absorbed_solar)
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
    # print('lake absorbed solar in turbulent mixing: ', lake_absorbed_solar)
    # print('radiation at bottom in turbulent mixing: ', radiation_at_bottom)
    # factor by which you want to scale the temporal resolution of this
    # calculation. it is very slow (taking up to half of the overall
    # model runtime when not using Numba).
    # Increasing this value up to the max value (dt) will make the
    # model run faster, but you increase the likelihood of
    # numerical instability
    dt_scaling = 1
    nsteps = int(dt / dt_scaling)
    dh = 0

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
        # temp change
        # signs - positive downward into lake
        temp_change = (flux_upper - flux_lower + lake_absorbed_solar) / (
            1000 * 4181 * cell["lake_depth"]
        )

        lake_core_temp += temp_change * dt_scaling
        net_lower_flux_for_dh = flux_lower + radiation_at_bottom
        # record energy removed from lake by the bottom flux this substep
        # (flux_lower positive downward => energy leaving lake if positive)

        # apply mixed core temp to interior nodes
        indices = np.arange(1, cell["vert_grid_lake"] - 1)
        cell["lake_temperature"][indices] = lake_core_temp
        dh_change, cap_reached = calc_height_adjustment(
            cell, k, net_lower_flux_for_dh
        )
        dh += dh_change
    if dh > (cell["firn_depth"] / cell["vert_grid"]):
        dh = cell["firn_depth"] / cell["vert_grid"]
        print('Melting entire layer')
    # print('dh in turbulent mixing: ', dh)
    # print('dh in turbulent mixing (original calc): ', dh_change * 3600)
    # return both the bottom flux (W/m^2) and the cumulative energy moved there (J/m^2)
    return flux_upper, dh


def lake_formation(
    cell, dt, met_data
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
        + air * k_air
        + cell["Lfrac"] * k_water
    )

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
    root, _, success, _ = solver.solve_firn_heateqn(
        cell, met_data, dt, dz, fixed_sfc=True, solver_method="hybr"
    )
    if success:
        cell["firn_temperature"] = root

    x = cell["lake_temperature"]
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

    old_T_sfc = solver.lake_seb_solver(cell, met_data, dt, dz, formation=True)[0][0]

    # Check for conservation of mass
    new_mass = utils.calc_mass_sum(cell)
    errflag = check_for_mass_conservation(
        cell, original_mass, new_mass, routine_name
    )

    if old_T_sfc >= 273.15 and Q > 0:  # melting occurring at the surface
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
                / (cell["Sfrac"][0] * L_ice * rho_ice)
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
            freeze_pre_lake(cell)
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
    cell, dt, met_data
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
        cell["albedo"] = surface_fluxes.sfc_albedo(
            cell["melt"],
            cell["exposed_water"],
            cell["lid"],
            cell["lake"],
            cell["v_lid"],
            cell["lake_depth"],
            cell["snow_on_lid"]
        )
        cell["lake_temperature"][0] = solver.lake_seb_solver(cell, met_data, dt, 0, formation=False)[0][0]
        # If surface cooled below freezing, create virtual lid
        if cell["lake_temperature"][0] < 273.15:
            cell["lid_temperature"][:] = cell["lake_temperature"][0]
            cell["lake_temperature"][0] = 273.15
            cell["v_lid"] = True
            cell["virtual_lid_temperature"] = cell["lid_temperature"][0]

        if cell["lake_temperature"][0] > 300:
            message = "Unrealistic lake surface temperature > 300K\n"
            message += f"\tT_sfc = {cell['lake_temperature'][0]}\n"
            generic_error(cell, routine_name, message)

    elif cell["lid"] or cell["v_lid"]:
        cell["lake_temperature"][0] = 273.15
    # print('Firn temperature = ', cell["firn_temperature"][0:10])

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
        + air * k_air
        + cell["Lfrac"] * k_water
    )
    # Lake is turbulent (>= 10 cm)
    # Compute dh using the actual fluxes over the timestep rather
    # than the instantaneous flux at the end of the timestep * dt
    """ Height change calculation is embedded in turbulent_mixing """
    Fu, boundary_change = turbulent_mixing(cell, met_data["SW_down"], dt, k)
    print('Boundary change in lake development: ', boundary_change)

    # Regrid the firn column to account for the change in boundary
    # (which is subtracted from the firn and added to the lake in
    # this subroutine)
    if boundary_change > 0:
        regrid_column.regrid_after_melt(cell, boundary_change, lake=True)
    elif boundary_change < 0:
        # remove water from lake, add ice to firn
        thickness_to_add = abs(boundary_change)
        water_loss = thickness_to_add * (rho_ice / rho_water)
        cell["lake_depth"] -= water_loss
        regrid_column.regrid_after_freeze(cell, thickness_to_add)
        cell["lake_boundary_change"] += boundary_change
    # Set end=False since we only care about the top cell, and in this
    # case we want to put this water into the lake.
    percolation.calc_saturation(cell, 0)

    # Bottom at freezing
    cell["lake_temperature"][-1] = 273.15

    new_mass = utils.calc_mass_sum(cell)
    check_for_mass_conservation(cell, original_mass, new_mass, routine_name)
    return Fu


def calc_height_adjustment(cell, k, Fl):
    routine_name = f"{MODULE_NAME}.calc_height_adjustment"
    # If lake is above freezing it will begin to melt the firn below it
    boundary_change = 0
    cap_reached = False
    # only do this if melting is possible!
    if cell["lake_temperature"][-2] > 273.15:
        # but only if the firn isn't completely melted.
        if cell["firn_depth"] > 0:
            # coordinate system is defined in depth terms. Therefore, if the
            # firn is colder than the lake, the gradient is positive, so heat
            # flows in the positive direction (downwards).
            # get avg_k from average of k from 0:2
            kdTdz = (
                (cell["firn_temperature"][0] - cell["firn_temperature"][1])
                * abs(k[0])
                / ((cell["firn_depth"] / cell["vert_grid"]))
            )
            # First work out the maximum available amount of melt based
            # on the solid fraction of the top cell
            # Limit sfrac_top to a minimum of 1e-2 to avoid zero division error
            sfrac_top = max(1e-2, cell["Sfrac"][0])
            if cell['Sfrac'][0] < 1e-2:
                print('Warning - Sfrac at top of firn very low in calc_height_adjustment')
            # print('Fl = ', Fl)
            # print('kdTdz = ', kdTdz)
            # print('Net flux = ', (Fl - kdTdz))

            # Fl is positive going *into* interface, kdTdz is positive going
            # *out of interface into the firn*, so to get the net flux we need to
            # subtract them (e.g. if Fl is 10 W/m^2 into interface, and kdTdz is 5 W/m^2
            # out of interface, net flux into interface is 5 W/m^2)
            boundary_change_raw = (
                (Fl - kdTdz)
                / (sfrac_top * L_ice * rho_ice)
            )

            cap = cell["firn_depth"] / cell["vert_grid"]

            # Take the minimum by absolute value, but keep the
            # sign of boundary_change_raw
            if abs(boundary_change_raw) > cap:
                boundary_change = np.sign(boundary_change_raw) * cap
                cap_reached = True
                print('Cap reached in calc_height_adjustment')
            else:
                boundary_change = boundary_change_raw

            # Raise an error if we have unphysical kdTdz (i.e. firn temperature
            # below the boundary is warmer than at the boundary)
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
