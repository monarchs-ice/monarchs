"""
Lake formation physics.

Contains functions for the early-stage evolution of exposed meltwater at the
firn surface, before a full lake (depth >= 10 cm) has developed.
"""

# TODO - module level docstring, split/refactor lake_formation and
#      - lake_development if possible
import numpy as np
from monarchs.core.kernels import kernel
from monarchs.physics import surface_fluxes, material_properties
from monarchs.physics.firn import percolation
from monarchs.core import utils
from monarchs.physics.firn import regrid_column
from monarchs.core.error_handling import (
    check_for_mass_conservation,
    generic_error,
)
from monarchs.physics import solver
from monarchs.physics.constants import (
    L_ice,
    rho_ice,
    rho_water,
    stefan_boltzmann,
    emissivity,
)

MODULE_NAME = "monarchs.physics.lake"


@kernel()
def lake_formation(cell, dt, met_data):
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
    met_data: numpy structured array
        Structured array containing meteorological data used in the model.
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
    # firn conductivity term
    k = material_properties.k_mixture(
        cell["firn_temperature"], cell["Sfrac"], cell["Lfrac"]
    )

    # Update cell albedo
    cell["albedo"] = surface_fluxes.sfc_albedo(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["v_lid"],
        cell["lake_depth"],
        cell["snow_on_lid"],
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
        cell["snow_on_lid"],
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
    check_for_mass_conservation(cell, original_mass, new_mass, routine_name)

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
                (Q - emissivity * stefan_boltzmann * 273.15**4 - kdTdz)
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
    # TODO - this currently doesn't do anything, but might be something to think
    # about for later.
    else:
        cell["exposed_water_refreeze_counter"] += 1
        if cell["exposed_water_refreeze_counter"] > 48 and cell["lake_depth"] < 0.1:
            # freeze_pre_lake(cell)
            pass

    cell["vertical_profile"] = np.linspace(0, cell["firn_depth"], cell["vert_grid"])
    if cell["lake_depth"] >= 0.1:
        cell["lake"] = True
    # Another round of mass conservation checks
    new_mass = utils.calc_mass_sum(cell)
    # end of timestep so not bothered about returning early if error
    check_for_mass_conservation(cell, original_mass, new_mass, routine_name)
    if np.isnan(cell["lake_depth"]):
        print("Error - lake depth is NaN (end of timestep)")
        cell["error_flag"] = 1


@kernel()
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
    cell["lid"] = False
