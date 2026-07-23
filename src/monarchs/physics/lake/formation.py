"""
Lake formation physics.

Contains functions for the early-stage evolution of exposed meltwater at the
firn surface, before a full lake (depth >= 10 cm) has developed.
"""

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
from monarchs.physics.firn.heateqn import firn_heateqn_solver
from monarchs.physics.lake.seb import lake_seb_solver
from monarchs.physics.constants import (
    L_ice,
    rho_ice,
)

MODULE_NAME = "monarchs.physics.lake.formation"


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
    cell["albedo"] = surface_fluxes.sfc_albedo(cell)
    root, success, _ = firn_heateqn_solver(cell, met_data, dt, dz, fixed_sfc=True)
    if success:
        cell["firn_temperature"] = root

    x = cell["lake_temperature"]
    # Update cell albedo
    cell["albedo"] = surface_fluxes.sfc_albedo(cell)
    Q = surface_fluxes.sfc_flux(cell, met_data, x[0])

    old_T_sfc = lake_seb_solver(cell, met_data, formation=True)[0]

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
            dHdt = (Q - kdTdz) / (cell["Sfrac"][0] * L_ice * rho_ice) * dt
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

    # If no melt and the surface is below freezing, count towards refreezing
    # the exposed water.
    else:
        cell["exposed_water_refreeze_counter"] += 1

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
