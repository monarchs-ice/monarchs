"""
TODO - module-level docstring, other docstring
"""

import numpy as np
from monarchs.physics import percolation
from monarchs.core import utils


def combine_lid_firn(cell):
    """
    Combines the lid and firn profiles into a single column when the lake is
    completely frozen.
    This avoids overwriting fixed-length arrays by creating new arrays for the
    combined profiles.

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
        "Combining lid and firn to create one profile..., column ="
        f" {cell["column"]}, row = {cell["row"]}"
    )

    if cell["v_lid"]:
        cell["lid_depth"] = cell["v_lid_depth"]
    # Create a deep copy of the original lid depth - used later
    orig_lid_depth = cell["lid_depth"] + 0
    # For calculating new vertical profile - add virtual lid and lake depth
    # to that of the lid.

    # Create new arrays for the combined profiles
    new_vert_grid = cell["vert_grid"]
    new_depth_grid = np.linspace(
        0, cell["firn_depth"] + cell["lake_depth"] + cell["lid_depth"],
        new_vert_grid
    )

    # If we have a lake, and we are combining the lid and lake profiles,
    # we need to ensure that the density of the lake is added to that of the
    # lid, accounting for the depths.
    rho_lake = cell["rho_water"] * np.ones(cell["vert_grid_lake"])

    # Add lake depth to lid depth. If the lake has frozen, then this will
    # ensure that mass is conserved.
    # We need to interpolate in such a way that we conserve mass.
    # Interpolate lid and firn properties to the new depth grid.
    # Three separate arrays - top to bottom of lid, bottom of lid to bottom of
    # lake, bottom of lake to bottom of firn.
    old_depth_grid = np.concatenate(
        (
            np.linspace(0, cell["lid_depth"], cell["vert_grid_lid"]),
            np.linspace(
                cell["lid_depth"],
                cell["lake_depth"] + cell["lid_depth"],
                cell["vert_grid_lake"],
            ),
            np.linspace(
                cell["lid_depth"] + cell["lake_depth"],
                cell["firn_depth"] + cell["lid_depth"] + cell["lake_depth"],
                cell["vert_grid"],
            ),
        )
    )

    new_firn_temperature = np.interp(
        new_depth_grid,
        old_depth_grid,
        np.concatenate(
            (
                cell["lid_temperature"],
                cell["lake_temperature"],
                cell["firn_temperature"],
            )
        ),
    )
    new_rho = np.interp(
        new_depth_grid,
        old_depth_grid,
        np.concatenate(
            (
                cell["rho_lid"],
                np.ones(cell["vert_grid_lake"]) * cell["rho_water"],
                cell["rho"],
            )
        ),
    )

    sfrac_new = mass_conserving_profile(cell, orig_lid_depth, var="Sfrac")
    lfrac_new = mass_conserving_profile(cell, orig_lid_depth, var="Lfrac")

    # Determine which parts of the new profile are saturated.
    percolation.calc_saturation(cell, cell["vert_grid"] - 1)
    # no meltwater present in the new profile
    new_meltflag = np.zeros(cell["vert_grid"])

    # Update cell properties with the new combined profiles
    cell["firn_temperature"] = new_firn_temperature
    cell["rho"] = new_rho
    cell["Lfrac"] = lfrac_new
    cell["Sfrac"] = sfrac_new
    cell["meltflag"] = new_meltflag
    # recall lid depth already accounts for lake depth here
    cell["firn_depth"] += cell["lid_depth"] + cell["lake_depth"]
    cell["vertical_profile"] = new_depth_grid
    # Recalculate saturation based on new Sfrac and Lfrac
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
    cell["lake_refreeze_counter"] = 0

    percolation.calc_saturation(cell, cell["vert_grid"] - 1)

    print("Saturation at point 0 = ", cell["saturation"][0])
    print("Lfrac at point 0 = ", cell["Lfrac"][0])
    print("Sfrac at point 0 = ", cell["Sfrac"][0])
    print("Lake depth = ", cell["lake_depth"])
    # Validate mass conservation
    new_mass = utils.calc_mass_sum(cell)
    try:
        assert abs(new_mass - original_mass) < original_mass / 1000
    except Exception:
        print(f"new mass = {new_mass}, original mass = {original_mass}")
        raise Exception
    pass


def mass_conserving_profile(cell, orig_lid_depth, var="Sfrac"):
    lid_dz = np.full(
        cell["vert_grid_lid"], orig_lid_depth / cell["vert_grid_lid"]
    )
    lake_dz = np.full(
        cell["vert_grid_lake"], cell["lake_depth"] / cell["vert_grid_lake"]
    )
    if var == "Sfrac":
        var_lid = np.concatenate((np.ones(cell["vert_grid_lid"]),
                                  np.zeros(cell["vert_grid_lake"])))
        rho = cell["rho_ice"]

    else:
        var_lid = np.concatenate((np.zeros(cell["vert_grid_lid"]),
                                  np.ones(cell["vert_grid_lake"])))
        rho = cell["rho_water"]

    column_dz = np.full(
        cell["vert_grid"], cell["firn_depth"] / cell["vert_grid"]
    )
    var_column = cell[var]

    # Combine into full profile
    dz_full = np.concatenate((lid_dz, lake_dz, column_dz))
    var_full = np.concatenate((var_lid, var_column))

    # Depth edges of full profile (top at 0)
    z_edges_full = np.concatenate((np.array([0]), np.cumsum(dz_full)))

    # Total depth
    total_depth = np.sum(dz_full)

    num_layers_new = cell["vert_grid"]
    dz_new = np.full(num_layers_new, total_depth / num_layers_new)
    z_edges_new = np.linspace(0, total_depth, num_layers_new + 1)

    # Solid mass per layer in old grid for the given variable
    mass_old = var_full * dz_full * rho

    # Create mass function to integrate
    mass_profile = np.zeros_like(z_edges_full)
    mass_profile[1:] = np.cumsum(mass_old)

    # Interpolate cumulative mass to new layer edges
    mass_interp_edges = np.interp(z_edges_new, z_edges_full, mass_profile)

    # New solid/liquid mass per layer
    mass_new = np.diff(mass_interp_edges)

    # Recover new solid/liquid fraction
    var_new = mass_new / (dz_new * rho)

    return np.clip(var_new, 0, 1)
