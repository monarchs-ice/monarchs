""" """

# TODO - module-level docstring, other docstring
# disable pylint warnings for broad exceptions as they are needed with Numba
# pylint: disable=broad-exception-raised, raise-missing-from
import numpy as np
from monarchs.physics import percolation
from monarchs.core import utils
from monarchs.core.error_handling import check_for_mass_conservation
from monarchs.physics.regrid_column import conservative_regrid

def combine_lid_firn(cell, freeze_lake=False):
    routine_name = 'monarchs.physics.reset_column.combine_lid_firn'
    original_mass = utils.calc_mass_sum(cell)

    if cell["v_lid"]:
        cell["lid_depth"] = cell["v_lid_depth"]
    orig_lid_depth = cell["lid_depth"] + 0.0

    # get effective frozen lake thickness (used only if we freeze)
    lake_depth_eff = (
        cell["lake_depth"] * (cell["rho_water"] / cell["rho_ice"])
        if freeze_lake
        else cell["lake_depth"]
    )

    # old depth grid - lid + lake + firn
    old_depth_grid = np.concatenate(
        (
            np.linspace(0.0, orig_lid_depth, cell["vert_grid_lid"]),
            np.linspace(
                orig_lid_depth,
                orig_lid_depth + lake_depth_eff,
                cell["vert_grid_lake"],
            ),
            np.linspace(
                orig_lid_depth + lake_depth_eff,
                orig_lid_depth + lake_depth_eff + cell["firn_depth"],
                cell["vert_grid"],
            ),
        )
    )

    # new depth grid over combined total
    new_vert_grid = cell["vert_grid"]
    new_depth_grid = np.linspace(
        0.0,
        orig_lid_depth + lake_depth_eff + cell["firn_depth"],
        new_vert_grid,
    )

    # temperature interpolation - fix lake to 273.15 K
    new_firn_temperature = np.interp(
        new_depth_grid,
        old_depth_grid,
        np.concatenate(
            (
                cell["lid_temperature"],
                273.15 * np.ones_like(cell["lake_temperature"]),
                cell["firn_temperature"],
            )
        ),
    )



    # Reconstruct phase fractions with mass conservation
    sfrac_new = mass_conserving_profile(
        cell, orig_lid_depth, var="Sfrac", freeze_lake=freeze_lake
    )
    lfrac_new = mass_conserving_profile(
        cell, orig_lid_depth, var="Lfrac", freeze_lake=freeze_lake
    )
    # Density is derived directly from the new mass-conserved fractions.
    # Interpolating rho directly causes mass errors when layer sizes change.
    new_rho = (sfrac_new * cell["rho_ice"]) + (lfrac_new * cell["rho_water"])
    # update cell with the new combined profiles/values
    cell["firn_temperature"] = new_firn_temperature
    cell["rho"] = new_rho
    cell["Sfrac"] = sfrac_new
    cell["Lfrac"] = lfrac_new
    cell["meltflag"] = np.zeros(cell["vert_grid"])
    cell["firn_depth"] = cell["firn_depth"] + orig_lid_depth + lake_depth_eff
    cell["vertical_profile"] = new_depth_grid

    # reset lake/lid flags and related fields
    cell["lid_depth"] = 0.0
    cell["v_lid"] = False
    cell["lid"] = False
    cell["lake"] = False
    cell["ice_lens"] = True
    cell["ice_lens_depth"] = 0.0
    cell["exposed_water"] = False
    cell["v_lid_depth"] = 0.0
    cell["has_had_lid"] = False
    cell["water"][0] = 0.0
    cell["melt"] = False
    cell["reset_combine"] = True
    cell["virtual_lid_temperature"] = 273.15
    cell["lid_temperature"] = np.ones(cell["vert_grid_lid"]) * 273.15
    cell["lake_temperature"] = np.ones(cell["vert_grid_lake"]) * 273.15
    cell["lid_melt_count"] = 0
    cell["lake_depth"] = 0.0
    cell["lid_sfc_melt"] = 0.0
    cell["lake_refreeze_counter"] = 0

    # get saturation of the new column explicitly rather than interpolating
    # the old values
    percolation.calc_saturation(cell, cell["vert_grid"] - 1)

    # validate mass conservation
    new_mass = utils.calc_mass_sum(cell)
    check_for_mass_conservation(cell, original_mass, new_mass, routine_name)

def mass_conserving_profile(
    cell, orig_lid_depth, var="Sfrac", freeze_lake=False
):

    if cell["vert_grid_lid"] > 0 and orig_lid_depth > 0:
        lid_dz = np.full(cell["vert_grid_lid"], orig_lid_depth / cell["vert_grid_lid"])
    else:
        lid_dz = np.array([])

    if cell["vert_grid_lake"] > 0 and cell["lake_depth"] > 0:
        if freeze_lake:
            lake_depth_eff = cell["lake_depth"] * (cell["rho_water"] / cell["rho_ice"])
        else:
            lake_depth_eff = cell["lake_depth"]
        lake_dz = np.full(cell["vert_grid_lake"], lake_depth_eff / cell["vert_grid_lake"])
    else:
        lake_dz = np.array([])

    column_dz = np.full(cell["vert_grid"], cell["firn_depth"] / cell["vert_grid"])

    dz_full = np.concatenate((lid_dz, lake_dz, column_dz))
    source_edges = np.concatenate((np.array([0.0]), np.cumsum(dz_full)))

    if var == "Sfrac":
        val_lid = np.ones(len(lid_dz))
        val_lake = np.ones(len(lake_dz)) if freeze_lake else np.zeros(len(lake_dz))
        val_firn = cell["Sfrac"]
    else: # Lfrac
        val_lid = np.zeros(len(lid_dz))
        val_lake = np.zeros(len(lake_dz)) if freeze_lake else np.ones(len(lake_dz))
        val_firn = cell["Lfrac"]

    source_vals = np.concatenate((val_lid, val_lake, val_firn))

    total_depth = source_edges[-1]
    nz = cell["vert_grid"]
    target_edges = np.linspace(0, total_depth, nz + 1)

    # 4. Regrid
    new_vals = conservative_regrid(source_edges, source_vals, target_edges)

    return np.clip(new_vals, 0.0, 1.0)