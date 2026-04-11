""" """

# TODO - module-level docstring, other docstring
# disable pylint warnings for broad exceptions as they are needed with Numba
# pylint: disable=broad-exception-raised, raise-missing-from
import numpy as np
from monarchs.physics import percolation
from monarchs.core import utils
from monarchs.core.error_handling import check_for_mass_conservation
from monarchs.physics.regrid_column import conservative_regrid
from monarchs.physics.constants import rho_ice, rho_water

def combine_lid_firn(cell, freeze_lake=False, surface_slush=False):
    routine_name = "monarchs.physics.reset_column.combine_lid_firn"

    # ensure that the density is up-to-date
    cell["rho"] = (cell["Sfrac"] * rho_ice) + (
        cell["Lfrac"] * rho_water
    )
    # add virtual lid to lid if we have one - we are combining either way
    if cell["v_lid"]:
        cell["lid_depth"] = cell["v_lid_depth"]
        cell["v_lid_depth"] = 0
    orig_lid_depth = cell["lid_depth"] + 0.0  # get a copy
    original_mass = utils.calc_mass_sum(cell)
    # get effective frozen lake thickness (used only if we freeze)
    lake_depth_eff = (
        cell["lake_depth"] * (rho_water / rho_ice)
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
                cell["lake_temperature"],
                cell["firn_temperature"],
            )
        ),
    )

    # get new solid and liquid fractions
    sfrac_new = mass_conserving_profile(
        cell, orig_lid_depth, var="Sfrac", freeze_lake=freeze_lake
    )
    lfrac_new = mass_conserving_profile(
        cell, orig_lid_depth, var="Lfrac", freeze_lake=freeze_lake
    )
    # Density is derived directly from the new mass-conserved fractions.
    # Interpolating rho directly causes mass errors when layer sizes change.
    new_rho = (sfrac_new * rho_ice) + (lfrac_new * rho_water)
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
    # Explicitly set lake_depth to 0.0 with proper dtype to eliminate floating-point residuals
    # that could cause divergence between different solver paths (e.g., Newton vs MinPACK).
    _old_lake_depth = cell["lake_depth"]
    cell["lake_depth"] = np.float64(0.0)
    # print(
    #     "LAKE_DEPTH_CHANGE reset_column:combine_reset",
    #     "day", cell["day"], "t_step", cell["t_step"],
    #     "row", cell["row"], "col", cell["column"],
    #     "old", _old_lake_depth, "new", cell["lake_depth"],
    #     "surface_slush", surface_slush,
    # )
    if surface_slush:
        cell["melt"] = True
        cell["exposed_water"] = True
        _old_lake_depth = cell["lake_depth"]
        cell["lake_depth"] = cell["lid_sfc_melt"]
        # print(
        #     "LAKE_DEPTH_CHANGE reset_column:combine_surface_slush",
        #     "day", cell["day"], "t_step", cell["t_step"],
        #     "row", cell["row"], "col", cell["column"],
        #     "old", _old_lake_depth, "new", cell["lake_depth"],
        #     "lid_sfc_melt", cell["lid_sfc_melt"],
        # )
    # # Remove near-zero residual water depths after combine to keep state
    # # deterministic across solver implementations.
    # if (not cell["lake"]) and cell["lake_depth"] <= 1e-5:
    #     cell["lake_depth"] = np.float64(0.0)
    #     # No physically meaningful surface water remains.
    #     cell["exposed_water"] = False
    #     cell["melt"] = False

    cell["lid_sfc_melt"] = 0.0
    cell["lake_refreeze_counter"] = 0
    cell["snow_on_lid"] = False
    cell["lid_snow_depth"] = 0.0
    # if we've combined a lid with firn due to melting on the lid surface,
    # then we have melt on the lid surface so the albedo will be reduced
    # compared to if it completely froze the lake


    # get saturation of the new column explicitly rather than interpolating
    # the old values.
    # NOTE: calc_saturation can write to cell["lake_depth"] if it finds excess
    # water at the surface (v_lev==0 path). After a combine, lake=False and any
    # such write is a phantom caused by tiny Lfrac overshoots from conservative
    # regridding, not real surface water. We zero it out unconditionally here
    # because lake=False has already been set above.
    percolation.calc_saturation(cell, cell["vert_grid"] - 1)
    if not cell["lake"]:
        cell["lake_depth"] = np.float64(0.0)
        # print(
        #     "LAKE_DEPTH_CHANGE reset_column:post_calc_saturation_zero_no_lake",
        #     "day", cell["day"], "t_step", cell["t_step"],
        #     "row", cell["row"], "col", cell["column"],
        #     "new", cell["lake_depth"],
        # )

    # validate mass conservation
    new_mass = utils.calc_mass_sum(cell)
    # use a slightly more relaxed tolerance (1%) for this function
    check_for_mass_conservation(
        cell, original_mass, new_mass, routine_name, tol=original_mass / 100
    )


def mass_conserving_profile(
    cell, orig_lid_depth, var="Sfrac", freeze_lake=False
):
    if cell["vert_grid_lid"] > 0 and orig_lid_depth > 0:
        lid_dz = np.full(
            cell["vert_grid_lid"], orig_lid_depth / cell["vert_grid_lid"]
        )
    else:
        lid_dz = np.zeros(0)

    if cell["vert_grid_lake"] > 0 and cell["lake_depth"] > 0:
        if freeze_lake:
            lake_depth_eff = cell["lake_depth"] * (
                rho_water / rho_ice
            )
        else:
            lake_depth_eff = cell["lake_depth"]
        lake_dz = np.full(
            cell["vert_grid_lake"], lake_depth_eff / cell["vert_grid_lake"]
        )
    else:
        lake_dz = np.zeros(0)

    column_dz = np.full(
        cell["vert_grid"], cell["firn_depth"] / cell["vert_grid"]
    )

    dz_full = np.concatenate((lid_dz, lake_dz, column_dz))
    source_edges = np.concatenate((np.array([0.0]), np.cumsum(dz_full)))

    if var == "Sfrac":
        val_lid = np.ones(len(lid_dz))
        val_lake = (
            np.ones(len(lake_dz)) if freeze_lake else np.zeros(len(lake_dz))
        )
        val_firn = cell["Sfrac"]
    else:  # Lfrac
        val_lid = np.zeros(len(lid_dz))
        val_lake = (
            np.zeros(len(lake_dz)) if freeze_lake else np.ones(len(lake_dz))
        )
        val_firn = cell["Lfrac"]

    source_vals = np.concatenate((val_lid, val_lake, val_firn))

    total_depth = source_edges[-1]
    nz = cell["vert_grid"]
    target_edges = np.linspace(0, total_depth, nz + 1)

    new_vals = conservative_regrid(source_edges, source_vals, target_edges)

    return np.clip(new_vals, 0.0, 1.0)
