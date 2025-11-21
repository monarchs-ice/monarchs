""" """

# TODO - module-level docstring, other docstring
# disable pylint warnings for broad exceptions as they are needed with Numba
# pylint: disable=broad-exception-raised, raise-missing-from
import numpy as np
from monarchs.physics import percolation
from monarchs.core import utils


def combine_lid_firn(cell, freeze_lake=False):
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

    # density interpolation: lake becomes ice if freezing
    lake_rho_profile = np.ones(cell["vert_grid_lake"]) * (
        cell["rho_ice"] if freeze_lake else cell["rho_water"]
    )
    new_rho = np.interp(
        new_depth_grid,
        old_depth_grid,
        np.concatenate((cell["rho_lid"], lake_rho_profile, cell["rho"])),
    )

    # Reconstruct phase fractions with mass conservation
    sfrac_new = mass_conserving_profile(
        cell, orig_lid_depth, var="Sfrac", freeze_lake=freeze_lake
    )
    lfrac_new = mass_conserving_profile(
        cell, orig_lid_depth, var="Lfrac", freeze_lake=freeze_lake
    )

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
    assert (
        abs(new_mass - original_mass) < original_mass / 1000
    ), f"new mass = {new_mass}, original mass = {original_mass}"


def mass_conserving_profile(
    cell, orig_lid_depth, var="Sfrac", freeze_lake=False
):
    lid_dz = np.full(
        cell["vert_grid_lid"], orig_lid_depth / cell["vert_grid_lid"]
    )

    # use ice-equivalent thickness when freezing the lake (water expands)
    if freeze_lake:
        lake_depth_eff = cell["lake_depth"] * (
            cell["rho_water"] / cell["rho_ice"]
        )
    else:
        lake_depth_eff = cell["lake_depth"]
    lake_dz = np.full(
        cell["vert_grid_lake"], lake_depth_eff / cell["vert_grid_lake"]
    )

    column_dz = np.full(
        cell["vert_grid"], cell["firn_depth"] / cell["vert_grid"]
    )

    if var == "Sfrac":
        # sfrac: lid is ice, lake is ice if freezing, else zero
        if freeze_lake:
            var_lidlake = np.concatenate(
                (
                    np.ones(cell["vert_grid_lid"]),
                    np.ones(cell["vert_grid_lake"]),
                )
            )
        else:
            var_lidlake = np.concatenate(
                (
                    np.ones(cell["vert_grid_lid"]),
                    np.zeros(cell["vert_grid_lake"]),
                )
            )
        var_column = cell["Sfrac"]
        rho_seg = (
            cell["rho_ice"],
            cell["rho_ice"] if freeze_lake else cell["rho_water"],
            cell["rho_ice"],
        )
    else:
        # lfrac: lid is zero, lake is zero if freezing, else one
        if freeze_lake:
            var_lidlake = np.concatenate(
                (
                    np.zeros(cell["vert_grid_lid"]),
                    np.zeros(cell["vert_grid_lake"]),
                )
            )
        else:
            var_lidlake = np.concatenate(
                (
                    np.zeros(cell["vert_grid_lid"]),
                    np.ones(cell["vert_grid_lake"]),
                )
            )
        var_column = cell["Lfrac"]
        rho_seg = (cell["rho_water"], cell["rho_water"], cell["rho_water"])

    # combine into full profile
    dz_full = np.concatenate((lid_dz, lake_dz, column_dz))
    var_full = np.concatenate((var_lidlake, var_column))

    rho_full = np.concatenate(
        (
            np.full(cell["vert_grid_lid"], rho_seg[0]),
            np.full(cell["vert_grid_lake"], rho_seg[1]),
            np.full(cell["vert_grid"], rho_seg[2]),
        )
    )

    # depth edges of full profile
    z_edges_full = np.concatenate((np.array([0.0]), np.cumsum(dz_full)))

    total_depth = z_edges_full[-1]
    num_layers_new = cell["vert_grid"]
    dz_new = np.full(num_layers_new, total_depth / num_layers_new)
    z_edges_new = np.linspace(0.0, total_depth, num_layers_new + 1)

    # mass per layer in old grid
    mass_old = var_full * dz_full * rho_full

    # cumulative mass at edges
    mass_profile = np.zeros_like(z_edges_full)
    mass_profile[1:] = np.cumsum(mass_old)

    # interpolate cumulative mass to new edges
    mass_interp_edges = np.interp(z_edges_new, z_edges_full, mass_profile)

    # new mass per layer and recover fraction
    mass_new = np.diff(mass_interp_edges)

    rho_var = cell["rho_ice"] if var == "Sfrac" else cell["rho_water"]
    # get our final Sfrac and Lfrac profiles weighted by the correct density
    var_new = mass_new / (dz_new * rho_var)

    return np.clip(var_new, 0.0, 1.0)
