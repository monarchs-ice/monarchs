"""
Functions to handle resetting the model state back to a single column of firn.

If a lid either fully freezes the lake below, or begins to melt from the top
due to the atmospheric conditions changing, we need to combine everything
back into one firn profile so the model can continue. This allows us to
run over multiple melt/freezing cycles without breaking the model.
"""

# disable pylint warnings for broad exceptions as they are needed with Numba
# pylint: disable=broad-exception-raised, raise-missing-from
import numpy as np
from monarchs.core.kernels import kernel
from monarchs.physics import material_properties
from monarchs.physics.firn import percolation
from monarchs.core import utils
from monarchs.core.error_handling import check_for_mass_conservation
from monarchs.physics.firn.regrid_column import conservative_regrid
from monarchs.physics.constants import rho_ice, rho_water, cp_water


@kernel()
def combine_lid_firn(cell, freeze_lake=False, surface_slush=False):
    routine_name = "monarchs.physics.reset_column.combine_lid_firn"

    # ensure that the density is up-to-date
    cell["rho"] = (cell["Sfrac"] * rho_ice) + (cell["Lfrac"] * rho_water)
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

    # Lid + lake + firn as one stack on its true layer thicknesses (dropping
    # any zero-thickness region), so mass and heat are remapped from a single
    # consistent source. The lid is solid ice, the lake liquid water (or ice
    # if freeze_lake), the firn keeps its own fractions and temperature.
    if cell["vert_grid_lid"] > 0 and orig_lid_depth > 0:
        n = cell["vert_grid_lid"]
        lid_dz = np.full(n, orig_lid_depth / n)
        lid_S = np.ones(n)
        lid_L = np.zeros(n)
        lid_T = cell["lid_temperature"].copy()
    else:
        lid_dz = np.zeros(0)
        lid_S = np.zeros(0)
        lid_L = np.zeros(0)
        lid_T = np.zeros(0)

    if cell["vert_grid_lake"] > 0 and cell["lake_depth"] > 0:
        n = cell["vert_grid_lake"]
        lake_dz = np.full(n, lake_depth_eff / n)
        if freeze_lake:
            lake_S = np.ones(n)
            lake_L = np.zeros(n)
        else:
            lake_S = np.zeros(n)
            lake_L = np.ones(n)
        lake_T = cell["lake_temperature"].copy()
    else:
        lake_dz = np.zeros(0)
        lake_S = np.zeros(0)
        lake_L = np.zeros(0)
        lake_T = np.zeros(0)

    firn_dz = np.full(cell["vert_grid"], cell["firn_depth"] / cell["vert_grid"])
    dz_full = np.concatenate((lid_dz, lake_dz, firn_dz))
    src_edges = np.concatenate((np.array([0.0]), np.cumsum(dz_full)))
    src_Sfrac = np.concatenate((lid_S, lake_S, cell["Sfrac"]))
    src_Lfrac = np.concatenate((lid_L, lake_L, cell["Lfrac"]))
    src_T = np.concatenate((lid_T, lake_T, cell["firn_temperature"]))

    total_depth = src_edges[-1]
    new_vert_grid = cell["vert_grid"]
    target_edges = np.linspace(0.0, total_depth, new_vert_grid + 1)
    new_depth_grid = np.linspace(0.0, total_depth, new_vert_grid)

    sfrac_new = np.clip(
        conservative_regrid(src_edges, src_Sfrac, target_edges), 0.0, 1.0
    )
    lfrac_new = np.clip(
        conservative_regrid(src_edges, src_Lfrac, target_edges), 0.0, 1.0
    )

    # Temperature by conservative remap of sensible heat content
    # h = cv (T - Tm) [J m^-3], with a fixed reference cp so the regrid
    # conserves sensible enthalpy; T is recovered from the new heat capacity.
    cp_ice_ref = material_properties.cp_ice(273.15)
    src_cv = src_Sfrac * rho_ice * cp_ice_ref + src_Lfrac * rho_water * cp_water
    h_new = conservative_regrid(src_edges, src_cv * (src_T - 273.15), target_edges)
    cv_new = sfrac_new * rho_ice * cp_ice_ref + lfrac_new * rho_water * cp_water
    new_firn_temperature = 273.15 + h_new / np.maximum(cv_new, 1e-6)

    # A layer holding no liquid cannot sit above the melting point.
    for i in range(len(new_firn_temperature)):
        if lfrac_new[i] < 1e-9 and new_firn_temperature[i] > 273.15:
            new_firn_temperature[i] = 273.15

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
    cell["lake_depth"] = 0.0
    if surface_slush:
        cell["melt"] = True
        cell["exposed_water"] = True
        # cell["lake_depth"] = cell["lid_sfc_melt"]

    cell["lid_sfc_melt"] = 0.0
    cell["lake_refreeze_counter"] = 0
    cell["snow_on_lid"] = False
    cell["lid_snow_depth"] = 0.0
    # if we've combined a lid with firn due to melting on the lid surface,
    # then we have melt on the lid surface so the albedo will be reduced
    # compared to if it completely froze the lake

    # get saturation of the new column explicitly rather than interpolating
    # the old values
    percolation.calc_saturation(cell, cell["vert_grid"] - 1)

    # validate mass conservation
    new_mass = utils.calc_mass_sum(cell)
    # use a slightly more relaxed tolerance (1%) for this function
    check_for_mass_conservation(
        cell, original_mass, new_mass, routine_name, tol=original_mass / 100
    )
