"""
Functions related to regridding (interpolation) of the firn column as a
response to melting.
"""

import numpy as np
from monarchs.physics import percolation
from monarchs.core import utils


def _integrate_piecewise_constant(edges, values, z0, z1):
    """
    Integrate layer-by-layer, assuming piecewise-constant values.
    (much like those in our firn column, which are constant between
    layer edges).
    This will yield a value in metres, if values is a fraction (i.e. Sfrac
    or Lfrac), and the depth coordinate is metres. This means that we get
    out the resulting thickness of the solid/liquid column in metres.
    Parameters
    ----------
    edges
    values
    z0
    z1

    Returns
    -------

    """
    z0 = max(z0, edges[0])
    z1 = min(z1, edges[-1])
    if z1 <= z0:
        return 0.0
    i0 = np.searchsorted(edges, z0, side="right") - 1
    i1 = np.searchsorted(edges, z1, side="left") - 1
    total = 0.0
    for i in range(i0, i1 + 1):
        seg_start = max(z0, edges[i])
        seg_end = min(z1, edges[i + 1])
        dz = seg_end - seg_start
        if dz > 0:
            total += values[i] * dz
    return total


def conservative_regrid(old_edges, old_values, new_edges):
    """
    Mass/volume-conserving interpolation for piecewise-constant values.

    Effectively, we take the integral of the old values, and interpolate that
    to the new grid, which ensures that the integral is conserved.
    """
    old_dz = np.diff(old_edges)
    old_mass_cum = np.zeros(len(old_edges))
    old_mass_cum[1:] = np.cumsum(old_values * old_dz)

    # interpolate the cumulative sum of the masses onto the new grid.
    # our function is piecewise constant, so this interpolation is exact
    new_values = np.zeros(len(new_edges) - 1)
    for i in range(len(new_values)):
        z0, z1 = new_edges[i], new_edges[i + 1]
        m0 = np.interp(z0, old_edges, old_mass_cum)
        m1 = np.interp(z1, old_edges, old_mass_cum)
        dz = z1 - z0
        new_values[i] = (m1 - m0) / dz if dz > 0 else 0.0
    return new_values


def merge_cells_into_lake(cell, height_change):
    """
    Check for and merge all fully liquid firn layers. e.g. if we
    melt through a solid layer, and reach a series of layers with
    Lfrac = 1.0, then add the height of all of these layers to
    height_change. These will then later be removed from the firn
    column and added to the lake depth.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    height_change: float
        Change in the firn height as a result of melting. [m]
    Returns
    -------
    height_change: float
        Updated height_change after merging fully liquid layers into the lake.
    """

    nz = int(cell["vert_grid"])
    old_depth = float(cell["firn_depth"])
    dz = old_depth / nz
    is_full_liq = cell['Lfrac'] >= 0.95
    # how many *full* layers have we melted through already?
    layers_melted = int(
        np.floor(height_change / (cell["firn_depth"] / cell["vert_grid"]))
    )

    # how many full layers have melted through? remove that many from
    # is_full_liq for the check
    is_full_liq = is_full_liq[layers_melted:]

    # count contiguous True values from the surface downward, after
    # accounting for any cells melted through.
    # in the edge case that everything is liquid, we have a problem...
    # but one that we will let a later part of the code handle.
    if np.all(is_full_liq):
        n_removed = nz
    # else, find where we get our first False value.
    else:
        # index of first False → number of contiguous True from top
        first_false = np.argmax(~is_full_liq)
        n_removed = int(first_false) if is_full_liq[0] else 0

    # If we haven't merged any layers, just return our initial height change
    # and regrid as normal
    if n_removed <= 0:
        return height_change
    # Otherwise, work out how many layers we have either melted or merged,
    # and set this as our new height change.
    height_change = (n_removed + layers_melted) * dz
    print('Combining', n_removed, 'fully liquid layers into lake.')
    cell['lake_boundary_change'] += n_removed * dz
    return height_change


def regrid_after_melt(cell, height_change, lake=False):
    """
    After melting occurs, subtract the amount of melting from the firn height,
    convert it into meltwater, and interpolate the entire column to the new
    vertical profile accounting for this height change.
    This meltwater is either converted into surface liquid water fraction,
    or if there is a lake, into lake height.


    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    height_change : float
        Change in the firn height as a result of melting. [m]
    lake : bool, optional
        Flag to determine whether a lake is present or not. This is contained
        here so that we can re-use the bulk of this algorithm, but with some
        small changes to reflect the different situation that occurs when a
        lake is present.

    Returns
    -------
    None
    """
    # check if we need to merge any fully liquid layers into the lake first, if
    # one exists.
    if lake:
        height_change = merge_cells_into_lake(cell, height_change)
    if height_change <= 0.0:
        return

    mass_before = utils.calc_mass_sum(cell)

    old_depth = float(cell["firn_depth"])
    nz = int(cell["vert_grid"])

    assert (
        height_change < old_depth
    ), "Height change must be less than the column depth."

    old_edges = np.linspace(0.0, old_depth, nz + 1)

    rem_S_thick = _integrate_piecewise_constant(
        old_edges, cell["Sfrac"], 0.0, height_change
    )
    rem_L_thick = _integrate_piecewise_constant(
        old_edges, cell["Lfrac"], 0.0, height_change
    )

    water_height_to_add = (
        cell["rho_ice"] / cell["rho_water"]
    ) * rem_S_thick + rem_L_thick

    new_depth = old_depth - height_change
    new_edges = np.linspace(height_change, old_depth, nz + 1)
    if height_change > old_depth/nz:
        print('Warning: height change greater than one layer thickness.')
        print('height_change =', height_change)

    cell["Sfrac"] = conservative_regrid(old_edges, cell["Sfrac"], new_edges)
    cell["Lfrac"] = conservative_regrid(old_edges, cell["Lfrac"], new_edges)
    cell["firn_temperature"] = conservative_regrid(
        old_edges, cell["firn_temperature"], new_edges
    )
    cell["firn_depth"] = new_depth

    dz_new = new_depth / nz
    # max pore space available in the top cell (as height)
    top_cell_space = max(
        0.0, (1.0 - (cell["Sfrac"][0] + cell["Lfrac"][0])) * dz_new
    )

    take = min(top_cell_space, water_height_to_add)
    cell["Lfrac"][0] += take / dz_new
    overflow = water_height_to_add - take

    if lake and overflow > 0.0:
        cell["lake_depth"] += overflow
        overflow = 0.0

    if not lake and overflow > 0.0:
        cell["Lfrac"][0] += overflow / dz_new
        percolation.calc_saturation(cell, 0, end=True)

    # find points where we have oversaturation and fix
    oversat = np.where((cell["Lfrac"] + cell["Sfrac"]) > 1.0)[0]
    if len(oversat) > 0:
        percolation.calc_saturation(cell, int(oversat[-1]), end=True)

    cell["vertical_profile"] = np.linspace(0.0, new_depth, nz)

    mass_after = utils.calc_mass_sum(cell)
    tol = max(1e-2, 1e-10 * mass_before)
    assert abs(mass_after - mass_before) < tol, (mass_after, mass_before)
