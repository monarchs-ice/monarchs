"""
Functions for determining much water is available to move from a grid cell,
and how much water can move out of the model.
"""

import numpy as np
from monarchs.core.kernels import kernel
from monarchs.core.utils import find_nearest
from monarchs.physics.lateral.directions import _DIR_COL
from monarchs.physics.constants import rho_ice, rho_water


@kernel()
def water_fraction(cell, m, timestep, direction, flow_speed_scaling=1.0):
    """
    Determine the fraction of water that is allowed to move through the solid
    firn, dependent on its density.
    Called in <move_water>.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    m : float
        difference in heights between this cell and the cell the water is
        moving to
    timestep : int
        time in which the water can move - default 3600 (set by user in
        runscript) [s]
    direction : int
        Direction index (0=NW, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W).
        Even indices are diagonal; odd indices are cardinal.
        Used to determine the effective cell size for flow calculations.

    Returns
    -------
    water_frac : np.ndarray or float
        Fraction of the total water that is allowed to move. Either a single
        number if a lake is present, or a Numpy array of length cell.vert_grid
        if not. M is biggest height difference
    """
    # cell size is determined by the direction - if in x or y then just the
    # difference between centres of cells, else the hypotenuse of the
    # respective triangle
    if direction % 2 == 0:  # diagonal
        cell_size = np.sqrt(cell["size_dx"] ** 2 + cell["size_dy"] ** 2)
    elif _DIR_COL[direction] == 0:  # N or S
        cell_size = cell["size_dy"]
    else:  # E or W
        cell_size = cell["size_dx"]
    # TODO - should cell_size be divided by 2? Since water is moving from the
    #      - centre of the cell. But if we consider that it is moving from
    #      - centre to centre, is probably fine assuming size of adjacent cells
    #      - is the same, but could implement a fix at some point.

    big_pi = -2.53 * 10**-10  # hydraulic permeability (m^2)
    eta = 1.787 * 10**-3  # viscosity(Pa/s)
    cell["rho"] = cell["Sfrac"] * rho_ice + cell["Lfrac"] * rho_water

    # This block ensures that u is not greater than 1, which would be
    # unphysical.
    # if lake, then we only are interested in rho_water so u is a float
    # (not an array)
    if cell["lake"]:
        # if in a lake all water moves
        water_frac = np.array([1.0])
    # otherwise we look at the density of a specific point in the firn so u
    # is an array
    else:
        cell_density = cell["rho"]
        u = big_pi / eta * (m / cell_size) * cell_density * -9.8  # flow speed (m/s)
        water_frac = u * timestep / cell_size
        water_frac[np.where(water_frac > 1)] = 1

        # JE - added flow_speed_scaling variable here.
        water_frac = water_frac * flow_speed_scaling
    water_frac = np.clip(water_frac, a_min=0, a_max=1)
    return water_frac


@kernel()
def calc_available_water_lake(cell, water_frac, split, neighbour_cell, outflow=False):
    if outflow:  # outflow case, so most of the calculation is irrelevant
        water_to_move = float(
            cell["lake_depth"]
        )  # if flowing out, whole lake drains away.
        return water_to_move, 0, 0, 0

    # Otherwise, proceed as normal
    # enough water should move such that we equalise water level,
    # accounting for moving to multiple cells.
    # [0] since we want a float not an array for Numba
    water_to_move = (
        water_frac
        * ((cell["water_level"] - neighbour_cell["water_level"]) / (split + 1))
    )[0]

    # Ensure we have enough water to fill our "quota" -
    # else everything goes but no more.
    # multiply by split since we want the total water that can move.
    # water frac is always 1 if the central cell is a lake
    # so doesn't pose a problem
    if cell["lake_depth"] < water_to_move * (split + 1):
        water_to_move = float(cell["lake_depth"] / (split + 1))
    if water_to_move < 0:
        if water_to_move > -1e-8:
            water_to_move = 0
        else:
            print("Water to move = ", water_to_move)
            raise ValueError("Water to move from lake is less than 0")
    return water_to_move, 0, 0, 0


@kernel()
def calc_available_water_ice_lens(
    cell, water_frac, split, neighbour_cell, outflow=False
):
    """
    Determine the amount of water that can be moved from one cell to the next.

    Parameters
    ----------
    cell
    water_frac
    split
    neighbour_cell
    outflow

    Returns
    -------

    """
    # central cell - (split * water_to_move) = outside cell + water_to_move
    # > water_to_move * split = central_cell - outside_cell
    # > water_to_move = central_cell - outside_cell / (split)
    # only the case if outside cells are equal so far - maybe change in future
    # We need to move enough water for the water levels to equalise.
    # If lake depth < difference between water levels, then all needs to move.

    cell["vertical_profile"] = np.linspace(0, cell["firn_depth"], cell["vert_grid"])
    # vertical profile measured from the surface downward (index 0 = bottom)
    vp = cell["vertical_profile"][::-1]

    if outflow:
        lowest_water_level = vp[cell["ice_lens_depth"]]
        move_from_index = find_nearest(vp, lowest_water_level)
        water_to_move = np.sum(cell["water"][: move_from_index + 1])
        return water_to_move, 0, 0, 0

    # Determine the highest point of the water - this is needed later
    top_saturation_level = np.argmax(
        cell["saturation"][: cell["ice_lens_depth"] + 1] > 0
    )
    # The water moves from the lowest level it can physically do so
    # i.e. the ice lens depth:
    if vp[cell["ice_lens_depth"]] > neighbour_cell["water_level"]:
        lowest_water_level = vp[cell["ice_lens_depth"]]
    else:  # or the midpoint of the water depths
        lowest_water_level = vp[
            find_nearest(
                vp,
                (cell["water_level"] + neighbour_cell["water_level"]) / (split + 1),
            )
        ]
    move_from_index = find_nearest(vp, lowest_water_level)
    water_to_move = (
        water_frac[move_from_index]
        * (cell["water_level"] - neighbour_cell["water_level"])
        / (split + 1)
    )

    if water_to_move < 0:
        if water_to_move > -1e-8:
            water_to_move = 0
        else:
            raise ValueError(f"Negative water in move_to_ice_lens: {water_to_move}")
    # If saturated firn doesn't hold enough water to fill the "quota",
    # then all of it moves but no more
    # +1 so we include the one where we move from
    if np.sum(cell["water"][: move_from_index + 1]) / (split + 1) < water_to_move:
        # i.e. if more water is allowed to move than we have in the cell
        water_to_move = np.sum(cell["water"][: move_from_index + 1]) / (split + 1)

    return (
        water_to_move,
        lowest_water_level,
        move_from_index,
        top_saturation_level,
    )


@kernel()
def calc_catchment_outflow(
    cell, temporary_cell, water_frac, split, outflow_proportion=1.0
):
    """
    Calculate the amount of water that can be moved from a cell to the land
    (i.e. outside of the model domain).

    Parameters
    ----------
    cell
    temporary_cell
    water_frac
    split

    Returns
    -------

    """
    # call cell twice as "cell" is the neighbour cell in this instance -
    # it isn't actually used but lets us jit-compile with Numba
    if cell["lake"]:
        water_to_move, _, __, ___ = calc_available_water_lake(
            cell, water_frac, split, cell, outflow=True
        )
    elif cell["ice_lens"] and not cell["lid"]:
        (
            water_to_move,
            lowest_water_level,
            move_from_index,
            top_saturation_level,
        ) = calc_available_water_ice_lens(cell, water_frac, split, cell, outflow=True)
    else:
        raise ValueError(
            "Trying to calculate outflow from a cell that is neither a "
            "lake or has an ice lens"
        )
    # Whether we are moving from a lake or ice lens, scale by the outflow
    # proportion.
    water_to_move = water_to_move * outflow_proportion
    water_out = 0.0
    # either: remove water from the lake directly.
    if cell["lake"] and not cell["lid"]:
        current_lake_depth = temporary_cell["lake_depth"]

        if cell["lake_depth"] < water_to_move:
            water_out = current_lake_depth
            temporary_cell["lake_depth"] = 0
        else:
            temporary_cell["lake_depth"] -= water_to_move
            water_out = water_to_move
        if temporary_cell["lake_depth"] < 0:
            # account for rounding errors
            if temporary_cell["lake_depth"] < -1e-12:
                raise ValueError("Lake depth has gone below 0")
            temporary_cell["lake_depth"] = 0
    # otherwise, loop through the column and remove water from it,
    # going from the top.
    else:
        for _l in range(0, cell["ice_lens_depth"] + 1):
            layer_share = cell["water"][_l] / (split + 1)
            current_actual = temporary_cell["water"][_l]

            if layer_share > current_actual:
                layer_share = current_actual

            if layer_share > water_to_move:
                amount_to_remove = water_to_move
                water_to_move = 0
            else:
                amount_to_remove = layer_share
                water_to_move -= layer_share
            # actually remove the water
            temporary_cell["water"][_l] -= amount_to_remove
            water_out += amount_to_remove

            if temporary_cell["water"][_l] <= 1e-12:
                temporary_cell["water"][_l] = 0
                temporary_cell["saturation"][_l] = 0
            # stop looping if we've flowed all the water
            if water_to_move <= 0:
                break
    return water_out
