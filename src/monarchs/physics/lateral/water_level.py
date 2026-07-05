"""
Function for determining the water level of a gridcell based on the state.
"""

import numpy as np
from monarchs.core.kernels import kernel


@kernel()
def update_water_level(cell):
    """
    Determine the water level of a single cell, so we can determine where water
    flows laterally to and from.
    This is determined by the presence of lakes, lids or ice lenses within the
    firn column.
    If there is no lake, lid or ice lens, then the entire grid cell is free for
    water to move into it.
    If there is no lake or lid, but there is an ice lens then the water level
    is the level of the highest point at which we have saturated firn.
    If a cell has a lake, but no lid, then the water level is the height of
    that lake.
    Finally, if we have a lid, we set the water level to be arbitrarily high,
    as we are not currently interested in water flow from a frozen lid as it is
    too complicated to model.
    Called in <move_water>.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.

    Returns
    -------

    """

    if not cell["valid_cell"]:
        # Invalid cell - not interested so set water level to something
        # unreasonably high
        cell["water"][:] = 0
        cell["saturation"][:] = 0
        cell["lake_depth"] = 0
        cell["Lfrac"][:] = 0
        cell["water_level"] = 1e11
        return

    elif not cell["lake"] and not cell["lid"]:
        if cell["ice_lens"]:
            # We find the water level by the topmost bit of saturated firn
            # above the ice lens.
            if not np.any(cell["saturation"][: cell["ice_lens_depth"] + 1] > 0):
                top_saturation_depth = cell["ice_lens_depth"]
            else:
                top_saturation_depth = np.argmax(
                    cell["saturation"][: cell["ice_lens_depth"] + 1] > 0
                )
            cell["water_level"] = cell["vertical_profile"][::-1][top_saturation_depth]

        # Otherwise, water is free to percolate all the way to the bottom,
        # so it doesn't move laterally from here.
        else:
            cell["water_level"] = 0

        # cell.water is only used for the lateral movement. So we first need to
        # update it based on Lfrac,
        # which is used in the rest of MONARCHS.
        cell["water"] = cell["Lfrac"] * (cell["firn_depth"] / cell["vert_grid"])

    # Add lake depth into water for the purposes of moving it around if a lake
    # is present.
    # TODO - previous defensive call didnt ever actually trigger, may need to revisit?
    elif cell["lake"] and not cell["lid"]:
        cell["water_level"] = cell["lake_depth"] + cell["firn_depth"]
        # Determine the water level from the water on top + the firn depth.
        cell["water"] = cell["Lfrac"] * (cell["firn_depth"] / cell["vert_grid"])

    elif cell["lid"]:
        # shouldn't matter, as water can't move from a lid
        cell["water_level"] = (
            cell["lid_depth"] + cell["firn_depth"] + cell["lake_depth"]
        )
        cell["water"] = cell["Lfrac"] * (cell["firn_depth"] / cell["vert_grid"])
