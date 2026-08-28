"""
Computed initial values for the few variables whose default is derived from the
grid sizes or the firn profile, rather than a constant.

`ctx` here refers to an instance of the ``InitContext`` class,
from ``definitions.py``. This defines the information that these
variables might need - e.g. the size of the vertical grid for the
ice lens position initialisation.
"""

import numpy as np

from monarchs.physics.constants import rho_ice


def vertical_profile(ctx):
    """Depth of each firn layer below the surface, 0 -> firn_depth."""
    firn_depth = ctx.inputs.get("firn_depth", 0.0)
    return np.moveaxis(np.linspace(0, firn_depth, ctx.vert_grid), 0, -1)


def sfrac_from_rho(ctx):
    """Solid (ice) fraction from density: rho / rho_ice."""
    return ctx.inputs.get("rho", 0.0) / rho_ice


def ice_lens_below_column(ctx):
    """Default ice-lens layer index: one past the bottom (i.e. no lens)."""
    return ctx.vert_grid + 1


def n_firn(ctx):
    return ctx.vert_grid


def n_lake(ctx):
    return ctx.vert_grid_lake


def n_lid(ctx):
    return ctx.vert_grid_lid
