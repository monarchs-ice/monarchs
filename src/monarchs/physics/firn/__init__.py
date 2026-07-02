"""
Firn column physics subpackage.

Public API
----------
firn_column      : Single-timestep firn column evolution (heat equation + melt).
percolation      : Meltwater percolation and refreezing within the firn column.
regrid_column    : Conservative regridding after melt or freeze events.
snow_accumulation: Snowfall accumulation on the firn surface.
"""

from monarchs.physics.firn import firn_column
from monarchs.physics.firn import percolation
from monarchs.physics.firn import regrid_column
from monarchs.physics.firn import snow_accumulation

__all__ = [
    "firn_column",
    "percolation",
    "regrid_column",
    "snow_accumulation",
]
