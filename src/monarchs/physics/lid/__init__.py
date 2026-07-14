"""
Lid physics subpackage.

Contains the development of frozen lids over melt lakes:
  - lid     : full lid (depth >= 10 cm) heat equation and surface energy balance
  - virtual_lid : thin lid (depth < 10 cm) treated as a single-layer slab

Public API
----------
lid_development          : Evolve a full frozen lid over a lake.
virtual_lid_development  : Evolve a thin virtual lid over a lake.
"""

from monarchs.physics.lid.lid import lid_development
from monarchs.physics.lid.virtual_lid import virtual_lid_development
from monarchs.physics.lid import lid, virtual_lid

__all__ = [
    "lid_development",
    "virtual_lid_development",
    "lid",
    "virtual_lid",
]
