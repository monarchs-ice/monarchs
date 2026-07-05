"""
Lateral water movement subpackage.

Public API
----------
move_water : move water laterally across the grid for one model day.
"""

from monarchs.physics.lateral.move_water import move_water

__all__ = ["move_water"]
