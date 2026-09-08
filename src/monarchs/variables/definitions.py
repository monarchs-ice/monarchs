"""
Definitions of what a variable is in terms of what the model needs.
Each variable needs to have some specific fields in order for the model
to use them, e.g. their dtype, shape and name.

We also define a class used to handle some edge cases - e.g. variables
that are defined based on *other* model variables, e.g. Sfrac which is
defined by the density.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# Define specific datatypes based on descriptors
# e.g. defining dtype = "FLOAT" goes to np.float64
FLOAT = np.float64
INT = np.int32
BOOL = np.bool_
# Default value for variables that are defined based on
# other variables. See initial_value_functions.py for
# examples of these.
INPUT = object()


class Dim(Enum):
    """
    Dimensions of a variable on the model grid.
    """

    SCALAR = "scalar"  # one value per cell
    FIRN = "firn"  # a profile in the firn column (vertical_points_firn)
    LAKE = "lake"  # a profile in the lake column (vertical_points_lake)
    LID = "lid"  # a profile in the lid column  (vertical_points_lid)
    # used only for water direction, 8 cardinal directions
    DIRECTIONS = "directions"


SCALAR, FIRN, LAKE, LID, DIRECTIONS = (
    Dim.SCALAR,
    Dim.FIRN,
    Dim.LAKE,
    Dim.LID,
    Dim.DIRECTIONS,
)


@dataclass(frozen=True, kw_only=True)
class Variable:
    """
    Defines a MONARCHS model variable.

    A variable needs to have these fields in order to work with every part of
    the model. Some of these have default values.
    """

    name: str
    dtype: type
    dim: Dim = SCALAR
    #: a constant, the INPUT sentinel, or a callable ``f(ctx) -> value``
    default_value: Any = 0
    # CF-style units string ("m", "K", "1", ...)
    # https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch03.html
    units: str = "1"
    # human-readable name for output/plots
    long_name: str = ""  # long name (as in CF)
    description: str = ""  # detailed description of the variable
    group: str = "state"  # grouping, for organisation/readability
    output: bool = True  # outputs metadata if going into the output netCDF


@dataclass
class InitContext:
    """Information a computed default (see ``initialisers.py``) may need."""

    num_rows: int
    num_cols: int
    vert_grid: int
    vert_grid_lake: int
    vert_grid_lid: int
    # default_factory ensures that the input dicts
    # are all separated, so we don't have any
    # issues with mutable defaults
    inputs: dict = field(default_factory=dict)
