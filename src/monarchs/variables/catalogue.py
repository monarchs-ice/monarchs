"""
The MONARCHS variable catalogue.

This is where all MONARCHS model variables (i.e. everything that you can
use as an argument to cell['<argument>']) are defined. So if you need
to add any new model variables, just define them here, and then you can
use it in the code freely.

Everything is defined here as a ``Variable``, which is a class defined
in ``definitions.py`` describing the main fields that a model variable needs
or wants in order to both a) run the code and b) be useful as an output.

To add a variable, add a row (see the other Variables for how to do this!)
and to remove it just delete it from here (and wherever it is used in the code).

i.e. use the Variable class and populate the fields below (keyword-only)
    Variable(name=..., dtype=..., dim=..., default_value=..., units=..., long_name=..., group=..., output=...)

The fields are:
    name        str    field name used in the code, e.g. "lake_depth"

    dtype       type   FLOAT, INT or BOOL, defines the datatype, needed for Numba as
                       this requires static typing

    dim         Dim    SCALAR | FIRN | LAKE | LID | DIRECTIONS
                       (SCALAR = one value per cell; FIRN/LAKE/LID = a profile
                        over that region's layers; DIRECTIONS = 8 cardinal dirs)

    default_value        the initial value. One of:
                         * a constant            e.g. 0.0, False, 273.15
                         * INPUT                 determined from the initial values
                                                 e.g. firn properties from a DEM
                         * a function            for values computed from grid
                                                 sizes or inputs - defined in
                                                 initial_value_functions.py and
                                                 referenced here by name

    units       str    Units in CF format, e.g. "m", "K", "kg m-3",
                        "1" (for dimensionless quantities)

    long_name   str    human-readable name, some variables already have this
                       implicitly (e.g. "lake_depth"), but can format more
                       nicely for say plotting e.g. "Lake depth"

    group       str    broad variable type, for organising the documentation
                       e.g. ("firn", "lake", ...)

    output      bool   whether the variable is designed for output or not,
                       e.g. if it is a diagnostic probably true, if an internal
                       counter probably not

    description str    Detailed description of the variable - i.e. what you would
                       want to read if you were reading a manual describing what
                       the thing does! This should describe intent if it is not
                       obvious why a variable is needed, and ideally where it is
                       actually used.

    Only ``name``, ``dtype`` and ``dim`` are required as these are what is needed
    by the actual code. The other variables are entirely for documentation and
    provenance/metadata purposes.

An example of this is:

Variable(name="firn_depth", dtype=FLOAT, dim=SCALAR, default_value=INPUT, units="m",
        long_name="Firn column depth",
        group="firn",
        description="Total depth of the firn column as represented in the model."),
"""

import numpy as np

from monarchs.variables.definitions import (
    BOOL,
    DIRECTIONS,
    FIRN,
    FLOAT,
    INPUT,
    INT,
    LAKE,
    LID,
    SCALAR,
    Variable,
)
from monarchs.variables.initial_value_functions import (
    ice_lens_below_column,
    n_firn,
    n_lake,
    n_lid,
    sfrac_from_rho,
    vertical_profile,
)

# catalogue is spaced by group
CATALOGUE = [
    # Fixed values - stuff that is immutable after a run starts
    Variable(
        name="column",
        dtype=INT,
        dim=SCALAR,
        default_value=INPUT,
        long_name="Grid column index",
        group="fixed values",
    ),
    Variable(
        name="row",
        dtype=INT,
        dim=SCALAR,
        default_value=INPUT,
        long_name="Grid row index",
        group="fixed values",
    ),
    Variable(
        name="vert_grid",
        dtype=INT,
        dim=SCALAR,
        default_value=n_firn,
        long_name="Firn layer count",
        group="fixed values",
    ),
    Variable(
        name="vert_grid_lake",
        dtype=INT,
        dim=SCALAR,
        default_value=n_lake,
        long_name="Lake layer count",
        group="fixed values",
    ),
    Variable(
        name="vert_grid_lid",
        dtype=INT,
        dim=SCALAR,
        default_value=n_lid,
        long_name="Lid layer count",
        group="fixed values",
    ),
    Variable(
        name="turbulent_mixing_substep",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=60.0,
        units="s",
        long_name="Lake turbulent-mixing substep",
        description="Substepping for the turbulent mixing routine. Check the description in the "
        "config schema for more details.",
        group="fixed values",
        output=False,
    ),
    Variable(
        name="lat",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=np.nan,
        units="degrees_north",
        long_name="Latitude",
        group="fixed values",
    ),
    Variable(
        name="lon",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=np.nan,
        units="degrees_east",
        long_name="Longitude",
        group="fixed values",
    ),
    Variable(
        name="size_dx",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=1000.0,
        units="m",
        long_name="Cell size, east-west",
        group="fixed values",
    ),
    Variable(
        name="size_dy",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=1000.0,
        units="m",
        long_name="Cell size, north-south",
        group="fixed values",
    ),
    Variable(
        name="valid_cell",
        dtype=BOOL,
        dim=SCALAR,
        default_value=True,
        long_name="Cell runs model physics (flag)",
        group="fixed values",
    ),
    # Firn column variables
    Variable(
        name="firn_depth",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=INPUT,
        units="m",
        long_name="Firn column total depth",
        group="firn",
    ),
    Variable(
        name="vertical_profile",
        dtype=FLOAT,
        dim=FIRN,
        default_value=vertical_profile,
        units="m",
        long_name="Depth of each firn layer",
        group="firn",
    ),
    Variable(
        name="firn_temperature",
        dtype=FLOAT,
        dim=FIRN,
        default_value=INPUT,
        units="K",
        long_name="Firn column temperature",
        group="firn",
    ),
    Variable(
        name="rho",
        dtype=FLOAT,
        dim=FIRN,
        default_value=INPUT,
        units="kg m-3",
        long_name="Firn density",
        group="firn",
    ),
    Variable(
        name="Sfrac",
        dtype=FLOAT,
        dim=FIRN,
        default_value=sfrac_from_rho,
        long_name="Solid (ice) volume fraction",
        group="firn",
    ),
    Variable(
        name="Lfrac",
        dtype=FLOAT,
        dim=FIRN,
        default_value=0.0,
        long_name="Liquid (water) volume fraction",
        group="firn",
    ),
    Variable(
        name="meltflag",
        dtype=FLOAT,
        dim=FIRN,
        default_value=0.0,
        long_name="Meltwater present at layer (flag)",
        group="firn",
    ),
    Variable(
        name="saturation",
        dtype=FLOAT,
        dim=FIRN,
        default_value=0.0,
        long_name="Layer saturated (flag)",
        group="firn",
    ),
    Variable(
        name="ice_lens",
        dtype=BOOL,
        dim=SCALAR,
        default_value=False,
        long_name="Ice lens present (flag)",
        group="firn",
    ),
    Variable(
        name="ice_lens_depth",
        dtype=INT,
        dim=SCALAR,
        default_value=ice_lens_below_column,
        long_name="Layer index of highest ice lens",
        group="firn",
    ),
    # Surface properties and flags
    Variable(
        name="albedo",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=0.0,
        long_name="Surface albedo",
        group="surface",
    ),
    Variable(
        name="melt",
        dtype=BOOL,
        dim=SCALAR,
        default_value=False,
        long_name="Surface melt this step (flag)",
        group="surface",
    ),
    Variable(
        name="exposed_water",
        dtype=BOOL,
        dim=SCALAR,
        default_value=False,
        long_name="Exposed surface water (flag)",
        group="surface",
    ),
    Variable(
        name="total_melt",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=0.0,
        units="m",
        long_name="Cumulative melt depth",
        group="surface",
    ),
    # Lake variables
    Variable(
        name="lake",
        dtype=BOOL,
        dim=SCALAR,
        default_value=False,
        long_name="Lake present (flag)",
        group="lake",
    ),
    Variable(
        name="lake_depth",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=0.0,
        units="m",
        long_name="Melt lake depth",
        group="lake",
    ),
    Variable(
        name="lake_temperature",
        dtype=FLOAT,
        dim=LAKE,
        default_value=273.15,
        units="K",
        long_name="Lake temperature profile",
        group="lake",
    ),
    # Frozen lid variables
    Variable(
        name="lid",
        dtype=BOOL,
        dim=SCALAR,
        default_value=False,
        long_name="Frozen lid present (flag)",
        group="lid",
    ),
    Variable(
        name="lid_depth",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=0.0,
        units="m",
        long_name="Frozen lid depth",
        group="lid",
    ),
    Variable(
        name="lid_temperature",
        dtype=FLOAT,
        dim=LID,
        default_value=273.15,
        units="K",
        long_name="Frozen lid temperature profile",
        group="lid",
    ),
    Variable(
        name="v_lid",
        dtype=BOOL,
        dim=SCALAR,
        default_value=False,
        long_name="Virtual lid present (flag)",
        group="lid",
    ),
    Variable(
        name="v_lid_depth",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=0.0,
        units="m",
        long_name="Virtual lid depth",
        group="lid",
    ),
    Variable(
        name="virtual_lid_temperature",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=273.15,
        units="K",
        long_name="Virtual lid temperature",
        group="lid",
    ),
    Variable(
        name="has_had_lid",
        dtype=BOOL,
        dim=SCALAR,
        default_value=False,
        long_name="Lid present this cycle (flag)",
        group="lid",
    ),
    Variable(
        name="lid_sfc_melt",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=0.0,
        units="m",
        long_name="Tracked lid surface melt",
        group="lid",
    ),
    Variable(
        name="lid_snow_depth",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=0.0,
        units="m",
        long_name="Snow depth on the lid",
        group="lid",
    ),
    Variable(
        name="snow_on_lid",
        dtype=INT,
        dim=SCALAR,
        default_value=0,
        long_name="Snow-on-lid state (0/1/2)",
        group="lid",
    ),
    # Lateral flow variables
    Variable(
        name="water",
        dtype=FLOAT,
        dim=FIRN,
        default_value=0.0,
        units="m",
        long_name="Liquid water depth per layer (lateral flow)",
        group="lateral",
    ),
    Variable(
        name="water_level",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=0.0,
        units="m",
        long_name="Water-table height for lateral flow",
        group="lateral",
    ),
    Variable(
        name="water_direction",
        dtype=INT,
        dim=DIRECTIONS,
        default_value=0,
        long_name="Lateral outflow direction (0=NW..7=W)",
        group="lateral",
    ),
    # Diagnostics
    Variable(
        name="firn_boundary_change",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=0.0,
        units="m",
        long_name="Firn boundary change this day",
        group="diagnostic",
    ),
    Variable(
        name="lake_boundary_change",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=0.0,
        units="m",
        long_name="Lake boundary change this day",
        group="diagnostic",
    ),
    Variable(
        name="lid_boundary_change",
        dtype=FLOAT,
        dim=SCALAR,
        default_value=0.0,
        units="m",
        long_name="Lid boundary change this day",
        group="diagnostic",
    ),
    # Internal counters
    # n.b. these are typically with output=False as we probably don't care to write them out
    Variable(
        name="melt_hours",
        dtype=INT,
        dim=SCALAR,
        default_value=0,
        units="h",
        long_name="Cumulative surface-melt hours",
        group="counter",
        output=False,
    ),
    Variable(
        name="lid_melt_count",
        dtype=INT,
        dim=SCALAR,
        default_value=0,
        long_name="Lid melt-step counter",
        group="counter",
        output=False,
    ),
    Variable(
        name="lake_refreeze_counter",
        dtype=INT,
        dim=SCALAR,
        default_value=0,
        long_name="Lake refreeze counter",
        group="counter",
        output=False,
    ),
    Variable(
        name="exposed_water_refreeze_counter",
        dtype=INT,
        dim=SCALAR,
        default_value=0,
        long_name="Exposed-water refreeze counter",
        group="counter",
        output=False,
    ),
    Variable(
        name="t_step",
        dtype=INT,
        dim=SCALAR,
        default_value=0,
        long_name="Timestep within the current day",
        group="counter",
        output=False,
    ),
    Variable(
        name="day",
        dtype=INT,
        dim=SCALAR,
        default_value=0,
        long_name="Model day",
        group="counter",
        output=False,
    ),
    Variable(
        name="visit_count",
        dtype=INT,
        dim=SCALAR,
        default_value=0,
        long_name="Times this cell has been visited",
        group="counter",
        output=False,
    ),
    Variable(
        name="reset_combine",
        dtype=BOOL,
        dim=SCALAR,
        default_value=False,
        long_name="Lid/firn just combined (flag)",
        group="internal",
        output=False,
        description="Flag indicating whether a lid/lake/firn system has reached a trigger that combines everything "
        "back into a single firn column.",
    ),
    Variable(
        name="error_flag",
        dtype=BOOL,
        dim=SCALAR,
        default_value=False,
        long_name="Cell hit an error state (flag)",
        group="internal",
        output=False,
    ),
]
