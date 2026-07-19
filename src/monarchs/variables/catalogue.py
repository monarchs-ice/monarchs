"""
The MONARCHS variable catalogue - the single place to add or edit model grid
variables.

Everything the model stores per grid cell is defined here as a
``Variable`` row. To add a variable, add a row, to remove it just delete
it from here (and wherever it is used in the code).

i.e. use the Variable class and populate the fields below
    Variable(name, dtype, dim, default_value, units=..., long_name=..., group=..., output=...)

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
                       the thing does!

    Only ``name``, ``dtype`` and ``dim`` are required.

An example of this is:

Variable("firn_depth", FLOAT, SCALAR, INPUT, units="m",
        long_name="Firn column depth",
        group="firn",
        description="Total depth of the firn column as represented in the model."),
"""

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

# fmt: off
# catalogue is spaced by group
CATALOGUE = [
    # Fixed values - stuff that is immutable after a run starts
    Variable("column", INT, SCALAR, INPUT, long_name="Grid column index", group="fixed values"),
    Variable("row", INT, SCALAR, INPUT, long_name="Grid row index", group="fixed values"),
    Variable("vert_grid", INT, SCALAR, n_firn, long_name="Firn layer count", group="fixed values"),
    Variable("vert_grid_lake", INT, SCALAR, n_lake, long_name="Lake layer count", group="fixed values"),
    Variable("vert_grid_lid", INT, SCALAR, n_lid, long_name="Lid layer count", group="fixed values"),
    Variable("lat", FLOAT, SCALAR, INPUT, units="degrees_north", long_name="Latitude", group="fixed values"),
    Variable("lon", FLOAT, SCALAR, INPUT, units="degrees_east", long_name="Longitude", group="fixed values"),
    Variable("size_dx", FLOAT, SCALAR, 1000.0, units="m", long_name="Cell size, east-west", group="fixed values"),
    Variable("size_dy", FLOAT, SCALAR, 1000.0, units="m", long_name="Cell size, north-south", group="fixed values"),
    Variable("valid_cell", BOOL, SCALAR, True, long_name="Cell runs model physics (flag)", group="fixed values"),

    # Firn column variables
    Variable("firn_depth", FLOAT, SCALAR, INPUT, units="m", long_name="Firn column total depth", group="firn"),
    Variable("vertical_profile", FLOAT, FIRN, vertical_profile, units="m", long_name="Depth of each firn layer", group="firn"),
    Variable("firn_temperature", FLOAT, FIRN, INPUT, units="K", long_name="Firn column temperature", group="firn"),
    Variable("rho", FLOAT, FIRN, INPUT, units="kg m-3", long_name="Firn density", group="firn"),
    Variable("Sfrac", FLOAT, FIRN, sfrac_from_rho, long_name="Solid (ice) volume fraction", group="firn"),
    Variable("Lfrac", FLOAT, FIRN, 0.0, long_name="Liquid (water) volume fraction", group="firn"),
    Variable("meltflag", FLOAT, FIRN, 0.0, long_name="Meltwater present at layer (flag)", group="firn"),
    Variable("saturation", FLOAT, FIRN, 0.0, long_name="Layer saturated (flag)", group="firn"),
    Variable("pore_closure", FLOAT, SCALAR, 0.0, units="kg m-3", long_name="Pore close-off density (unused; see constants)", group="firn"),
    Variable("ice_lens", BOOL, SCALAR, False, long_name="Ice lens present (flag)", group="firn"),
    Variable("ice_lens_depth", INT, SCALAR, ice_lens_below_column, long_name="Layer index of highest ice lens", group="firn"),

    # Surface properties and flags
    Variable("albedo", FLOAT, SCALAR, 0.0, long_name="Surface albedo", group="surface"),
    Variable("melt", BOOL, SCALAR, False, long_name="Surface melt this step (flag)", group="surface"),
    Variable("exposed_water", BOOL, SCALAR, False, long_name="Exposed surface water (flag)", group="surface"),
    Variable("total_melt", FLOAT, SCALAR, 0.0, units="m", long_name="Cumulative melt depth", group="surface"),
    Variable("snow_added", FLOAT, SCALAR, 0.0, units="m", long_name="Snow depth added", group="surface"),

    # Lake variables
    Variable("lake", BOOL, SCALAR, False, long_name="Lake present (flag)", group="lake"),
    Variable("lake_depth", FLOAT, SCALAR, 0.0, units="m", long_name="Melt lake depth", group="lake"),
    Variable("lake_temperature", FLOAT, LAKE, 273.15, units="K", long_name="Lake temperature profile", group="lake"),

    # Frozen lid variables
    Variable("lid", BOOL, SCALAR, False, long_name="Frozen lid present (flag)", group="lid"),
    Variable("lid_depth", FLOAT, SCALAR, 0.0, units="m", long_name="Frozen lid depth", group="lid"),
    Variable("lid_temperature", FLOAT, LID, 273.15, units="K", long_name="Frozen lid temperature profile", group="lid"),
    Variable("rho_lid", FLOAT, LID, 0.0, units="kg m-3", long_name="Frozen lid density", group="lid"),
    Variable("v_lid", BOOL, SCALAR, False, long_name="Virtual lid present (flag)", group="lid"),
    Variable("v_lid_depth", FLOAT, SCALAR, 0.0, units="m", long_name="Virtual lid depth", group="lid"),
    Variable("virtual_lid_temperature", FLOAT, SCALAR, 273.15, units="K", long_name="Virtual lid temperature", group="lid"),
    Variable("has_had_lid", BOOL, SCALAR, False, long_name="Lid present this cycle (flag)", group="lid"),
    Variable("lid_sfc_melt", FLOAT, SCALAR, 0.0, units="m", long_name="Tracked lid surface melt", group="lid"),
    Variable("lid_snow_depth", FLOAT, SCALAR, 0.0, units="m", long_name="Snow depth on the lid", group="lid"),
    Variable("snow_on_lid", INT, SCALAR, 0, long_name="Snow-on-lid state (0/1/2)", group="lid"),

    # Lateral flow variables
    Variable("water", FLOAT, FIRN, 0.0, units="m", long_name="Liquid water depth per layer (lateral flow)", group="lateral"),
    Variable("water_level", FLOAT, SCALAR, 0.0, units="m", long_name="Water-table height for lateral flow", group="lateral"),
    Variable("water_direction", INT, DIRECTIONS, 0, long_name="Lateral outflow direction (0=NW..7=W)", group="lateral"),

    # Diagnostics
    Variable("firn_boundary_change", FLOAT, SCALAR, 0.0, units="m", long_name="Firn boundary change this day", group="diagnostic"),
    Variable("lake_boundary_change", FLOAT, SCALAR, 0.0, units="m", long_name="Lake boundary change this day", group="diagnostic"),
    Variable("lid_boundary_change", FLOAT, SCALAR, 0.0, units="m", long_name="Lid boundary change this day", group="diagnostic"),

    # Internal counters
    # n.b. these are typically with output=False as we probably don't care to write them out
    Variable("melt_hours", INT, SCALAR, 0, units="h", long_name="Cumulative surface-melt hours", group="counter", output=False),
    Variable("lid_melt_count", INT, SCALAR, 0, long_name="Lid melt-step counter", group="counter", output=False),
    Variable("lake_refreeze_counter", INT, SCALAR, 0, long_name="Lake refreeze counter", group="counter", output=False),
    Variable("exposed_water_refreeze_counter", INT, SCALAR, 0, long_name="Exposed-water refreeze counter", group="counter", output=False),
    Variable("t_step", INT, SCALAR, 0, long_name="Timestep within the current day", group="counter", output=False),
    Variable("day", INT, SCALAR, 0, long_name="Model day", group="counter", output=False),
    Variable("visit_count", INT, SCALAR, 0, long_name="Times this cell has been visited", group="counter", output=False),
    Variable("reset_combine", BOOL, SCALAR, False, long_name="Lid/firn just combined (flag)", group="internal", output=False),
    Variable("error_flag", BOOL, SCALAR, False, long_name="Cell hit an error state (flag)", group="internal", output=False),
    Variable("numba", BOOL, SCALAR, False, long_name="Running under Numba (flag)", group="internal", output=False),
]
# fmt: on
