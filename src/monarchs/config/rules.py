"""
Cross-setting consistency checks for the settings catalogue.

Kept separate from `catalogue.py` so that file stays a clean settings table.
Each `Rule` says when a combination of settings is invalid and what to raise.
 Rules are run by `apply.check_rules` *before* defaults are filled, so use
`getattr``/``hasattr` for settings that may not be present yet.

The `fail_condition` parameter is what determines whether a rule is violated.
This should be in the form of a function, so that it can be evaluated only
when checking rules, rather than when loading in the catalogue. Most of these
functions are implemented here using lambdas, so that they are inlined with
the rules themselves.

`ms` in the rule descriptions refers to a ModelSetup object, i.e. the
class filled in by reading the config file before performing any validation.
"""

from monarchs.config.definitions import Rule


def _wants_dem_bounds(ms):
    """True when DEM lat/long bounds are requested but no DEM was provided."""
    lat_bounds = getattr(ms, "lat_bounds", None)
    return (
        isinstance(lat_bounds, str)
        and lat_bounds.lower() == "dem"
        and not hasattr(ms, "DEM_path")
    )


RULES = [
    # check for non-square grids
    Rule(
        failed_when=lambda ms: ms.row_amount != ms.col_amount,
        message="row_amount != col_amount. Non-square grids are not yet tested.",
        error=NotImplementedError,
    ),
    # check for MPI flag being enabled
    Rule(
        failed_when=lambda ms: getattr(ms, "use_mpi", False),
        message="MPI support is not yet implemented.",
        error=UserWarning,
    ),
    # check to ensure that a valid DEM is specified if using `lat_bounds = "dem"`
    Rule(
        failed_when=_wants_dem_bounds,
        message='You must provide a DEM file using the "DEM_path" argument to use'
        " DEM lat/long bounds.",
    ),
    # ensure that we have a valid filepath for a checkpoint file if we are
    # writing them out
    Rule(
        failed_when=lambda ms: (
            getattr(ms, "dump_data", False) is True and not hasattr(ms, "dump_filepath")
        ),
        message="<dump_data> is specified but <dump_filepath> is empty - please"
        " specify in model_setup a filepath to write the dump into via the"
        " <dump_filepath> attribute.",
        error=NameError,
    ),
    # check that we have a valid checkpoint file to load from if restarting
    # from one
    Rule(
        failed_when=lambda ms: (
            getattr(ms, "reload_from_dump", False) is True
            and not hasattr(ms, "dump_filepath")
        ),
        message="<reload_from_dump> is specified but <dump_filepath> is empty -"
        " please specify in model_setup a filepath to write the dump into"
        " via the <dump_filepath> attribute.",
        error=NameError,
    ),
    # check that we have a valid scientific output file to write into
    # if we are writing output
    Rule(
        failed_when=lambda ms: (
            getattr(ms, "save_output", False) is True
            and not hasattr(ms, "output_filepath")
        ),
        message="<save_output> is specified but <output_filepath> is empty - please"
        " specify in model_setup a filepath to write the output into via the"
        " <output_filepath> attribute.",
        error=NameError,
    ),
    # make sure that we use netCDF4 dump formatting only
    # TODO - can deprecate this rule and the setup parameter
    Rule(
        failed_when=lambda ms: getattr(ms, "dump_format", "NETCDF4") != "NETCDF4",
        message="dump_format must be 'NETCDF4'. Pickle dumps are no longer supported.",
    ),
    # check we have at least one of a firn depth profile or a DEM to read one in from
    Rule(
        failed_when=lambda ms: (
            not hasattr(ms, "DEM_path") and not hasattr(ms, "firn_depth")
        ),
        message="no initial firn geometry provided - set either <DEM_path> (a DEM to"
        " read firn depth from) or <firn_depth> (a number or (row, col) array)"
        " in model_setup.",
    ),
    # check we have a meteorological forcing data source - either specified as
    # arrays or read in from an ERA5-format file
    Rule(
        failed_when=lambda ms: (
            not hasattr(ms, "met_input_filepath") and not hasattr(ms, "met_data")
        ),
        message="no meteorological data source provided - set either"
        " <met_input_filepath> (an ERA5-format netCDF) or <met_data> (a dict"
        " of user-defined data) in model_setup.",
    ),
]
