"""
Cross-setting consistency checks for the settings catalogue.

Kept separate from `catalogue.py` so that file stays a clean settings table.
Each `Rule` says when a combination of settings is invalid and what to raise.
 Rules are run by `apply.check_rules` *before* defaults are filled, so use
`getattr``/``hasattr` for settings that may not be present yet.

The `fail_condition` parameter is what determines whether a rule is violated.
This should be in the form of a function, so that it can be evaluated only
when checking rules, rather than when loading in the catalogue. Most of these
functions are implemented here using lambdas, which lets us define the functions
with the rules together. These typically work on "ms" - which is shorthand for
ModelSetup.

`ms` in the rule descriptions refers to a ModelSetup object, i.e. the
class filled in by reading the config file before performing any validation.
"""

from monarchs.config.definitions import Rule

# the model_setup setting each met source reads its input from. e.g. ERA5
# reads an input filepath.
MET_SOURCE_INPUTS = {
    "ERA5": "met_input_filepath",
    "user_defined": "met_data",
}

# toggles that make the model read or write a checkpoint, so all need a
# dump_filepath to point at - this lets us apply the Rule to all of these
# flags
_NEEDS_DUMP_FILEPATH = (
    "dump_data",
    "reload_from_dump",
    "dump_data_pre_lateral_movement",
)


def _no_met_source(ms):
    """True when the setup provides no input for any met data source.
    acceptable values are "era5" and "user_defined"."""
    return not any(hasattr(ms, setting) for setting in MET_SOURCE_INPUTS.values())


def _met_source_input_missing(ms):
    """True when met_data_source names a source whose input was not given."""
    setting = MET_SOURCE_INPUTS.get(getattr(ms, "met_data_source", None))
    return setting is not None and not hasattr(ms, setting)


def _no_dump_filepath(ms):
    """True when something wants to write a checkpoint but no path was given."""
    return any(
        getattr(ms, flag, False) for flag in _NEEDS_DUMP_FILEPATH
    ) and not hasattr(ms, "dump_filepath")


RULES = [
    # check for non-square grids
    Rule(
        failed_when=lambda ms: ms.row_amount != ms.col_amount,
        message="row_amount != col_amount. Non-square grids are not yet tested.",
        error=NotImplementedError,
    ),
    # parallelism comes from Numba's prange, so <parallel> does nothing on the
    # pure-Python path. Only warn when the user asked for both explicitly -
    # rules run before defaults are filled, so an absent use_numba here means
    # "not specified", which resolves to the catalogue default (Numba on).
    Rule(
        failed_when=lambda ms: (
            getattr(ms, "parallel", False)
            and hasattr(ms, "use_numba")
            and not ms.use_numba
        ),
        message="<parallel> has no effect when <use_numba> is False - the"
        " pure-Python grid loop is always serial. Set use_numba=True to run"
        " in parallel.",
        error=UserWarning,
    ),
    # check for MPI flag being enabled
    Rule(
        failed_when=lambda ms: getattr(ms, "use_mpi", False),
        message="MPI support is not yet implemented.",
        error=UserWarning,
    ),
    # ensure we have a filepath for the checkpoints, whichever toggle asked
    # for them to be read or written
    Rule(
        failed_when=_no_dump_filepath,
        message=f"one of {list(_NEEDS_DUMP_FILEPATH)} is specified but"
        " <dump_filepath> is empty - please specify in model_setup a filepath"
        " to read/write the dump via the <dump_filepath> attribute.",
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
    # check we have at least one of a firn depth profile or a DEM to read one in from
    Rule(
        failed_when=lambda ms: (
            not hasattr(ms, "DEM_path") and not hasattr(ms, "firn_depth")
        ),
        message="no initial firn geometry provided - set either <DEM_path> (a DEM to"
        " read firn depth from) or <firn_depth> (a number or (row, col) array)"
        " in model_setup.",
    ),
    # check we have the input for one of the meteorological forcing sources
    Rule(
        failed_when=_no_met_source,
        message="no meteorological data source provided - set either"
        " <met_input_filepath> (an ERA5-format netCDF) or <met_data> (a dict"
        " of user-defined data) in model_setup.",
    ),
    # and that an explicitly chosen source is the one that was given an input
    Rule(
        failed_when=_met_source_input_missing,
        message="<met_data_source> names a source whose input is missing -"
        " 'ERA5' reads <met_input_filepath>, 'user_defined' reads <met_data>.",
    ),
]
