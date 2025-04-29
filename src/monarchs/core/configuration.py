import argparse
import warnings
import os
import numpy as np


def parse_args():
    """
    Parse input. Most things are controlled by `model_setup.py`; the only input here is (optionally) the location
    (as a filepath, so including the filename) of that setup file.
    """
    run_dir = os.getcwd().replace("\\", "/")
    warning_flag = False
    if "PYTEST_CURRENT_TEST" in os.environ:
        if "numba" in run_dir.split("/")[-1]:
            runscript = "model_test_setup_numba.py"
        elif "serial" in run_dir.split("/")[-1]:
            runscript = "model_test_setup_serial.py"
        elif "parallel" in run_dir.split("/")[-1]:
            runscript = "model_test_setup_parallel.py"
        else:
            warning_flag = True
        if not warning_flag:
            print("\nmonarchs.core.configuration.parse_args:")
            print(
                f"Setting runscript to {runscript} since MONARCHS has identified this as a unit test"
            )
            return runscript
        else:
            warnings.warn(
                "Unrecognised test case. This warning occurs if you are trying to run a unit test from the wrong place, MONARCHS will try and continue using the input arguments you specified, but if running an incorrectly-setup test case will fail."
            )
    parser = argparse.ArgumentParser(
        prog="MONARCHS",
        description="A model of ice shelf development, written by Sammie Buzzard, Jon Elsey and Alex Robel.",
    )
    parser.add_argument(
        "--input_path",
        "-i",
        help="Absolute or relative path to an input file, in the formatof <model_setup.py>",
        default="model_setup.py",
        required=False,
    )
    args, unknown = parser.parse_known_args()
    model_setup_path = args.input_path
    return model_setup_path


def create_output_folders(model_setup):
    """
    Create the output folders for the model output, meteorological data and dump files, if they do not already exist.
    """
    if not os.path.exists(model_setup.output_filepath.rsplit("/", 1)[0]):
        os.makedirs(model_setup.output_filepath.rsplit("/", 1)[0])
    if not os.path.exists(model_setup.dump_filepath.rsplit("/", 1)[0]):
        os.makedirs(model_setup.dump_filepath.rsplit("/", 1)[0])
    if not os.path.exists(model_setup.met_output_filepath.rsplit("/", 1)[0]):
        os.makedirs(model_setup.met_output_filepath.rsplit("/", 1)[0])


def handle_incompatible_flags(model_setup):
    """
    Handle incompatible model flags that could cause issues with the code.
    This mostly consists of generating errors that the user should check for. This is one of many reasons why
    long model runs should be run in a test queue first to ensure that the setup is correct.

    Parameters
    ----------

    Returns
    -------
    None

    Raises
    ------
    """
    print("\n")
    if hasattr(model_setup, "lat_bounds") and not hasattr(model_setup, "DEM_path"):
        if model_setup.lat_bounds.lower() == "dem":
            raise ValueError(
                f'monarchs.core.configuration.handle_incompatible_flags(): You must provide a DEM file using the "DEM_path" argument to use DEM lat/long bounds.'
            )
    dump_attrs = ["dump_data", "reload_from_dump"]
    for attr in dump_attrs:
        if hasattr(model_setup, attr) and not hasattr(model_setup, "dump_filepath"):
            if getattr(model_setup, attr) is True:
                raise NameError(
                    f"monarchs.core.configuration.handle_incompatible_flags(): <{attr}> is specified but <dump_filepath> is empty - please specify in model_setup a filepath to write the dump into via the <dump_filepath> attribute."
                )
    save_attrs = ["save_output"]
    for attr in dump_attrs:
        if hasattr(model_setup, attr) and not hasattr(model_setup, "output_filepath"):
            if getattr(model_setup, attr) is True:
                raise NameError(
                    f"monarchs.core.configuration.handle_incompatible_flags(): <{attr}> is specified but <output_filepath> is empty - please specify in model_setup a filepath to write the saved data into via the <output_filepath> attribute."
                )
    dump_formats = ["NETCDF4", "pickle"]
    if hasattr(model_setup, "dump_format"):
        if model_setup.dump_format not in dump_formats:
            raise ValueError(
                f"monarchs.core.configuration.handle_incompatible_flags(): dump_format must be one of {dump_formats}, not {model_setup.dump_format}"
            )
        if (
            model_setup.dump_format == "pickle"
            and hasattr(model_setup, "use_numba")
            and model_setup.use_numba
        ):
            raise ValueError(
                f"monarchs.core.configuration.handle_incompatible_flags(): dump_format is set to `'pickle'` but use_numba is `True`. This is not supported since Numba jitclasses are not picklable"
            )
    valid_solvers = ["hybr", "df-sane", "brentq", "lm", "trust-ncg", "broyden1"]
    if hasattr(model_setup, "solver") and model_setup.solver not in valid_solvers:
        raise ValueError(
            f"monarchs.core.configuration.handle_incompatible_flags(): solver must be one of {valid_solvers}, not {model_setup.solver}"
        )


def create_defaults_for_missing_flags(model_setup):
    """
    Prevent the model from crashing out if certain flags are not specified in the model_setup file.
    This will not prevent the code from stopping if key information is not provided (e.g. a DEM GeoTIFF file or
    a NumPy array of firn column depth matching the chosen grid size, or a netCDF of input meteorological data).
    It is intended to ensure that the code runs even if the setup file does not contain every possible argument.
    (for example, not having flags that don't affect the model physics such as <met_dem_diagnostic_plots> set,
    or likewise the debugging toggles such as `firn_heat_toggle`).

    You can also amend this function to set defaults for certain variables, e.g. the default surface density
    `rho_sfc` = 500 as defined below.

    Args
    -------
    model_setup - loaded in model setup file (see <load_in_model_setup>)

    Returns
    -------
    None
    """
    optional_args_to_true = [
        "lake_development_toggle",
        "lid_development_toggle",
        "lateral_movement_toggle",
        "lateral_movement_percolation_toggle",
        "percolation_toggle",
        "flow_into_land",
        "perc_time_toggle",
        "snowfall_toggle",
        "firn_heat_toggle",
        "firn_column_toggle",
        "single_column_toggle",
    ]
    optional_args_to_false = [
        "densification_toggle",
        "simulated_water_toggle",
        "ignore_errors",
        "heateqn_res_toggle",
        "dump_data",
        "verbose_logging",
        "spinup",
        "reload_from_dump",
        "met_dem_diagnostic_plots",
        "bbox_top_right",
        "bbox_bottom_left",
        "bbox_top_left",
        "bbox_bottom_right",
        "dump_data_pre_lateral_movement",
        "use_numba",
        "use_mpi",
        "dem_diagnostic_plots",
        "parallel",
        "use_numba",
        "catchment_outflow",
    ]
    for attr in optional_args_to_true:
        if not hasattr(model_setup, attr):
            setattr(model_setup, attr, True)
            print(
                f"monarchs.core.configuration.create_defaults_for_missing_flags: Setting missing model_setup attribute <{attr}> to default value True"
            )
    for attr in optional_args_to_false:
        if not hasattr(model_setup, attr):
            setattr(model_setup, attr, False)
            print(
                f"monarchs.core.configuration.create_defaults_for_missing_flags: Setting missing model_setup attribute <{attr}> to default value False"
            )
    inits = ["rho_init", "T_init"]
    for attr in inits:
        if not hasattr(model_setup, attr):
            setattr(model_setup, attr, "default")
            print(
                f'monarchs.core.configuration.create_defaults_for_missing_flags: Setting missing model_setup attribute <{attr}> to default value "default"'
            )
    bounds = ["latmax", "latmin", "lonmax", "lonmin"]
    for attr in bounds:
        if not hasattr(model_setup, attr):
            setattr(model_setup, attr, np.nan)
            print(
                f"monarchs.core.configuration.create_defaults_for_missing_flags: Setting missing model_setup attribute <{attr}> to default value np.nan"
            )
    vardict = {}
    vardict["output_grid_size"] = model_setup.vertical_points_firn
    vardict["met_timestep"] = "hourly"
    vardict["met_output_filepath"] = "interpolated_met_data.nc"
    vardict["met_start"] = 0
    vardict["rho_sfc"] = 500
    vardict["t_steps_per_day"] = 24
    vardict["lateral_timestep"] = model_setup.t_steps_per_day * 3600
    vardict["firn_max_height"] = 150
    vardict["firn_min_height"] = 20
    vardict["min_height_handler"] = "filter"
    vardict["max_height_handler"] = "filter"
    vardict["output_timestep"] = 1
    vardict["vars_to_save"] = (
        "firn_temperature",
        "Sfrac",
        "Lfrac",
        "firn_depth",
        "lake_depth",
        "lid_depth",
        "lake",
        "lid",
        "v_lid",
        "ice_lens_depth",
    )
    vardict["dump_format"] = "NETCDF4"
    vardict["input_crs"] = 3031
    vardict["cores"] = "all"
    vardict["solver"] = "hybr"
    if hasattr(model_setup, "met_input_filepath"):
        vardict["met_data_source"] = "ERA5"
    elif hasattr(model_setup, "met_data") and isinstance(model_setup.met_data, dict):
        vardict["met_data_source"] = "user_defined"
    else:
        raise ValueError(
            "monarchs.core.configuration.create_defaults_for_missing_flags: No meteorological data source was detected. Either specify input data as an ERA5-formatnetCDF file, or as a ``dict`` of user-defined data in model_setup.py. See ``examples/1D_test_case/model_setup.py`` for an example of user-defined data, or ``examples/10x10_gaussian_threelake/model_setup.py`` for an example of ERA5 data."
        )
    special_keys = ["lateral_timestep"]
    for key in vardict.keys():
        if not hasattr(model_setup, key):
            setattr(model_setup, key, vardict[key])
            if key not in special_keys:
                print(
                    f"monarchs.core.configuration.create_defaults_for_missing_flags: Setting missing model_setup attribute <{key}> to default value <{vardict[key]}>"
                )
            elif key == "lateral_timestep":
                print(
                    f"monarchs.core.configuration.create_defaults_for_missing_flags: Setting missing model_setup attribute <{key}> to default value model_setup.t_steps_per_day * 3600"
                )
    if hasattr(model_setup, "DEM_path"):
        if not hasattr(model_setup, "lat_grid_size"):
            setattr(model_setup, "lat_grid_size", "dem")
            print(
                f"monarchs.core.configuration.create_defaults_for_missing_flags: Setting missing model_setup attribute <lat_grid_size> to default value 'dem' since a valid DEMwas provided."
            )


class ModelSetup:

    def __init__(self, script_path):
        try:
            with open(script_path, "r") as file:
                script_content = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"monarchs.core.configuration: Path to runscript ({script_path}) not found. Please either run from a directory containing a valid model_setup.py, or pass the -i flag with a valid runscript path."
            )
        local_vars = {}
        exec(script_content, {}, local_vars)
        for var_name, var_value in local_vars.items():
            setattr(self, var_name, var_value)


def jit_modules():
    """
    If using Numba, then we need to apply the `numba.jit` decorator to several functions
    in `physics` and `core`, and ensure that MONARCHS loads in the Numba-compatible solvers.
    This function handles this process, by loading in the modules and using `setattr`
    to overwrite the initial implementation with the Numba-compatible
    versions, either by applying the `jit` decorator, or in the case where the source code
    differs between the Numba and non-Numba versions, by overwriting the pure-Python
    version with the Numba-compatible one.

    This is only called when `use_numba` is `True`, so the import only happens if needed
    (important for if the user does not have Numba installed).

    This function was designed this way (as opposed to applying jit to each function in its
    own module locally) so that we only need to import `monarchs.core.configuration`
    once; in `monarchs.core.driver`. This allows us to run in parallel with `multiprocessing`,
    as otherwise each thread would try and load in the configuration file with null arguments,
    whenever importing the physics functions (which needed to know whether `use_numba` was `True`)
    which was causing an error.
    """
    from numba import jit
    from inspect import getmembers, isfunction

    fastmath = False
    from monarchs.physics import percolation_functions, snow_accumulation
    from monarchs.physics import lid_functions
    from monarchs.physics import surface_fluxes
    from monarchs.physics import lake_functions
    from monarchs.physics import firn_functions
    from monarchs.physics import timestep
    from monarchs.core import utils

    module_list = [
        surface_fluxes,
        utils,
        firn_functions,
        lake_functions,
        lid_functions,
        percolation_functions,
        snow_accumulation,
        timestep,
    ]
    for module in module_list:
        functions_list = getmembers(module, isfunction)
        for name, function in functions_list:
            if hasattr(function, "__wrapped__") or name.startswith("__"):
                continue
            print(f"Applying Numba jit decorator to {module.__name__}.{name}")
            jitted_function = jit(function, nopython=True, fastmath=fastmath)
            setattr(module, name, jitted_function)
    from monarchs.physics import solver
    from monarchs.physics.Numba import solver as numba_solver

    jit_functions_list = getmembers(numba_solver)
    for name, jitfunc in jit_functions_list:
        if not name.startswith("__"):
            print(
                f"Setting {solver.__name__}.{name} to the equivalent Numba-compatible version"
            )
            setattr(solver, name, jitfunc)


def jit_classes():
    from numba.experimental import jitclass
    from monarchs.core import iceshelf_class
    from monarchs.met_data import metdata_class

    iceshelf_spec = iceshelf_class.get_spec()
    iceshelf_class.IceShelf = jitclass(iceshelf_class.IceShelf, iceshelf_spec)
    metdata_spec = metdata_class.get_spec()
    metdata_class.MetData = jitclass(metdata_class.MetData, metdata_spec)


if os.environ.get("MONARCHS_MPI", None) is not None:
    mpi = True
    print("Setting MPI to True")
else:
    mpi = False
if mpi:
    if os.environ.get("MONARCHS_MODEL_SETUP_PATH") is not None:
        model_setup_path = os.environ.get("MONARCHS_MODEL_SETUP_PATH")
    else:
        model_setup_path = "model_setup.py"
else:
    model_setup_path = parse_args()
model_setup = ModelSetup(model_setup_path)
if hasattr(model_setup, "use_numba") and model_setup.use_numba:
    jit_modules()
    jit_classes()
