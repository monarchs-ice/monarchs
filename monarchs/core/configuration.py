import argparse
import os
import warnings

import numpy as np
from numba import jit_module


def parse_args():
    """
    Parse input. Most things are controlled by model_setup.py; the only input here is (optionally) the location
    (as a filepath, so including the filename) of that setup file.
    """
    # If we are calling from the test suite rather than MONARCHS-main, then we need to ensure that we don't parse
    # arguments. We do this since various parts of the code that we might want to test will look for parameters in
    # monarchs.core.configuration.model_setup to determine their behaviour (particularly regarding use of Numba).
    run_dir = os.getcwd().replace("\\", "/")  # platform-agnostic by replacing \\ with /
    warning_flag = False
    # Check if pytest is currently running by checking the current environment variables. If so, then
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
                "Unrecognised test case. This warning occurs if you are trying to run a unit test from "
                "the wrong place, MONARCHS will try and continue using the input arguments you specified,"
                " but if running an incorrectly-setup test case will fail."
            )

    parser = argparse.ArgumentParser(
        prog="MONARCHS",
        description="A model of ice shelf development, written by Sammie Buzzard, Jon Elsey and Alex Robel.",
    )
    parser.add_argument(
        "--input_path",
        "-i",
        help="Absolute or relative path to an input file, in the format"
             "of <model_setup.py>",
        default="model_setup.py",
        required=False,
    )

    args = parser.parse_args()
    model_setup_path = args.input_path
    return model_setup_path


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

    print(
        "\n"
    )  # print a newline so that we can separate out configuration steps from other console output
    # Handle issue where the weather data is set to map onto a lat/long grid but an input DEM (and thus input lat/long)
    # is not specified.
    if hasattr(model_setup, "lat_bounds") and not hasattr(model_setup, "DEM_path"):
        if model_setup.lat_bounds.lower() == "dem":
            raise ValueError(
                f"monarchs.core.configuration.handle_incompatible_flags(): "
                'You must provide a DEM file using the "DEM_path" argument to use DEM lat/long bounds.'
            )

    dump_attrs = ["dump_data", "reload_state"]
    for attr in dump_attrs:
        if hasattr(model_setup, attr) and not hasattr(model_setup, "dump_filepath"):
            if getattr(model_setup, attr) is True:
                raise NameError(
                    f"monarchs.core.configuration.handle_incompatible_flags(): "
                    f"<{attr}> is specified but <dump_filepath> is empty - please specify in model_setup "
                    f"a filepath to write the dump into via the <dump_filepath> attribute."
                )
    save_attrs = ["save_output"]
    for attr in dump_attrs:
        if hasattr(model_setup, attr) and not hasattr(model_setup, "output_filepath"):
            if getattr(model_setup, attr) is True:
                raise NameError(
                    f"monarchs.core.configuration.handle_incompatible_flags(): "
                    f"<{attr}> is specified but <output_filepath> is empty - please specify in model_setup "
                    f"a filepath to write the saved data into via the <output_filepath> attribute."
                )


def create_defaults_for_missing_flags(model_setup):
    """
    Prevent the model from crashing out if certain flags are not specified in the model_setup file.
    This will not prevent the code from stopping if key information is not provided (e.g. a DEM GeoTIFF file or
    a NumPy array of firn column depth matching the chosen grid size, or a netCDF of input meteorological data).
    It is intended to ensure that the code runs even if the setup file does not contain every possible argument.
    (for example, not having flags that don't affect the model physics such as <met_dem_diagnostic_plots> set,
    or likewise the debugging toggles such as firn_heat_toggle).

    You can also amend this function to set defaults for certain variables, e.g. the default surface density
    <rho_sfc> = 500 as defined below.

    Args
    -------
    model_setup - loaded in model setup file (see <load_in_model_setup>)

    Returns
    -------
    None
    """

    # Some arguments get set to True or False, so we group these here for convenience.
    optional_args_to_true = [
        "lake_development_toggle",
        "lid_development_toggle",
        "lateral_movement_toggle",
        "lateral_movement_percolation_toggle",
        "percolation_toggle",
        "catchment_outflow",
        "perc_time_toggle",
        "snowfall_toggle",
        "firn_heat_toggle",
        "firn_column_toggle",
        "single_column_toggle"
    ]
    optional_args_to_false = [
        "densification_toggle",
        "simulated_water_toggle",
        "ignore_errors",
        "heateqn_res_toggle",
        "dump_data",
        "verbose_logging",
        "spinup",
        "reload_state",
        "met_dem_diagnostic_plots",
        "bbox_top_right",
        "bbox_bottom_left",
        "bbox_top_left",
        "bbox_bottom_right",
        "dump_data_pre_lateral_movement",
    ]

    for attr in optional_args_to_true:
        if not hasattr(model_setup, attr):
            setattr(model_setup, attr, True)
            print(
                f"monarchs.core.configuration.create_defaults_for_missing_flags: "
                f"Setting missing model_setup attribute <{attr}> to default value True"
            )

    for attr in optional_args_to_false:
        if not hasattr(model_setup, attr):
            setattr(model_setup, attr, False)
            print(
                f"monarchs.core.configuration.create_defaults_for_missing_flags: "
                f"Setting missing model_setup attribute <{attr}> to default value False"
            )

    inits = ["rho_init", "T_init"]
    for attr in inits:
        if not hasattr(model_setup, attr):
            setattr(model_setup, attr, "default")
            print(
                f"monarchs.core.configuration.create_defaults_for_missing_flags: "
                f'Setting missing model_setup attribute <{attr}> to default value "default"'
            )

    bounds = ["latmax", "latmin", "lonmax", "lonmin"]

    for attr in bounds:
        if not hasattr(model_setup, attr):
            setattr(model_setup, attr, np.nan)
            print(
                f"monarchs.core.configuration.create_defaults_for_missing_flags: "
                f"Setting missing model_setup attribute <{attr}> to default value np.nan"
            )

    # These parameters have bespoke values - use a dictionary to create these
    vardict = {}
    vardict["output_grid_size"] = model_setup.vertical_points_firn
    vardict["met_timestep"] = "hourly"
    vardict["met_output_filepath"] = "interpolated_met_data.nc"
    vardict["met_start"] = 0
    vardict["rho_sfc"] = 500
    vardict["t_steps_per_day"] = 24
    vardict["lateral_timestep"] = model_setup.t_steps_per_day * 3600
    vardict['firn_max_height'] = 150
    vardict["firn_min_height"] = 20
    vardict["min_height_handler"] = "filter"
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
    # Keys that have special print messages - e.g. those that have default values that depend on model_setup variables
    # go in here, and a specific print message is written for them
    special_keys = ["lateral_timestep"]
    for key in vardict.keys():
        if not hasattr(model_setup, key):
            setattr(model_setup, key, vardict[key])
            if key not in special_keys:
                print(
                    f"monarchs.core.configuration.create_defaults_for_missing_flags: "
                    f"Setting missing model_setup attribute <{key}> to default value <{vardict[key]}>"
                )
            elif key == "lateral_timestep":
                print(
                    f"monarchs.core.configuration.create_defaults_for_missing_flags: "
                    f"Setting missing model_setup attribute <{key}> to default value model_setup.t_steps_per_day * 3600"
                )


class ModelSetup:
    def __init__(self, script_path):
        # Execute the script to get the variables
        try:
            with open(script_path, "r") as file:
                script_content = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f'monarchs.core.configuration: Path to runscript ({script_path}) not found. '
                                    'Please either run from a directory containing a valid model_setup.py, or '
                                    'pass the -i flag with a valid runscript path.')

        # Create a local dictionary to capture the variables
        local_vars = {}
        exec(script_content, {}, local_vars)

        # Assign the variables to the class attributes
        for var_name, var_value in local_vars.items():
            setattr(self, var_name, var_value)

def jit_modules():
    if model_setup.use_numba:
        from numba import jit
        fastmath = False
        from inspect import getmembers, isfunction
        from monarchs.physics import (
                                      surface_fluxes,
                                      firn_functions,
                                      lake_functions,
                                      lid_functions,
                                      percolation_functions,
                                      snow_accumulation
                                      )
        from monarchs.core import utils, timestep

        module_list = [surface_fluxes, utils, firn_functions, lake_functions, lid_functions,
                       percolation_functions, snow_accumulation, timestep]

        for module in module_list:
            functions_list = getmembers(module, isfunction)
            for name, function in functions_list:
                if hasattr(function, '__wrapped__') or name.startswith('__'):
                    continue
                print(f'Applying Numba jit decorator to {module.__name__}.{name}')
                jitted_function = jit(function, nopython=True, fastmath=fastmath)
                setattr(module, name, jitted_function)

        from monarchs.physics import solver
        from monarchs.physics.Numba import solver as numba_solver
        # relax the isfunction stipulation for numba_solver since it is mostly jitted functions
        jit_functions_list = getmembers(numba_solver)
        for name, jitfunc in jit_functions_list:
            # ignore builtins - which we did not filter out with getmembers
            if not name.startswith('__'):
                print(f'Setting {solver.__name__}.{name} to the equivalent Numba-compatible version')
                setattr(solver, name, jitfunc)



def jit_classes():

        from numba.experimental import jitclass
        from monarchs.core import iceshelf_class
        from monarchs.met_data import metdata_class

        # define the "spec" for the Numba jitclass here - effectively a list of
        # all the variables and their datatype (arrays denoted by [:])
        iceshelf_spec = iceshelf_class.get_spec()
        iceshelf_class.IceShelf = jitclass(iceshelf_class.IceShelf, iceshelf_spec)

        metdata_spec = metdata_class.get_spec()
        metdata_class.MetData = jitclass(metdata_class.MetData, metdata_spec)


model_setup_path = parse_args()
model_setup = ModelSetup(model_setup_path)

if model_setup.use_numba:
    jit_modules()
    jit_classes()
    from monarchs.physics import solver
    print(solver.firn_heateqn_solver)