""" """

# TODO - module-level docstring, other docstrings
import argparse
import warnings
import os
import numpy as np


def parse_args():
    """
    Parse input. Most things are controlled by `model_setup.py`; the only input
    here is (optionally) the location (as a filepath, so including the
    filename) of that setup file.
    """
    func_name = "monarchs.core.configuration.parse_args"
    run_dir = os.getcwd().replace("\\", "/")
    warning_flag = False
    runscript = ""
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
            print(f"\n{func_name}:")
            print(
                f"Setting runscript to {runscript} since MONARCHS has"
                " identified this as a unit test"
            )
            return runscript

        warnings.warn(
            "Unrecognised test case. This warning occurs if you are trying"
            " to run a unit test from the wrong place, MONARCHS will try"
            " and continue using the input arguments you specified, but if"
            " running an incorrectly-setup test case will fail."
        )
    parser = argparse.ArgumentParser(
        prog="MONARCHS",
        description=(
            "A model of ice shelf development, written by Sammie Buzzard, Jon"
            " Elsey and Alex Robel."
        ),
    )
    parser.add_argument(
        "--input_path",
        "-i",
        help=(
            "Absolute or relative path to an input file, in the format of"
            " <model_setup.py>"
        ),
        default="model_setup.py",
        required=False,
    )
    args, _ = parser.parse_known_args()
    model_setup_path = args.input_path
    return model_setup_path


def create_output_folders(model_setup):
    """
    Create the output folders for the model output, meteorological data and
    dump files, if they do not already exist.
    """
    for filepath in (
        model_setup.output_filepath,
        model_setup.dump_filepath,
        model_setup.met_output_filepath,
    ):
        # os.path.dirname is "" by default so writes to cwd
        folder = os.path.dirname(filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)


def handle_incompatible_flags(model_setup):
    """
    Handle incompatible model flags that could cause issues with the code.
    This mostly consists of generating errors that the user should check for.
    This is one of many reasons why long model runs should be run in a test
    queue first to ensure that the setup is correct.

    Parameters
    ----------

    Returns
    -------
    None

    Raises
    ------
    """
    func_name = "monarchs.core.configuration.handle_incompatible_flags"
    print("\n")
    # Model doesn't properly handle non-square grids yet, so raise error at
    # the start rather than potentially crashing out
    if model_setup.row_amount != model_setup.col_amount:
        raise NotImplementedError(
            f"{func_name}: row_amount ({model_setup.row_amount}) !="
            f" col_amount ({model_setup.col_amount}). Non-square grids are"
            " not yet supported - see the to-fix list in _review."
        )
    # MPI support has been removed pending a full rework in future so
    # warn here
    if getattr(model_setup, "use_mpi", False):
        warnings.warn(
            f"{func_name}: MPI support has been removed from MONARCHS -"
            " <use_mpi> is ignored. Use Numba (use_numba=True) or Dask"
            " (parallel=True) parallelism instead."
        )
    if hasattr(model_setup, "lat_bounds") and not hasattr(model_setup, "DEM_path"):
        if model_setup.lat_bounds.lower() == "dem":
            raise ValueError(
                f"{func_name}: You"
                ' must provide a DEM file using the "DEM_path" argument to'
                " use DEM lat/long bounds."
            )
    dump_attrs = ["dump_data", "reload_from_dump"]
    for attr in dump_attrs:
        if hasattr(model_setup, attr) and not hasattr(model_setup, "dump_filepath"):
            if getattr(model_setup, attr) is True:
                raise NameError(
                    f"{func_name}:"
                    f" <{attr}> is specified but <dump_filepath> is empty -"
                    " please specify in model_setup a filepath to write the"
                    " dump into via the <dump_filepath> attribute."
                )
    for attr in dump_attrs:
        if hasattr(model_setup, attr) and not hasattr(model_setup, "output_filepath"):
            if getattr(model_setup, attr) is True:
                raise NameError(
                    f"{func_name}:"
                    f" <{attr}> is specified but <output_filepath> is empty -"
                    " please specify in model_setup a filepath to write the"
                    " saved data into via the <output_filepath> attribute."
                )
    if hasattr(model_setup, "dump_format") and model_setup.dump_format != "NETCDF4":
        raise ValueError(
            f"{func_name}:"
            f" dump_format must be 'NETCDF4', not {model_setup.dump_format}."
            " Pickle dumps are no longer supported - netCDF checkpoints are"
            " portable and can be reloaded with reload_from_dump."
        )


def handle_invalid_values(model_setup):
    """
    Handle cases where model attributes are set to values that will cause
    the model to fail.

    Parameters
    ----------
    model_setup - load_model_setup.ModelSetup instance
        Loaded in model setup file (see <load_in_model_setup>)

    Returns
    -------
    None

    """
    func_name = "monarchs.core.configuration.handle_invalid_values"
    valid_solvers = [
        "hybr",
        "df-sane",
        "brentq",
        "lm",
        "trust-ncg",
        "broyden1",
    ]
    if hasattr(model_setup, "solver") and model_setup.solver not in valid_solvers:
        raise ValueError(
            f"{func_name}:"
            f" solver must be one of {valid_solvers}, not {model_setup.solver}"
        )
    if hasattr(model_setup, "outflow_proportion") and not (
        0.0 <= model_setup.outflow_proportion <= 1.0
    ):
        raise ValueError(
            f"{func_name}:"
            " outflow_proportion must be between 0.0 and 1.0, not"
            f" {model_setup.outflow_proportion}"
        )


def create_defaults_for_missing_flags(model_setup):
    """
    Prevent the model from crashing out if certain flags are not specified in
    the model_setup file.
    This will not prevent the code from stopping if key information is not
    provided (e.g. a DEM GeoTIFF file or a NumPy array of firn column depth
    matching the chosen grid size, or a netCDF of input meteorological data).
    It is intended to ensure that the code runs even if the setup file does not
    contain every possible argument. (for example, not having flags that don't
    affect the model physics such as <met_dem_diagnostic_plots> set,
    or likewise the debugging toggles such as `firn_heat_toggle`).

    You can also amend this function to set defaults for certain variables,
    e.g. the default surface density `rho_sfc` = 500 as defined below.

    Args
    -------
    model_setup - loaded in model setup file (see <load_in_model_setup>)

    Returns
    -------
    None
    """
    func_name = "monarchs.core.configuration.create_defaults_for_missing_flags"
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
        "ignore_errors",
        "dump_data",
        "spinup",
        "reload_from_dump",
        "met_dem_diagnostic_plots",
        "bbox_top_right",
        "bbox_bottom_left",
        "bbox_top_left",
        "bbox_bottom_right",
        "dump_data_pre_lateral_movement",
        "use_numba",
        "dem_diagnostic_plots",
        "parallel",
        "catchment_outflow",
        "load_precalculated_met_data",
        "dump_checkpoint_frequency",
    ]
    for attr in optional_args_to_true:
        if not hasattr(model_setup, attr):
            setattr(model_setup, attr, True)
            print(
                f"{func_name}"
                f" Setting missing model_setup attribute <{attr}> to default"
                " value True"
            )
    for attr in optional_args_to_false:
        if not hasattr(model_setup, attr):
            setattr(model_setup, attr, False)
            print(
                f"{func_name}"
                f" Setting missing model_setup attribute <{attr}> to default"
                " value False"
            )
    inits = ["rho_init", "T_init"]
    for attr in inits:
        if not hasattr(model_setup, attr):
            setattr(model_setup, attr, "default")
            print(
                f"{func_name}"
                f" Setting missing model_setup attribute <{attr}> to default"
                " value default"
            )

    bounds = ["latmax", "latmin", "longmax", "longmin"]
    for attr in bounds:
        if not hasattr(model_setup, attr):
            setattr(model_setup, attr, np.nan)
            print(
                f"{func_name}"
                f" Setting missing model_setup attribute <{attr}> to default"
                " value np.nan"
            )
    vardict = {}
    vardict["output_grid_size"] = model_setup.vertical_points_firn
    vardict["met_timestep"] = "hourly"
    vardict["met_output_filepath"] = "interpolated_met_data.nc"
    vardict["rho_sfc"] = 500
    vardict["t_steps_per_day"] = 24
    vardict["lateral_timestep"] = model_setup.t_steps_per_day * 3600
    vardict["firn_max_height"] = 150
    vardict["firn_min_height"] = 20
    vardict["min_height_handler"] = "filter"
    vardict["max_height_handler"] = "filter"
    vardict["output_timestep"] = 1
    vardict["dump_timestep"] = 1
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
    vardict["input_crs"] = 3031
    vardict["cores"] = "all"
    vardict["solver"] = "hybr"
    vardict["dask_scheduler"] = (
        "processes"  # set to "distributed" if using HPC across multiple nodes
    )
    vardict["flow_speed_scaling"] = 1.0
    vardict["outflow_proportion"] = 1.0
    if hasattr(model_setup, "met_input_filepath"):
        vardict["met_data_source"] = "ERA5"
    elif hasattr(model_setup, "met_data") and isinstance(model_setup.met_data, dict):
        vardict["met_data_source"] = "user_defined"
    else:
        raise ValueError(
            f"{func_name}:"
            " No meteorological data source was detected. Either specify"
            " input data as an ERA5-formatnetCDF file, or as a ``dict`` of"
            " user-defined data in model_setup.py. See"
            " ``examples/1D_test_case/model_setup.py`` for an example of"
            " user-defined data, or"
            " ``examples/10x10_gaussian_threelake/model_setup.py`` for an"
            " example of ERA5 data."
        )
    special_keys = ["lateral_timestep"]
    for key, value in vardict.items():
        if not hasattr(model_setup, key):
            setattr(model_setup, key, value)
            if key not in special_keys:
                print(
                    f"{func_name}"
                    f" Setting missing model_setup attribute <{key}> to"
                    f" default value <{value}>"
                )
            elif key == "lateral_timestep":
                print(
                    f"{func_name}"
                    f" Setting missing model_setup attribute <{key}> to"
                    " default value model_setup.t_steps_per_day * 3600"
                )
    if hasattr(model_setup, "DEM_path"):
        if not hasattr(model_setup, "lat_grid_size"):
            setattr(model_setup, "lat_grid_size", "dem")
            print(
                f"{func_name}"
                " Setting missing model_setup attribute <lat_grid_size> to"
                " default value 'dem' since a valid DEMwas provided."
            )
