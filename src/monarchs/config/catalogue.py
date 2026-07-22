"""
The MONARCHS settings catalogue.

Similarly to the variable catalogue (`monarchs.variables.catalogue`), this
defines all of the settings that the user can define in a setup script to
be passed into the model. The aim is to have one centralised place where
we define all settings, and then all application of these settings (putting
them into a format the model understands) and documentation is generated
purely from this one place.

Each setting is a `Setting` object - which defines the allowed type(s) of the
setting (e.g. True/False, a number or a string), a default value,
and some validation rules or constraints on the allowed values.

Any fields left out of a run setup script are filled in if appropriate by
default values from this schema. If there are values that are missing and
need to be included, the appropriate validator should be set here to ensure
that it is included. We also include a message to raise if this is the case.

The cross-setting consistency checks (`RULES`) live in `rules.py`.

Both `Setting` and `Rule` are keyword-only, so every row reads as
``Setting(name=..., dtype=..., ...)`` - clearer than a run of positional
arguments.
"""

import numpy as np

from monarchs.config.definitions import REQUIRED, UNSET, Setting
from monarchs.config.computed_defaults import (
    lat_grid_size,
    lateral_timestep,
    met_data_source,
    output_grid_size,
)

# default set of grid fields written to the output netCDF (kept out of the
# Setting row below so that row stays readable)
_DEFAULT_VARS_TO_SAVE = (
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


SETTINGS = [
    # Model grid settings
    Setting(name="row_amount", dtype=int, default=REQUIRED, group="grid"),
    Setting(name="col_amount", dtype=int, default=REQUIRED, group="grid"),
    Setting(name="vertical_points_firn", dtype=int, default=REQUIRED, group="grid"),
    Setting(name="vertical_points_lake", dtype=int, default=REQUIRED, group="grid"),
    Setting(name="vertical_points_lid", dtype=int, default=REQUIRED, group="grid"),
    Setting(
        name="lat_grid_size",
        dtype=str,
        default=lat_grid_size,
        group="grid",
        default_doc="'dem' when a DEM is provided",
    ),
    # Timestepping
    Setting(name="num_days", dtype=int, default=REQUIRED, group="timestepping"),
    Setting(name="t_steps_per_day", dtype=int, default=24, group="timestepping"),
    Setting(
        name="lateral_timestep",
        dtype=int,
        default=lateral_timestep,
        group="timestepping",
        default_doc="model_setup.t_steps_per_day * 3600",
    ),
    # DEM / geography
    Setting(name="latmax", dtype=float, default=np.nan, group="dem"),
    Setting(name="latmin", dtype=float, default=np.nan, group="dem"),
    Setting(name="longmax", dtype=float, default=np.nan, group="dem"),
    Setting(name="longmin", dtype=float, default=np.nan, group="dem"),
    Setting(name="bbox_top_right", dtype=list, default=False, group="dem"),
    Setting(name="bbox_bottom_left", dtype=list, default=False, group="dem"),
    Setting(name="bbox_top_left", dtype=list, default=False, group="dem"),
    Setting(name="bbox_bottom_right", dtype=list, default=False, group="dem"),
    Setting(name="input_crs", dtype=int, default=3031, group="dem"),
    Setting(name="firn_max_height", dtype=float, default=150, group="dem"),
    Setting(name="firn_min_height", dtype=float, default=20, group="dem"),
    Setting(name="max_height_handler", dtype=str, default="filter", group="dem"),
    Setting(name="min_height_handler", dtype=str, default="filter", group="dem"),
    Setting(name="dem_diagnostic_plots", dtype=bool, default=False, group="dem"),
    Setting(
        name="DEM_path",
        dtype=str,
        default=UNSET,
        group="dem",
        description="Path to a DEM used to initialise firn depth. Provide this or firn_depth.",
    ),
    Setting(
        name="lat_bounds",
        dtype=str,
        default=UNSET,
        group="dem",
        description="Set to 'dem' to take the lat/long bounds from the DEM.",
    ),
    # Initial conditions
    Setting(
        name="rho_init",
        dtype=(str, float),
        default="default",
        group="initial conditions",
    ),
    Setting(
        name="T_init",
        dtype=(str, float),
        default="default",
        group="initial conditions",
    ),
    Setting(name="rho_sfc", dtype=float, default=500, group="initial conditions"),
    Setting(
        name="firn_depth",
        dtype=float,
        default=UNSET,
        group="initial conditions",
        description="Initial firn depth, a number or (row, col) array. Provide this or a DEM_path.",
    ),
    Setting(
        name="initial_conditions",
        dtype=dict,
        default=UNSET,
        group="initial conditions",
        description="Optional {grid variable: value} overrides applied when building"
        " the initial grid, e.g. {'lake_depth': 0.5}. Keys must be names from the"
        " variable catalogue (monarchs.variables); scalars broadcast over the grid,"
        " arrays set per-cell or per-layer profiles.",
    ),
    # Met data
    Setting(
        name="met_data_source",
        dtype=str,
        default=met_data_source,
        group="met",
        default_doc="'ERA5' or 'user_defined', inferred from the inputs",
    ),
    Setting(name="met_timestep", dtype=str, default="hourly", group="met"),
    Setting(
        name="met_output_filepath",
        dtype=str,
        default="interpolated_met_data.nc",
        group="met",
    ),
    Setting(name="met_dem_diagnostic_plots", dtype=bool, default=False, group="met"),
    Setting(name="load_precalculated_met_data", dtype=bool, default=False, group="met"),
    Setting(
        name="met_input_filepath",
        dtype=str,
        default=UNSET,
        group="met",
        description="Path to an ERA5-format netCDF. Provide this or met_data.",
    ),
    Setting(
        name="met_data",
        dtype=dict,
        default=UNSET,
        group="met",
        description="User-defined met data as a dict. Provide this or met_input_filepath.",
    ),
    Setting(
        name="radiation_forcing_factor",
        dtype=float,
        default=1,
        group="met",
        description="Multiplier applied to downwelling SW/LW (testing only; 1 = no forcing).",
    ),
    # Physics toggles (debug switches - True for normal operation)
    Setting(
        name="snowfall_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        kernel_toggle=True,
    ),
    Setting(
        name="firn_column_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        kernel_toggle=True,
    ),
    Setting(
        name="firn_heat_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        kernel_toggle=True,
    ),
    Setting(
        name="percolation_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        kernel_toggle=True,
    ),
    Setting(
        name="perc_time_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        kernel_toggle=True,
    ),
    Setting(
        name="lake_development_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        kernel_toggle=True,
    ),
    Setting(
        name="lid_development_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        kernel_toggle=True,
    ),
    Setting(name="lateral_movement_toggle", dtype=bool, default=True, group="toggles"),
    Setting(
        name="lateral_movement_percolation_toggle",
        dtype=bool,
        default=True,
        group="toggles",
    ),
    Setting(name="single_column_toggle", dtype=bool, default=True, group="toggles"),
    Setting(
        name="densification_toggle",
        dtype=bool,
        default=False,
        group="toggles",
        kernel_toggle=True,
    ),
    # Lateral flow
    Setting(name="flow_into_land", dtype=bool, default=True, group="lateral"),
    Setting(name="catchment_outflow", dtype=bool, default=False, group="lateral"),
    Setting(name="flow_speed_scaling", dtype=float, default=1.0, group="lateral"),
    Setting(
        name="outflow_proportion",
        dtype=float,
        default=1.0,
        group="lateral",
        validator=lambda v: 0.0 <= v <= 1.0,
        validator_doc="between 0.0 and 1.0",
    ),
    # Output / checkpointing
    Setting(
        name="output_grid_size",
        dtype=int,
        default=output_grid_size,
        group="io",
        default_doc="vertical_points_firn",
    ),
    Setting(name="output_timestep", dtype=int, default=1, group="io"),
    Setting(
        name="vars_to_save",
        dtype=tuple,
        default=_DEFAULT_VARS_TO_SAVE,
        group="io",
        description="Grid fields written to the output netCDF each output step.",
    ),
    Setting(name="save_output", dtype=bool, default=False, group="io"),
    Setting(
        name="output_filepath",
        dtype=str,
        default=UNSET,
        group="io",
        description="Where the output netCDF is written (required if save_output/dump_data).",
    ),
    Setting(name="dump_data", dtype=bool, default=False, group="io"),
    Setting(
        name="dump_filepath",
        dtype=str,
        default=UNSET,
        group="io",
        description="Where checkpoints are written (required if dump_data/reload_from_dump).",
    ),
    Setting(name="dump_format", dtype=str, default="NETCDF4", group="io"),
    Setting(name="dump_timestep", dtype=int, default=1, group="io"),
    Setting(name="dump_checkpoint_frequency", dtype=int, default=False, group="io"),
    Setting(
        name="dump_data_pre_lateral_movement", dtype=bool, default=False, group="io"
    ),
    Setting(name="reload_from_dump", dtype=bool, default=False, group="io"),
    # Runtime / performance
    Setting(name="use_numba", dtype=bool, default=True, group="runtime"),
    Setting(name="parallel", dtype=bool, default=True, group="runtime"),
    Setting(
        name="use_mpi",
        dtype=bool,
        default=False,
        group="runtime",
        description="MPI support is not yet implemented.",
    ),
    Setting(name="cores", dtype=(str, int), default="all", group="runtime"),
    Setting(name="dask_scheduler", dtype=str, default="processes", group="runtime"),
    Setting(
        name="ignore_errors",
        dtype=bool,
        default=False,
        group="runtime",
        kernel_toggle=True,
    ),
]
