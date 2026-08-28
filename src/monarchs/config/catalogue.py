"""
The MONARCHS settings catalogue.

Similarly to the variable catalogue (`monarchs.variables.catalogue`), this
defines all the settings that the user can define in a setup script to
be passed into the model. The aim is to have one centralised place where
we define all settings, and then all application of these settings (putting
them into a format the model understands) and documentation is generated
from this one place.

Each setting is a `Setting` object - which defines the allowed type(s) of the
setting (e.g. True/False, a number or a string), a default value,
and some validation rules or constraints on the allowed values.

This catalogue defines default values - so if you are adding a new
setting ensure that the default behaviour is defined too.

The cross-setting consistency checks (`RULES`) live in `rules.py`.
"""

import numpy as np

from monarchs.config.definitions import REQUIRED, UNSET, Setting
from monarchs.config.rules import MET_SOURCE_INPUTS
from monarchs.config.computed_defaults import (
    lat_grid_size,
    lateral_timestep,
    met_data_source,
    output_grid_size,
)

# default set of grid fields written to the output netCDF
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
    Setting(
        name="row_amount",
        dtype=int,
        default=REQUIRED,
        group="grid",
        description="Number of rows (i.e. ``y``-points) in your model grid, looking from top-down. "
        "MONARCHS indexes the model grid via ``grid[col][row]``, i.e. the ``y``-coordinate "
        "is the second index. This has only been tested with the same # of points as "
        "``col_amount``, so use caution if using different values for each of these.",
    ),
    Setting(
        name="col_amount",
        dtype=int,
        default=REQUIRED,
        group="grid",
        description="Number of columns (i.e. `x`-points) in your model grid, looking from top-down. "
        "MONARCHS indexes the model grid via ``grid[col][row]``, i.e. the ``x``-coordinate "
        "is the first index. This has only been tested with the same # of points as "
        "``row_amount``, so use caution if using different values for each of these.",
    ),
    Setting(
        name="vertical_points_firn",
        dtype=int,
        default=REQUIRED,
        group="grid",
        description="Number of vertical grid cells in the firn profile. Default is 400. "
        "This determines the resolution of the model vertically. "
        "For best results ensure that your vertical resolution is on the order of ~10 cm "
        "(i.e. your maximum firn height divided by ``vertical_points_firn`` is approximately 0.1)",
    ),
    Setting(
        name="vertical_points_lake",
        dtype=int,
        default=REQUIRED,
        group="grid",
        description="Number of vertical grid cells in the lake profile. Default 20, as lakes are both much "
        "smaller than the firn column and less sensitive to vertical resolution since they are "
        "turbulently mixed.",
    ),
    Setting(
        name="vertical_points_lid",
        dtype=int,
        default=REQUIRED,
        group="grid",
        description="Number of vertical grid cells in the lid profile. Default 20.",
    ),
    Setting(
        name="lat_grid_size",
        # 'dem' to take the cell size from the DEM, otherwise the cell size in metres
        dtype=(str, float),
        default=lat_grid_size,
        group="grid",
        default_doc="'dem' when a DEM is provided",
        description="Size of each grid cell in m, if a DEM is provided.",
    ),
    # Timestepping
    Setting(
        name="num_days",
        dtype=int,
        default=REQUIRED,
        group="timestepping",
        description="Number of days (i.e. full model iterations) to run for. A day is ``t_steps_per_day`` "
        "steps of the single-column physics, followed by one pass of the lateral flow algorithm.",
    ),
    Setting(
        name="t_steps_per_day",
        dtype=int,
        default=24,
        group="timestepping",
        description="Number of timesteps to run for each day. 24 = 1h resolution, 8 = 3h resolution, etc.",
    ),
    Setting(
        name="lateral_timestep",
        dtype=int,
        default=lateral_timestep,
        group="timestepping",
        default_doc="model_setup.t_steps_per_day * 3600",
        description="Timestep for each iteration of lateral water flow calculation (in s). "
        "It is highly unlikely this should be anything other than 3600 * 24.",
    ),
    # DEM / geography
    Setting(
        name="latmax",
        dtype=float,
        default=np.nan,
        group="dem",
    ),
    Setting(
        name="latmin",
        dtype=float,
        default=np.nan,
        group="dem",
    ),
    Setting(
        name="longmax",
        dtype=float,
        default=np.nan,
        group="dem",
    ),
    Setting(
        name="longmin",
        dtype=float,
        default=np.nan,
        group="dem",
    ),
    Setting(
        name="bbox_top_right",
        dtype=list,
        default=False,
        group="dem",
        description="Corner of a bounding box constraining the model domain, given as ``[lat, long]``. "
        "Set all four corners to run on a subset of a large DEM. "
        "With ``lat_bounds == 'dem'`` the input met data is regridded to these bounds. "
        "(i.e. the met data shifts with the choices made here).",
    ),
    Setting(
        name="bbox_bottom_left",
        dtype=list,
        default=False,
        group="dem",
        description="Corner of the model bounding box - see ``bbox_top_right``.",
    ),
    Setting(
        name="bbox_top_left",
        dtype=list,
        default=False,
        group="dem",
        description="Corner of the model bounding box - see ``bbox_top_right``.",
    ),
    Setting(
        name="bbox_bottom_right",
        dtype=list,
        default=False,
        group="dem",
        description="Corner of the model bounding box - see ``bbox_top_right``.",
    ),
    Setting(
        name="input_crs",
        dtype=int,
        default=3031,
        group="dem",
    ),
    Setting(
        name="firn_max_height",
        dtype=float,
        default=150,
        group="dem",
        description="Maximum height that your firn column can be at. "
        "Use this if you're loading in a DEM which has large height ranges.",
    ),
    Setting(
        name="firn_min_height",
        dtype=float,
        default=20,
        group="dem",
        description='Minimum height that we consider to be "firn". Anything below this we consider to be solid ice, '
        "which affects some of the physics. ",
    ),
    Setting(
        name="max_height_handler",
        dtype=str,
        default="filter",
        group="dem",
        description="How to handle cells that exceed ``firn_max_height``, designed to help us filter out land cells. "
        "``'filter'`` sets ``cell.valid_cell = False``, so no physics is run on the cell and it stays "
        "the same throughout the model. ``'clip'`` caps those cells at ``firn_max_height`` "
        "and keeps running physics on them.",
    ),
    Setting(
        name="min_height_handler",
        dtype=str,
        default="filter",
        group="dem",
        description="Handle regions on a DEM below the specified minimum height. Possible options are "
        "``'extend'`` (add height to all "
        "cells to bring the lowest area up to the minimum), ``'normalise'`` (keep heights as-is, adjust "
        "density to account for the lower height), ``'filter'`` (set the cells invalid so physics doesn't run), "
        "and ``'clip'`` (bring everything up to the minimum without keeping the relative values).",
    ),
    Setting(
        name="dem_diagnostic_plots",
        dtype=bool,
        default=False,
        group="dem",
        description="If ``True``, generate some plots to check that we have read in the DEM correctly. "
        "This is useful if using a bounding box to select a subset of the original DEM, "
        "so you can check visually that it is working as intended.",
    ),
    Setting(
        name="DEM_path",
        dtype=str,
        default=UNSET,
        group="dem",
        description="Path to a digital elevation model (DEM) to be read in by MONARCHS, according to its filetype and"
        " interpolated to shape ``(row_amount, col_amount)``. Provide this or ``firn_depth``. "
        "If using a relative path, it is relative to the folder you are running MONARCHS from, "
        "not the folder that the code repository is included in - e.g. if your DEM is in "
        "``/home/data/DEM/test_dem.tif`` and you run MONARCHS from ``/home/model_runs/test_run``, "
        "the DEM path would be ``'../../data/DEM/test_dem.tif'``.",
    ),
    Setting(
        name="lat_bounds",
        dtype=str,
        default=UNSET,
        group="dem",
        choices=("dem",),
        description="Toggle whether to constrain the input met data file to lat/long bounds specified by a "
        "digital elevation model (DEM) or not. If set to ``'dem'``, then the model grid and input "
        "meteorological data are constrained to the lat/long of the DEM, i.e. "
        "the data from the met data netCDF is matched/regridded to the DEM, accounting for changes "
        "in e.g. the coordinate reference systems between the two. "
        "See ``examples/50x50_numba_parallel`` for an example, which has "
        "``met_dem_diagnostic_plots == True`` so a plot is generated showing what this does visually.",
    ),
    # Initial conditions
    Setting(
        name="rho_init",
        dtype=(str, float),
        default="default",
        group="initial conditions",
        description="Initial density profile. This follows Paterson, W. (2000). The Physics of Glaciers. "
        "Butterworth-Heinemann, using the formula of *Schytt, V. (1958). Glaciology. "
        "Norwegian-British-Swedish Antarctic Expedition, 1949-5, IV*. With 'default', "
        "MONARCHS calculates an empirical density profile with ``rho_sfc`` = ``500`` "
        "and ``z_t`` = ``37``. Alternatively, specify either a) a pair of points in the form "
        "``[rho_sfc, zt]`` to use this equation and specify ``rho_sfc`` and ``z_t`` yourself, "
        "b) a 1D array of length ``vertical_points_firn`` to specify a uniform density profile "
        "across the whole grid, or c) an array of dimension(``row_amount``, ``col_amount``, "
        "``vertical_points_firn``) to specify different density profiles across your model grid.",
    ),
    Setting(
        name="T_init",
        dtype=(str, float),
        default="default",
        group="initial conditions",
        description="Initial temperature profile. With 'default', MONARCHS uses an assumed firn top temperature of "
        "260 K and bottom temperature of 240 K, linearly interpolated between these points. "
        "Alternatively, specify either a) a pair of points in the form [top, bottom] to assume a "
        "linear temperature profile across the whole grid, b) a 1D array of length "
        "``vertical_points_firn`` to specify a uniform temperature profile across the whole grid, "
        "or c) an array of dimension(``row_amount``, ``col_amount``, ``vertical_points_firn``) "
        "to specify different temperature profiles across your model grid.",
    ),
    Setting(
        name="rho_sfc",
        dtype=float,
        default=500,
        group="initial conditions",
        description="Initial surface density used to calculate the profile if using ``rho_init`` = 'default'. "
        "Defaults to 500.",
    ),
    Setting(
        name="firn_depth",
        dtype=float,
        default=UNSET,
        group="initial conditions",
        description="Initial depth of the firn columns making up the MONARCHS model grid. "
        "Provide this or ``DEM_path`` - **if a valid DEM path is specified, then this is overridden "
        "by the DEM**. Specify as either a number, assumed as the firn depth across the whole grid, "
        "or an array of dimension(``row_amount``, ``col_amount``) giving the firn depth across the "
        "whole grid."
        "e.g. ``gaussian_testcase.export_gaussian_dem`` in the ``10x10_gaussian_threelake`` "
        "example generates a usable Gaussian three-lake profile.",
    ),
    Setting(
        name="initial_conditions",
        dtype=dict,
        default=UNSET,
        group="initial conditions",
        description="Optional {grid variable: value} overrides applied when building the initial grid, e.g. "
        "{'lake_depth': 0.5}. Keys must be names from the variable catalogue (monarchs.variables), "
        "with appropriate values (scalar or array depending on the field).",
    ),
    # Forcing data
    Setting(
        name="met_data_source",
        dtype=str,
        default=met_data_source,
        group="forcing",
        choices=tuple(MET_SOURCE_INPUTS),
        default_doc="inferred from the input given",
        description="Format of the meteorological forcing. Depending on the source, the met data will be set up from "
        "the corresponding data input - e.g. an ERA5 forcing file if 'era5' is specified - see "
        "``monarchs.met_data.sources``. If you do not set this, it is inferred from whichever input you "
        "provide - ``met_input_filepath`` gives 'ERA5', ``met_data`` gives 'user_defined'.",
    ),
    Setting(
        name="met_timestep",
        dtype=str,
        default="hourly",
        group="forcing",
        description="Temporal resolution of your input meteorological data. Ideally, MONARCHS would read in hourly "
        "gridded data. However, it is possible that the user may want to run long climate simulation runs, "
        "which may necessitate lower temporal resolution. If str, the value should be 'hourly', "
        "'three-hourly' or 'daily'. For other resolutions, please specify an integer, corresponding "
        "to how many hours each point in your data corresponds to - 'hourly' is 1, 'three_hourly' "
        "is 3, and 'daily' is 24.",
    ),
    Setting(
        name="met_output_filepath",
        dtype=str,
        default="interpolated_met_data.nc",
        group="forcing",
        description="Filepath for the interpolated grid used by MONARCHS to be saved. This is used to save memory, "
        "and prevent us from having to repeatedly interpolate our input data. This file can be large "
        "if running for large domains and timescales. Therefore,this setting is useful for those who "
        "e.g. want to save this file into scratch space rather than locally.",
    ),
    Setting(
        name="met_dem_diagnostic_plots",
        dtype=bool,
        default=False,
        group="forcing",
        description="If ``True``, generate some plots to show the regridding of the meteorological data onto the "
        "DEM lat/long grid. Useful as a sanity check to make sure that this has worked as intended. "
        "Typically you might run a test (in serial, on a local machine) where you cancel the run "
        "during the first model day to check these plots, then re-run (in parallel, possibly on HPC) "
        "with this set to ``False``.",
    ),
    Setting(
        name="load_precalculated_met_data",
        dtype=bool,
        default=False,
        group="forcing",
    ),
    Setting(
        name="met_input_filepath",
        dtype=str,
        default=UNSET,
        group="forcing",
        description="Path to a file of meteorological data to be used as a driver to MONARCHS. "
        "Provide this or ``met_data``. At the moment, only ERA5 format (in netCDF) is supported. "
        "If this is a relative filepath, then you should ensure that is relative to the folder "
        "in which you are running MONARCHS from, not the source code directory.",
    ),
    Setting(
        name="met_data",
        dtype=dict,
        default=UNSET,
        group="forcing",
        description="User-defined met data as a dict. Provide this or met_input_filepath.",
    ),
    Setting(
        name="radiation_forcing_factor",
        dtype=float,
        default=1,
        group="forcing",
        description="Multiply the shortwave/longwave radiation variables by this factor for testing purposes, e.g. "
        "if running a 1D case and you want to ensure that lake formation occurs. This is left in "
        "mostly as an example; see ``monarchs.met_data.setup_met_data.scale_by_factor`` for more details.",
    ),
    # Physics toggles (debug switches)
    Setting(
        name="snowfall_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        kernel_toggle=True,
        description="Determines whether to add height to the firn column via snowfall over time, or not. "
        "e.g. can be turned off if you don't have a source of snowfall data.",
    ),
    Setting(
        name="firn_column_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        kernel_toggle=True,
        description="Determines whether the firn column is allowed to evolve or not, i.e. if "
        "``physics.firn_column`` is ever invoked.",
    ),
    Setting(
        name="firn_heat_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        kernel_toggle=True,
        description="Determines whether the temperature of the firn is allowed to evolve, "
        "i.e. if ``physics.heateqn`` is ever invoked.",
    ),
    Setting(
        name="percolation_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        kernel_toggle=True,
        description="Determines whether water can percolate during the firn column evolution step.",
    ),
    Setting(
        name="perc_time_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        kernel_toggle=True,
        description="Determines if percolation occurs over timescales (if ``True``), or all water can percolate "
        "forever until it saturates a cell (as in the old MATLAB model).",
    ),
    Setting(
        name="lake_development_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        kernel_toggle=True,
        description="Determines whether lakes are allowed to form, i.e. "
        "if ``physics.lake_development`` is ever invoked.",
    ),
    Setting(
        name="lid_development_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        kernel_toggle=True,
        description="Determines whether frozen lids are allowed to form, "
        "i.e. if ``physics.lid_development`` is ever invoked.",
    ),
    Setting(
        name="lateral_movement_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        description="Determines whether water can move between grid points laterally, "
        "or if we treat each column as entirely independent.",
    ),
    Setting(
        name="lateral_movement_percolation_toggle",
        dtype=bool,
        default=True,
        group="toggles",
        description="Determines whether water can percolate during the lateral movement step, "
        "assuming that ``lateral_movement_toggle`` is ``True``.",
    ),
    Setting(
        name="single_column_toggle",
        dtype=bool,
        default=True,
        group="toggles",
    ),
    Setting(
        name="densification_toggle",
        dtype=bool,
        default=False,
        group="toggles",
        kernel_toggle=True,
        description="Determines whether snow densification is enabled. "
        "This is currently always False since our implementation of snow densification is WIP.",
    ),
    # Lateral flow
    Setting(
        name="flow_into_land",
        dtype=bool,
        default=True,
        group="lateral",
        description="If ``True``, then similarly to ``catchment_outflow``, water that reaches "
        "the edge of the grid and is at a local minimum will flow out of the model if it "
        "is adjacent to a land cell. This is motivated by the presence of large lakes at "
        "the edge of the ice shelf in the validation runs, which are not seen in "
        "observational datasets. This occurs since the water has nowhere else to go, "
        "and thus a positive feedback loop occurs where the lake grows, melts the firn underneath, "
        "and more water flows in.",
    ),
    Setting(
        name="catchment_outflow",
        dtype=bool,
        default=False,
        group="lateral",
        description="If ``True``, then water that a) reaches the edge of the grid and b) is at a local "
        "minimum in terms of the cell's water level will disappear from the model, i.e. "
        "it moves outside of the model domain. This may or may not be a good assumption "
        "depending on location.",
    ),
    Setting(
        name="flow_speed_scaling",
        dtype=float,
        default=1.0,
        group="lateral",
        description="Determines how much water can flow from a cell in the firn to its neighbours during the "
        "lateral flow step. The amount of water that can flow is determined by the density of the cell, "
        "and the available water in the cell. This flag scales the amount of water that *can* flow by "
        "the value given - it will not move more water than there is in the cell. It has no effect "
        "on the amount of water that can flow from lakes - since it is assumed that all water is "
        "available to move from lakes.",
    ),
    Setting(
        name="outflow_proportion",
        dtype=float,
        default=1.0,
        group="lateral",
        validator=lambda v: 0.0 <= v <= 1.0,
        validator_doc="between 0.0 and 1.0",
        description="Determines how much water will flow out of the catchment as a proportion of the total "
        "water available to flow out. If ``1``, then all water that is at a local minimum and "
        "adjacent to the edge of the grid will flow out. Setting it to ``0`` will effectively "
        "set ``catchment_outflow`` to ``False``.",
    ),
    # Output / checkpointing
    Setting(
        name="output_grid_size",
        dtype=int,
        default=output_grid_size,
        group="io",
        default_doc="vertical_points_firn",
        description="Size of the vertical grid that you want to write to. This can be different from the "
        "size of the grid used in the actual model calculations, in which case the results are "
        "interpolated to this grid size. Useful to reduce the size of output files, which can be large.",
    ),
    Setting(
        name="output_timestep",
        dtype=int,
        default=1,
        group="io",
        description="Write model output every ``output_timestep`` model days. "
        "Useful if you want to save data less regularly than every timestep, e.g. "
        "if filesizes are getting too large and you don't need daily resolution.",
    ),
    Setting(
        name="vars_to_save",
        dtype=tuple,
        default=_DEFAULT_VARS_TO_SAVE,
        group="io",
        description="Tuple containing the names of the variables that we wish to save during the evolution "
        "of MONARCHS over time. If you want to save a particular diagnostic, then you should add it here. "
        "See the variable catalogue in ``monarchs.variables`` for the full list of variables that "
        "``vars_to_save`` accepts.",
    ),
    Setting(
        name="save_output",
        dtype=bool,
        default=False,
        group="io",
        description="Flag to determine whether you want to save the output of MONARCHS to netCDF. "
        "If True, save the variables defined in ``vars_to_save`` into a netCDF file at "
        "``output_filepath`` every timestep, i.e. save spatial and temporal data for "
        "the selected variables. File sizes can get rather large for large model grids "
        "and long runs. Note that this is separate from dumping, where only a snapshot "
        "of the current iteration is saved - it is not possible to restart MONARCHS "
        "from the output defined here.",
    ),
    Setting(
        name="output_filepath",
        dtype=str,
        default=UNSET,
        group="io",
        description="Path to the file that you want to save output into, including file extension. "
        "Required if ``save_output`` is ``True``. MONARCHS uses netCDF for saving output data, "
        "so this may be e.g. ``/work/monarchs/monarchs_run1.nc``.",
    ),
    Setting(
        name="dump_data",
        dtype=bool,
        default=False,
        group="io",
        description="Flag that determines whether to dump the current model state at the end of each iteration (day). "
        "Doing so will allow the user to restart MONARCHS in the event of a crash. "
        "If this is ``True``, then you also need to specify ``dump_filepath``. "
        "Note that dumping the model state is separate to setting model output - "
        "this only dumps a snapshot of the model in its current state, needed to restart the model.",
    ),
    Setting(
        name="dump_filepath",
        dtype=str,
        default=UNSET,
        group="io",
        description="File path to dump the current model state into at the end of each timestep. "
        "Required if ``dump_data`` or ``reload_from_dump`` are True.",
    ),
    Setting(
        name="dump_timestep",
        dtype=int,
        default=1,
        group="io",
        description="How often to write model checkpoint states. Default is every timestep. "
        "Set to a value greater than 1 to write checkpoints less frequently.",
    ),
    Setting(
        name="dump_checkpoint_frequency",
        dtype=int,
        default=0,
        group="io",
        description="Write an extra numbered checkpoint every N days. 0 = off. Used for debugging mainly.",
    ),
    Setting(
        name="dump_data_pre_lateral_movement",
        dtype=bool,
        default=False,
        group="io",
    ),
    Setting(
        name="reload_from_dump",
        dtype=bool,
        default=False,
        group="io",
        description="Flag to determine whether we want to reload from a dump (see ``dump_data`` for details). "
        "If ``True``, reload model state from file at the path determined by ``dump_filepath``.",
    ),
    # Runtime / performance
    Setting(
        name="use_numba",
        dtype=bool,
        default=True,
        group="runtime",
        description="Toggle whether to jit-compile the code using Numba or not. "
        "Gives a performance boost, but may not always work and adds a few complications. "
        "See the Numba page of the documentation for more details.",
    ),
    Setting(
        name="parallel",
        dtype=bool,
        default=True,
        group="runtime",
        description="Determines whether or not to run in parallel, or serially. If running in parallel, "
        "then performance is improved since the model will run many of the single-column gridpoints "
        "at the same time. Parallelism comes via Numba's ``prange`` function, which works similarly to "
        "an OpenMP ``parallel do`` loop, so this requires ``use_numba``.",
    ),
    Setting(
        name="use_mpi",
        dtype=bool,
        default=False,
        group="runtime",
        description="MPI support is not yet implemented.",
    ),
    Setting(
        name="cores",
        dtype=(str, int),
        default="all",
        group="runtime",
        description="Number of processing cores to use. 'all' or ``False`` will instruct MONARCHS to use all "
        "available CPU cores, else it will use however many you specify. You may want to manually "
        "specify this to something lower than the number of cores on your system if e.g. running "
        "on a laptop which you are using for other purposes, or if running on HPC and you are "
        "experiencing memory bottleneck issues.",
    ),
    Setting(
        name="ignore_errors",
        dtype=bool,
        default=False,
        group="runtime",
        kernel_toggle=True,
        description="If ``True``, then ``monarchs.core.utils.check_correct`` will never be invoked, i.e. "
        "the model may be free to evolve into an unphysical state. Errors may still occur, "
        "but these will be Python errors rather than MONARCHS errors if so. "
        "May be useful for debugging.",
    ),
]
