# TODO - check TODOs in docstrings, move docstrings to documentation file but don't include in final runscript
"""
Run script for MONARCHS.
All parameters that have default values will use these defaults if they are not specified in the runscript.
The template includes all MONARCHS parameters explicitly.
Since this is a Python script, you can specify parameters e.g. as numpy arrays.
"""

import os
import numpy as np

print(f"Loading runscript from {os.getcwd()}/model_setup.py")
"""
Spatial parameters

    row_amount : int
        Number of rows (i.e. `y`-points) in your model grid, looking from top-down. 
        MONARCHS indexes the model grid via `grid[col][row]`, i.e. the `y`-coordinate is the second index.

    col_amount : int
        Number of columns (i.e. `x`-points) in your model grid, looking from top-down.
        MONARCHS indexes the model grid via `grid[col][row]`, i.e. the `x`-coordinate is the first index.
    
    lat_grid_size : float, or str
        Size of each grid cell in m. This is used to determine how much water can flow during the lateral
        flow calculations. If set to a number, then the cells are assumed square. If set to `'dem'`, then the `x` and
        `y` dimensions are calculated separately - in which case the cells are not necessarily assumed to be square.
        This value is stored in the IceShelf class as `cell.grid_size_dx` and `cell.grid_size_dy`
"""
col_amount = 50  # Number of columns in your model grid, looking from top-down.
row_amount = 50  # Number of rows in your model grid, looking from top-down.
lat_grid_size = 'dem'  # size of each lateral grid cell in m - can either be a number or 'dem' to calculate
# x and y grid sizes from the DEM itself.
vertical_points_firn = 400  # Number of vertical grid cells
# (i.e. firn_depth/vertical_points_firn = height of each grid cell)
vertical_points_lake = 20  # Number of vertical grid cells in lake
vertical_points_lid = 20  # Number of vertical grid cells in ice lid
# Latitude/longitude. Set to 'dem' to use the boundaries from the DEM itself if using. Set np.nan to ignore entirely.
# Set to a number if you want to manually specify a bounding box.
lat_bounds = 'dem'
latmax = np.nan  # Maximum latitude to use in our DEM and met data files.
latmin = np.nan  # Minimum latitude to use in our DEM and met data files.
longmax = np.nan  # Maximum longitude to use in our DEM and met data files.
longmin = np.nan  # Minimum longitude to use in our DEM and met data files.

bbox_top_right = [
    (-66.52, -62.814)
]  # bounding box top right coordinates, [(lat, long)]
bbox_bottom_left = [
    (-66.289, -64.68)
]  # bounding box bottom left coordinates, [(lat, long)]
bbox_top_left = [
    (-66.04, -63.42)
]  # bounding box top left coordinates, [(lat, long)]
bbox_bottom_right = [
    (-66.778, -64.099)
]  # bounding box bottom right coordinates, [(lat, long)]

"""
Timestepping parameters
"""
num_days = 1000  # number of days to run the model for (assuming t_steps = 24 below)
t_steps_per_day = 24  # hours to run in each iteration, i.e. 24 = 1h resolution
lateral_timestep = 3600 * t_steps_per_day  # Timestep for each iteration of lateral
# water flow calculation (in s)
# It is highly unlikely this should be anything other than 3600 * t_steps.

"""
DEM/initial firn profile

    DEM_path : str
        Path to a digital elevation model (DEM) to be read in by MONARCHS.
        This will be read in by MONARCHS according to its filetype, and 
        interpolated to size(<row_amount>, <col_amount>).
        If using a relative import, it is a relative import from the folder you are running
        MONARCHS from, not the folder that the code repository is included in. 

    firn_depth : float, or array_like, float, dimension(<row_amount, <col_amount>)
        Initial depth of the firn columns making up the MONARCHS model grid.
        If a valid DEM path is specified, then this is overridden by the DEM. Use this only if you want to manually
        specify your own DEM path. Specify as either a number or an array. 
        If a number is specified, this number is assumed as the firn depth across the whole grid.
        If an array is specified, this should be an array of dimension(<row_amount>, <col_amount>), 
        i.e. the firn depth is user-specified across the whole grid. This is likely the safest option if you want to
        pre-process your firn profile, or don't trust MONARCHS to interpolate it to your desired model grid for you.

    firn_max_height : float
        Maximum height that your firn column can be at. Use this if you're loading in a DEM which has large height
        ranges. 
    firn_min_height : float
        Minimum height that we consider to be "firn". Anything below this we consider to be solid ice, which affects
        some of the physics. Notably, we will see a lot more surface water in these cells. 
    max_height_handler : str
        How to handle cells where that exceed the maximum firn height. This is designed to help us filter out land cells.
        These are still part of the overall grid, since they occupy geographic space, but are not useful in terms of 
        the physics. This variable can be one of the following:
            'filter' - Set the variable cell.valid_cell = False, which prevents MONARCHS from running any of the physics
            to these cells. This means they effectively stay the same throughout the whole model.
            'clip' - Set all cells above the max firn height to ``firn_max_height``. This will not prevent MONARCHS
            from running physics on these cells.
    input_crs: int
        Coordinate reference system of the input data. Default is 3038, i.e. WGS84 Antarctic Polar Stereographic.
        
"""

DEM_path = 'DEM/38_12_32m_v2.0/38_12_32m_v2.0_dem.tif'
# DEM_path = "DEM/42_07_32m_v2.0/42_07_32m_v2.0_dem.tif"

# firn_depth - by default overridden by the presence of a valid DEM
firn_depth = np.ones((row_amount, col_amount)) * 35
firn_max_height = 100
firn_min_height = 35
max_height_handler = "filter"
min_height_handler = "extend"
input_crs = 3031  # Coordinate reference system of the input data

"""
Model initial conditions (density/temperature profiles)

    rho_init : str, or array_like, float
        Initial density profile. 
        This follows Paterson, W. (2000). The Physics of Glaciers. Butterworth-Heinemann, 
        using the formula of Schytt, V. (1958). Glaciology. A: Snow studies at Maudheim. Glaciology. B: Snow studies
        inland. Glaciology. C: The inner structure of the ice shelf at Maudheim as shown by
        core drilling. Norwegian- British- Swedish Antarctic Expedition, 1949-5, IV.)

        Defaults to 'default', in which case MONARCHS will calculate an empirical density profile with <rho_sfc> = 500 
        and <z_t> = 37.        
        Alternatively, specify as either a) a pair of points in the form [rho_sfc, zt] to use this equation and specify 
        <rho_sfc> and <z_t> yourself, b) a 1D array of length <vertical_points> to specify a user-specified
        uniform density profile across the whole grid, or c) an array of 
        dimension(<row_amount>, <col_amount>, <vertical_points_firn>) to specify different density profiles across your
        model grid. 

    T_init : str, or array_like, float
        Initial temperature profile. 
        Defaults to 'default', which MONARCHS reads in and uses an assumed firn top temperature of 260 K and
        bottom temperature of 240 K, linearly interpolated between these points.
        Alternatively, specify as either a) a pair of points in the form [top, bottom] to assume a linear
        temperature profile across the whole grid, b) a 1D array of length <vertical_points> to specify a user-specified
        uniform temperature profile across the whole grid, or c) an array of 
        dimension(<row_amount>, <col_amount>, <vertical_points_firn>) to specify different temperature profiles across 
        your model grid. 

    rho_sfc: float
        Initial surface density used to calculate the profile if using `rho_init` = 'default'.
"""
rho_init = "default"  # Initial density, use 'default' to use empirical formula for initial density profile
T_init = "default"  # Initial temperature profile.
rho_sfc = 500  # Initial surface density, if using empirical formula for initial density profile. Otherwise, it is 500.

"""
Meteorological parameters and input
    met_input_filepath : str
        Path to a file of meteorological data to be used as a driver to MONARCHS.
         At the moment, only ERA5 format (in netCDF) is supported. 
         If this is a relative filepath, then you should ensure that is relative to the folder in which
         you are running MONARCHS from, not the source code directory.
         # TODO - write full list of variable names that can be read into MONARCHS

    met_start_index : int
        If specified, start reading the data from <met_input> at this index. Useful if you e.g. have a met data file
        that starts at a point sooner than you want to run MONARCHS from.
        This only affects runs starting at iteration 0, i.e. runs that have not been reloaded from a dump. 
        Such runs will continue from the index it would have run next were the code not to have stopped regardless
        of this parameter.

    met_timestep : str, or int
        Default 'hourly'.
        Temporal resolution of your input meteorological data. 
        Ideally, MONARCHS would read in hourly gridded data. However, it is possible that the user may want
        to run long climate simulation runs, which may necessitate lower temporal resolution. This flag tells
        MONARCHS how often the meteorological input data should be run for.
        If str - the value should be 'hourly', 'three-hourly' or 'daily'. For other resolutions, please 
        specify an integer, corresponding to how many hours each point in your data corresponds to. 
        In this integer form, 'hourly' corresponds to met_timestep = 1, 'three_hourly' to met_timestep = 3, and 
        'daily' to met_timestep = 24.

    met_output_filepath : str
        Filepath for the interpolated grid used by MONARCHS to be saved. 
        Default 'interpolated_met_data.nc'.
        This is used to save memory, and prevent us from having to repeatedly interpolate our input data.
        This file can be large if running for large domains and timescales. Therefore,this setting is useful 
        for those who e.g. want to save this file into scratch space rather than locally.
"""

# met_input_filepath = "data/ERA5_new_dem_fixed.nc"
met_input_filepath = "data/ERA5_small.nc"

met_start = 0  # Index at which to start the met data, in case you want to start the model from an intermediate point.
# It will roll the array so that it fits this length.

met_timestep = "hourly"
met_output_filepath = "met_data.nc"

"""
Model output

    save_output : bool
        Default True.
        Flag to determine whether you want to save the output of MONARCHS to netCDF. If True, save the variables
        defined in <vars_to_save> into a netCDF file at <output_filepath> every timestep (i.e. save spatial and temporal
        data for the selected variables). Note that the file sizes can get rather large for large model grids and long
        runs.
        Note that this is separate from dumping, where only a snapshot of the current iteration is saved. It is not 
        possible to restart MONARCHS from the output defined here. See the documentation on dumping and reload
        parameters for information on how to enable restarting MONARCHS.
        # TODO - hyperlink to documentation on dumping

    vars_to_save : tuple, str
        Default ('firn_temperature', 'Sfrac', 'Lfrac', 'firn_depth', 'lake_depth', 'lid_depth', 'lake', 'lid', 'v_lid').
        Tuple containing the names of the variables that we wish to save during the evolution of MONARCHS over time.
        See <iceshelf_class> for details on the full list of variables that <vars_to_save> accepts.
        # TODO - flag so that if var in vars to save not in iceshelf_class, flag this and either exit or write a warning

    output_filepath : str
        Path to the file that you want to save output into, including file extension. 
        MONARCHS uses netCDF for saving output data, so you should include ".nc" at the end of your filepath.
    output_grid_size : int

        Size of the vertical grid that you want to write to. This can be different from the size of the grid used in the 
        actual model calculations, in which case the results are interpolated to this grid size. Useful to reduce the 
        size of output files, which can be large.
"""
save_output = True
vars_to_save = (
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
output_filepath = "../MONARCHS_runs/sample_output.nc"  # Filename for model output, including file extension (.nc for netCDF).
output_grid_size = 200  # Size of interpolated output
output_timestep = 1
"""
Dumping and reloading parameters

    dump_data : bool
        Flag that determines whether to dump the current model state at the end of each iteration (day). Doing so
        will allow the user to restart MONARCHS in the event of a crash. Set True to enable this behaviour.
        If this is True, then you also need to specify <reload_filename>. Note that dumping the model state is separate
        to setting model output - this only dumps a snapshot of the model in its current state, needed to restart the 
        model. If you desire output over time, see the section on model output.
        # TODO - hyperlink to model output doc 
    dump_filepath : str
        File path to dump the current model state into at the end of each timestep, 
        for use if <dump_data> or <reload_state> are True. 
    reload_state : bool
        Flag to determine whether we want to reload from a dump (see <dump_data> for details). If True, reload model
        state from file at the path determined by <reload_filepath>.
"""
dump_data = True
dump_filepath = (
    "../MONARCHS_runs/progress_df.nc"  # Filename of our previously dumped state
)
reload_state = False  # Flag to determine whether to reload the state or not
dump_format = 'NETCDF4' # Format to save the dump file in. Default is NETCDF4, but can be changed to "pickle"
"""
Computing and numerical parameters
"""
use_numba = False  # Use Numba-optimised version (faster, but harder to debug)
parallel = True  # run in parallel or serial. Parallel is of course much faster for large model grids, but you mayTru
# wish to run serial if doing single-column calculations.
use_mpi = False  # Enable to use MPI-based parallelism for HPC, if running on a non-cluster machine set this False
# Note that this is not yet compatible with Numba. The code will fail if you attempt to run with both
# this switch and use_numba both True.
spinup = False  # Try and force the firn column heat equation to converge at the start of the run?
verbose_logging = False  # if True, output logs every "timestep" (hour). # Otherwise, log only every "iteration" (day).
cores = 24  # number of processing cores to use. 'all' or False will tell MONARCHS to use all available cores.

"""
Toggles to turn on or off various parts of the model. These should only be changed for testing purposes. 
All of these default to True.
"""
snowfall_toggle = True
firn_column_toggle = True
firn_heat_toggle = (
    True  # if firn_column_toggle is False, this just triggers during lake formation
)
lake_development_toggle = True  # also triggers lake formation
lid_development_toggle = True  # also triggers lid formation
lateral_movement_toggle = True
lateral_movement_percolation_toggle = True
densification_toggle = False
percolation_toggle = True  # only works if firn_column_toggle also True
perc_time_toggle = True  # Determines if percolation occurs over timescales,
# or all water can percolate until it can no longer move
catchment_outflow = False  # Determines if water on the edge of the catchment area will
# preferentially stay within the model grid,
# or flow out of the catchment area (resulting in us 'losing' water)
flow_into_land = False  # Determines if water will flow into land cells at local minima
"""
Other flags for doing tests - e.g. adding water from outside catchment area
"""
simulated_water_toggle = False  # 0.001  # False if off, otherwise float
if simulated_water_toggle:
    print("model_setup: Simulated water is on")
ignore_errors = False  # don't flag if model reaches unphysical state
heateqn_res_toggle = False  # True for testing low resolution heat equation runs

met_dem_diagnostic_plots = False
radiation_forcing_factor = 1

if __name__ == '__main__':
    from monarchs.core.driver import monarchs

    monarchs()