"""
Run script for MONARCHS.
All parameters that have default values will use these defaults if they are
not specified in the runscript.
Since this is a Python script, you can specify parameters e.g. as numpy arrays.

This template includes all MONARCHS parameters explicitly, along with docstrings
indicating their use. This may be a bit much for a production runscript,
since many parameters are optional and the docstrings may hinder readability.

You can alternatively get a full reference of the available model_setup parameters
at monarchs-ice.github.io/monarchs/model_setup_reference.
"""

import os

import numpy as np

print(f"Loading runscript from {os.getcwd()}/model_setup.py")
"""
Spatial parameters
"""
row_amount = 50  # Number of rows in your model grid, looking from top-down.
col_amount = 50  # Number of columns in your model grid, looking from top-down.
lat_grid_size = 2000  # size of each lateral grid cell in m - possible to automate
# TODO - calc based on DEM automatically
# lat_grid_size = 'dem'
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

# bbox_top_right = [
#     (-66.52, -62.814)
# ]  # bounding box top right coordinates, [(lat, long)]
# bbox_bottom_left = [
#     (-66.289, -64.68)
# ]  # bounding box bottom left coordinates, [(lat, long)]
# bbox_top_left = [
#     (-66.04, -63.42)
# ]  # bounding box top left coordinates, [(lat, long)]
# bbox_bottom_right = [
#     (-66.778, -64.099)
# ]  # bounding box bottom right coordinates, [(lat, long)]

"""
Timestepping parameters
"""
num_days = 20  # number of days to run the model for (assuming t_steps = 24 below)
t_steps_per_day = 24  # hours to run in each iteration, i.e. 24 = 1h resolution
lateral_timestep = 3600 * t_steps_per_day  # Timestep for each iteration of lateral
# water flow calculation (in s)
# It is highly unlikely this should be anything other than 3600 * t_steps.

"""
DEM/firn profile parameters
"""
DEM_path = 'DEM/38_12_32m_v2.0/38_12_32m_v2.0_dem.tif'
# firn_depth - by default overridden by the presence of a valid DEM
# firn_depth = np.ones((row_amount, col_amount)) * 35
firn_max_height = 100
firn_min_height = 15
max_height_handler = "filter"
min_height_handler = "extend"

rho_init = "default"  # Initial density, use 'default' to use empirical formula for initial density profile
T_init = "default"  # Initial temperature profile.
rho_sfc = 500  # Initial surface density, if using empirical formula for initial density profile. Otherwise, it is 500.

"""
Met data parameters
"""
met_input_filepath = "data/ERA5_small.nc"
met_start = 0  # Index at which to start the met data, in case you want to start the model from an intermediate point.
# It will roll the array so that it fits this length.

met_timestep = "hourly"
met_output_filepath = "met_data.nc"

"""
Output parameters
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
output_filepath = "model_output.nc"  # Filename for model output, including file extension (.nc for netCDF).
output_grid_size = 400  # Size of interpolated output
dump_data = True
dump_filepath = (
    "../MONARCHS_runs/progress.nc"  # Filename of our previously dumped state
)
reload_state = False  # Flag to determine whether to reload the state or not

"""
Computing and numerical parameters
"""
use_numba = True  # Use Numba-optimised version (faster, but harder to debug)
parallel = True  # run in parallel or serial. Parallel is of course much faster for large model grids, but you may
# wish to run serial if doing single-column calculations.
use_mpi = False  # Enable to use MPI-based parallelism for HPC, if running on a non-cluster machine set this False
# Note that this is not yet compatible with Numba. The code will fail if you attempt to run with both
# this switch and use_numba both True.
spinup = False  # Try and force the firn column heat equation to converge at the start of the run?
verbose_logging = False  # if True, output logs every "timestep" (hour). # Otherwise, log only every "iteration" (day).
cores = "all"  # number of processing cores to use. 'all' or False will tell MONARCHS to use all available cores.

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
"""
Other flags for doing tests - e.g. adding water from outside catchment area
"""
simulated_water_toggle = False  # 0.001  # False if off, otherwise float
if simulated_water_toggle:
    print("model_setup: Simulated water is on")
ignore_errors = False  # don't flag if model reaches unphysical state
heateqn_res_toggle = False  # True for testing low resolution heat equation runs

met_dem_diagnostic_plots = True
radiation_forcing_factor = 1
