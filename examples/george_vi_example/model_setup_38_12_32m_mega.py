# TODO - check TODOs in docstrings, move docstrings to documentation file but don't include in final runscript
"""
Run script for MONARCHS - 38_12_32m_v2.0 case.
"""

import os
import numpy as np

print(f"Loading runscript from {os.getcwd()}/model_setup.py")
"""
Spatial parameters
"""
work_dir = '/work/ta170/ta170/jelsey92/MONARCHS/runs/38m_dem_10year' # shorthand useful for specifying output filepaths later

row_amount = 100  # Number of rows in your model grid, looking from top-down.
col_amount = 100  # Number of columns in your model grid, looking from top-down.
lat_grid_size = 'dem'  # size of each lateral grid cell in m
vertical_points_firn = 500  # Number of vertical grid cells
vertical_points_lake = 20  # Number of vertical grid cells in lake
vertical_points_lid = 20  # Number of vertical grid cells in ice lid
# Latitude/longitude. Set to 'dem' to use the boundaries from the DEM itself if using. Set np.nan to ignore entirely.
# Set to a number if you want to manually specify a bounding box.
lat_bounds = 'dem'

"""
Timestepping parameters
"""
num_days = 365 * 10  # number of days to run the model for (assuming t_steps = 24 below)
t_steps_per_day = 24  # hours to run in each iteration, i.e. 24 = 1h resolution
lateral_timestep = 3600 * t_steps_per_day  # Timestep for each iteration of lateral water flow calculation (in s)


"""
DEM parameters
"""
DEM_path = '/work/ta170/ta170/jelsey92/MONARCHS/data/DEM/38_12_32m_v2.0/38_12_32m_v2.0_dem.tif'
# these options say - filter out all cells above 100m (not ice shelf), and set everything below 35m to 35m 
# (adjusting the height of other cells up to accommodate)
firn_max_height = 100  
firn_min_height = 35
max_height_handler = "filter"
min_height_handler = "extend"

"""
Physical parameters
"""
rho_init = "default"  # Initial density, use 'default' to use empirical formula for initial density profile
T_init = "default"  # Initial temperature profile.
rho_sfc = 500  # Initial surface density, if using empirical formula for initial density profile. Otherwise, it is 500.

met_input_filepath = "/work/ta170/ta170/jelsey92/MONARCHS/data/met_data/era5_megafile.nc"
met_output_filepath = f"{work_dir}/met_data.nc"

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
) # this can be the name of any variable contained within IceShelf.
output_filepath = f"{work_dir}/model_output.nc"  # Filename for model output, including file extension (.nc for netCDF).
output_grid_size = 100  # Size of interpolated output - 400 is the default vert_grid_size so no interpolation in this case
output_timestep = 30
"""
Dumping and reloading parameters

    If the code fails, and we have a dump file, then by setting <reload_state> = True the model will pick up
    the dump file and run from there.
"""

dump_data = True
dump_filepath = (
    f"{work_dir}/progress.nc"  # Filename of our previously dumped state
)
reload_state = False  # Flag to determine whether to reload the state or not

"""
Computing and numerical parameters
"""
use_numba = False  # Use Numba-optimised version (faster, but harder to debug)
parallel = True  # run in parallel or serial. Parallel is of course much faster for large model grids, but you may
# wish to run serial if doing single-column calculations.
cores = 'all'  # number of processing cores to use. 'all' or False will tell MONARCHS to use all available cores.
# I would be a little careful with this on the CPOM server, as we don't have a scheduler. Using all available
# cores will make the system unusable for anyone else. 

catchment_outflow = True
flow_into_land = True
