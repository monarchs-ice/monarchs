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
"""
row_amount = 1  # Number of rows in your model grid, looking from top-down.
col_amount = 1  # Number of columns in your model grid, looking from top-down.
lat_grid_size = 1000  # size of each lateral grid cell in m - possible to automate
vertical_points_firn = 400  # Number of vertical grid cells
vertical_points_lake = 20  # Number of vertical grid cells in lake
vertical_points_lid = 20  # Number of vertical grid cells in ice lid


"""
Timestepping parameters
"""
num_days = 20  # number of days to run the model for (assuming t_steps = 24 below)
t_steps_per_day = 24  # hours to run in each iteration, i.e. 24 = 1h resolution
lateral_timestep = 3600 * t_steps_per_day  # Timestep for each iteration of lateral
# water flow calculation (in s)
# It is highly unlikely this should be anything other than 3600 * t_steps.
firn_depth = np.ones((row_amount, col_amount)) * 35


"""
Model initial conditions (density/temperature profiles)
"""
rho_init = "default"  # Initial density, use 'default' to use empirical formula for initial density profile
T_init = "default"  # Initial temperature profile.
rho_sfc = 500  # Initial surface density, if using empirical formula for initial density profile. Otherwise, it is 500.

"""
Meteorological parameters and input
"""

met_input_filepath = "../../data/ERA5_small.nc"
met_start = 0  # Index at which to start the met data, in case you want to start the model from an intermediate point.
# It will roll the array so that it fits this length.

met_timestep = "hourly"
met_output_filepath = "output/met_data_1d_testcase.nc"

"""
Model output
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
output_filepath = "output/1d_testcase_output.nc"  # Filename for model output, including file extension (.nc for netCDF).
output_grid_size = 400  # Size of interpolated output

"""
Dumping and reloading parameters
"""
dump_data = True
dump_filepath = (
    "output/1d_testcase_dump.nc"  # Filename of our previously dumped state
)
reload_state = False  # Flag to determine whether to reload the state or not

if __name__ == '__main__':
    from monarchs.core.driver import monarchs
    monarchs()