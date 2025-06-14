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
vertical_points_firn = 500  # Number of vertical grid cells
vertical_points_lake = 20  # Number of vertical grid cells in lake
vertical_points_lid = 20  # Number of vertical grid cells in ice lid

"""
Timestepping parameters
"""
num_days = 105  # number of days to run the model for (assuming t_steps = 24 below)
t_steps_per_day = 24  # hours to run in each iteration, i.e. 24 = 1h resolution
lateral_timestep = 3600 * t_steps_per_day  # Timestep for each iteration of lateral
# water flow calculation (in s)
# It is highly unlikely this should be anything other than 3600 * t_steps.

"""
Model initial conditions (density/temperature profiles)
"""
rho_init = "default"  # Initial density, use 'default' to use empirical formula for initial density profile
T_init = "default"  # Initial temperature profile.
rho_sfc = 500  # Initial surface density, if using empirical formula for initial density profile. Otherwise, it is 500.
firn_depth = 35.0 * np.ones(
    (row_amount, col_amount)
)  # needs to be an array as MONARCHS expects a 2D field

"""
Meteorological parameters and input
"""

met_output_filepath = "output/met_data_1d_testcase.nc"

# Set up user-defined meteorological data.
# We exploit the fact that this is a Python script, so can set up our data in Numpy arrays.
# If we wanted to extend this to a 2x2 case, we could make use of
# np.broadcast_to to broadcast our data to the correct shape.
# e.g. np.broadcast_to(met_data["LW_surf"], (row_amount, col_amount, len(met_data["LW_surf"])))
met_data = {}
met_data["LW_surf"] = np.append(
    800 * np.ones(800), 50 * np.ones(1720)
)  # Incoming longwave radiation. [W m^-2].
met_data["SW_surf"] = np.append(
    800 * np.ones(800), 100 * np.ones(1720)
)  # Incoming shortwave (solar) radiation. [W m^-2].
met_data["temperature"] = np.append(
    267 * np.ones(800), 250 * np.ones(1720)
)  # Surface-layer air temperature. [K].
met_data["pressure"] = 1000 * np.ones(
    num_days * t_steps_per_day
)  # Surface-layer air pressure. [hPa].
met_data["dew_point_temperature"] = np.append(
    265 * np.ones(800), 240 * np.ones(1720)
)  # Dew-point temperature. [K].
met_data["wind"] = 5 * np.ones(num_days * t_steps_per_day)  # Wind speed. [m s^-1].
met_data["snowfall"] = 0 * np.ones(
    num_days * t_steps_per_day
)  # Snowfall rate. [m s^-1].
met_data["snow_dens"] = 300 * np.ones(
    num_days * t_steps_per_day
)  # Snow density. [kg m^-3].

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
output_filepath = "output/1d_testcase_outputnew.nc"  # Filename for model output, including file extension (.nc for netCDF).
output_grid_size = vertical_points_firn  # Size of interpolated output
lateral_movement_toggle = False
firn_column_toggle = True
lake_development_toggle = True
lid_development_toggle = True

"""
Dumping and reloading parameters
"""
dump_data = True
dump_filepath = "output/1d_testcase_dumpnew.nc"  # Filename of our previously dumped state
reload_from_dump = False  # Flag to determine whether to reload the state or not
use_numba = True
if __name__ == "__main__":
    from monarchs.core.driver import monarchs

    grid = monarchs()
