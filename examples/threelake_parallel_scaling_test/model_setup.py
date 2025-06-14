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
from monarchs.DEM import create_DEM_GaussianTestCase as cgt

"""
Spatial parameters
"""
row_amount = 50  # Number of rows in your model grid, looking from top-down.
col_amount = 50  # Number of columns in your model grid, looking from top-down.
lat_grid_size = 1000  # size of each lateral grid cell in m - possible to automate
vertical_points_firn = 400  # Number of vertical grid cells
# (i.e. firn_depth/vertical_points_firn = height of each grid cell)
vertical_points_lake = 20  # Number of vertical grid cells in lake
vertical_points_lid = 20  # Number of vertical grid cells in ice lid
# Latitude/longitude. Set to 'dem' to use the boundaries from the DEM itself if using. Set np.nan to ignore entirely.
# Set to a number if you want to manually specify a bounding box.

"""
Timestepping parameters
"""
num_days = 10  # number of days to run the model for (assuming t_steps = 24 below)
t_steps_per_day = 24  # hours to run in each iteration, i.e. 24 = 1h resolution
lateral_timestep = 3600 * t_steps_per_day  # Timestep for each iteration of lateral
# water flow calculation (in s)
# It is highly unlikely this should be anything other than 3600 * t_steps.

"""
DEM/firn profile parameters
"""

firn_depth = 35 * cgt.export_gaussian_DEM(row_amount, diagnostic_plots=False)
rho_init = "default"  # Initial density, use 'default' to use empirical formula for initial density profile
T_init = "default"  # Initial temperature profile.
rho_sfc = 500  # Initial surface density, if using empirical formula for initial density profile. Otherwise, it is 500.
firn_max_height = 100
firn_min_height = 35
max_height_handler = "filter"
min_height_handler = "extend"

"""
Met data parameters
"""
met_data = {}
timesteps_warm = 24 * 10
timesteps_cold = 0
met_data["LW_surf"] = np.append(
    800 * np.ones(timesteps_warm), 100 * np.ones(timesteps_cold)
)  # Incoming longwave radiation. [W m^-2].
met_data["SW_surf"] = np.append(
    800 * np.ones(timesteps_warm), 100 * np.ones(timesteps_cold)
)  # Incoming shortwave (solar) radiation. [W m^-2].
met_data["dew_point_temperature"] = np.append(
    265 * np.ones(timesteps_warm), 240 * np.ones(timesteps_cold)
)  # Dew-point temperature. [K].
met_data["temperature"] = np.append(
    267 * np.ones(timesteps_warm), 250 * np.ones(timesteps_cold)
)  # Surface-layer air temperature. [K].

met_data["pressure"] = 1000 * np.ones(
    num_days * t_steps_per_day
)  # Surface-layer air pressure. [hPa].
met_data["wind"] = 5 * np.ones(num_days * t_steps_per_day)  # Wind speed. [m s^-1].
met_data["snowfall"] = 0 * np.ones(
    num_days * t_steps_per_day
)  # Snowfall rate. [m s^-1].
met_data["snow_dens"] = 300 * np.ones(
    num_days * t_steps_per_day
)  # Snow density. [kg m^-3].
# take these 1D met parameters and broadcast them to the grid we actually want
for key in met_data.keys():
    met_data[key] = np.broadcast_to(
        met_data[key][:, np.newaxis, np.newaxis],
        (len(met_data[key]), row_amount, col_amount),
    )

met_output_filepath = "output/met_data_threelake.nc"
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
    "water_level",
    "water_direction"
)
output_filepath = "output/gaussian_threelake_example_output_new.nc"  # Filename for model output, including file extension (.nc for netCDF).
# output_grid_size = 400  # Size of array outputs for each column (e.g. firn depth). Commented out for this example.
# output_timestep = 1  # How often to save output, in days. Commented out for this example.
dump_data = True
dump_filepath = "output/gaussian_threelake_example_dump.nc"  # Filename of our previously dumped state
reload_from_dump = False  # Flag to determine whether to reload the state or not

"""
Computing and numerical parameters
"""
use_numba = True  # Use Numba-optimised version (faster, but harder to debug)
parallel = True  # run in parallel or serial. Parallel is of course much faster for large model grids, but you may
# wish to run serial if doing single-column calculations.
use_mpi = False

spinup = False  # Try and force the firn column heat equation to converge at the start of the run?
verbose_logging = False  # if True, output logs every "timestep" (hour). # Otherwise, log only every "iteration" (day).
cores = 10  # number of processing cores to use. 'all' or False will tell MONARCHS to use all available cores.
solver = "hybr"

"""
Toggles to turn on or off various parts of the model. These should only be changed for testing purposes. 
All of these default to True.
"""
catchment_outflow = False  # Determines if water on the edge of the catchment area will
# preferentially stay within the model grid,
# or flow out of the catchment area (resulting in us 'losing' water)
flow_into_land = False  # As above, but for flowing into invalid cells in addition to the model edge boundaries.
lateral_movement_toggle = True

# Just for this specific case - assert that the DEM is symmetric
import numpy.testing as npt

npt.assert_array_equal(firn_depth, firn_depth[::-1, ::-1])

if __name__ == "__main__":
    print(f"Loading runscript from {os.getcwd()}/model_setup.py")

    from monarchs.core.driver import monarchs

    from monarchs.core.utils import get_2d_grid
    profile = False
    if profile:
        import cProfile

        def run_target():
            grid = monarchs()
            return grid

        result_container = {}
        cProfile.runctx(
            "result_container['grid'] = run_target()",
            globals(),
            locals(),
            filename="profile_output.prof"
        )
        grid = result_container['grid']
    else:
        grid = monarchs()

    # from matplotlib import pyplot as plt
    #
    # plt.figure()
    # for i in range(len(grid)):
    #     for j in range(len(grid[0])):
    #         if grid[i][j]['lid']:
    #             grid[i][j]['water_level'] = 0
    #
    # plt.imshow(get_2d_grid(grid, 'water_level'))
    # plt.title('water_level')
    # plt.figure()
    # plt.imshow(get_2d_grid(grid, 'lake_depth'))
    # plt.title('Lake depth')
    #
    # import sys
    #
    # sys.path.append('../../scripts')
    # import flow_plot as fp
    #
    # flow_plot = fp.flow_plot
    #
    # from netCDF4 import Dataset
    #
    # a = Dataset(output_filepath)
    #
    # idx = 45
    #
    # def make_fd_plot(a, idx=0):
    #     fig, ax = plt.subplots()
    #     ax.imshow(a.variables['water_level'][idx])
    #     return fig, ax
    #
    #
    # def make_both(a, idx=0):
    #     fig, ax = make_fd_plot(a, idx=idx)
    #     flow_plot(a, netcdf=True, index=idx, fig=fig, ax=ax)
    #
    #
    # make_both(a, idx=30)
    # a.close()