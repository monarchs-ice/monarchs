"""
Sample script to do some basic plotting of model diagnostics. This script will generate contour maps
of the firn and lake depths on the dem_utils.
"""

from netCDF4 import Dataset
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy

path = "C:/Users/jdels/Documents/Work/MONARCHS_runs/ARCHER2_flow_into_land/38m_dem/progress.nc"
data = Dataset(path)

# # contour plots
# projection = ccrs.PlateCarree()
# lat = data.variables['lat']
# lon = data.variables['lon']
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection=projection)
# ax.coastlines()
# vmin = 0
# vmax = firndepth[:].max()
# levels = np.linspace(vmin, vmax, 20)
# ax.contourf(lon, lat, firndepth[:], transform=projection, levels=levels)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection=projection)
# ax.coastlines()
# vmin = 0
# vmax = 2
# levels = np.linspace(vmin, vmax, 20)
# ax.contourf(lon, lat, lakedepth[:], transform=projection, levels=levels)
