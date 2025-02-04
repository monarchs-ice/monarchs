from monarchs.core.dump_model_state import reload_state
from monarchs.core.utils import get_2d_grid
from matplotlib import pyplot as plt
import numpy as np

path = 'C:/Users/jdels/Documents/Work/MONARCHS_runs/ARCHER2_030225/dump.nc'
#path = "../MONARCHS_runs/progress_df.nc"

# Set up a dummy IceShelf instance, create a grid of these, then write out our dumpfile into this.
class IceShelf():
    pass


row_amount = 100
col_amount = 100

grid = []
for i in range(col_amount):
    _l = []
    for j in range(row_amount):
        _l.append(IceShelf())
    grid.append(_l)

test, _, _, _ = reload_state(path, grid)

# Now we can interact with the model, plot stuff out etc.
lakedepth = get_2d_grid(grid, 'lake_depth')
plt.imshow(lakedepth)
plt.colorbar()
plt.title('Lake depth')
plt.figure()
lakedepth = get_2d_grid(grid, 'lake_depth')
plt.imshow(lakedepth, vmax=2, cmap='magma')
plt.colorbar()
plt.title('Lake depth (max shown = 2)')
plt.figure()
firndepth = get_2d_grid(grid, 'firn_depth')
plt.imshow(firndepth, vmax=80)
plt.colorbar()
plt.title('Firn depth')
plt.figure()
bothdepth = lakedepth + firndepth
plt.imshow(bothdepth, vmax=80)
plt.colorbar()
plt.title('Lake depth + firn depth')
plt.figure()
liddepth = get_2d_grid(grid, 'lid_depth')
plt.imshow(liddepth)
plt.colorbar()
plt.title('lid depth')
plt.figure()
alldepth = lakedepth + firndepth + liddepth
plt.imshow(alldepth, vmax=80)
plt.colorbar()
plt.title('all depth')
#

plt.figure()
lakepresent = get_2d_grid(grid, 'lake')
plt.imshow(lakepresent)
plt.colorbar()
plt.title('lake present')
lidpresent = get_2d_grid(grid, 'lid')
plt.figure()
plt.imshow(lidpresent)
plt.colorbar()
plt.title('lid present')
lidpresent = get_2d_grid(grid, 'v_lid')

#

# # contour plots
# import cartopy.crs as ccrs
# import cartopy
# import numpy as np
# projection = ccrs.PlateCarree()
# lat = get_2d_grid(grid, 'lat')
# lon = get_2d_grid(grid, 'lon')
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection=projection)
# ax.coastlines()
# vmin = 0
# vmax = firndepth.max()
# levels = np.linspace(vmin, vmax, 20)
# ax.contourf(lon, lat, firndepth, transform=projection, levels=levels)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection=projection)
# ax.coastlines()
# vmin = 0
# vmax = 2
# levels = np.linspace(vmin, vmax, 20)
# ax.contourf(lon, lat, lakedepth, transform=projection, levels=levels)
