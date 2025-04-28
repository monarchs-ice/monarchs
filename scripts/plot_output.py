"""
Sample script to plot some basic images from the model output.
In this script, I load in both the model dump (from progress.nc), and the
diagnostic output (model_output.nc). The dump file gives an indication of the current state of the model.
The diagnostic file is only used here to plot the initial conditions of the firn column, so I can do a diff to see
visually how it has changed over time.
"""

from netCDF4 import Dataset
from matplotlib import pyplot as plt
import numpy.ma as ma
import matplotlib
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from pyproj import CRS, Transformer
import numpy as np

matplotlib.use('TkAgg')
dumppath = r'C:\Users\jdels\Documents\Work\MONARCHS_runs\ARCHER2_140425\progress.nc'
diagpath = r'C:\Users\jdels\Documents\Work\MONARCHS_runs\ARCHER2_140425\model_output.nc'
# dumppath = r'C:\Users\jdels\Documents\Work\MONARCHS_runs\ARCHER2_10year\progress.nc'
# diagpath = r'C:\Users\jdels\Documents\Work\MONARCHS_runs\ARCHER2_10year\model_output.nc'
# dumppath = '../examples/10x10_gaussian_threelake/output/gaussian_threelake_example_dump.nc'
# diagpath = '../examples/10x10_gaussian_threelake/output/gaussian_threelake_example_output.nc'
big_data_path = r'C:\Users\jdels\Documents\Work\MONARCHS_runs\ARCHER2_10year\progress.nc'
flowdata = Dataset(dumppath)
t0data = Dataset(diagpath)
big_data = Dataset(big_data_path)

lat_big = big_data.variables['lat'][:]
lon_big = big_data.variables['lon'][:]


def plot_variable(dset, variable_name, cmap='Blues', vmax=None):
    """
    Plot a variable from an input dump file as a 2D grid with plt.imshow. This is a shorthand to generate
    plots from the final model state.

    Parameters
    ----------
    dset : netCDF4.Dataset
        The dataset containing the variable to plot.
    variable_name : str
        The name of the variable to plot.
    cmap : str, optional
        The colormap to use for the plot. Default is 'Blues'.
    vmax : float, optional
        The maximum value for the color scale. Default is None.

    Returns
    -------
    variable : netCDF4.Variable
        The variable from the input netCDF that was plotted.
    """

    variable = dset.variables[variable_name]
    plt.figure()
    plt.imshow(variable, vmax=vmax, cmap=cmap)
    plt.colorbar()
    title_str = variable_name.replace('_', ' ')
    if vmax:
        title_str += ', (max shown = {})'.format(vmax)
    plt.title(title_str)
    return variable

lakedepth = plot_variable(flowdata, 'lake_depth')
lake = plot_variable(flowdata, 'lake')
# plot_variable(flowdata,'lake_depth', vmax=0.2)
# ice_lens = plot_variable(flowdata,'ice_lens')
# ice_lens_depth = plot_variable(flowdata,'ice_lens_depth', vmax=500)
# firndepth = plot_variable(flowdata,'firn_depth', vmax=150)
# liddepth = plot_variable(flowdata,'lid_depth')
# lake = plot_variable(flowdata,'lake')
# lid = plot_variable(flowdata,'lid')

plt.figure()
ofd = t0data.variables['firn_depth'][0]
plt.imshow(ofd, vmax=80)
plt.colorbar()
plt.title('Firn depth (original)')
#
# from matplotlib.colors import TwoSlopeNorm
# plt.figure()
# diff = firndepth[:] - ofd[:]
# plt.imshow(diff, cmap='Blues', norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3))
# plt.colorbar()
# plt.title('Firn depth difference (m)')


lons = flowdata.variables['lon'][:]
lats = flowdata.variables['lat'][:]

crs_cartopy = ccrs.SouthPolarStereo()
proj_string = crs_cartopy.proj4_init  # or crs_cartopy.proj4_params
print(proj_string)

# Project from EPSG:4326 → EPSG:3031
crs_cartopy_proj = CRS.from_proj4(proj_string)
transformer = Transformer.from_crs("EPSG:4326", crs_cartopy_proj, always_xy=True)
x, y = transformer.transform(lons, lats)  # still 2D, same shape
x_big, y_big = transformer.transform(lon_big, lat_big)  # still 2D, same shape
xmin, xmax, ymin, ymax = x_big.min(), x_big.max(), y_big.min(), y_big.max()
# Transform back as a sanity check
transformer = Transformer.from_crs(crs_cartopy_proj, "EPSG:4326", always_xy=True)
lon2, lat2 = transformer.transform(x, y)  # still 2D, same shape

# start date is 2010-01-01
# data is in 30 day chunks
# 0 = 2010-01-01
# Moussavi validation data is for 2018-01-03
# 2018-01-03 = 2925 days
# 2925 / 30 = 97.5 - so index ~98

lake = t0data.variables['lake'][98]
lakedepth = t0data.variables['lake_depth'][98]
def lake_thresh(thresh):
    lake_plot = np.zeros_like(lake)
    for i in range(len(lake)):
        for j in range(len(lake[0])):
            if lakedepth[i][j] < thresh:
                lake_plot[i][j] = 0
            else:
                lake_plot[i][j] = 1
    return lake_plot

def make_lake_plot(lake_plot, thresh):

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={
        'projection': ccrs.SouthPolarStereo()
    })
    ax.pcolormesh(lons, lats, lake_plot, cmap='Blues', shading='auto', transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN)
    ax.gridlines(draw_labels=True)
    ax.set_title(f'Lake present (model), depth threshold = {thresh} m')

lake_plot = lake_thresh(0.5)
make_lake_plot(lake_plot, 0.5)
lake_plot = lake_thresh(1)
make_lake_plot(lake_plot, 1)
lake_plot = lake_thresh(0.2)
make_lake_plot(lake_plot, 0.2)

# Moussavi data
moussavi_lake_depth = np.load('../validation/lake_depth_moussavi.npy')
x_moussavi = np.load('../validation/x_moussavi_pooled.npy')
y_moussavi = np.load('../validation/y_moussavi_pooled.npy')

def plot_on_map(x, y, mask_array, labelstr='Moussavi', vmax=False):
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={
        'projection': ccrs.SouthPolarStereo()
    })

    # Plot the data using pcolormesh
    if vmax:
        mesh = ax.pcolormesh(x, y, mask_array, cmap='Blues', shading='auto',
                             transform=None, vmax=vmax)
    else:
        mesh = ax.pcolormesh(x, y, mask_array, cmap='Blues', shading='auto',
                             transform=None)

    # Add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN)
    ax.gridlines(draw_labels=True)
    #ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.SouthPolarStereo())
    # Colorbar and title
    plt.colorbar(mesh, ax=ax, orientation='vertical', label='Lake depth (m)')
    ax.set_title(f'Lake depth - {labelstr}')


# Plot up Moussavi lake data
plot_on_map(x, y, moussavi_lake_depth)
plot_on_map(x, y, lakedepth, labelstr='MONARCHS')

# fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={
#     'projection': ccrs.SouthPolarStereo()
# })
# ax.pcolormesh(x, y, lake_plot, cmap='Blues', shading='auto', transform=None)
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
# ax.add_feature(cfeature.OCEAN)
# ax.gridlines(draw_labels=True)
# ax.set_title(f'Lake present (model, projected coordinates)')

# flowdata.close()
# t0data.close()