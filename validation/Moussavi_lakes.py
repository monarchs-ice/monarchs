import matplotlib.pyplot as plt
import numpy as np
from geotiff import GeoTiff

# from monarchs.DEM.import_DEM import export_DEM_geotiff
# from monarchs.core.utils import find_nearest

# Files are labelled in the same way as Landsat-8 images
# Line 26-33 of .txt metadata gives lat lon coordinates of location
# e.g
# CORNER_UL_LAT_PRODUCT = -70.82877
# CORNER_UL_LON_PRODUCT = 163.16603
# CORNER_UR_LAT_PRODUCT = -70.16954
# CORNER_UR_LON_PRODUCT = 157.62932
# CORNER_LL_LAT_PRODUCT = -68.97158
# CORNER_LL_LON_PRODUCT = 164.72008
# CORNER_LR_LAT_PRODUCT = -68.37323
# CORNER_LR_LON_PRODUCT = 159.61407

moussavi_file_path = "LC08_L1GT_218111_20200101_20200113_01_T2_All_Masks.tif"
model_output_filepath = ""

# Coordinates used for model output
latmax = -71.18899625333428
latmin = -72.36057124339845
lonmax = -66.03420319510812
lonmin = -69.77777888105943

# Landsat 218211

# Coordinates from Moussavi file
##These are the corners of the product but from them you have lat/lon of the whole box in the file...
CORNER_UL_LAT_PRODUCT = -70.62378
CORNER_UL_LON_PRODUCT = -69.08794
CORNER_UR_LAT_PRODUCT = -72.91601
CORNER_UR_LON_PRODUCT = -66.06738
CORNER_LL_LAT_PRODUCT = -71.35901
CORNER_LL_LON_PRODUCT = -76.32295
CORNER_LR_LAT_PRODUCT = -73.76178
CORNER_LR_LON_PRODUCT = -74.21682

# Create grid of lat/lons of the Moussavi product


geo_tiff = GeoTiff(moussavi_file_path, crs_code=4326)  # Had to guess crs
shape = geo_tiff.tif_shape
# Bbox = geo_tiff.tif_bBox_wgs_84
zarr_array = geo_tiff.read()
mask_array = np.array(zarr_array)

fig = plt.figure()
plt.imshow(mask_array)
plt.colorbar()
plt.show()
# plt.close()

# TODO
# percent of surface lakes
# percent of surface ice
# percent of ice that's lakes (i.e. total-land-not sampled etc=ice)
# lake depth mean, median
# co-locate and then...
# 1 minus other, RMSE?

# create function to sample a smaller bounding box (use read box exisiting function https://pypi.org/project/geotiff/)
# create function to get histogram of lake depths for general comparison (can check against auto generated)
# direct comparison- need to remove masks

# Use Max/min lat/lon from DEM or bounding box used for model run to remove data from Moussavi file not needed

# lat_indices = np.zeros((model_setup.row_amount, model_setup.col_amount))
# lon_indices = np.zeros((model_setup.row_amount, model_setup.col_amount))
new_Moussavi_grid = {}

# old_Moussavi grid = interpolate_grid(ERA5_vars, model_setup.row_amount, model_setup.col_amount)


masks = np.ndarray.flatten(mask_array)
masks[masks < 0] = 9
count = np.bincount(masks)
# 0 other imaged areas
# 1 lake mask
# 2 rock/seawater mask
# 3 cloud mask
# 9 anything that was below 0 (non-imaged areas? Should be 255 in metadata but this is -1 in another file type in the dataset)


# Get grid from DEM to cut off Moussavi file. Proably don't want to interpolate due to nature of Moussavi file, just crop and plot over each other/ do the stats.

# Questions for histogram- does Mousaavi have a min lake depth?

##############################
#     old_ERA5_grid = ERA5_grid
#     for var in ERA5_grid.keys():
#         if var in ['lat', 'long', 'time']:
#             new_ERA5_grid[var] = ERA5_grid[var]
#             continue
#
#         new_ERA5_grid[var] = np.zeros((len(ERA5_grid['time']), model_setup.row_amount, model_setup.col_amount))
#         for i in range(model_setup.row_amount):
#             for j in range(model_setup.col_amount):
#                 lat_indices[i, j] = find_nearest(ERA5_grid['lat'], lat_array[i, j])
#                 lon_indices[i, j] = find_nearest(ERA5_grid['long'], lon_array[i, j])
#                 new_ERA5_grid[var][:, i, j] = ERA5_grid[var][:, int(lat_indices[i, j]), int(lon_indices[i, j])]
#     ERA5_grid = new_ERA5_grid
#
#     if diagnostic_plots:
#         """
#         Diagnostic plotting - ensure that this method works outside of our test case (see import_ERA5.py)
#         """
#         fig0 = plt.figure()
#         cmap = 'viridis'
#         projection = ccrs.PlateCarree()
#         ax0 = fig0.add_subplot(111, projection=projection)
#         ax0.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
#
#         ax0.coastlines()
#         lons = old_ERA5_grid['long'][:]
#         lats = old_ERA5_grid['lat'][:]
#         temperature = old_ERA5_grid['temperature'][:]
#         ax0.contourf(lons, lats, temperature[0],
#                      cmap=cmap, transform=projection,
#                      levels=20)
#         ax0.gridlines(draw_labels=True)
#
#         fig1 = plt.figure()
#         ax1 = fig1.add_subplot(111, projection=projection)
#         ax1.coastlines()
#         ax1.gridlines(draw_labels=True)
#
#         fig2 = plt.figure()
#         ax2 = fig2.add_subplot(111, projection=projection)
#
#         ax1.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
#
#         cont = ax1.contourf(ilons, ilats, ERA5_grid['temperature'][0],
#                             cmap=cmap, transform=projection, levels=20,
#                             )  # np.sqrt(ERA5_data['u10'][0] **2 + ERA5_data['v10'][0] **2)
#         fig1.colorbar(cont)
#         ax1.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
#
#         cont = ax2.contourf(ilons, ilats, iheights, cmap=cmap, transform=ccrs.PlateCarree(), )
#         # levels=bounds, vmin=0, vmax=50)
#         ax2.coastlines()
#         ax2.gridlines(draw_labels=True)
#         ax2.title.set_text('Initial DEM height profile')
#         plt.show()
#     return ERA5_grid
#
#
# #Transform to lat lon to match REMA DEM (not needed)
# #from pyproj import Transformer
# #transformer = Transformer.from_crs({"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'}, '+proj=longlat +datum=WGS84 +no_defs +type=crs')
# #pointA = transformer.transform(Bbox[0][0], Bbox[0][1])
# #pointB = transformer.transform(Bbox[1][0], Bbox[1][1])
