import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from pyproj import CRS, Transformer
import rioxarray
from shapely.geometry import box
import geopandas as gpd
from netCDF4 import Dataset
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import xy

# 0 other imaged areas
# 1 lake mask
# 2 rock/seawater mask
# 3 cloud mask
# 9 anything that was below 0 (non-imaged areas? Should be 255 in metadata but this is -1 in another file type in the dataset)

# set up some variables
proj = ccrs.PlateCarree()
matplotlib.use('TkAgg')

crs_cartopy = ccrs.SouthPolarStereo()
proj_string = crs_cartopy.proj4_init  # or crs_cartopy.proj4_params
print(proj_string)

# Project from EPSG:4326 → EPSG:3031
crs_cartopy_proj = CRS.from_proj4(proj_string)


def get_bbox_from_model_data(modeldata):
    lons = modeldata.variables['lon'][:]
    lats = modeldata.variables['lat'][:]
    crs_cartopy = ccrs.SouthPolarStereo()
    proj_string = crs_cartopy.proj4_init  # or crs_cartopy.proj4_params
    print(proj_string)

    # Project from EPSG:4326 → EPSG:3031
    crs_cartopy_proj = CRS.from_proj4(proj_string)
    transformer = Transformer.from_crs("EPSG:4326", crs_cartopy_proj, always_xy=True)
    x, y = transformer.transform(lons, lats)  # still 2D, same shape
    return x, y


# print(f'Box boundary = {input_raster.tif_bBox_converted}')
def plot_on_map(x, y, mask_array, label=''):
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={
        'projection': ccrs.SouthPolarStereo()
    })

    # Plot the data using pcolormesh
    mesh = ax.pcolormesh(x, y, mask_array, cmap='Blues', shading='auto',
                         transform=None)

    # Add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN)
    ax.gridlines(draw_labels=True)

    # Colorbar and title
    plt.colorbar(mesh, ax=ax, orientation='vertical', label='Lake depth (m)')
    ax.set_title(f'Lake present (size = {np.shape(mask_array)})' + label)


def max_pooling_to_shape(data, output_shape):
    """
    Max-pools the data into target_shape, preserving full spatial extent.
    """
    H, W = data.shape
    new_H, new_W = output_shape

    # Compute bin edges
    row_bins = np.linspace(0, H, new_H + 1, dtype=int)
    col_bins = np.linspace(0, W, new_W + 1, dtype=int)

    pooled = np.zeros((new_H, new_W), dtype=data.dtype)

    for i in range(new_H):
        for j in range(new_W):
            block = data[row_bins[i]:row_bins[i+1], col_bins[j]:col_bins[j+1]]
            pooled[i, j] = block.max()  # or use np.any(block) if binary

    return pooled


def apply_lake_mask(mask_array):
    # Apply mask so all values are either -1 (non-lake areas) or 1 (lakes)
    mask_array[mask_array > 1] = -1
    mask_array[mask_array == 0] = -1
    mask_array[mask_array == -1] = 0
    return mask_array

# Load in model data
model_data_path = r'C:\Users\jdels\Documents\Work\MONARCHS_runs\ARCHER2_140425\model_output.nc'
model_data = Dataset(model_data_path)
x_model, y_model = get_bbox_from_model_data(model_data)

xmin, ymin = np.min(x_model), np.min(y_model)
xmax, ymax = np.max(x_model), np.max(y_model)
bbox = box(xmin, ymin, xmax, ymax)
gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs=crs_cartopy_proj.to_wkt())

model_data.close()

# Load in Moussavi data
#moussavi_file_path = "LC08_L1GT_218111_20200101_20200113_01_T2_All_Masks.tif"
moussavi_file_path = (
    r"C:\Users\jdels\Documents\Work\GVI\GVI\216-111\LC08_L1GT_216111_20171228_20180103_01_T2\LC08_L1GT_216111_20171228_20180103_01_T2_All_Masks.tif"
)
moussavi_file_path_depth = (
    r"C:\Users\jdels\Documents\Work\GVI\GVI\216-111\LC08_L1GT_216111_20171228_20180103_01_T2\LC08_L1GT_216111_20171228_20180103_01_T2_Average_Red_And_Panchromatic_Depth.tif"
)
#moussavi_file_path = (
#    r"C:\Users\jdels\Documents\Work\GVI\GVI\216-111\LC08_L1GT_216111_20190217_20190222_01_T2\LC08_L1GT_216111_20190217_20190222_01_T2_All_Masks.tif"
#)
# with rioxarray.open_rasterio(moussavi_file_path) as moussavi_riox:
#     xn, yn = moussavi_riox.x.values, moussavi_riox.y.values
#     data = apply_lake_mask(moussavi_riox.squeeze().values)
#     plot_on_map(xn, yn, data, label='no masking')
#
#     da_proj = moussavi_riox.rio.reproject(crs_cartopy_proj)
#     clipped = da_proj.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)
#     xn, yn = clipped.x.values, clipped.y.values
#     data = apply_lake_mask(clipped.squeeze().values)
#
#     fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.SouthPolarStereo()})
#
#     # Plot the data (using the correct CRS for interpretation)
#     clipped.plot(ax=ax, transform=ccrs.SouthPolarStereo(), cmap='Blues', robust=True)
#
#     # Add coastlines and features
#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS, linestyle=':')
#     ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
#     ax.add_feature(cfeature.OCEAN)
#
#     plot_on_map(xn, yn, data, label='Moussavi data - cropped')
#
#     # Perform max pooling
#     pooled_data = max_pooling_to_shape(data, (100, 100))
#     # Interpolate lat/long to this 100 x 100 grid
#     pooled_xn = np.linspace(xn.min(), xn.max(), 100)
#     pooled_yn = np.linspace(yn.min(), yn.max(), 100)[::-1]
#     plot_on_map(pooled_xn, pooled_yn, pooled_data, label='Moussavi data - pooled')
#
#     plt.figure()
#     plt.imshow(pooled_data)
#     plt.title('Pooled Moussavi data')
#     plt.figure()
#     plt.imshow(data)
#     plt.title('Cropped Moussavi data')
#
# # Plot up original data - in PlateCarree
# fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={
#     'projection': ccrs.SouthPolarStereo()
# })
# da_proj = moussavi_riox.rio.reproject(4326)
# xn, yn = da_proj.x.values, da_proj.y.values
# # Plot the data using pcolormesh
# mesh = ax.pcolormesh(xn, yn, da_proj.squeeze().values, cmap='Set1',
#                      transform=ccrs.PlateCarree())
# # Add map features
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
# ax.add_feature(cfeature.OCEAN)
# ax.gridlines(draw_labels=True)
# # Colorbar and title
# cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', label='Value')
# cbar.set_ticks([-1, 0, 1, 2, 3])

with rioxarray.open_rasterio(moussavi_file_path_depth) as moussavi_riox:
    xn, yn = moussavi_riox.x.values, moussavi_riox.y.values
    data = apply_lake_mask(moussavi_riox.squeeze().values)
    plot_on_map(xn, yn, data, label='no masking')

    da_proj = moussavi_riox.rio.reproject(crs_cartopy_proj)
    clipped = da_proj.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)
    xn, yn = clipped.x.values, clipped.y.values
    data = clipped.squeeze().values

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.SouthPolarStereo()})

    # Plot the data (using the correct CRS for interpretation)
    clipped.plot(ax=ax, transform=ccrs.SouthPolarStereo(), cmap='Blues', robust=True)

    # Add coastlines and features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN)

    plot_on_map(xn, yn, data, label='Moussavi data - cropped')

    # Perform max pooling
    pooled_data = max_pooling_to_shape(data, (100, 100))
    # Interpolate lat/long to this 100 x 100 grid
    pooled_xn = np.linspace(xn.min(), xn.max(), 100)
    pooled_yn = np.linspace(yn.min(), yn.max(), 100)[::-1]
    plot_on_map(pooled_xn, pooled_yn, pooled_data/1000, label='Moussavi lake depth')

    plt.figure()
    plt.imshow(pooled_data)
    plt.title('Pooled Moussavi data')
    plt.figure()
    plt.imshow(data)
    plt.title('Cropped Moussavi data')

# Plot up original data - in PlateCarree
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={
    'projection': ccrs.SouthPolarStereo()
})
da_proj = moussavi_riox.rio.reproject(4326)
xn, yn = da_proj.x.values, da_proj.y.values
# Plot the data using pcolormesh
mesh = ax.pcolormesh(xn, yn, da_proj.squeeze().values, cmap='Set1',
                     transform=ccrs.PlateCarree())
# Add map features
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
ax.add_feature(cfeature.OCEAN)
ax.gridlines(draw_labels=True)
# Colorbar and title
cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', label='Value')


np.save('lake_depth_moussavi_subset.npy', pooled_data/1000)
np.save('x_moussavi_pooled_subset.npy', pooled_xn)
np.save('y_moussavi_pooled_subset.npy', pooled_yn)





# # Reproject this subset to a 100x100 grid.
# #resampled_data = moussavi_riox.rio.reproject(crs, shape=(100, 100), resampling=Resampling.bilinear)
#
# # Landsat 218211
# plot_on_map(lons, lats, mask_array, label='no masking')
#

#
# data[data > 1] = -1
# data[data == 0] = -1
# data[data == -1] = 0
#
# plt.figure()
# plt.imshow(mask_array)
# # Coordinates from Moussavi file
# ##These are the corners of the product but from them you have lat/lon of the whole box in the file...
# #plot_on_map(lons, lats, mask_array)
# # plot up map of the Moussavi data
#
#
# #plot_on_map(xs_subset, ys_subset, data, label='Moussavi data - downscaled')
# mask_array_p = max_pooling_to_shape(mask_array, (100, 100))
#
# plt.figure()
# plt.imshow(mask_array_p)
# plt.title('Pooled Moussavi data')
# # Interpolate lat/long to this 100 x 100 grid
#
#
# # plt.close()

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

#
# masks = np.ndarray.flatten(mask_array)
# masks[masks < 0] = 9
# count = np.bincount(masks)

# Get grid from DEM to cut off Moussavi file. Proably don't want to interpolate due to nature of Moussavi file, just crop and plot over each other/ do the stats.

# Questions for histogram- does Mousaavi have a min lake depth?

# Open the original raster with the custom WKT

# Select only the region within the bounding box defined above.
# with rasterio.open(moussavi_file_path) as src:
#     dst_path = 'moussavi_reprojected.tif'
#     # first do the projection in the correct CRS
#     convert_to_3031(src, dst_path)
# moussavi_riox = rioxarray.open_rasterio(moussavi_file_path)
# crs = CRS.from_wkt(moussavi_riox.spatial_ref.crs_wkt)
# transformer = Transformer.from_crs(crs, crs_cartopy_proj, always_xy=True)
# x, y = np.meshgrid(moussavi_riox.x.values, moussavi_riox.y.values)
# mask_array = moussavi_riox.squeeze().values
# x_flat, y_flat = x.flatten(), y.flatten()
# lon_array, lat_array = transformer.transform(x_flat, y_flat)
# lats = lat_array.reshape(x.shape)
# lons = lon_array.reshape(y.shape)
# def convert_to_3031(src, dst_path):
#     # Define the target CRS
#     dst_crs = crs_cartopy_proj
#
#     # Calculate the transform and new dimensions
#     transform, width, height = calculate_default_transform(
#         src.crs, dst_crs, src.width, src.height, *src.bounds
#     )
#
#     # Copy and update the metadata
#     kwargs = src.meta.copy()
#     kwargs.update({
#         "crs": dst_crs,
#         "transform": transform,
#         "width": width,
#         "height": height
#     })
#
#     # Create the destination raster and reproject
#     with rasterio.open(dst_path, "w", **kwargs) as dst:
#         for i in range(1, src.count + 1):
#             reproject(
#                 source=rasterio.band(src, i),
#                 destination=rasterio.band(dst, i),
#                 src_transform=src.transform,
#                 src_crs=src.crs,
#                 dst_transform=transform,
#                 dst_crs=dst_crs,
#                 resampling=Resampling.nearest  # Use 'nearest' for masks/binary data
#             )