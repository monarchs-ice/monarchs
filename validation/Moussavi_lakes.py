import matplotlib.pyplot as plt
import numpy as np
from geotiff import GeoTiff
import matplotlib

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

#This like makes the plots appear in Pycharm
matplotlib.use('TkAgg')

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

# TODO
# Create grid of lat/lons of the Moussavi product
# Get correct date for Moussavi
# create function to sample a smaller bounding box (use read box exisiting function https://pypi.org/project/geotiff/)
# create function to get histogram of lake depths for general comparison (can check against auto generated)
# direct comparison- need to remove masks
# percent of surface lakes
# percent of surface ice
# percent of ice that's lakes (i.e. total-land-not sampled etc=ice)
# lake depth mean, median
# Questions for histogram- does Mousaavi have a min lake depth?
# co-locate and then...
# 1 minus other, RMSE?


geo_tiff = GeoTiff(moussavi_file_path, crs_code=4326)  # Had to guess crs
shape = geo_tiff.tif_shape
# Bbox = geo_tiff.tif_bBox_wgs_84
zarr_array = geo_tiff.read()
mask_array = np.array(zarr_array)

fig = plt.figure()
plt.imshow(mask_array)
plt.colorbar()

# Use Max/min lat/lon from DEM or bounding box used for model run to remove data from Moussavi file not needed
# Get grid from DEM to cut off Moussavi file. Proably don't want to interpolate due to nature of Moussavi file, just crop and plot over each other/ do the stats.


masks = np.ndarray.flatten(mask_array)
masks[masks < 0] = 9
count = np.bincount(masks)
# 0 other imaged areas
# 1 lake mask
# 2 rock/seawater mask
# 3 cloud mask
# 9 anything that was below 0 (non-imaged areas? Should be 255 in metadata but this is -1 in another file type in the dataset)
