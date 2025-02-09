# Read in lake extent maps from Dirscherl et al. (2021) TC paper
# Ice shelves available are AP: GeorgeVI, Bach & Wilkins, plus Riiser-Larsen, Nivlisen and Amery


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS
import geotiff as gt
from cartopy import crs as ccrs

Image.MAX_IMAGE_PIXELS = 3172163365  # Number of pixels in image from error
iceshelves = ["George_VI"]#, "Wilkins", "Riiser-Larsen", "Nivlisen", "Amery", "Bach"]
for iceshelf in iceshelves:
    im = gt.GeoTiff(f"AntarcticLakes/{iceshelf}/202001_1_max_extent.tif")
    print(f'{iceshelf} coordinates ((min long, min lat), (max long, max lat)):', im.tif_bBox_converted)
    projection = ccrs.PlateCarree()
    area_box = ((-69.5, -72.3), (-66.5, -71.25))
    lake_data = np.array(im.read_box(area_box))

    lons, lats = im.get_coord_arrays(area_box)
    plt.figure()
    plt.imshow(lake_data)

    fig = plt.figure()
    cmap = "viridis"
    projection = ccrs.PlateCarree()
    # first panel

    ax = fig.add_subplot(111, projection=projection)
    cont = ax.contourf(
        lons,
        lats,
        lake_data,
        cmap=cmap,
        transform=projection
    )

    #meta_dict = {TAGS[key]: im.tag[key] for key in im.tag_v2}
    # plt.colorbar(label="Pixel values")
    plt.title(f"TIF Image, {iceshelf}")
    ax.coastlines()
    ax.gridlines(draw_labels=True)
plt.show(block=True)


# Function to read and map a .tif file
# def read_and_map_tif(file_path):
#     # Open the .tif file using rasterio
#     with rasterio.open(file_path) as monarchs:
#         # Read the data into a numpy array
#         data = monarchs.read(1)  # Read the first band
#
#         # Get metadata and transform information
#         metadata = monarchs.meta
#         transform = monarchs.transform
#
#         # Print some metadata
#         print("Metadata:", metadata)
#
#     # Plot the data using matplotlib
#     plt.imshow(data, cmap='gray')
#     plt.colorbar(label='Pixel values')
#     plt.title('TIF Image')
#     plt.xlabel('Column')
#     plt.ylabel('Row')
#     plt.show()
#
#     return data, metadata, transform

# Example usage
# file_path = 'path_to_your_file.tif'
# data, metadata, transform = read_and_map_tif(file_path)
