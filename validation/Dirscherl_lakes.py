# Read in lake extent maps from Dirscherl et al. (2021) TC paper
# Ice shelves available are AP: GeorgeVI, Bach & Wilkins, plus Riiser-Larsen, Nivlisen and Amery


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS

Image.MAX_IMAGE_PIXELS = 3172163365  # Number of pixels in image from error
im = Image.open("202001_1_max_extent.tif")
# im=np.loadtxt(tiffname, delimiter=',')
lake_data = np.array(im)
test1 = lake_data[100][:]
test2 = lake_data[1000][:]
meta_dict = {TAGS[key]: im.tag[key] for key in im.tag_v2}


plt.imshow(lake_data, cmap="gray")
plt.colorbar(label="Pixel values")
plt.title("TIF Image")
plt.xlabel("Column")
plt.ylabel("Row")
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
