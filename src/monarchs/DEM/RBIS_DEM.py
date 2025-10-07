import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS

tiffname = "RBIS_GTDX_45m.tif"

im = Image.open(tiffname)
DEM_grid = np.array(im)
meta_dict = {TAGS[key]: im.tag[key] for key in im.tag_v2}
print(meta_dict)
plt.imshow(DEM_grid, vmax=100, vmin=0)
plt.colorbar()
plt.show()
plt.clf()
print(DEM_grid.shape)
heights = DEM_grid[1080:2080, 760:1760]
water_level = 0 * heights
font = {"family": "normal", "weight": "bold", "size": 18}
plt.rc("font", **font)
fig = plt.figure(figsize=(4, 2))
plt.imshow(heights, vmin=0, vmax=100)
plt.set_cmap("Reds")
cbar = plt.colorbar()
cbar.set_label("Height (m)")
plt.title("Initial Height")
plt.savefig("RBISInit_height.jpg")
plt.show()
max_grid_row = len(heights)
max_grid_col = len(heights[0])
water = np.random.rand(max_grid_row, max_grid_col) / 10
water = np.ma.masked_where(heights > 100, water)
