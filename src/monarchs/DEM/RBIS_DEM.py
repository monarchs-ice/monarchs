# RBIS from Sophie

tiffname = "RBIS_GTDX_45m.tif"

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS

im = Image.open(tiffname)
# im=np.loadtxt(tiffname, delimiter=',')
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
# cells = []
# cells = [
#     [IceShelf(heights[i][j] + water[i][j], water[i][j], 0) for j in range(max_grid_col)]
#     for i in range(max_grid_row)
# ]
# plot_water(cells, max_grid_row, max_grid_col)
# data = [[cells[i][j].water for j in range(max_grid_col)] for i in range(max_grid_row)]
#
# data = np.array(data)
# fig = plt.figure(figsize=(6, 3))
# plt.imshow(data, vmin=0, vmax=0.4)
# plt.set_cmap("Blues")
# plt.title("RBIS Water")
# plt.xticks((500, 1000), ["100km", "200km"])
# plt.ylim((0, 1000))
# plt.yticks((500, 1000), ["100km", "200km"])
# cbar = plt.colorbar()
# cbar.set_label("Water (m)")
# plt.savefig("RBIS_GIF/000initRBISwater.jpg")
# plt.show()
# plt.clf()


# plot_water(cells, max_grid_row, max_grid_col,1)
# for timestep in range(0, 200):
#     cells = add_random_water(cells, max_grid_row, max_grid_col)
#     cells = move_water(cells, max_grid_row, max_grid_col)
#     # plot_water(cells, max_grid_row, max_grid_col)
#     data = [
#         [cells[i][j].water for j in range(max_grid_col)] for i in range(max_grid_row)
#     ]
#     data = np.array(data)
#     fig = plt.figure(figsize=(6, 3))
#     plt.imshow(data, vmin=0, vmax=0.4)
#     plt.set_cmap("Blues")
#     plt.title("RBIS Water")
#     plt.xticks((500, 1000), ["100km", "200km"])
#     plt.ylim((0, 1000))
#     plt.yticks((500, 1000), ["100km", "200km"])
#     cbar = plt.colorbar()
#     cbar.set_label("Water (m)")
#     plt.savefig("RBIS_GIF/" + str("%02d" % (timestep)) + "RBISwater.jpg")
#     plt.show()
#     plt.clf()
#
# # plot_water(cells, max_grid_row, max_grid_col,1)
#
# data = [[cells[i][j].water for j in range(max_grid_col)] for i in range(max_grid_row)]
# data = np.array(data)
# fig = plt.figure(figsize=(6, 3))
# plt.imshow(data, vmin=0, vmax=0.4)
# plt.set_cmap("Blues")
# plt.title("Roi Bedoin Ice Shelf Water")
# plt.xticks((200, 400), ["40km", "80km"])
# plt.ylim((500, 499))
# plt.yticks((100, 300), ["80km", "40km"])
# cbar = plt.colorbar()
# cbar.set_label("Water (m)")
# plt.savefig("RBISwater.jpg")
# plt.show()
# plt.clf()
