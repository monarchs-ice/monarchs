import numpy as np
import matplotlib.pyplot as plt
from geotiff import GeoTiff
import matplotlib.pyplot as plt
import numpy as np
from geotiff import GeoTiff

tiffname = "38_12_32m_v2.0_dem.tif"
gt = GeoTiff(tiffname)
# print(f'Box boundary = {gt.tif_bBox_converted}')
print("Reading in firn depth from DEM")
# if all the box boundary coordinates aren't False, then take a subset defined by these bounds
heights = np.array(gt.read())
lon_array, lat_array = gt.get_coord_arrays()
# print(max(lon_array))
# plt.imshow(lon_array)
# plt.show()
# plt.imshow(lat_array)
# plt.show()
plt.imshow(heights)
plt.show()


# lat and lon are on a diagonal. Better to specify a box within this?
# Check format of Moussavi lakes
