import numpy as np
import matplotlib.pyplot as plt
from geotiff import GeoTiff
import matplotlib.pyplot as plt
import numpy as np
from geotiff import GeoTiff
tiffname = '38_12_32m_v2.0_dem.tif'
gt = GeoTiff(tiffname)
print('Reading in firn depth from DEM')
heights = np.array(gt.read())
lon_array, lat_array = gt.get_coord_arrays()
plt.imshow(heights)
plt.show()
