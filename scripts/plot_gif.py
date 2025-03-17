from netCDF4 import Dataset
from matplotlib import pyplot as plt
import numpy.ma as ma
import matplotlib
from matplotlib.animation import ArtistAnimation
matplotlib.use('TkAgg')
dumppath = '../examples/10x10_gaussian_threelake/output/gaussian_threelake_example_dump_df.nc'
diagpath = '../examples/10x10_gaussian_threelake/output/gaussian_threelake_example_output_df.nc'

# Load in the data - just focusing on lake depth from the diagnostics file

diagnostic_data = Dataset(diagpath)

lake_depth = diagnostic_data.variables['lake_depth']

diagnostic_data.close()
# Plot the lake depth - we want to generate an ArtistAnimation to do this so we can create a gif over time
fig, ax = plt.subplots()
ims = []
for i in range(len(lake_depth)):
    im = ax.imshow(lake_depth[i])
    ims.append([im])

ani = ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
ani.save('lake_depth.gif', writer='imagemagick')
