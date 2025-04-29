from netCDF4 import Dataset
from matplotlib import pyplot as plt
import numpy.ma as ma
import matplotlib
from matplotlib.animation import ArtistAnimation

matplotlib.use("TkAgg")
dumppath = r"C:\Users\jdels\Documents\Work\MONARCHS_runs\ARCHER2_10year\progress.nc"
diagpath = r"C:\Users\jdels\Documents\Work\MONARCHS_runs\ARCHER2_10year\model_output.nc"

# Load in the data - just focusing on lake depth from the diagnostics file

diagnostic_data = Dataset(diagpath)

lake_depth = diagnostic_data.variables["lake_depth"]
lid_depth = diagnostic_data.variables["lid_depth"]
# Plot the lake depth - we want to generate an ArtistAnimation to do this so we can create a gif over time
fig, ax = plt.subplots()
ims = []
for i in range(len(lake_depth)):
    im = ax.imshow(lake_depth[i])
    title = ax.text(
        0.5,
        1.05,
        f"Lake depth at month {i}, vmax = {lake_depth[i].max()}",
        size=plt.rcParams["axes.titlesize"],
        ha="center",
        transform=ax.transAxes,
    )
    ims.append([im, title])

ani = ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
ani.save("lake_depth.gif", writer="imagemagick")

# Plot the lake depth - we want to generate an ArtistAnimation to do this so we can create a gif over time
fig, ax = plt.subplots()
ims = []
for i in range(len(lid_depth)):
    im = ax.imshow(lid_depth[i])
    title = ax.text(
        0.5,
        1.05,
        f"Lid depth at month {i}, vmax = {lid_depth[i].max()}",
        size=plt.rcParams["axes.titlesize"],
        ha="center",
        transform=ax.transAxes,
    )
    ims.append([im, title])

ani = ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
ani.save("lid_depth.gif", writer="imagemagick")
diagnostic_data.close()
