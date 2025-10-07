from matplotlib import pyplot as plt
import numpy as np
from netCDF4 import Dataset
import matplotlib.cm as cm
import matplotlib.colors as mcolors


resolutions = [200, 300, 400, 500, 700, 1000, 2500, 5000, 10000]

folder = r"C:\Users\jdels\Documents\Work\monarchs\MONARCHS\tests\resolution_benchmarking\output"

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

cmap = plt.cm.viridis
colours = cmap(np.linspace(0, 1, len(resolutions)))

for idx, res in enumerate(resolutions):
    colour = colours[idx]
    path = f"{folder}/1d_testcase_dump_{res}.nc"
    file = Dataset(path)
    rho = file.variables["rho"][0, 0][:]
    vp = file.variables["vertical_profile"][0, 0][::-1]
    T = file.variables["firn_temperature"][0, 0][:]

    ax.plot(rho, vp, label=f"{res} cm", color=colour)
    ax.set_xlabel("Density (kg m$^{-3}$)")
    ax.set_ylabel("Depth (m)")
    ax.grid(
        True, which="both", linestyle="--", linewidth=0.5, alpha=1, zorder=0
    )
    ax.legend()
    plt.title("Old solver")

    ax2.plot(T, vp, label=f"{res} cm", color=colour)
    ax2.set_xlabel("Temperature (K)")
    ax2.set_ylabel("Depth (m)")
    ax2.grid(
        True, which="both", linestyle="--", linewidth=0.5, alpha=1, zorder=0
    )
    ax2.legend()
    plt.title("Old solver")
    file.close()
