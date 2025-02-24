"""
Sample script to plot some basic images from the model output.
In this script, I load in both the model dump (from progress.nc), and the
diagnostic output (model_output.nc). The dump file gives an indication of the current state of the model.
The diagnostic file is only used here to plot the initial conditions of the firn column, so I can do a diff to see
visually how it has changed over time.
"""

from netCDF4 import Dataset
from matplotlib import pyplot as plt
import numpy.ma as ma

dumppath = 'C:/Users/jdels/Documents/Work/MONARCHS_runs/ARCHER2_flow_into_land/38m_dem/progress.nc'
diagpath = 'C:/Users/jdels/Documents/Work/MONARCHS_runs/ARCHER2_flow_into_land/38m_dem/model_output.nc'

flowdata = Dataset(dumppath)
t0data = Dataset(diagpath)

def plot_variable(dset, variable_name, cmap='viridis', vmax=None, title_flow=True):
    variable = dset.variables[variable_name]
    plt.figure()
    plt.imshow(variable, vmax=vmax, cmap=cmap)
    plt.colorbar()
    title_str = variable_name.replace('_', ' ')
    if vmax:
        title_str += ', (max shown = {})'.format(vmax)
    if title_flow:
        title_str += ', flow_into_land = `True`'
    plt.title(title_str)
    return variable

lakedepth = plot_variable(flowdata, 'lake_depth')
plot_variable(flowdata,'lake_depth', vmax=1)
ice_lens = plot_variable(flowdata,'ice_lens')
ice_lens_depth = plot_variable(flowdata,'ice_lens_depth', vmax=500)
firndepth = plot_variable(flowdata,'firn_depth', vmax=150)
liddepth = plot_variable(flowdata,'lid_depth')
lake = plot_variable(flowdata,'lake')
lid = plot_variable(flowdata,'lid')

plt.figure()
ofd = t0data.variables['firn_depth'][0]
plt.imshow(ofd, vmax=80)
plt.colorbar()
plt.title('Firn depth (original)')

plt.figure()
diff = firndepth[:] - ofd[:]
plt.imshow(diff)
plt.colorbar()
plt.title('Firn depth difference (m)')

lakedepthbool = ma.masked_outside(lakedepth[:], 0.001, 0.1)
plt.figure()
plt.imshow(lakedepthbool)
plt.colorbar()
plt.title('Lake depth boolean')

