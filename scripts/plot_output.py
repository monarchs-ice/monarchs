"""
Script to plot basic images from the model output.
In this script, I load in both the model dump (from progress.nc), and the
diagnostic output (model_output.nc). The dump file gives an indication of the current state of the model.
The diagnostic file is only used here to plot the initial conditions of the firn column, so I can do a diff to see
visually how it has changed over time.
"""

from netCDF4 import Dataset
from matplotlib import pyplot as plt
import numpy.ma as ma
import matplotlib
matplotlib.use('TkAgg')
dumppath = '/media/dmwq2/2C40-69DF/progress.nc'
diagpath = '/media/dmwq2/2C40-69DF/model_output.nc'

flowdata = Dataset(dumppath)
t0data = Dataset(diagpath)

def plot_variable(dset, variable_name, cmap='viridis', vmax=None):
    """
    Plot a variable from an input dump file as a 2D grid with plt.imshow. This is a shorthand to generate
    plots from the final model state.

    Parameters
    ----------
    dset : netCDF4.Dataset
        The dataset containing the variable to plot.
    variable_name : str
        The name of the variable to plot.
    cmap : str, optional
        The colormap to use for the plot. Default is 'viridis'.
    vmax : float, optional
        The maximum value for the color scale. Default is None.

    Returns
    -------
    variable : netCDF4.Variable
        The variable from the input netCDF that was plotted.
    """

    variable = dset.variables[variable_name]
    plt.figure()
    plt.imshow(variable, vmax=vmax, cmap=cmap)
    plt.colorbar()
    title_str = variable_name.replace('_', ' ')
    if vmax:
        title_str += ', (max shown = {})'.format(vmax)
    plt.title(title_str)
    return variable

lakedepth = plot_variable(flowdata, 'lake_depth')
lakedepth_max1 = plot_variable(flowdata,'lake_depth', vmax=1)
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

from matplotlib.colors import TwoSlopeNorm
plt.figure()
diff = firndepth[:] - ofd[:]
plt.imshow(diff, cmap='coolwarm', norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3))
plt.colorbar()
plt.title('Firn depth difference (m)')

flowdata.close()
t0data.close()