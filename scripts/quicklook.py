"""
Pick a variable from a model_output file,
and visualise it over time using a slider.
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from datetime import datetime, timedelta
import matplotlib

matplotlib.use('WebAgg')
nc_path = '/Users/jonathanelsey/mo_tests/subset/output/model_output_subset.nc'
var_name = 'firn_melt_cumulative'
start_date = datetime(2014, 1, 1)
dt_days = 5

with Dataset(nc_path) as ds:
	lake_depth = ds[var_name][:]

nt, nx, ny = lake_depth.shape

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.18)
im = ax.imshow(lake_depth[0], cmap='viridis')
cb = plt.colorbar(im, ax=ax)
cb.set_label('Lake depth (m)')

# initial title
def get_date_str(t):
	return (start_date + timedelta(days=dt_days * t)).strftime('%Y-%m-%d')

title = ax.set_title(f'Timestep 0: {get_date_str(0)}')

# set up the slider
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.05])
slider = Slider(ax_slider, 'Timestep', 0, nt - 1, valinit=0, valstep=1)


def update(val):
    # update figure vals/title based on slider position
	t = int(slider.val)
	im.set_data(lake_depth[t])
	title.set_text(f'Timestep {t}: {get_date_str(t)}')
	im.set_clim(np.nanmin(lake_depth[t]), np.nanmax(lake_depth[t]))
	fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()