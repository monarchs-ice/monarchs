"""
Sets up a test case using a planar DEM that slopes from a maximum to a minimum
from left to right.
"""

import matplotlib.pyplot as plt
import numpy as np
from monarchs.dem_utils.create_dem_gaussian_test_case import (
    interpolate_func_to_dem,
)


def export_planar_dem(num_points=20, diagnostic_plots=False):
    """
    Generate an elevation map that goes from a maximum on the LHS to a minimum
    on the RHS.
    The output is normalised between 0 and 1, so scale it by some factor to
    obtain a realistic firn depth.

    Parameters
    ----------
    num_points : float, optional
        Number of points to generate for our grid. Set this to the amount of
        points in the x and y directions of your model grid. Currently only
        works for square grids.
        Default 20.
    diagnostic_plots : bool, optional
        Flag to determine whether to generate some figures to visualise the
        elevation map.
        Default False.

    Returns
    -------
    interpolated_heights : array_like, float, dimension(x, y)
        Planar elevation map interpolated to num_points points.

    """

    x = y = np.linspace(-1, 1, 10)

    # As with the Gaussian dem_utils - generate a heights array
    heights = np.zeros((len(x), len(y)), dtype=np.float64)
    for i in range(len(heights)):
        heights[:, i] = (len(heights) - i) / len(heights)
    scale = len(heights) / num_points

    interpolated_heights = interpolate_func_to_dem(heights, scale)
    if diagnostic_plots:
        plt.figure(figsize=(4, 2))
        plt.imshow(interpolated_heights, vmin=0, vmax=1)
        plt.set_cmap("Reds")
        cbar = plt.colorbar()
        cbar.set_label("Height (m)")
        plt.show()
    return np.clip(interpolated_heights, 0, 1)


if __name__ == "__main__":
    h = export_planar_dem(num_points=20, diagnostic_plots=True)
