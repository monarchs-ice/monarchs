import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def export_gaussian_DEM(num_points=20, diagnostic_plots=False):
    """
    Generate a Gaussian elevation map with two lakes in the upper left and bottom right corners.
    The output is normalised between 0 and 1, so scale it by some factor to obtain a realistic firn depth.

    Parameters
    ----------
    num_points : float, optional
        Number of points to generate for our grid. Set this to the amount of points in the x and y
        directions of your model grid. Currently only works for square grids.
        Default 20.
    diagnostic_plots : bool, optional
        Flag to determine whether to generate some figures to visualise the elevation map.
        Default False.

    Returns
    -------
    interpolated_heights : array_like, float, dimension(x, y)
        Gaussian elevation map interpolated to num_points points.

    """

    def Gaussian(x, y):
        mu_x = mu_y = 0
        sigma = 0.3
        height = (
            1
            / (2 * np.pi * sigma**2)
            * np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma**2))
        )
        return 1 - height

    x = y = np.linspace(-1, 1, 10)
    x1, y1 = np.meshgrid(x, y)

    heights = Gaussian(x1, y1)
    heights[1, 1] = 0.5
    heights[8, 8] = 0.5
    # Get the scale factor from the number of points originally generated and the number of points requested
    scale = len(heights) / num_points

    def interpolate_DEM(heights, scale):
        """Interpolate the DEM from the original Gaussian to the scale that we want"""
        # new resolution is 1/scale
        x = np.linspace(0, 1, len(heights))
        y = np.linspace(0, 1, len(np.transpose(heights)))
        interp = RegularGridInterpolator(
            (x, y), heights, bounds_error=False, fill_value=None
        )
        xx = np.linspace(0, 1, int(len(heights) / scale))
        yy = np.linspace(0, 1, int(len(np.transpose(heights)) / scale))
        X, Y = np.meshgrid(xx, yy, indexing="ij")
        return interp((X, Y))

    interpolated_heights = interpolate_DEM(heights, scale)
    # Scale of 2 is ok here, 4 looses the 2 smaller lakes (but they could be made deeper if needed?)

    # Force DEM to be completely symmetric
    interpolated_heights = (interpolated_heights + interpolated_heights[::-1,::-1]) / 2

    if diagnostic_plots:
        fig = plt.figure(figsize=(4, 2))
        plt.imshow(interpolated_heights, vmin=0, vmax=1)
        plt.set_cmap("Reds")
        cbar = plt.colorbar()
        cbar.set_label("Height (m)")
        plt.title("Initial Height Gaussian & Two Lakes")
        # plt.savefig('RBISInit_height.jpg')
        plt.show()
    return np.clip(interpolated_heights, 0, 1)


if __name__ == "__main__":
    """
    Run the script to generate a test case DEM and plot it up.
    """
    h = export_gaussian_DEM(num_points=10, diagnostic_plots=True)
