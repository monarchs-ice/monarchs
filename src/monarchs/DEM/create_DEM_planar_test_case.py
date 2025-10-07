import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def export_planar_DEM(num_points=20, diagnostic_plots=False):
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

    # As with the Gaussian DEM - generate a heights array
    heights = np.zeros((len(x), len(y)), dtype=np.float64)
    for i in range(len(heights)):
        heights[:, i] = (len(heights) - i) / len(heights)
    scale = len(heights) / num_points

    def interpolate_DEM(heights, scale):
        """Interpolate the DEM from the original Gaussian to the scale that we
        want"""
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
    if diagnostic_plots:
        fig = plt.figure(figsize=(4, 2))
        plt.imshow(interpolated_heights, vmin=0, vmax=1)
        plt.set_cmap("Reds")
        cbar = plt.colorbar()
        cbar.set_label("Height (m)")
        plt.show()
    return np.clip(interpolated_heights, 0, 1)


if __name__ == "__main__":
    """
    Run the script to generate a test case DEM and plot it up.
    """
    h = export_planar_DEM(num_points=20, diagnostic_plots=True)
