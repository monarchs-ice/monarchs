# At the moment have a subset of RBIS, should work out how to do with masking out the non-ice shelf areas to put in MONARCHS
# RBIS from Sophie

import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS
from geotiff import GeoTiff
from scipy.interpolate import RegularGridInterpolator


def interpolate_DEM(heights, num_points):
    """

    Parameters
    ----------
    heights : array_like, float, dimension(lat, long)
        Array of heights from the DEM, creating a heightmap of the area we want to interpolate to our model space.
    num_points : int
        Number of points we want to interpolate the DEM onto. This currently just works for square grids, i.e.
        row_amount == col_amount.

    Returns
    -------
    interp((X, Y)) - DEM interpolated onto the regular grid defined by num_points.
    """
    # new resolution is 1/scale
    x = np.linspace(0, 1, len(heights))
    y = np.linspace(0, 1, len(np.transpose(heights)))
    interp = RegularGridInterpolator(
        (x, y), heights, bounds_error=False, fill_value=None
    )
    xx = np.linspace(0, 1, num_points)
    yy = np.linspace(0, 1, num_points)
    X, Y = np.meshgrid(xx, yy, indexing="ij")
    return interp((X, Y))


def export_DEM(tiffname, num_points=10, diagnostic_plots=False):
    """
    Load in a DEM TIFF file, and return an array with the elevation map covering the
    requested number of model gridpoints.

    Parameters
    ----------
    tiffname : str
        Name of the geoTIFF file containing the digital elevation map (DEM) we want to load in.
    num_points : int, optional
        number of gridpoints. Currently only supports squares (i.e.
        gridpoint number is same in x and y). Default 10.
    diagnostic_plots : bool, optional
        Flag to trigger plotting of the exported DEM, for testing. Default False (i.e. no plots).

    Returns
    -------
    heights : array_like, float, dimension(lat, long)
        Raw elevation data from the DEM.
    interpolated_heights : array_like, float, dimension(lat, long)
        DEM elevation data interpolated to the number of points specified by num_points.
        Likely the desired output of the function.
    meta_dict : dict
        Dictionary containing metadata about the input TIFF file, in case there is something the user wants to check.
    """
    im = Image.open(tiffname)
    # im=np.loadtxt(tiffname, delimiter=',')
    DEM_grid = np.array(im)

    meta_dict = {TAGS[key]: im.tag[key] for key in im.tag_v2}

    # The indices in "heights" reflect how much of the DEM we are interested in - in
    # this case a 45000 x 45000 m subset of the total?

    heights = DEM_grid  # [1580:2080, 760:1760]
    water_level = 0 * heights
    # print(DEM_grid)

    # Regridding to different resolution
    scale = len(heights) / num_points

    interpolated_heights = interpolate_DEM(heights, scale)

    # Show plots for diagnostics if running this as as script
    if diagnostic_plots:
        plt.imshow(DEM_grid, vmax=100, vmin=0)
        plt.colorbar()
        plt.title("Initial Height Whole RBIS")

        fig = plt.figure(figsize=(4, 2))
        plt.imshow(heights, vmin=0, vmax=100)
        # plt.set_cmap("Reds")
        cbar = plt.colorbar()
        cbar.set_label("Height (m)")
        plt.title("Initial Height Subset")
        # plt.savefig('RBISInit_height.jpg')

        fig = plt.figure(figsize=(4, 2))
        plt.imshow(interpolated_heights, vmin=0, vmax=100)
        # plt.set_cmap("Reds")
        cbar = plt.colorbar()
        cbar.set_label("Height (m)")
        plt.title("Initial Height Subset_Interpolated")
        # plt.savefig('RBISInit_height.jpg')
        plt.show()

    return heights, interpolated_heights, meta_dict


def export_DEM_geotiff(
        tiffname,
        num_points=50,
        top_right=False,
        top_left=False,
        bottom_right=False,
        bottom_left=False,
        diagnostic_plots=False,
        all_outputs=False,
        return_lats=True,
):
    """
    Load in a DEM from a GeoTIFF file, and turn it into an elevation map in the form of a Numpy array.
    This can optionally be done using a bounding box, to restrict the output to a specific region of the DEM.
    Since our DEM is in polar stereographic coordinates, this bounding box is not "nice". Therefore, we need to
    specify the corner coordinates of the bounding box rather than just a max/min lat/long.

    Parameters
    ----------
    tiffname : str
        Name of the geoTIFF file that you wish to load in.
    num_points : int, optional
        Number of points to interpolate the DEM onto. Currently only supports square grids as output.
        Default 50.

    top_right : bool or array_like, optional, dimension([lat, long])
        Latitude/longitude of the top right of the rectangle
        to be used as the bounding box to extract the part of the DEM we want.
        Requires latmin, longmax and latmax to not be False to do anything.
        Default False
    top_left : bool or array_like, optional, dimension([lat, long])
        Latitude/longitude of the top left of the rectangle
        to be used as the bounding box to extract the part of the DEM we want.
        Requires longmin, longmax and latmax to not be False to do anything.
        Default False.
    bottom_right : bool or array_like, optional, dimension([lat, long])
        Latitude/longitude of the bottom right of the rectangle
        to be used as the bounding box to extract the part of the DEM we want.
        Requires latmin, longmin and latmax to not be False to do anything.
        Default False.
    bottom_left : bool or array_like, optional, dimension([lat, long])
        Latitude/longitude of the bottom left of the rectangle
        to be used as the bounding box to extract the part of the DEM we want.
        Requires latmin, longmin and longmax to not be False to do anything.
        Default False.
    diagnostic_plots : bool, optional
        Flag that triggers whether to generate plots of the output for testing.
        Default False.
    all_outputs : bool, optional, optional
        Flag to determine whether to output intermediate points in the process, for testing. Default False, as we
        only need the zoomed and interpolated elevation map.
        Default False.

    Returns
    -------
    new_heights_interpolated : array_like, float, dimension(lat, long)
        Array containing the zoomed and interpolated elevation map. This is the main output of the function.
    heights : array_like, float, dimension(lat, long), optional
        Array containing the elevation data from the model, either within the bounding box or the whole grid.
        Only output when all_outputs is True.
    lat_array : array_like, float, dimension(y)
        Array containing the latitude coordinates of the elevation data.
        Only output when all_outputs is True.
    lon_array : array_like, float, dimension(x)
        Array containing the longitude coordinates of the elevation data.
        Only output when all_outputs is True.
    new_heights : array_like, float, dimension(lat, long), optional
        Heights after applying scipy.ndimage.zoom, but before interpolation.
        Only output when all_outputs is True.
    newlons : array_like, float, dimension(x)
        Array containing the latitude coordinates of the elevation data after applying scipy.ndimage.zoom.
        Only output when all_outputs is True.
    newlats : array_like, float, dimension(y)
        Array containing the longitude coordinates of the elevation data after applying scipy.ndimage.zoom.
        Only output when all_outputs is True.
    """
    gt = GeoTiff(tiffname)
    # print(f'Box boundary = {gt.tif_bBox_converted}')
    print("Reading in firn depth from DEM")
    # if all the box boundary coordinates aren't False or Nan, then take a subset defined by these bounds
    # use concatenate to turn into a single vector and check all values are ok, index by [0]

    if not any(val == False for val in[top_right, top_left, bottom_right, bottom_left])  and not any(
            np.isnan(val) for val in np.concatenate([top_right, top_left, bottom_right, bottom_left], axis=1)[0]
    ):
        heights = np.array(gt.read())
        lon_array, lat_array = gt.get_coord_arrays()
        heights[heights < -100] = np.nan  # filter out overly negative values
        # smooth over NaNs using nearest neighbour interpolation
        mask = np.where(~np.isnan(heights))
        from scipy import interpolate, spatial
        interp = interpolate.NearestNDInterpolator(np.transpose(mask), heights[mask])
        heights = interp(*np.indices(heights.shape))

        # find the nearest points in the DEM array to the chosen bounding box points
        coord_array = np.c_[lat_array.ravel(), lon_array.ravel()]
        coord_ref = np.dstack((lat_array, lon_array))
        trd, tri = spatial.KDTree(coord_array).query(top_right, k=1)  # top right index is what we want (tri)
        bld, bli = spatial.KDTree(coord_array).query(bottom_left, k=1)
        brd, bri = spatial.KDTree(coord_array).query(bottom_right, k=1)
        tld, tli = spatial.KDTree(coord_array).query(top_left, k=1)

        tri = np.unravel_index(tri, lat_array.shape)
        bli = np.unravel_index(bli, lat_array.shape)
        bri = np.unravel_index(bri, lat_array.shape)
        tli = np.unravel_index(tli, lat_array.shape)
        print('Top right - at ', tri)
        print('Bottom left - at ', bli)
        print('Bottom right - at ', bri)
        print('Top left - at ', tli)

        # Now we need to convert these into our slice. Take the average of our arrays.
        lat_start = int(np.floor((tri[0] + tli[0]) / 2)[0])
        lat_end = int(np.floor((bri[0] + bli[0]) / 2)[0])
        lon_start = int(np.floor((tli[1] + bli[1]) / 2)[0])
        lon_end = int(np.floor((tri[1] + bri[1]) / 2)[0])

        print('Length of lat arrays = ', lat_end - lat_start)
        print('Length of lon arrays = ', lon_end - lon_start)

        # Slice to get the final result
        lon_array = lon_array[lat_start:lat_end, lon_start:lon_end]
        lat_array = lat_array[lat_start:lat_end, lon_start:lon_end]
        heights = heights[lat_start:lat_end, lon_start:lon_end]

    else:  # otherwise get the whole array
        # print('Reading whole array')
        heights = np.array(gt.read())
        lon_array, lat_array = gt.get_coord_arrays()

    # Remove NaN/overly negative values
    heights[heights < -10] = 0
    # interpolate the lat/long coordinates
    from scipy.ndimage import zoom

    zoomlevel = 1 / 3
    newlons = zoom(lon_array, zoomlevel)
    newlats = zoom(lat_array, zoomlevel)
    nans = np.where(np.isnan(heights))
    heights[nans] = 0
    new_heights = zoom(heights, zoomlevel)
    new_heights_interpolated = interpolate_DEM(heights, num_points)
    lat_interp = interpolate_DEM(newlats, num_points)
    lon_interp = interpolate_DEM(newlons, num_points)

    if diagnostic_plots:
        generate_diagnostic_plots(
            lon_array,
            lat_array,
            heights,
            newlons,
            newlats,
            new_heights,
            new_heights_interpolated,
        )

    if all_outputs:
        return (
            heights,
            lat_array,
            lon_array,
            new_heights,
            new_heights_interpolated,
            newlons,
            newlats,
        )  # , meta_dict
    elif return_lats:
        return new_heights_interpolated, lat_interp, lon_interp
    else:
        return new_heights_interpolated
    # return new_heights_interpolated#, newlons, newlats


def generate_diagnostic_plots(
        lons, lats, heights, newlons, newlats, newheights, new_heights_interpolated
):
    import cartopy.crs as ccrs

    plt.figure()
    plt.imshow(heights)

    plt.figure()
    plt.imshow(newheights)
    plt.show()
    projection = ccrs.PlateCarree(central_longitude=0)
    cmap = "viridis"
    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection=projection)
    ax1.coastlines()
    bounds = np.arange(0, 500, 1)

    plt.figure()

    cont = ax1.contourf(
        lons,
        lats,
        heights,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
    )
    # levels=bounds, vmin=0, vmax=50)
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)
    ax1.title.set_text("Initial DEM height profile")

    ax2 = fig.add_subplot(212, projection=projection)
    cont2 = ax2.contourf(
        newlons,
        newlats,
        newheights,
        cmap=cmap,
        transform=projection,
        levels=bounds,
        vmin=0,
        vmax=500,
    )
    ax2.coastlines()
    ax2.gridlines(draw_labels=True)
    ax2.title.set_text("After using scipy.zoom")
    cax = plt.axes((0.85, 0.1, 0.075, 0.8))
    fig.colorbar(cont2, cax=cax)

    """
    This figure shows the final result - the DEM on our chosen grid. Note that if the aspect ratio 
    is different between the chosen grid and the size of the initial DEM grid (accounting for your choice 
    of boundaries), then the final DEM will be stretched to compensate. 
    """
    fig, ax3 = plt.subplots()
    cs = ax3.imshow(new_heights_interpolated, vmin=0, vmax=200)
    ax3.title.set_text(f"After interpolating to our grid size "
                       f"({len(new_heights_interpolated)}x{len(new_heights_interpolated[0])})")
    cb = fig.colorbar(cs)
    plt.show()


if __name__ == "__main__":
    """
    Run script directly to generate some test data and plot it up.
    """
    tiffname = "DEM/42_07_32m_v2.0/42_07_32m_v2.0_dem.tif"

    heights, lats, lons, newheights, new_heights_interpolated, newlons, newlats = (
        export_DEM_geotiff(
            tiffname,
            num_points=50,
            diagnostic_plots=False,
            top_right=[(-66.52, -62.814)],  # bounding box top right coordinates, [(lat, long)]
            bottom_left=[(-66.289, -64.68)],  # bounding box bottom left coordinates, [(lat, long)]
            top_left=[(-66.04, -63.42)],  # bounding box top left coordinates, [(lat, long)]
            bottom_right=[(-66.778, -64.099)],  # bounding box bottom right coordinates, [(lat, long)]
            all_outputs=True,
        )
    )
    # iheights, ilats, ilons = export_DEM_geotiff(
    #     tiffname,
    #     num_points=50,
    #     diagnostic_plots=False,
    #     latmin=-72.15,
    #     latmax=-71.55,
    #     longmin=-68.28,
    #     longmax=-67.1,
    #     all_outputs=False,
    # )
    import matplotlib.pyplot as plt

    generate_diagnostic_plots(
        lons, lats, heights, newlons, newlats, newheights, new_heights_interpolated
    )
