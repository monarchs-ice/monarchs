# At the moment have a subset of RBIS, should work out how to do with masking out the non-ice shelf areas to put in MONARCHS
# RBIS from Sophie

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import rioxarray

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




def export_DEM(
        tiffname,
        num_points=50,
        top_right=False,
        top_left=False,
        bottom_right=False,
        bottom_left=False,
        diagnostic_plots=False,
        all_outputs=False,
        return_lats=True,
        input_crs=3031 # Antarctic polar stereographic projection by default
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
        Default False
    top_left : bool or array_like, optional, dimension([lat, long])
        Latitude/longitude of the top left of the rectangle
        to be used as the bounding box to extract the part of the DEM we want.
        Default False.
    bottom_right : bool or array_like, optional, dimension([lat, long])
        Latitude/longitude of the bottom right of the rectangle
        to be used as the bounding box to extract the part of the DEM we want.
        Default False.
    bottom_left : bool or array_like, optional, dimension([lat, long])
        Latitude/longitude of the bottom left of the rectangle
        to be used as the bounding box to extract the part of the DEM we want.
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
    from pyproj import CRS, Transformer
    from scipy.ndimage import zoom

    # print(f'Box boundary = {input_raster.tif_bBox_converted}')
    print("Reading in firn depth from DEM")
    input_raster = rioxarray.open_rasterio(tiffname).rio.reproject("EPSG:3031")
    # Get the lat/long coordinates of the DEM

    transformer = Transformer.from_crs(input_crs, CRS.from_epsg(4326), always_xy=True)
    x, y = np.meshgrid(input_raster.x.values, input_raster.y.values)
    x_flat, y_flat = x.flatten(), y.flatten()
    lon_array, lat_array = transformer.transform(x_flat, y_flat)
    lat_array = lat_array.reshape(x.shape)
    lon_array = lon_array.reshape(y.shape)

    # if all the box boundary coordinates aren't False or Nan, then take a subset defined by these bounds
    # use concatenate to turn into a single vector and check all values are ok, index by [0]
    if not any(val in [False, np.nan] for val in np.ravel([top_right, top_left, bottom_right, bottom_left])):
        custom_bbox = True
        projected_raster = input_raster.rio.reproject("EPSG:4326")
        # we need to transform our bounding box coordinates to the EPSG:3031 projection, then back again
        # into lat/long.
        from matplotlib.path import Path
        import xarray as xr
        corners = [bottom_left, bottom_right, top_right, top_left]

        lon_min, lon_max = min([corner[1] for corner in corners]), max([corner[1] for corner in corners])
        lat_min, lat_max = min([corner[0] for corner in corners]), max([corner[0] for corner in corners])
        subset_raster = projected_raster.rio.clip_box(minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max,
                                                   crs="EPSG:4326")
        heights = subset_raster.values[0]
        lat_subset = subset_raster.y.values
        lon_subset = subset_raster.x.values
        if len(lat_subset) != 0 and len(lon_subset) != 0:
            dx, dy = get_xy_distance(lat_subset, lon_subset)

            from matplotlib import pyplot as plt
            if diagnostic_plots:
                bounding_box_diagnostic_plots(input_raster, subset_raster, lon_array, lat_array, lon_subset, lat_subset)
            # set the values that we use from here to the subset values
            lon_array, lat_array = np.meshgrid(lon_subset, lat_subset)
            plt.figure()
            plt.imshow(subset_raster.values[0])
            plt.title('reprojected')
    else:
        heights = input_raster.values[0]
        custom_bbox = False

    # Remove NaN/overly negative values
    heights[heights < -10] = -10
    # interpolate the lat/long coordinates
    # speed up the interpolation step by first applying a zoom
    newlons = lon_array
    newlats = lat_array
    new_heights = heights
    nans = np.where(np.isnan(heights))
    heights[nans] = 999
    new_heights_interpolated = interpolate_DEM(heights, num_points)
    lat_interp = interpolate_DEM(newlats, num_points)
    lon_interp = interpolate_DEM(newlons, num_points)
    if np.size(lat_interp) > 1 and np.size(lon_interp) > 1:
        dx, dy = get_xy_distance(lat_interp, lon_interp)
    else:
        dx, dy = np.array([[0]]), np.array([[0]])

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
            dx,
            dy
        )  # , meta_dict
    elif return_lats:
        return new_heights_interpolated, lat_interp, lon_interp, dx, dy
    else:
        return new_heights_interpolated, dx, dy
    # return new_heights_interpolated#, newlons, newlats

def get_xy_distance(latitudes, longitudes):
    import numpy as np
    from pyproj import Geod

    # Load the raster (already in EPSG:4326)

    # Define the WGS84 geodetic model
    geod = Geod(ellps="WGS84")

    # Create 2D meshgrid of lat/lon if not passing a mesh grid already
    if np.ndim(latitudes) == 1 and np.ndim(longitudes) == 1:
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    else:
        lon_grid, lat_grid = longitudes, latitudes
    # Compute dy (North-South size of each grid cell in metres)
    lat1_dy = lat_grid[:, :-1]  # Take all but last row
    lat2_dy = lat_grid[:, 1:]  # Take all but first row
    dy_metres = geod.inv(lon_grid[:, :-1], lat1_dy, lon_grid[:, :-1], lat2_dy)[-1]

    # Compute dx (East-West size of each grid cell in metres)
    lon1_dx = lon_grid[:-1, :]  # Take all but last column
    lon2_dx = lon_grid[1:, :]  # Take all but first column
    dx_metres = geod.inv(lon1_dx, lat_grid[:-1, :], lon2_dx, lat_grid[:-1, :])[-1]

    # Convert dx/dy to full-sized arrays by padding (so they match raster size)
    dy_metres = np.pad(dy_metres, ((0, 0), (0, 1)), mode='edge')  # Pad last row
    dx_metres = np.pad(dx_metres, ((0, 1), (0, 0)), mode='edge')  # Pad last column

    print("Grid cell sizes computed for all points!")
    return dx_metres, dy_metres

def bounding_box_diagnostic_plots(input_raster, subset_raster, lon_array, lat_array, lon_subset, lat_subset):
    """

    Parameters
    ----------
    input_raster
    subset_raster
    lon_array
    lat_array
    lon_subset
    lat_subset

    Returns
    -------

    """
    from matplotlib import pyplot as plt
    import cartopy.crs as ccrs
    projection = ccrs.PlateCarree(central_longitude=0)
    cmap = "viridis"
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection=projection)
    levels = np.arange(np.nanmin(subset_raster.values[0]), np.nanmax(subset_raster.values[0]), 1)
    cont = ax1.contourf(
        lon_array,
        lat_array,
        input_raster.values[0],
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        vmax=np.nanmax(subset_raster.values[0]),
        vmin=np.nanmin(subset_raster.values[0]),
        levels=levels
    )
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)
    ax1.title.set_text('DEM height profile with subset')
    cont = ax1.contourf(
        lon_subset,
        lat_subset,
        subset_raster.values[0],
        cmap='magma',
        transform=ccrs.PlateCarree(),
        levels=levels
    )

    plt.figure()
    plt.imshow(subset_raster.values[0])
    plt.figure()
    plt.imshow(input_raster.values[0])
    plt.show()

def generate_diagnostic_plots(
        lons, lats, heights, newlons, newlats, newheights, new_heights_interpolated
):
    """
    Diagnostics for the DEM interpolation process.

    Parameters
    ----------
    lons

    lats

    heights

    newlons

    newlats

    newheights

    new_heights_interpolated

    Returns
    -------

    """
    import cartopy.crs as ccrs
    from matplotlib import pyplot as plt
    projection = ccrs.PlateCarree(central_longitude=0)
    cmap = "viridis"
    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection=projection)
    ax1.coastlines()
    bounds = np.arange(0, 500, 1)

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

    heights, lats, lons, newheights, new_heights_interpolated, newlons, newlats, dx, dy = (
        export_DEM(
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
    # iheights, ilats, ilons = export_DEM(
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
