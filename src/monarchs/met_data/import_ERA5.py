import netCDF4
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def ERA5_to_variables(ERA5_input):
    """
    Take in an input ERA5 netCDF file, and convert it into a dictionary that can be read in by MONARCHS.
    This step also performs the necessary unit conversions from the Copernicus default units to the ones used in
    MONARCHS.
    If the input netCDF doesn't have some parameters, use default values for these instead. These are denoted by the
    try/except blocks.

    Parameters
    ----------
    ERA5_input : str
        Path to a netCDF file of meteorological input.

    Returns
    -------
    var_dict : dict
        Dictionary of gridded output, with variable names and formatting suitable for loading into MONARCHS.
    """
    # Load in dimensions first
    var_dict = {}
    ERA5_data = netCDF4.Dataset(ERA5_input)
    assert "latitude" in ERA5_data.variables.keys(), ('"latitude" not found in the input ERA5 netCDF.'
                                                      'This indicates that the data is not in the correct format.')
    assert "longitude" in ERA5_data.variables.keys(), ('"longitude" not found in the input ERA5 netCDF. '
                                                       'This indicates that the data is not in the correct format.')
    var_dict["long"] = ERA5_data.variables["longitude"][:]
    var_dict["lat"] = ERA5_data.variables["latitude"][:]
    try:
        var_dict["time"] = ERA5_data.variables["time"][:]
    except KeyError:
        try:
            var_dict["time"] = ERA5_data.variables['valid_time'][:]
        except:
            raise KeyError(
                'Time variable "time" or "valid_time" not found in the input ERA5 netCDF. Check your input data,'
                "or amend <monarchs.met_data.import_ERA5.ERA5_to_variables> to use the key that is in "
                "your data."
            )

    assert "u10" in ERA5_data.variables.keys(), '"u10" (10m u-component of wind) not found in the input ERA5 netCDF.'
    assert "v10" in ERA5_data.variables.keys(), '"v10" (10m v-component of wind) not found in the input ERA5 netCDF.'
    # get overall wind speed via addition in quadrature of x and y components
    var_dict["wind"] = np.sqrt(
        ERA5_data.variables["u10"][:] ** 2 + ERA5_data.variables["v10"][:] ** 2
    )

    # Load in temperature and dew point temperature
    assert "t2m" in ERA5_data.variables.keys(), '"t2m" (2m temperature) not found in the input ERA5 netCDF.'
    assert "d2m" in ERA5_data.variables.keys(), '"d2m" (2m dew point temperature) not found in the input ERA5 netCDF.'

    var_dict["temperature"] = ERA5_data.variables["t2m"][:]  # [K]
    var_dict["dew_point_temperature"] = ERA5_data.variables["d2m"][:]  # [K]

    # Load in either mean-sea-level pressure or surface pressure.
    try:
        var_dict["pressure"] = ERA5_data.variables["sp"][:] / 100  # [Pa] -> [hPa]
    except KeyError:
        try:
            var_dict["pressure"] = ERA5_data.variables["msl"][:] / 100  # [Pa] -> [hPa]
        except:
            raise KeyError(
                'Pressure variable "sp" or "msl" not found in the input ERA5 netCDF. Check your input data,'
                "or amend <monarchs.met_data.import_ERA5.ERA5_to_variables> to use the key that is in "
                "your data."
            )
    assert "sf" in ERA5_data.variables.keys(), '"sf" (snowfall) not found in the input ERA5 netCDF.'
    var_dict["snowfall"] = ERA5_data.variables["sf"][:]  # [m water equiv.]

    # Load in shortwave radiation - first try all-sky, then clear-sky if that doesn't work, else fail
    try:
        var_dict["SW_surf"] = (
            ERA5_data.variables["ssrd"][:] / 3600
        )  # [J m^-2] -> [W m^-2]
    except KeyError:
        try:
            var_dict["SW_surf"] = (
                ERA5_data.variables["ssrdc"][:] / 3600
            )  # [J m^-2] -> [W m^-2]
            print(
                "Reading in clear-sky rather than all-sky radiation data since ssrd was not in the input netCDF"
            )
        except:
            raise KeyError(
                'Downwelling shortwave radiation variable "ssrd" or "ssrdc" not found in the '
                "input ERA5 netCDF. "
                "Check your input data, or amend <monarchs.met_data.import_ERA5.ERA5_to_variables> "
                "to use the key that is in your data."
            )

    # likewise for longwave - try and read in strd, if not strdc, else fail
    try:
        var_dict["LW_surf"] = (
            ERA5_data.variables["strd"][:] / 3600
        )  # [J m^-2] -> [W m^-2]
    except KeyError:
        try:
            print(
                "Reading in clear-sky rather than all-sky radiation data since strd was not in the input netCDF"
            )
            var_dict["LW_surf"] = (
                ERA5_data.variables["strdc"][:] / 3600
            )  # [J m^-2] -> [W m^-2]
        except:
            raise KeyError(
                'Downwelling longwave radiation variable "strd" or "strdc" not found in the '
                "input ERA5 netCDF. "
                "Check your input data, or amend <monarchs.met_data.import_ERA5.ERA5_to_variables> "
                "to use the key that is in your data."
            )
    # [0-1], can use up until melt occurs then we need to calc our own albedo
    try:
        var_dict["snow_albedo"] = ERA5_data.variables["asn"][:]
    except KeyError:
        var_dict["snow_albedo"] = 0.85 * np.ones(np.shape(ERA5_data.variables["t2m"]))
        print('monarchs.met_data.import_ERA5: No snow albedo data found in the input netCDF. '
              'Using default value of 0.85.')
    try:
        var_dict["snow_dens"] = ERA5_data.variables["rsn"][:]  # [kgm^-3]
    except KeyError:
        var_dict["snow_dens"] = 300 * np.ones(np.shape(ERA5_data.variables["t2m"]))
        print('monarchs.met_data.import_ERA5: No snow density data found in the input netCDF. '
              'Using default value of 300')
    ERA5_data.close()
    return var_dict


def grid_subset(
    var_dict,
    lat_upper_bound,
    lat_lower_bound,
    long_upper_bound,
    long_lower_bound,
):
    """
    Obtain a subset of a met data dictionary from ERA5_to_variables using a set of user-defined latitude and
    longitude boundaries.
    As this model is used for Antarctic ice shelves, ensure that your max and min bounds are the correct ones and not
    swapped around, as they may well be negative (S).

    Parameters
    ----------
    var_dict : dict
        Dictionary of variables from the input netCDF.
    lat_upper_bound : float
        User-defined upper latitude boundary.

    lat_lower_bound : float
        User-defined lower latitude boundary.

    long_upper_bound : float
        User-defined upper longitude boundary.

    long_lower_bound : float
        User-defined lower longitude boundary.

    Returns
    -------
    var_dict : dict
        Amended input dictionary.
    """
    # set up dictionary
    var_dict = dict(var_dict)
    # Obtain lat/long indices within the chosen boundaries
    lat_indices = np.where(
        (var_dict["lat"] <= lat_upper_bound) & (var_dict["lat"] > lat_lower_bound)
    )[0]
    long_indices = np.where(
        (var_dict["long"] <= long_upper_bound) & (var_dict["long"] >= long_lower_bound)
    )[0]

    for key in var_dict.keys():
        if key in ["time"]:
            continue
        elif key == "lat":
            var_dict[key] = var_dict[key][lat_indices]
        elif key == "long":
            var_dict[key] = var_dict[key][long_indices]
        else:
            var_dict[key] = var_dict[key][:, lat_indices, :]
            var_dict[key] = var_dict[key][:, :, long_indices]

    return var_dict


def interpolate_grid(var_dict, num_rows, num_cols):
    """
    Take an input grid, and interpolate it from the native grid to the MONARCHS model grid.
    Ensure that you have first restricted your met data file to the correct bounds using grid_subset.

    Parameters
    ----------
    var_dict : dict
        Dictionary of variables from the input netCDF.
    num_rows : int
        Number of rows in the model grid.
    num_cols : int
        Number of columns in the model grid.
    Returns
    -------
    var_dict : dict
        Amended input dictionary.

    """
    var_dict = dict(var_dict)
    print("monarchs.met_data.interpolate_grid: Interpolating ERA5 data")
    new_lat = np.nan
    new_long = np.nan
    for key in var_dict.keys():
        # print(key)
        if key in ["lat", "long", "time"]:
            continue
        interp = RegularGridInterpolator(
            (var_dict["time"][:], var_dict["lat"], var_dict["long"][:]),
            var_dict[key][:, :, :],
        )
        new_lat = np.linspace(var_dict["lat"][-1], var_dict["lat"][0], num_cols)
        new_long = np.linspace(var_dict["long"][0], var_dict["long"][-1], num_rows)
        X, Y, Z = np.meshgrid(var_dict["time"], new_lat, new_long, indexing="ij")
        var_dict[key] = interp((X, Y, Z))

    var_dict["lat"] = new_lat
    var_dict["long"] = new_long
    return var_dict


def generate_met_dem_diagnostic_plots(old_ERA5_grid, ERA5_grid, ilats, ilons, iheights):
    """
    Generate diagnostic plots to visualise the regridding of ERA5 data onto the DEM mesh.

    Parameters
    ----------
    old_ERA5_grid
    ERA5_grid
    ilats
    ilons
    iheights

    Returns
    -------

    """
    import cartopy.crs as ccrs
    import cartopy
    print('monarchs.met_data.import_ERA5: Generating diagnostic plots')
    latmax = ilats.max()
    latmin = ilats.min()
    lonmax = ilons.max()
    lonmin = ilons.min()
    # First figure - original met data on the Cartesian grid. Set up figure, colour map and projection:
    fig1 = plt.figure()
    cmap = "viridis"
    projection = ccrs.PlateCarree()
    # first panel

    ax1 = fig1.add_subplot(211, projection=projection)
    ax1.set_title("Original vs regridded met data")

    ax1.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)
    lons = old_ERA5_grid["long"][:]
    lats = old_ERA5_grid["lat"][:]
    temperature = old_ERA5_grid["temperature"][:]
    vmin = np.min(temperature[0])
    vmax = np.max(temperature[0])
    # use the levels kwarg to ensure that the colour maps are consistent
    levels = np.linspace(vmin, vmax, 20)
    cont = ax1.contourf(
        lons,
        lats,
        temperature[0],
        cmap=cmap,
        transform=projection,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
    )

    # Figure 2 - regridded data
    ax2 = fig1.add_subplot(212, projection=projection)
    ax2.coastlines()
    ax2.gridlines(draw_labels=True)
    ax2.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    ax2.add_feature(cartopy.feature.LAND)

    cont = ax2.contourf(
        ilons,
        ilats,
        ERA5_grid["temperature"][0],
        cmap=cmap,
        transform=projection,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
    )
    cbar_ax = fig1.add_axes((0.85, 0.15, 0.05, 0.7))
    fig1.colorbar(cont, ticks=levels, extend="both", cax=cbar_ax)

    # Figure 3 - plot the DEM data on the rotated-polar grid
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111, projection=projection)
    ax3.coastlines()
    ax3.gridlines(draw_labels=True)
    cont = ax3.contourf(
        ilons,
        ilats,
        iheights,
        cmap=cmap,
        levels=20,
        transform=ccrs.PlateCarree(),
    )

    ax3.title.set_text("Initial DEM height profile")
    fig2.colorbar(cont, extend="both")

    plt.show()


def get_met_bounds_from_DEM(
    model_setup, ERA5_grid, lat_array, lon_array, diagnostic_plots=False
):
    from monarchs.DEM.load_DEM import export_DEM
    from monarchs.core.utils import find_nearest

    # Handle cases where we don't have lat/long bounds set in the model setup file to avoid errors
    bounds = [
        "bbox_top_right",
        "bbox_bottom_left",
        "bbox_top_left",
        "bbox_bottom_right",
    ]
    bdict = {}
    for bound in bounds:
        if not hasattr(model_setup, bound):
            bdict[bound] = np.nan
        else:
            bdict[bound] = getattr(model_setup, bound)
    # Get lat/long from DEM
    iheights, ilats, ilons,dx, dy = export_DEM(
        model_setup.DEM_path,
        top_right=bdict["bbox_top_right"],
        bottom_left=bdict["bbox_bottom_left"],
        top_left=bdict["bbox_top_left"],
        bottom_right=bdict["bbox_bottom_right"],
        num_points=model_setup.row_amount,
        diagnostic_plots=False,
        all_outputs=False,
        input_crs=model_setup.input_crs
    )

    print("Testing - loading lat/long bounds from DEM")
    # Select lat/long boundaries.
    # Find indices that relate to points in the DEM, then set the values accordingly.
    lat_indices = np.zeros((model_setup.row_amount, model_setup.col_amount))
    lon_indices = np.zeros((model_setup.row_amount, model_setup.col_amount))
    new_ERA5_grid = {}
    old_ERA5_grid = ERA5_grid

    # Go through each met data variable, and set to the new values directly (if lat, long). If a met
    # variable, then loop over the rows and columns, find the lat/long point in our DEM that is closest
    # to a lat/long point in the met data grid, and set the new met values to this closest value at
    # each gridpoint. Use the met_dem_diagnostic_plots model_setup flag to check this works visually.

    for var in ERA5_grid.keys():
        if var in ["time"]:
            new_ERA5_grid[var] = ERA5_grid[var]
            continue
        elif var in ["lat"]:
            new_ERA5_grid[var] = lat_array
            continue
        elif var in ["long"]:
            new_ERA5_grid[var] = lon_array
            continue

        # set up an empty array
        new_ERA5_grid[var] = np.zeros(
            (
                len(ERA5_grid["time"]),
                model_setup.row_amount,
                model_setup.col_amount,
            )
        )

        # loop over and find the relevant values. This is done very naively by finding the closest
        # lat/long separately. The assumption is that the resolution of the DEM is high enough
        # (and the resolution of the met data is low enough) that this does not matter too much.
        # A better way is to find the shortest Euclidean distance from the centre of a gridpoint
        # in the met data to the centre of a point on the DEM - a possible TODO in future.

        for i in range(model_setup.row_amount):
            for j in range(model_setup.col_amount):
                lat_indices[i, j] = find_nearest(ERA5_grid["lat"], lat_array[i, j])
                lon_indices[i, j] = find_nearest(ERA5_grid["long"], lon_array[i, j])
                new_ERA5_grid[var][:, i, j] = ERA5_grid[var][
                    :, int(lat_indices[i, j]), int(lon_indices[i, j])
                ]
    ERA5_grid = new_ERA5_grid

    #  Diagnostic plotting - ensure that this method works to a reasonable degree of accuracy.
    if diagnostic_plots:
        generate_met_dem_diagnostic_plots(
            old_ERA5_grid, ERA5_grid, ilats, ilons, iheights
        )

    return ERA5_grid


if __name__ == "__main__":
    #  Run this script directly to test some output generation and plot out the results.

    ERA5_input = "../../../data/ERA5_small.nc"
    ERA5_data = netCDF4.Dataset(ERA5_input)
    ERA5_vars = ERA5_to_variables(ERA5_input)
    row_amount = 5
    col_amount = 5
    ERA5_vars_subset_interpolated = interpolate_grid(ERA5_vars, row_amount, col_amount)

    import cartopy.crs as ccrs
    fig, ax = plt.subplots()
    projection = ccrs.PlateCarree(central_longitude=0)
    lons = ERA5_vars_subset_interpolated["long"][:]
    lats = ERA5_vars_subset_interpolated["lat"][:]
    T = ERA5_vars_subset_interpolated["temperature"][0]
    plt.figure()
    plt.imshow(T)
    plt.figure()
    plt.imshow(ERA5_vars["temperature"][0])
    vmin = np.min(T)
    vmax = np.max(T)
    cmap = "viridis"
    ax1 = fig.add_subplot(211, projection=projection)
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)
    frames = 364 * 24
    cont = ax1.contourf(
        lons,
        lats,
        T,
        cmap=cmap,
        transform=projection,
        levels=20,
        vmin=vmin,
        vmax=vmax,
    )
    # ax1.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())


    # temperature = ERA5_data.variables['t2m'][:]
    ax2 = fig.add_subplot(212, projection=projection)
    ax2.coastlines()
    ax2.gridlines(draw_labels=True)
    orig_lons = ERA5_data.variables["longitude"][:]
    orig_lats = ERA5_data.variables["latitude"][:]
    orig_T = ERA5_data.variables["t2m"][0]
    cont = ax2.contourf(
        orig_lons,
        orig_lats,
        orig_T,
        cmap=cmap,
        transform=projection,
        levels=20,
        vmin=vmin,
        vmax=vmax,
    )

    def animate_T(frame, im):
        day = int(np.floor(frame / 24))
        hour = (frame % 24) + 1
        # vmax = np.max(T[frame])
        im.set_data(T[frame])
        ax.set_title(f"Temperature, day = {day}, hour = {hour}")
        # im.set_clim(vmin, vmax)
