""" """

# TODO - module-level docstring
import netCDF4
import numpy as np
from monarchs.met_data.index_map import apply_index_map, build_coarse_index_map
from monarchs.physics.constants import rho_water

MODULE_NAME = "monarchs.met_data.import_ERA5"


def ERA5_to_variables(
    era5_input, met_timestep, total_days, start_index=0, chunk_size=365
):
    """
    Take in an input ERA5 netCDF file, and convert it into a dictionary that
    can be read in by MONARCHS.
    This step also performs the necessary unit conversions from the Copernicus
    default units to the ones used in MONARCHS.
    If the input netCDF doesn't have some parameters, use default values for
    these instead. These are denoted by the try/except blocks.

    Parameters
    ----------
    era5_input : str
        Path to a netCDF file of meteorological input.

    Returns
    -------
    var_dict : dict
        Dictionary of gridded output, with variable names and formatting
        suitable for loading into MONARCHS.
    """
    routine_name = "ERA5_to_variables"
    var_dict = {}
    # met_timestep is the number of met timesteps per day (24 = hourly,
    # 8 = three-hourly, ...). ERA5 radiation is *accumulated* [J m^-2] over
    # each step, so the conversion to W m^-2 must divide by the step length.
    seconds_per_step = 86400 // met_timestep
    # Determine indices for start and end of the year.
    # We write only in one-yearly segments.

    if total_days * met_timestep > start_index + (met_timestep * chunk_size):
        end_index = start_index + (met_timestep * chunk_size)
    else:
        end_index = start_index + (met_timestep * total_days - start_index)

    start_index = int(start_index)
    end_index = int(end_index)
    era5_data = netCDF4.Dataset(era5_input)
    errflag = False
    try:
        if len(era5_data.variables["time"]) < end_index:
            errflag = True
    except KeyError:
        try:
            if len(era5_data.variables["valid_time"]) < end_index:
                errflag = True
        except KeyError:
            raise ValueError(
                f"{MODULE_NAME}.{routine_name}: No time"
                " variable found in the input netCDF file. Please check your"
                " input data."
            )
    finally:
        if errflag:
            raise ValueError(
                f"{MODULE_NAME}.{routine_name}: End index"
                f" {end_index} is greater than the length of the data"
                f" available ({len(era5_data.variables['time'])} timesteps) in"
                " the input netCDF file. Please check your input data is"
                " large enough, or adjust your chosen number of days to"
                " compensate."
            )

    var_dict["long"] = era5_data.variables["longitude"][:]
    var_dict["lat"] = era5_data.variables["latitude"][:]
    try:
        var_dict["time"] = era5_data.variables["time"][start_index:end_index]
    except KeyError:
        try:
            var_dict["time"] = era5_data.variables["valid_time"][start_index:end_index]
        except KeyError:
            raise KeyError(
                f"{MODULE_NAME}.{routine_name}: Time variable 'time' or 'valid_time' not found in the input"
                " ERA5 netCDF. Check your input data,or amend"
                " <monarchs.met_data.import_ERA5.ERA5_to_variables> to use the"
                " key that is in your data."
            )
    var_dict["wind"] = np.sqrt(
        era5_data.variables["u10"][start_index:end_index] ** 2
        + era5_data.variables["v10"][start_index:end_index] ** 2
    )
    var_dict["temperature"] = era5_data.variables["t2m"][start_index:end_index]
    try:
        var_dict["dew_point_temperature"] = era5_data.variables["d2m"][
            start_index:end_index
        ]
    except KeyError:
        # deprecated fallback based on 95% of true temperature - now raise an error
        # if no dewpoint temperature provided. may relax this in future
        raise KeyError(
            f"{MODULE_NAME}.{routine_name}: Dewpoint temperature 'd2m' not"
            " found in the input ERA5 netCDF. Check your input data, or amend"
            " <monarchs.met_data.import_ERA5.ERA5_to_variables> to use the"
            " key that is in your data."
        )
    try:
        var_dict["pressure"] = era5_data.variables["sp"][start_index:end_index] / 100
    except KeyError:
        try:
            var_dict["pressure"] = (
                era5_data.variables["msl"][start_index:end_index] / 100
            )
        except KeyError:
            raise KeyError(
                f"{MODULE_NAME}.{routine_name}: Pressure variable 'sp' or 'msl' not found in the input ERA5"
                " netCDF. Check your input data,or amend"
                " <monarchs.met_data.import_ERA5.ERA5_to_variables> to use the"
                " key that is in your data."
            )
    var_dict["snowfall"] = era5_data.variables["sf"][start_index:end_index]

    try:
        var_dict["SW_surf"] = (
            era5_data.variables["ssrd"][start_index:end_index] / seconds_per_step
        )
    except KeyError:
        try:
            var_dict["SW_surf"] = (
                era5_data.variables["ssrdc"][start_index:end_index] / seconds_per_step
            )
            print(
                "Reading in clear-sky rather than all-sky radiation data since"
                " ssrd was not in the input netCDF"
            )
        except KeyError:
            raise KeyError(
                f"{MODULE_NAME}.{routine_name}: Downwelling shortwave radiation variable `ssrd` or `ssrdc`"
                " not found in the input ERA5 netCDF. Check your input data,"
                " or amend <monarchs.met_data.import_ERA5.ERA5_to_variables>"
                " to use the key that is in your data."
            )
    try:
        var_dict["LW_surf"] = (
            era5_data.variables["strd"][start_index:end_index] / seconds_per_step
        )
    except KeyError:
        try:
            print(
                f"{MODULE_NAME}.{routine_name}: "
                "Reading in clear-sky rather than all-sky radiation data since"
                " strd was not in the input netCDF"
            )
            var_dict["LW_surf"] = (
                era5_data.variables["strdc"][start_index:end_index] / seconds_per_step
            )
        except KeyError:
            raise KeyError(
                f"{MODULE_NAME}.{routine_name}: "
                "Downwelling longwave radiation variable `strd` or `strdc` not"
                " found in the input ERA5 netCDF. Check your input data, or"
                " amend <monarchs.met_data.import_ERA5.ERA5_to_variables> to"
                " use the key that is in your data."
            )
    try:
        var_dict["snow_albedo"] = era5_data.variables["asn"][start_index:end_index]
    except KeyError:
        var_dict["snow_albedo"] = 0.85 * np.ones(
            np.shape(era5_data.variables["t2m"][start_index:end_index])
        )
    try:
        var_dict["snow_dens"] = era5_data.variables["rsn"][start_index:end_index]
    except KeyError:
        var_dict["snow_dens"] = 350 * np.ones(
            np.shape(era5_data.variables["t2m"][start_index:end_index])
        )  # Kuipers Munekke 2015

    # Convert snow amount from ERA5 units (in water equivalent) to actual snow depth
    # downstream code treats this as a snow depth in m and gets mass via snow_dens
    var_dict["snowfall"] = var_dict["snowfall"] * rho_water / var_dict["snow_dens"]
    era5_data.close()
    return var_dict


def grid_subset(
    var_dict,
    lat_upper_bound,
    lat_lower_bound,
    long_upper_bound,
    long_lower_bound,
):
    """
    Obtain a subset of a met data dictionary from ERA5_to_variables using a set
    of user-defined latitude and longitude boundaries.
    As this model is used for Antarctic ice shelves, ensure that your max and
    min bounds are the correct ones and not swapped around, as they may well be
    negative (S).

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
    var_dict = dict(var_dict)
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


def get_met_bounds_from_DEM(
    model_setup, era5_grid, lat_array, lon_array, diagnostic_plots=False
):
    from monarchs.dem_utils.load_dem import export_DEM

    routine_name = "get_met_bounds_from_DEM"
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
    iheights, ilats, ilons, dx, dy = export_DEM(
        model_setup.DEM_path,
        top_right=bdict["bbox_top_right"],
        bottom_left=bdict["bbox_bottom_left"],
        top_left=bdict["bbox_top_left"],
        bottom_right=bdict["bbox_bottom_right"],
        num_points=model_setup.row_amount,
        input_crs=model_setup.input_crs,
    )
    print(f"{MODULE_NAME}.{routine_name}: Loading in lat/long bounds from DEM")

    # Build 2-D index maps using vectorised nearest-neighbour (same result as
    # the previous find_nearest loop, but in one place with index_map module)
    lat_indices, lon_indices = build_coarse_index_map(
        era5_grid["lat"],
        era5_grid["long"],
        lat_array,
        lon_array,
    )

    # Preserve original coarse axes for netCDF writing (lat/long are overwritten)
    coarse_lat = np.asarray(era5_grid["lat"])
    coarse_lon = np.asarray(era5_grid["long"])

    # ── Update lat/long in the grid to the DEM geographic coords,
    #    but leave all met variables at coarse resolution ─────────────────
    era5_grid = dict(era5_grid)
    era5_grid["lat"] = lat_array
    era5_grid["long"] = lon_array
    era5_grid["coarse_lat"] = coarse_lat
    era5_grid["coarse_lon"] = coarse_lon

    if diagnostic_plots:
        from monarchs.met_data.diagnostics import generate_met_dem_diagnostic_plots

        # diagnostic plots still need the expanded data, so build it here
        # only if actually needed — avoids the cost in normal runs
        expanded = dict(era5_grid)
        for var in era5_grid.keys():
            if var in ["lat", "long", "time"]:
                continue
            expanded[var] = apply_index_map(era5_grid[var], lat_indices, lon_indices)
        generate_met_dem_diagnostic_plots(era5_grid, expanded, ilats, ilons, iheights)

    return era5_grid, lat_indices, lon_indices
