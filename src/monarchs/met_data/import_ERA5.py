""" """

# TODO - module-level docstring
import netCDF4
import numpy as np
from monarchs.met_data.index_map import apply_index_map, build_coarse_index_map
from monarchs.physics.constants import rho_water
from monarchs.met_data.catalogue import MET_CATALOGUE

MODULE_NAME = "monarchs.met_data.import_ERA5"

# met field name -> its MetVariable, so the ERA5 source names / derivations /
# conversions all come from the catalogue (monarchs.met_data.catalogue)
_MET = {var.name: var for var in MET_CATALOGUE}


def _read_era5(era5_data, field, start, end):
    """
    Read a met field's ERA5 source variable, named by the catalogue's
    ``era5_name`` (falling back to ``era5_fallback`` if the primary variable is
    absent), returning the ``[start:end]`` time slice. Raises ``KeyError`` with
    a pointer to the catalogue if neither variable is present.
    """
    var = _MET[field]
    candidates = [name for name in (var.era5_name, var.era5_fallback) if name]
    for i, name in enumerate(candidates):
        if name in era5_data.variables:
            if i > 0:
                print(
                    f"{MODULE_NAME}: '{var.era5_name}' not found for met field"
                    f" '{field}' - falling back to '{name}'"
                )
            return era5_data.variables[name][start:end]
    tried = " or ".join(f"'{name}'" for name in candidates)
    raise KeyError(
        f"{MODULE_NAME}.ERA5_to_variables: ERA5 variable {tried} (for met field"
        f" '{field}') not found in the input netCDF. Check your input data, or"
        f" amend the era5_name for '{field}' in monarchs.met_data.catalogue."
    )


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

    # divide 24h in seconds by the met timestep (default 1h)
    # to get the correct unit conversion for the radiation
    # variables, which are aggregated over 1h periods by default
    # in ERA5 ([J m^-2] -> [W m^-2])
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

    var_dict["long"] = era5_data.variables[_MET["lon"].era5_name][:]
    var_dict["lat"] = era5_data.variables[_MET["lat"].era5_name][:]
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

    # Generic pass over the time-sliced fields, driven by the catalogue's
    # era5_name / derived_from / convert. lat/lon (read in full above) and
    # snow_dens (shape-dependent fallback below) are handled separately.
    def read(name):
        return era5_data.variables[name][start_index:end_index]

    for var in MET_CATALOGUE:
        if var.name in ("lat", "lon", "snow_dens"):
            continue
        if var.derived_from is not None:
            value = var.derived_from(read)
        else:
            value = _read_era5(era5_data, var.name, start_index, end_index)
        if var.convert is not None:
            value = var.convert(value, seconds_per_step)
        var_dict[var.name] = value

    # snow albedo is read from ERA5 but is not a model met field; default if
    # absent (needs the field shape, so done after the loop reads temperature)
    try:
        var_dict["snow_albedo"] = read("asn")
    except KeyError:
        var_dict["snow_albedo"] = 0.85 * np.ones(np.shape(var_dict["temperature"]))
    # snow density falls back to a constant (Kuipers Munneke 2015) if absent
    try:
        var_dict["snow_dens"] = _read_era5(
            era5_data, "snow_dens", start_index, end_index
        )
    except KeyError:
        var_dict["snow_dens"] = 350 * np.ones(np.shape(var_dict["temperature"]))

    # convert snowfall from mwe to a height analogous to the firn height (this
    # couples snowfall to snow_dens, so it stays a step here rather than in the
    # catalogue)
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
