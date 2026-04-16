"""
Define the model grid datatype. This is a Numpy structured array.
"""

import numpy as np


def initialise_iceshelf(
    num_rows,
    num_cols,
    vert_grid,
    vert_grid_lake,
    vert_grid_lid,
    dtype,
    x,
    y,
    firn_depth,
    rho,
    firn_temperature,
    Sfrac=np.array([np.nan]),
    Lfrac=np.array([np.nan]),
    meltflag=np.array([np.nan]),
    saturation=np.array([np.nan]),
    lake_depth=0,
    lake_temperature=np.array([np.nan]),
    lid_depth=0,
    lid_temperature=np.array([np.nan]),
    melt=False,
    exposed_water=False,
    lake=False,
    v_lid=False,
    lid=False,
    water_level=0,
    water=np.array([np.nan]),
    ice_lens=False,
    ice_lens_depth=999,
    has_had_lid=False,
    lid_sfc_melt=0,
    lid_melt_count=0,
    melt_hours=0,
    exposed_water_refreeze_counter=0,
    virtual_lid_temperature=273.15,
    total_melt=0.0,
    daily_melt=0.0,
    valid_cells=True,
    lat=0,
    lon=0,
    numba=False,
    size_dx=1000,
    size_dy=1000,
):
    """
    Initialize a NumPy structured array representing the ice shelf.

    Parameters
    ----------
    All parameters correspond to the attributes specified in get_spec() below.

    Returns
    -------
    iceshelf : np.ndarray
        A NumPy structured array initialized with the provided parameters.
    """
    iceshelf = np.zeros((num_rows, num_cols), dtype=dtype)
    iceshelf["column"] = x
    iceshelf["row"] = y
    iceshelf["firn_depth"] = firn_depth
    iceshelf["vert_grid"] = vert_grid
    iceshelf["vertical_profile"] = np.moveaxis(
        np.linspace(0, firn_depth, vert_grid), 0, -1
    )
    iceshelf["vert_grid_lake"] = vert_grid_lake
    iceshelf["vert_grid_lid"] = vert_grid_lid
    iceshelf["rho"] = rho
    iceshelf["rho_lid"] = 917.0 * np.ones((num_rows, num_cols, vert_grid_lid))
    iceshelf["firn_temperature"] = firn_temperature
    if np.isnan(Sfrac).all():
        iceshelf["Sfrac"] = np.ones((num_rows, num_cols, vert_grid)) * rho / 917
    else:
        iceshelf["Sfrac"][:] = Sfrac
    if np.isnan(Lfrac).all():
        iceshelf["Lfrac"] = np.zeros((num_rows, num_cols, vert_grid))
    else:
        iceshelf["Lfrac"][:] = Lfrac
    if np.isnan(meltflag).all():
        iceshelf["meltflag"] = np.zeros((num_rows, num_cols, vert_grid))
    else:
        iceshelf["meltflag"][:] = meltflag
    if np.isnan(saturation).all():
        iceshelf["saturation"] = np.zeros((num_rows, num_cols, vert_grid))
    else:
        iceshelf["saturation"][:] = saturation
    if np.isnan(lake_temperature).any():
        iceshelf["lake_temperature"] = 273.15 * np.ones(
            (num_rows, num_cols, vert_grid_lake)
        )
    else:
        iceshelf["lake_temperature"][:] = lake_temperature
    if np.isnan(lid_temperature).any():
        iceshelf["lid_temperature"] = 273.15 * np.ones(
            (num_rows, num_cols, vert_grid_lid)
        )
    else:
        iceshelf["lid_temperature"][:] = lid_temperature
    if isinstance(lake_depth, np.ndarray) and len(lake_depth) > 1:
        iceshelf["lake_depth"] = lake_depth
    else:
        iceshelf["lake_depth"][:] = lake_depth
    if isinstance(lid_depth, np.ndarray) and len(lid_depth) > 1:
        iceshelf["lid_depth"] = lid_depth
    else:
        iceshelf["lid_depth"][:] = lid_depth
    iceshelf["water_level"][:] = water_level
    if np.isnan(water).any():
        iceshelf["water"][:] = np.zeros((num_rows, num_cols, vert_grid))
    else:
        iceshelf["water"][:] = water
    iceshelf["melt_hours"][:] = melt_hours
    iceshelf["exposed_water_refreeze_counter"][:] = exposed_water_refreeze_counter
    iceshelf["lid_sfc_melt"][:] = lid_sfc_melt
    iceshelf["melt"][:] = melt
    iceshelf["exposed_water"][:] = exposed_water
    iceshelf["lake"][:] = lake
    iceshelf["v_lid"][:] = v_lid
    iceshelf["virtual_lid_temperature"][:] = virtual_lid_temperature
    iceshelf["lid"][:] = lid
    iceshelf["ice_lens"][:] = ice_lens
    iceshelf["ice_lens_depth"][:] = ice_lens_depth
    iceshelf["has_had_lid"][:] = has_had_lid
    iceshelf["lid_melt_count"][:] = lid_melt_count
    iceshelf["total_melt"][:] = total_melt
    iceshelf["daily_melt"][:] = daily_melt
    iceshelf["rho_ice"][:] = 917
    iceshelf["rho_water"][:] = 1000
    iceshelf["L_ice"][:] = 334000
    iceshelf["pore_closure"][:] = 830
    iceshelf["k_air"][:] = 0.022
    iceshelf["cp_air"][:] = 1004
    iceshelf["k_water"][:] = 0.5818
    iceshelf["cp_water"][:] = 4217
    iceshelf["t_step"][:] = 0
    iceshelf["day"][:] = 0
    iceshelf["snow_added"][:] = 0
    iceshelf["reset_combine"][:] = False
    iceshelf["valid_cell"][:] = valid_cells
    iceshelf["lat"][:] = lat
    iceshelf["lon"][:] = lon
    iceshelf["numba"][:] = numba
    iceshelf["size_dx"][:] = size_dx
    iceshelf["size_dy"][:] = size_dy
    iceshelf["water_direction"] = np.zeros((num_rows, num_cols, 8))  # 8 possible directions

    return iceshelf


def get_spec( vert_grid_size, vert_grid_lid, vert_grid_lake):
    """
    Define the structured array dtype for the model grid, with explicit sizes for dimensions.

    Parameters
    ----------
    num_rows : int
        Number of rows in the grid.
    num_cols : int
        Number of columns in the grid.
    vert_grid_size : int
        Number of vertical grid points in the firn column.
    vert_grid_lid : int
        Number of vertical grid points in the frozen lid.
    vert_grid_lake : int
        Number of vertical grid points in the lake.

    Returns
    -------
    dtype : np.dtype
        Structured array dtype for the model grid.
    """
    dtype = np.dtype(
        [
            ("column", np.int32),
            ("row", np.int32),
            ("firn_depth", np.float64),
            ("vert_grid", np.int32),
            ("vertical_profile", np.float64, vert_grid_size),
            ("vert_grid_lake", np.int32),
            ("vert_grid_lid", np.int32),
            ("rho", np.float64, vert_grid_size),
            ("rho_lid", np.float64, vert_grid_lid),
            ("firn_temperature", np.float64, vert_grid_size),
            ("Sfrac", np.float64, vert_grid_size),
            ("Lfrac", np.float64, vert_grid_size),
            ("meltflag", np.float64, vert_grid_size),
            ("saturation", np.float64, vert_grid_size),
            ("lake_temperature", np.float64, vert_grid_lake),
            ("lid_temperature", np.float64, vert_grid_lid),
            ("water_level", np.float64),
            ("water", np.float64, vert_grid_size),
            ("melt", np.bool_),
            ("exposed_water", np.bool_),
            ("lake", np.bool_),
            ("lake_depth", np.float64),
            ("v_lid", np.bool_),
            ("virtual_lid_temperature", np.float64),
            ("lid", np.bool_),
            ("lid_depth", np.float64),
            ("ice_lens", np.bool_),
            ("ice_lens_depth", np.int32),
            ("rho_ice", np.float64),
            ("rho_water", np.float64),
            ("L_ice", np.float64),
            ("pore_closure", np.float64),
            ("k_air", np.float64),
            ("cp_air", np.float64),
            ("k_water", np.float64),
            ("cp_water", np.float64),
            ("v_lid_depth", np.float64),
            ("has_had_lid", np.bool_),
            ("melt_hours", np.int32),
            ("exposed_water_refreeze_counter", np.int32),
            ("lid_sfc_melt", np.float64),
            ("lid_melt_count", np.int32),
            ("total_melt", np.float64),
            ("daily_melt", np.float64),
            ("t_step", np.int32),
            ("day", np.int32),
            ("log", "U256"),
            ("snow_added", np.float64),
            ("reset_combine", np.bool_),
            ("valid_cell", np.bool_),
            ("lat", np.float64),
            ("lon", np.float64),
            ("size_dx", np.float64),
            ("size_dy", np.float64),
            ("numba", np.bool_),
            ("water_direction", np.int32, 8),
        ]
    )
    return dtype
