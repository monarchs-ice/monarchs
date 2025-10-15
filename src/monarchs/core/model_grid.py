"""
Define the model grid datatype. This is a Numpy structured array.
"""

import numpy as np


# This function handles loading in *all* the variables to the model grid,
# so the pylint warnings are not too relevant here. Splitting this would
# likely make it harder rather than easier to follow.
# pylint: disable=too-many-arguments,too-many-locals
# pylint: disable=too-many-positional-arguments
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
    sfrac=np.array([np.nan]),
    lfrac=np.array([np.nan]),
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
    def validate_inputs(value, default_value, z):
        """Check for NaNs and assign default values with correct shape."""
        shape = (num_rows, num_cols, z)

        # if value is not an array or contains NaNs, use default_value
        if not isinstance(value, np.ndarray) or np.isnan(value).any():
            # if default_value is a scalar, broadcast it
            if np.isscalar(default_value):
                return np.full(shape, default_value)
            # if default_value is an array, ensure it matches
            # the expected shape
            if isinstance(default_value, np.ndarray):
                if default_value.shape == shape:
                    return default_value

                return np.broadcast_to(default_value, shape)
        return value

    # want a nested dict here with the values to check with
    # their default values and their name
    values_to_check = {
        "sfrac": (sfrac, rho / 917, vert_grid),
        "lfrac": (lfrac, 0, vert_grid),
        "meltflag": (meltflag, 0, vert_grid),
        "saturation": (saturation, 0, vert_grid),
        "water": (water, 0, vert_grid),
        "lake_temperature": (lake_temperature, 273.15, vert_grid_lake),
        "lid_temperature": (lid_temperature, 273.15, vert_grid_lid),
    }
    validated_values = {}
    for key, default in values_to_check.items():
        validated_values[key] = validate_inputs(*default)

    # Set up dictionaries with the various groups of variables that are used
    # in the model.
    # We could throw everything into a big bucket here, but splitting them
    # into types can be useful as a reference.
    fixed_model_values = {
        "lat": lat,
        "lon": lon,
        "size_dx": size_dx,
        "size_dy": size_dy,
        "row": y,
        "column": x,
        "vert_grid": vert_grid,
        "vert_grid_lake": vert_grid_lake,
        "vert_grid_lid": vert_grid_lid,
    }

    model_vectors = {
        "firn_temperature": firn_temperature,
        "rho": rho,
        "Sfrac": validated_values["sfrac"],
        "Lfrac": validated_values["lfrac"],
        "meltflag": validated_values["meltflag"],
        "saturation": validated_values["saturation"],
        "lake_temperature": validated_values["lake_temperature"],
        "lid_temperature": validated_values["lid_temperature"],
        "water": validated_values["water"],
        "vertical_profile": np.moveaxis(
            np.linspace(0, firn_depth, vert_grid), 0, -1
        ),
    }

    model_scalars = {
        "ice_lens_depth": ice_lens_depth,
        "lid_depth": lid_depth,
        "lake_depth": lake_depth,
        "firn_depth": firn_depth,
        "water_level": water_level,
        "virtual_lid_temperature": virtual_lid_temperature,
        "t_step": 0,
        "day": 0,
    }
    flags = {
        "has_had_lid": has_had_lid,
        "valid_cell": valid_cells,
        "numba": numba,
        "melt": melt,
        "exposed_water": exposed_water,
        "lake": lake,
        "v_lid": v_lid,
        "lid": lid,
        "ice_lens": ice_lens,
        "reset_combine": False,
    }
    counters = {
        "lid_melt_count": lid_melt_count,
        "melt_hours": melt_hours,
        "exposed_water_refreeze_counter": exposed_water_refreeze_counter,
        "lake_refreeze_counter": 0,
    }

    constants = {
        "rho_ice": 917,
        "rho_water": 1000,
        "L_ice": 334000,
        "pore_closure": 830,
        "k_air": 0.022,
        "cp_air": 1004,
        "k_water": 0.5818,
        "cp_water": 4217,
    }

    # for water direction - 8 possible directions
    diagnostics = {
        "water_direction": np.zeros((num_rows, num_cols, 8)),
        "firn_boundary_change": 0,
        "lake_boundary_change": 0,
        "lid_boundary_change": 0,
        "snow_added": 0,
        "total_melt": total_melt,
        "lid_sfc_melt": lid_sfc_melt,
    }

    def add_keys_to_grid(input_dict):
        """Add keys and values from a dictionary to the
        iceshelf structured array."""
        for key, value in input_dict.items():
            iceshelf[key][:] = value

    list_of_dicts = [
        diagnostics,
        constants,
        flags,
        counters,
        fixed_model_values,
        model_vectors,
        model_scalars,
    ]
    for dict_item in list_of_dicts:
        add_keys_to_grid(dict_item)

    return iceshelf


def get_spec(vert_grid_size, vert_grid_lake, vert_grid_lid):
    """
    Define the structured array dtype for the model grid, with explicit sizes
    for dimensions.

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
            ("firn_boundary_change", np.float64),
            ("lake_boundary_change", np.float64),
            ("lid_boundary_change", np.float64),
            ("lake_refreeze_counter", np.int32),
        ]
    )
    return dtype
