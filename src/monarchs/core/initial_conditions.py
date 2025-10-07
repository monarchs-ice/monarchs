"""
Functions used by run_MONARCHS.py to convert a model runscript
(default model_setup.py) to the format actually usedb y MONARCHS.
This includes setting up the initial firn profile information,
loading in meteorological data and interpolating it, and loading
in/interpolating the digital elevation model (DEM) if applicable.

TODO - further refactor initialise_firn_profile, docstrings
"""

import numpy as np
from monarchs.DEM.load_DEM import export_DEM
from monarchs.core.model_grid import initialise_iceshelf, get_spec


def initialise_firn_profile(model_setup, diagnostic_plots=False):
    """
    DEM/initial firn profile
    TODO - docstring
    TODO - This function has grown rather complex - perhaps abstract out.
    """
    func_name = "monarchs.core.initial_conditions.initialise_firn_profile"
    print(
        f"monarchs.core.initial_conditions.initialise_firn_profile: Setting up"
        f" firn profile"
    )
    if hasattr(model_setup, "DEM_path"):
        print(
            "monarchs.core.initial_conditions.initialise_firn_profile: Reading"
            " in firn depth from DEM"
        )

        firn_depth, lat_array, lon_array, dx, dy = export_DEM(
            model_setup.DEM_path,
            num_points=model_setup.row_amount,
            diagnostic_plots=diagnostic_plots,
            top_right=model_setup.bbox_top_right,
            top_left=model_setup.bbox_top_left,
            bottom_right=model_setup.bbox_bottom_right,
            bottom_left=model_setup.bbox_bottom_left,
            input_crs=model_setup.input_crs,
        )
    elif hasattr(model_setup, "firn_depth"):
        firn_depth = model_setup.firn_depth
        dx = model_setup.lat_grid_size
        dy = model_setup.lat_grid_size
    else:
        raise ValueError(
            f"{func_name}:"
            " Neither a path to a DEM or a firn depth profile exists. Please"
            " specify this in your model configuration file."
        )
    valid_cells = np.ones(
        (model_setup.row_amount, model_setup.col_amount), dtype=bool
    )
    if hasattr(model_setup, "firn_max_height"):
        if model_setup.max_height_handler == "clip":
            firn_depth = np.clip(firn_depth, 0, model_setup.firn_max_height)
        elif model_setup.max_height_handler == "filter":
            valid_cells[np.where(firn_depth > model_setup.firn_max_height)] = (
                False
            )
            with np.printoptions(threshold=np.inf):
                print(
                    f"{func_name}:"
                    " Filtering out cells according to the following mask"
                    " (False = filtered out), since they exceed the firn"
                    " height threshold:"
                )
                print("Valid cells = ", valid_cells)
    firn_depth_under_35_flag = False
    if hasattr(model_setup, "firn_min_height"):
        if model_setup.min_height_handler == "clip":
            firn_depth = np.clip(firn_depth, model_setup.firn_min_height)
        elif model_setup.min_height_handler == "filter":
            valid_cells[np.where(firn_depth < model_setup.firn_min_height)] = (
                False
            )
            with np.printoptions(threshold=np.inf):
                print(
                    f"{func_name}:"
                    " Filtering out cells according to the following mask"
                    " (False = filtered out), since they are below the firn"
                    " height threshold:"
                )
                print("Valid cells = ", valid_cells)
        elif model_setup.min_height_handler == "extend":
            if firn_depth.min() < model_setup.firn_min_height:
                firn_depth += model_setup.firn_min_height - firn_depth.min()
        elif model_setup.min_height_handler == "normalise":
            firn_depth_under_35_flag = True

    valid_cells_old = valid_cells
    valid_cells = check_for_isolated_cells(valid_cells)

    if not np.array_equal(valid_cells_old, valid_cells):
        print("Removed some isolated cells - new grid = ", valid_cells)

    firn_columns = np.moveaxis(
        np.linspace(0, firn_depth, int(model_setup.vertical_points_firn)),
        0,
        -1,
    )

    if hasattr(model_setup, "rho_init") and model_setup.rho_init != "default":
        rho = model_setup.rho_init
    else:
        if hasattr(model_setup, "rho_sfc"):
            rho_sfc = model_setup.rho_sfc
        else:
            rho_sfc = 500
        if not hasattr(model_setup, "rho_init"):
            print(
                f"{func_name}:"
                " rho_init not specified in run configuration file - using"
                " default profile (empirical formula with z_t = 37 and rho_sfc"
                f" = {rho_sfc})"
            )
        rho = rho_init_emp(firn_columns, rho_sfc, 37)
        if firn_depth_under_35_flag:
            print("Correcting firn profile\n\n\n")
            for rowidx, row in enumerate(firn_columns):
                for colidx, column in enumerate(row):
                    if column.max() < model_setup.firn_min_height:
                        profile_temp = np.linspace(
                            0,
                            model_setup.firn_min_height,
                            model_setup.vertical_points_firn,
                        )
                        rho_temp = rho_init_emp(profile_temp, rho_sfc, 37)
                        rho[rowidx, colidx] = np.interp(
                            model_setup.firn_min_height - column,
                            profile_temp,
                            rho_temp,
                        )[::-1]

    if hasattr(model_setup, "T_init") and model_setup.rho_init != "default":
        T = model_setup.T_init
    else:
        T_init = np.linspace(253.15, 263.15, model_setup.vertical_points_firn)[
            ::-1
        ]
        T = np.zeros(
            (
                model_setup.row_amount,
                model_setup.col_amount,
                model_setup.vertical_points_firn,
            )
        )
        T[:][:] = T_init
        if not hasattr(model_setup, "T_init"):
            print(f"{func_name}: ")
            print(
                "T_init not specified in run configuration file - using"
                " default profile (linear 260->240 K top to bottom"
            )
    print("\n")
    if (
        hasattr(model_setup, "DEM_path")
        and hasattr(model_setup, "lat_bounds")
        and model_setup.lat_bounds == "dem"
    ):
        return T, rho, firn_depth, valid_cells, lat_array, lon_array, dx, dy
    else:
        return T, rho, firn_depth, valid_cells, dx, dy


def check_for_isolated_cells(valid_cells):
    """
    Ensure that cells aren't isolated - e.g. a cell in the middle of the
    land doesn't pointlessly run any physics when it can't flow laterally.
    """
    for i in range(len(valid_cells)):
        for j in range(len(valid_cells[0])):
            if valid_cells[i, j]:
                neighbours = np.ones((3, 3))
                adjustments = (
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 0),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                )
                for adj in adjustments:
                    try:
                        if i + adj[0] < 0 or j + adj[1] < 0:
                            neighbours[adj[0] + 1, adj[1] + 1] = -999
                        elif not valid_cells[i + adj[0], j + adj[1]]:
                            neighbours[adj[0] + 1, adj[1] + 1] = 0
                    except IndexError:
                        neighbours[adj[0] + 1, adj[1] + 1] = -999
                neighbours[1, 1] = 2
                if not np.any(neighbours == 1) and not np.any(
                    neighbours == -999
                ):
                    valid_cells[i, j] = False
    return valid_cells


def rho_init_emp(z, rho_sfc, z_t):
    """
    Initialise the firn column with a density from an empirical formula.
    This follows Paterson, W. (2000). The Physics of Glaciers.
    Butterworth-Heinemann, using the formula of Schytt, V. (1958).
    Glaciology. A: Snow studies at Maudheim. Glaciology. B: Snow studies
    inland. Glaciology. C: The inner structure of the ice shelf at Maudheim as
    shown by core drilling. Norwegian-British- Swedish Antarctic Expedition,
    1949-5, IV.)

    Parameters
    ----------
    z : float
    rho_sfc : float
        Density that you desire for the surface firn layer. [kg m^-3]
    z_t : float


    Returns
    -------

    """
    rho = 917 - (917 - rho_sfc) * np.exp(-(1.9 / z_t) * z)
    return rho


def create_model_grid(
    row_amount,
    col_amount,
    firn_depth,
    vert_grid,
    vert_grid_lake,
    vert_grid_lid,
    rho,
    firn_temperature,
    Sfrac=np.array([np.nan]),
    Lfrac=np.array([np.nan]),
    meltflag=np.array([np.nan]),
    saturation=np.array([np.nan]),
    lake_depth=0.0,
    lake_temperature=np.array([np.nan]),
    lid_depth=0.0,
    lid_temperature=np.array([np.nan]),
    melt=False,
    exposed_water=False,
    lake=False,
    v_lid=False,
    lid=False,
    water_level=0,
    water=np.array([np.nan]),
    ice_lens=False,
    has_had_lid=False,
    lid_sfc_melt=0.0,
    lid_melt_count=0,
    melt_hours=0,
    exposed_water_refreeze_counter=0,
    virtual_lid_temperature=273.15,
    total_melt=0.0,
    valid_cells=np.array([np.nan]),
    use_numba=False,
    lats=np.array([np.nan]),
    lons=np.array([np.nan]),
    size_dx=1000.0,
    size_dy=1000.0,
):
    """
    Creates the model grid by initializing the ice shelf with the provided
    parameters.
    """
    y, x = np.meshgrid(
        np.arange(0, row_amount, 1), np.arange(0, col_amount, 1), indexing="ij"
    )
    dtype = get_spec(vert_grid, vert_grid_lake, vert_grid_lid)
    grid = initialise_iceshelf(
        row_amount,
        col_amount,
        vert_grid,
        vert_grid_lake,
        vert_grid_lid,
        dtype,
        x,
        y,
        firn_depth,
        rho,
        firn_temperature,
        Sfrac=Sfrac,
        Lfrac=Lfrac,
        meltflag=meltflag,
        saturation=saturation,
        lake_depth=lake_depth,
        lake_temperature=lake_temperature,
        lid_depth=lid_depth,
        lid_temperature=lid_temperature,
        melt=melt,
        exposed_water=exposed_water,
        lake=lake,
        v_lid=v_lid,
        lid=lid,
        water_level=water_level,
        water=water,
        ice_lens=ice_lens,
        ice_lens_depth=vert_grid + 1,
        has_had_lid=has_had_lid,
        lid_sfc_melt=lid_sfc_melt,
        lid_melt_count=lid_melt_count,
        melt_hours=melt_hours,
        exposed_water_refreeze_counter=exposed_water_refreeze_counter,
        virtual_lid_temperature=virtual_lid_temperature,
        total_melt=total_melt,
        valid_cells=valid_cells,
        numba=use_numba,
        lat=lats,
        lon=lons,
        size_dx=size_dx,
        size_dy=size_dy,
    )
    return grid
