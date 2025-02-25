"""
Functions used by run_MONARCHS.py to convert a model runscript (default model_setup.py) to the format actually used
by MONARCHS. This includes setting up the initial firn profile information, loading in meteorological data and
interpolating it, and loading in/interpolating the digital elevation map (DEM) if applicable.
"""

import numpy as np
from netCDF4 import Dataset

from monarchs.DEM.load_DEM import export_DEM
from monarchs.core.iceshelf_class import IceShelf
from monarchs.met_data.import_ERA5 import (
    ERA5_to_variables,
    interpolate_grid,
    grid_subset,
    get_met_bounds_from_DEM,
)


def initialise_firn_profile(model_setup, diagnostic_plots=False):
    """
    DEM/initial firn profile
    TODO - docstring
    TODO - This function has grown rather complex - perhaps abstract out.
    """
    print(
        f"monarchs.core.initial_conditions.initialise_firn_profile: Setting up firn profile"
    )

    # Check if we have the relevant parameters in our model setup file, and if so, either load the DEM, or
    # initialise the firn depth from the values in the model setup file.
    if hasattr(model_setup, "DEM_path"):
        firn_depth, lat_array, lon_array, dx, dy = export_DEM(
            model_setup.DEM_path,
            num_points=model_setup.row_amount,
            diagnostic_plots=diagnostic_plots,
            top_right=model_setup.bbox_top_right,
            top_left=model_setup.bbox_top_left,
            bottom_right=model_setup.bbox_bottom_right,
            bottom_left=model_setup.bbox_bottom_left,
            input_crs=model_setup.input_crs
        )
    elif hasattr(model_setup, "firn_depth"):
        firn_depth = model_setup.firn_depth
        dx = model_setup.lat_grid_size
        dy = model_setup.lat_grid_size

    else:
        raise ValueError(
            f"monarchs.core.initial_conditions.initialise_firn_profile: "
            "Neither a path to a DEM or a firn depth profile exists. Please specify this in your "
            "model configuration file."
        )

    valid_cells = np.ones((model_setup.row_amount, model_setup.col_amount), dtype=bool)

    # Sort out cells above the maximum height - likely land
    if hasattr(model_setup, "firn_max_height"):
        if model_setup.max_height_handler == "clip":
            firn_depth = np.clip(firn_depth, 0, model_setup.firn_max_height)
        # if filtering - set all cells that don't fit the criteria to False, so no physics is run on them
        elif model_setup.max_height_handler == "filter":
            valid_cells[np.where(firn_depth > model_setup.firn_max_height)] = False

            with np.printoptions(
                threshold=np.inf
            ):  # context manager so we can see the whole array, not a subset
                print(
                    f"monarchs.core.initial_conditions.initialise_firn_profile: "
                    f"Filtering out cells according to the following mask (False = filtered out), "
                    f"since they exceed the firn height threshold:"
                )
                print("Valid cells = ", valid_cells)


    firn_depth_under_35_flag = False
    # Sort out cells below the mininum height
    if hasattr(model_setup, "firn_min_height"):
        if model_setup.min_height_handler == "clip":
            firn_depth = np.clip(firn_depth, model_setup.firn_min_height)

        elif model_setup.min_height_handler == "filter":
            valid_cells[np.where(firn_depth < model_setup.firn_min_height)] = False

            with np.printoptions(threshold=np.inf):  # context manager so we print the whole array
                print(
                    f"monarchs.core.initial_conditions.initialise_firn_profile: "
                    f"Filtering out cells according to the following mask (False = filtered out), "
                    f"since they are below the firn height threshold:"
                )
                print("Valid cells = ", valid_cells)

        elif model_setup.min_height_handler == "extend":
            if firn_depth.min() < model_setup.firn_min_height:
                # Add some metres of firn to the column everywhere to ensure that everywhere is at least
                # <firn_min_height> metres in height. We do this everywhere to retain the correct water level.
                firn_depth += model_setup.firn_min_height - firn_depth.min()
        elif model_setup.min_height_handler == "normalise":
            # Generate a profile for 35 m, then interpolate it. This will mean that lower-down regions will be made
            # up of mostly dense ice. Here we just set up the flag to do so later, since we are collating the
            # min_height_handler options here.
            firn_depth_under_35_flag = True
    valid_cells_old = valid_cells
    valid_cells = check_for_isolated_cells(valid_cells)
    if not np.array_equal(valid_cells_old, valid_cells):
        print("Removed some isolated cells - new grid = ", valid_cells)

    # Height for each vertical grid cell, converted into a coordinate space
    firn_columns = np.transpose(
        np.linspace(0, firn_depth, int(model_setup.vertical_points_firn))
    )

    # Check to see if we want to use a user defined profile, or create it from the firn depth.
    if hasattr(model_setup, "rho_init") and model_setup.rho_init != "default":
        rho = model_setup.rho_init  # user-defined density profile

    else:
        # If we specify a surface density, use that, else use the default value (500)
        if hasattr(model_setup, "rho_sfc"):
            rho_sfc = model_setup.rho_sfc
        else:
            rho_sfc = 500
        # Print a warning message to say we are using the default profile if rho_init is not explicitly defined
        if not hasattr(model_setup, "rho_init"):
            print(
                f"monarchs.core.initial_conditions.initialise_firn_profile: "
                f"rho_init not specified in run configuration file - using default profile (empirical formula with "
                f"z_t = 37 and rho_sfc = {rho_sfc})"
            )
        # Empirical formula for initial density profile
        rho = rho_init_emp(firn_columns, rho_sfc, 37)

        # If we are using the 'normalise' flag set up when checking minimum height, then calculate density from the
        # bottom-up rather than the top-down, i.e. a firn depth of 10m will be dense ice, rather than firn.
        # This basically means that we define the profile using *height* as the coordinate rather than *depth*.
        if firn_depth_under_35_flag:
            print('Correcting firn profile\n\n\n')
            for rowidx, row in enumerate(firn_columns):
                for colidx, column in enumerate(row):
                    if column.max() < model_setup.firn_min_height:
                        profile_temp = np.linspace(0, model_setup.firn_min_height,
                                                   model_setup.vertical_points_firn)
                        rho_temp = rho_init_emp(profile_temp, rho_sfc, 37)
                        rho[rowidx, colidx] = np.interp(model_setup.firn_min_height - column,
                                                        profile_temp, rho_temp)[::-1]
                        # need to reverse direction using [::-1] to revert back to depth coordinates



    # Sort out cells below the minimum height - these are solid ice rather than firn in this case
    # TODO - commented out for testing a different method
    # if hasattr(model_setup, 'firn_min_height'):
    #     wheremin = np.where(firn_depth < model_setup.firn_min_height)
    #     print('Setting the following cells to solid ice as they are below the minimum firn height threshold (True = '
    #           'unchanged, False = ice)')
    #     edited_cells = np.ones_like(firn_depth, dtype=bool)
    #     edited_cells[wheremin] = False
    #     rho[wheremin] = 917
    #     print(edited_cells)

    # Temperature profile
    if hasattr(model_setup, "T_init") and model_setup.rho_init != "default":
        T = model_setup.T_init  # user-defined temperature profile

    else:
        # Otherwise use default temperature profile everywhere
        T_init = np.linspace(253.15, 263.15, model_setup.vertical_points_firn)[::-1]
        T = np.zeros(
            (
                model_setup.row_amount,
                model_setup.col_amount,
                model_setup.vertical_points_firn,
            )
        )
        T[:][:] = T_init
        # warn user we're using default profile if not explicitly specified in runscript
        if not hasattr(model_setup, "T_init"):
            print(f"monarchs.core.initial_conditions.initialise_firn_profile: ")
            print(
                "T_init not specified in run configuration file - using default profile"
                " (linear 260->240 K top to bottom"
            )
    print("\n")  # newline to separate from other config steps
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
    Ensure that cells aren't isolated - e.g. a cell in the middle of the land doesn't pointlessly
    run any physics when it can't flow laterally.
    """

    # Loop over both dimensions of valid cells
    for i in range(len(valid_cells)):
        for j in range(len(valid_cells[0])):
            # if a cell in this is valid, then set up a 3x3 grid of its neighbours in
            # each direction (including itself)
            if valid_cells[i, j]:
                neighbours = np.ones((3, 3))
                # pairs of indices relative to the "central" cell
                adjustments = ((-1, -1), (-1, 0), (-1, 1),
                               (0, -1), (0, 0), (0, 1),
                               (1, -1), (1, 0), (1, 1))

                # loop through these. If it is at a boundary, i.e. the cell is at an edge and therefore
                # borders "empty" space not an invalid cell, then we want to keep it.
                # This stops e.g. a single-column case being marked invalid.
                for adj in adjustments:
                    # try-except clause since for cells on the right-boundary of grids
                    # you can get adj + 1 = 2, which raises IndexError - but also need to account for
                    # -1 cases which loop back around, hence both the try-except and if-else
                    try:
                        if i + adj[0] < 0 or j + adj[1] < 0:
                            neighbours[adj[0] + 1, adj[1] + 1] = -999
                        else:
                            if not valid_cells[i+adj[0], j+adj[1]]:
                                neighbours[adj[0] + 1, adj[1] + 1] = 0

                    except IndexError:
                        neighbours[adj[0] + 1, adj[1] + 1] = -999
                neighbours[1, 1] = 2  # set central cell to 2 as it shouldn't count itself for
                # checking whether to keep it or not
                # set cells not meeting the criteria (at least one valid cell or a boundary next to it)
                if not np.any(neighbours == 1) and not np.any(neighbours == -999):
                    valid_cells[i, j] = False

    return valid_cells

def rho_init_emp(z, rho_sfc, z_t):
    """
    Initialise the firn column with a density derived from an empirical formula.
    This follows Paterson, W. (2000). The Physics of Glaciers. Butterworth-Heinemann,
    using the formula of Schytt, V. (1958). Glaciology. A: Snow studies at Maudheim. Glaciology. B: Snow studies
    inland. Glaciology. C: The inner structure of the ice shelf at Maudheim as shown by
    core drilling. Norwegian-British- Swedish Antarctic Expedition, 1949-5, IV.)

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


def interpolate_met_data(model_setup, lat_array=False, lon_array=False):
    """
    Meteorological data input
    Here I use ERA5 data for testing. This won't necessarily be realistic yet,
    but should work for testing the data pipeline.
    TODO - docstring
    """

    if model_setup.met_timestep == "hourly":
        index = 1
    elif model_setup.met_timestep == "three_hourly":
        index = 3
    elif model_setup.met_timestep == "daily":
        index = 24
    elif isinstance(model_setup.met_timestep, int):
        index = model_setup.met_timestep
    else:
        raise ValueError(
            f"monarchs.core.initial_conditions.interpolate_met_data: "
            'met_timestep should be an integer, "hourly", "three_hourly" or "daily". See documentation for'
            " model_setup.met_timestep for details."
        )

    # convert the variable names from the netCDF to those used in this code
    ERA5_vars = ERA5_to_variables(model_setup.met_input_filepath)

    # Select some bounds - lat upper/lower, long upper/lower (in degrees, -90/90, -180/180)
    bounds = ["latmax", "latmin", "longmax", "longmin"]

    if all(hasattr(model_setup, attr) for attr in bounds) and (
        all(~np.isnan(getattr(model_setup, attr)) for attr in bounds)
    ):
        ERA5_vars = grid_subset(
            ERA5_vars,
            model_setup.latmax,
            model_setup.latmin,
            model_setup.longmax,
            model_setup.longmin,
        )

    # Interpolate to our grid size
    ERA5_grid = interpolate_grid(
        ERA5_vars, model_setup.row_amount, model_setup.col_amount
    )

    # Restrict to bounds of our input DEM if desired.
    if hasattr(model_setup, "lat_bounds") and model_setup.lat_bounds.lower() == "dem":
        ERA5_grid = get_met_bounds_from_DEM(
            model_setup,
            ERA5_grid,
            lat_array,
            lon_array,
            diagnostic_plots=model_setup.met_dem_diagnostic_plots,
        )

    # Save it so we don't need to keep the whole thing in memory
    ERA5_grid_path = model_setup.met_output_filepath

    # Arbitrary forcing of met data in case we want to do some testing.
    if hasattr(model_setup, "radiation_forcing_factor"):
        if model_setup.radiation_forcing_factor not in [False, 1]:
            ERA5_grid["SW_surf"] *= model_setup.radiation_forcing_factor
            ERA5_grid["LW_surf"] *= model_setup.radiation_forcing_factor
            print(f"monarchs.core.initial_conditions.interpolate_met_data: ")
            print(
                f"Scaling SW_surf and LW_surf by a factor of {model_setup.radiation_forcing_factor} for testing"
            )
    # If our index is not 1, then we need to repeat each point <index> times, e.g. if our data is daily, then each
    # point actually corresponds to 24 model timesteps so we need to account for this

    if index > 1:
        selected_keys = [key for key in ERA5_grid.keys() if key not in ["lat", "long"]]
        for var in selected_keys:
            ERA5_grid[var] = np.repeat(ERA5_grid[var], index, axis=0)  # along time axis

    with Dataset(ERA5_grid_path, "w") as f:
        f.createGroup("variables")
        f.createDimension("time", len(ERA5_grid["SW_surf"]))
        f.createDimension("column", model_setup.col_amount)
        f.createDimension("row", model_setup.row_amount)

        for key, value in ERA5_grid.items():
            if key in ["long", "lat", "time"]:
                if key == 'long':
                    var = f.createVariable('cell_longitude',
                    np.dtype("float64").char, ("column", "row"))
                    var.long_name = 'Longitude of grid cell'
                    var[:] = value
                if key == 'lat':
                    var = f.createVariable('cell_latitude',
                     np.dtype("float64").char, ("column", "row"))
                    var.long_name = 'Latitude of grid cell'
                    var[:] = value
                else:
                    continue
            var = f.createVariable(
                key, np.dtype("float64").char, ("time", "column", "row")
            )
            var.long_name = key
            var[:] = value
        print(
            f"monarchs.core.initial_conditions.interpolate_met_data: "
            f"Saved meteorological data used for the model run into {ERA5_grid_path}"
        )
    return ERA5_grid_path


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
    ice_lens_depth=999,
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
    size_dy=1000.0
):
    """
    Create a grid of IceShelf objects based on a set of input parameters.
    Called in the runscript (run_MONARCHS.py)
    When changing optional arguments, ensure that any inputs are in the correct format, particularly if using Numba.
    e.g. if a variable is default np.array([np.nan]), ensure that the changed variable is array_like.

    Parameters
    ----------
    row_amount : int
        Number of rows in the model grid.
    col_amount: int
        Number of columns in the model grid.
    firn_depth : float
        Height of the firn column [m]
    lake_depth : float
        Height of the surface lake, 0 if not present [m]
    lid_depth : float
        Height of the frozen lid on top of the surface lake, 0 if not present [m]
     vert_grid : int
        Number of vertical grid points in the firn column.
        vert_grid_lake : int
        Number of vertical grid points in the lake. int
    vert_grid_lid : int
        Number of vertical grid points in the frozen lid. int
    rho : array_like, float, dimension(vert_grid),
        Firn density profile. Mostly used for convenience as all calculations use Sfrac and Lfrac. [kg m^-3]
    firn_temperature : array_like, float, dimension(vert_grid),
        Temperature profile of t
    Sfrac : array_like, float, dimension(vert_grid), optional
        Solid fraction of the firn.
    Lfrac : array_like, float, dimension(vert_grid), optional
        Liquid fraction of the firn.
    meltflag : array_like, bool, dimension(vert_grid), optional
        Boolean flag to determine if there is melt at a given vertical point
    saturation : array_like, bool, dimension(vert_grid), optional
        Boolean flag to determine whether a vertical layer is saturated.
    lake_temperature : array_like, float, dimension(vert_grid_lake), optional
        Temperature of the lake as a function of vertical level [K]
    lid_temperature : array_like, float, dimension(vert_grid_lake), optional
        Temperature of the lid as a function of vertical level [K]
    water_level : float, optional
        Water level of the cell. Can be either at the height of the lake, infinite (in the case of a lid), or
        at the height of the highest vertical level with saturated firn.
    water : array_like, float, dimension(vert_grid), optional
        Water content of each grid cell. Used only in lateral_functions.move_water, where it is necessary to combine
        water from the lake and from the firn (which is determined by Lfrac).
    melt_hours : int, optional
        Tracks the number of hours of melt that have occurred. Currently only tracks melting of the firn due to
        the temperature of the lake above it.
    exposed_water_refreeze_counter : int, optional
        Tracks the number of times that exposed water at the surface freezes due to surface conditions.
    lid_sfc_melt : float, optional
        Tracks the amount of meltwater resulting from melting of the frozen lid. Used to
    melt : bool, optional
        Flag to determine whether the model is in a melting state or not. This affects the surface albedo for the
        surface flux calculation, used variously but mostly in the surface energy balance for the lake and lid, and for
        the calculation of the heat equation.
    exposed_water : bool, optional
        Flag to track whether there is exposed water due to surface melting.
    lake : bool, optional
        Flag to track whether the model is in a state that includes a lake or not. Is True even if there is a
        frozen lid, until the lid freezes enough to create a single firn profile.
    v_lid : bool, optional
        Flag to track whether a virtual lid is present. Set to False if a true lid is present.
    virtual_lid_temperature : float, optional
        Temperature of the virtual lid, if there is one. The virtual lid is so small that this can be a single number
        rather than a vertical profile. [K]
    lid : bool, optional
        Flag to track whether a frozen lid has formed.
    ice_lens: bool, optional
        Flag to track whether there is pore closure and the formation of a lens of solid ice. Necessary for
        saturation to occur and lakes to form.
    ice_lens_depth : int, optional
        Vertical layer (not physical depth) of the ice lens if there is one. Default 999, i.e. no lens present.
    has_had_lid : bool, optional
        Flag to determine whether the model has undergone lid development.
        Set True if lid depth exceeds 0.1.
        Resets to False if lid melts below 0.1 m depth.
    lid_melt_count : int, optional
        Track the number of times that the lid has undergone melting. Used as a tracker.
    total_melt : float, optional
        Total amount of melting that has occurred. Not used for any physics, but as a tracker.
    valid_cells : array_like, bool
        Mask to filter out invalid (e.g. land) cells, so that we don't waste time running any physics on them.
    use_numba : ara
    lats : array, float, optional
    lons : array, float, optional
    # TODO - finish docstring
    """
    # Create a Numba typed list instead of a nested list. Each element of the
    # typed list is also a typed list, consisting of an instance of the
    # IceShelf class. This lets us use Numba's prange
    if use_numba:
        from numba.typed import List

        grid = List()
    else:
        grid = []

    if isinstance(size_dx, float) or isinstance(size_dy, float):
        size_dx = np.ones((row_amount, col_amount)) * size_dx
        size_dy = np.ones((row_amount, col_amount)) * size_dy
    for i in range(col_amount):
        if use_numba:
            _l = List()
        else:
            _l = []

        for j in range(row_amount):
            _l.append(
                IceShelf(
                    j,  # x
                    i,  # y
                    firn_depth[i, j],
                    vert_grid,
                    vert_grid_lake,
                    vert_grid_lid,
                    rho[i, j],
                    firn_temperature[i, j],
                    Sfrac,
                    Lfrac,
                    meltflag,
                    saturation,
                    lake_depth,
                    lake_temperature,
                    lid_depth,
                    lid_temperature,
                    melt,
                    exposed_water,
                    lake,
                    v_lid,
                    lid,
                    water_level,
                    water,
                    ice_lens,
                    ice_lens_depth,
                    has_had_lid,
                    lid_sfc_melt,
                    lid_melt_count,
                    melt_hours,
                    exposed_water_refreeze_counter,
                    virtual_lid_temperature,
                    total_melt,
                    valid_cell=valid_cells[i, j],
                    lat=lats[i, j],
                    lon=lons[i, j],
                    size_dx=size_dx[i, j],
                    size_dy=size_dy[i, j]
                )
            )
        grid.append(_l)

    return grid
