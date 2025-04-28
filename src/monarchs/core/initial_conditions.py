"""
Functions used by run_MONARCHS.py to convert a model runscript (default model_setup.py) to the format actually used
by MONARCHS. This includes setting up the initial firn profile information, loading in meteorological data and
interpolating it, and loading in/interpolating the digital elevation model (DEM) if applicable.
"""
import numpy as np
from monarchs.DEM.load_DEM import export_DEM
from monarchs.core.iceshelf_class import initialise_iceshelf, get_spec


def initialise_firn_profile(model_setup, diagnostic_plots=False):
    """
    DEM/initial firn profile
    TODO - docstring
    TODO - This function has grown rather complex - perhaps abstract out.
    """
    print(
        f'monarchs.core.initial_conditions.initialise_firn_profile: Setting up firn profile'
        )
    if hasattr(model_setup, 'DEM_path'):
        firn_depth, lat_array, lon_array, dx, dy = export_DEM(model_setup.
            DEM_path, num_points=model_setup.row_amount, diagnostic_plots=
            diagnostic_plots, top_right=model_setup.bbox_top_right,
            top_left=model_setup.bbox_top_left, bottom_right=model_setup.
            bbox_bottom_right, bottom_left=model_setup.bbox_bottom_left,
            input_crs=model_setup.input_crs)
    elif hasattr(model_setup, 'firn_depth'):
        firn_depth = model_setup.firn_depth
        dx = model_setup.lat_grid_size
        dy = model_setup.lat_grid_size
    else:
        raise ValueError(
            f'monarchs.core.initial_conditions.initialise_firn_profile: Neither a path to a DEM or a firn depth profile exists. Please specify this in your model configuration file.'
            )
    valid_cells = np.ones((model_setup.row_amount, model_setup.col_amount),
        dtype=bool)
    if hasattr(model_setup, 'firn_max_height'):
        if model_setup.max_height_handler == 'clip':
            firn_depth = np.clip(firn_depth, 0, model_setup.firn_max_height)
        elif model_setup.max_height_handler == 'filter':
            valid_cells[np.where(firn_depth > model_setup.firn_max_height)
                ] = False
            with np.printoptions(threshold=np.inf):
                print(
                    f'monarchs.core.initial_conditions.initialise_firn_profile: Filtering out cells according to the following mask (False = filtered out), since they exceed the firn height threshold:'
                    )
                print('Valid cells = ', valid_cells)
    firn_depth_under_35_flag = False
    if hasattr(model_setup, 'firn_min_height'):
        if model_setup.min_height_handler == 'clip':
            firn_depth = np.clip(firn_depth, model_setup.firn_min_height)
        elif model_setup.min_height_handler == 'filter':
            valid_cells[np.where(firn_depth < model_setup.firn_min_height)
                ] = False
            with np.printoptions(threshold=np.inf):
                print(
                    f'monarchs.core.initial_conditions.initialise_firn_profile: Filtering out cells according to the following mask (False = filtered out), since they are below the firn height threshold:'
                    )
                print('Valid cells = ', valid_cells)
        elif model_setup.min_height_handler == 'extend':
            if firn_depth.min() < model_setup.firn_min_height:
                firn_depth += model_setup.firn_min_height - firn_depth.min()
        elif model_setup.min_height_handler == 'normalise':
            firn_depth_under_35_flag = True
    valid_cells_old = valid_cells
    valid_cells = check_for_isolated_cells(valid_cells)
    if not np.array_equal(valid_cells_old, valid_cells):
        print('Removed some isolated cells - new grid = ', valid_cells)
    firn_columns = np.transpose(np.linspace(0, firn_depth, int(model_setup.
        vertical_points_firn)))
    if hasattr(model_setup, 'rho_init') and model_setup.rho_init != 'default':
        rho = model_setup.rho_init
    else:
        if hasattr(model_setup, 'rho_sfc'):
            rho_sfc = model_setup.rho_sfc
        else:
            rho_sfc = 500
        if not hasattr(model_setup, 'rho_init'):
            print(
                f'monarchs.core.initial_conditions.initialise_firn_profile: rho_init not specified in run configuration file - using default profile (empirical formula with z_t = 37 and rho_sfc = {rho_sfc})'
                )
        rho = rho_init_emp(firn_columns, rho_sfc, 37)
        if firn_depth_under_35_flag:
            print('Correcting firn profile\n\n\n')
            for rowidx, row in enumerate(firn_columns):
                for colidx, column in enumerate(row):
                    if column.max() < model_setup.firn_min_height:
                        profile_temp = np.linspace(0, model_setup.
                            firn_min_height, model_setup.vertical_points_firn)
                        rho_temp = rho_init_emp(profile_temp, rho_sfc, 37)
                        rho[rowidx, colidx] = np.interp(model_setup.
                            firn_min_height - column, profile_temp, rho_temp)[:
                            :-1]
    if hasattr(model_setup, 'T_init') and model_setup.rho_init != 'default':
        T = model_setup.T_init
    else:
        T_init = np.linspace(253.15, 263.15, model_setup.vertical_points_firn)[
            ::-1]
        T = np.zeros((model_setup.row_amount, model_setup.col_amount,
            model_setup.vertical_points_firn))
        T[:][:] = T_init
        if not hasattr(model_setup, 'T_init'):
            print(f'monarchs.core.initial_conditions.initialise_firn_profile: '
                )
            print(
                'T_init not specified in run configuration file - using default profile (linear 260->240 K top to bottom'
                )
    print('\n')
    if hasattr(model_setup, 'DEM_path') and hasattr(model_setup, 'lat_bounds'
        ) and model_setup.lat_bounds == 'dem':
        return T, rho, firn_depth, valid_cells, lat_array, lon_array, dx, dy
    else:
        return T, rho, firn_depth, valid_cells, dx, dy


def check_for_isolated_cells(valid_cells):
    """
    Ensure that cells aren't isolated - e.g. a cell in the middle of the land doesn't pointlessly
    run any physics when it can't flow laterally.
    """
    for i in range(len(valid_cells)):
        for j in range(len(valid_cells[0])):
            if valid_cells[i, j]:
                neighbours = np.ones((3, 3))
                adjustments = (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (
                    0, 1), (1, -1), (1, 0), (1, 1)
                for adj in adjustments:
                    try:
                        if i + adj[0] < 0 or j + adj[1] < 0:
                            neighbours[adj[0] + 1, adj[1] + 1] = -999
                        elif not valid_cells[i + adj[0], j + adj[1]]:
                            neighbours[adj[0] + 1, adj[1] + 1] = 0
                    except IndexError:
                        neighbours[adj[0] + 1, adj[1] + 1] = -999
                neighbours[1, 1] = 2
                if not np.any(neighbours == 1) and not np.any(neighbours ==
                    -999):
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


def create_model_grid(row_amount, col_amount, firn_depth, vert_grid,
    vert_grid_lake, vert_grid_lid, rho, firn_temperature, Sfrac=np.array([
    np.nan]), Lfrac=np.array([np.nan]), meltflag=np.array([np.nan]),
    saturation=np.array([np.nan]), lake_depth=0.0, lake_temperature=np.
    array([np.nan]), lid_depth=0.0, lid_temperature=np.array([np.nan]),
    melt=False, exposed_water=False, lake=False, v_lid=False, lid=False,
    water_level=0, water=np.array([np.nan]), ice_lens=False, ice_lens_depth
    =999, has_had_lid=False, lid_sfc_melt=0.0, lid_melt_count=0, melt_hours
    =0, exposed_water_refreeze_counter=0, virtual_lid_temperature=273.15,
    total_melt=0.0, valid_cells=np.array([np.nan]), use_numba=False, lats=
    np.array([np.nan]), lons=np.array([np.nan]), size_dx=1000.0, size_dy=1000.0
    ):
    """
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
    use_numba : bool, optional
        Flag to determine whether to use Numba typed lists or not. Default False.
    lat : float, optional
        Input latitude.
    lon : float, optional
        Input longitude.
    size_dx : float, optional
        Size of the grid cell in the x direction [m]
    size_dy : float, optional
        Size of the grid cell in the y direction [m]

    """
    x, y = np.meshgrid(np.arange(0, row_amount, 1), np.arange(0, col_amount,
        1), indexing='ij')
    dtype = get_spec(row_amount, col_amount, vert_grid, vert_grid_lake,
        vert_grid_lid)
    grid = initialise_iceshelf(row_amount, col_amount, vert_grid,
        vert_grid_lake, vert_grid_lid, dtype, x, y, firn_depth, rho,
        firn_temperature, Sfrac=np.array([np.nan]), Lfrac=np.array([np.nan]
        ), meltflag=np.array([np.nan]), saturation=np.array([np.nan]),
        lake_depth=0.0, lake_temperature=np.array([np.nan]), lid_depth=0.0,
        lid_temperature=np.array([np.nan]), melt=False, exposed_water=False,
        lake=False, v_lid=False, lid=False, water_level=0, water=np.array([
        np.nan]), ice_lens=False, ice_lens_depth=999, has_had_lid=False,
        lid_sfc_melt=0.0, lid_melt_count=0, melt_hours=0,
        exposed_water_refreeze_counter=0, virtual_lid_temperature=273.15,
        total_melt=0.0, valid_cells=np.array([np.nan]), numba=False, lat=np
        .array([np.nan]), lon=np.array([np.nan]), size_dx=1000.0, size_dy=
        1000.0)
    return grid
