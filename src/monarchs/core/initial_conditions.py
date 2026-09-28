"""
Functions used by run_monarchs.py to convert a model runscript
(default model_setup.py) to the format actually used by MONARCHS.
This includes setting up the initial firn profile information,
loading in meteorological data and interpolating it, and loading
in/interpolating the digital elevation model (DEM) if applicable.

"""

import numpy as np
from monarchs.dem_utils.load_dem import export_DEM
from monarchs.variables import make_grid


def initialise_firn_profile(model_setup, diagnostic_plots=False):
    """
    DEM/initial firn profile

    """
    #     TODO - docstring
    #     TODO - This function has grown rather complex - perhaps abstract out.
    func_name = "monarchs.core.initial_conditions.initialise_firn_profile"

    # some dummy values that may be returned later
    lat_array = 0
    lon_array = 0

    print(f"{func_name}: Setting up firn profile")
    if model_setup.DEM_path is not None:
        print(f"{func_name}: Reading in firn depth from DEM")

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
    elif model_setup.firn_depth is not None:
        firn_depth = model_setup.firn_depth
        dx = model_setup.lat_grid_size
        dy = model_setup.lat_grid_size
    else:
        raise ValueError(
            f"{func_name}:"
            " Neither a path to a DEM or a firn depth profile exists. Please"
            " specify this in your model configuration file."
        )
    valid_cells = np.ones((model_setup.row_amount, model_setup.col_amount), dtype=bool)
    # handle DEM heights above the user-defined maximum
    if model_setup.max_height_handler == "clip":
        firn_depth = np.clip(firn_depth, 0, model_setup.firn_max_height)
    elif model_setup.max_height_handler == "filter":
        valid_cells[np.where(firn_depth > model_setup.firn_max_height)] = False

    # likewise for heights below the user-defined minimum
    firn_depth_under_35_flag = False
    if model_setup.min_height_handler == "clip":
        firn_depth = np.clip(firn_depth, a_min=model_setup.firn_min_height, a_max=None)
    elif model_setup.min_height_handler == "filter":
        valid_cells[np.where(firn_depth < model_setup.firn_min_height)] = False
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

    # initialise density from the model setup script (an array), or from the
    # empirical profile for the "default" keyword
    rho_init = model_setup.rho_init
    if not isinstance(rho_init, str):
        rho = rho_init
    else:
        rho_sfc = model_setup.rho_sfc
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

    T_init = model_setup.T_init
    if not isinstance(T_init, str):
        temperature = T_init
    else:
        t_init = np.linspace(253.15, 263.15, model_setup.vertical_points_firn)[::-1]
        temperature = np.zeros(
            (
                model_setup.row_amount,
                model_setup.col_amount,
                model_setup.vertical_points_firn,
            )
        )
        temperature[:][:] = t_init

    # else return null values for the lat/long arrays which aren't used
    # pylint: disable=duplicate-code
    return (
        temperature,
        rho,
        firn_depth,
        valid_cells,
        dx,
        dy,
        lat_array,
        lon_array,
    )
    # pylint: enable=duplicate-code


def check_for_isolated_cells(valid_cells):
    """
    Ensure that cells aren't isolated - e.g. a cell in the middle of the
    land doesn't pointlessly run any physics when it can't flow laterally.
    """
    # relative coordinates of all the adjacent cells
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
    for i in range(len(valid_cells)):
        for j in range(len(valid_cells[0])):
            # only process currently valid cells
            if valid_cells[i, j]:
                # Initialise all cells to 1 (i.e. assume all are valid)
                neighbours = np.ones((3, 3))
                # Check all cells and adjust values to be not 1 if they
                # match some criterion that marks them as invalid
                for adj in adjustments:
                    # check if we are at the edge of the grid
                    try:
                        # if the neighbour is beyond the edge, set to -999
                        if i + adj[0] < 0 or j + adj[1] < 0:
                            neighbours[adj[0] + 1, adj[1] + 1] = -999
                        # if the neighbour is invalid, set to 0
                        elif not valid_cells[i + adj[0], j + adj[1]]:
                            neighbours[adj[0] + 1, adj[1] + 1] = 0
                    # if the cell is out of bounds, then set to -999 as off
                    # edge of the grid
                    except IndexError:
                        neighbours[adj[0] + 1, adj[1] + 1] = -999
                # centre cell gets marked with a 2 so we don't do anything
                # with it by accident
                neighbours[1, 1] = 2
                # if there are no valid neighbours, mark this cell as invalid
                # and move on
                if not np.any(neighbours == 1) and not np.any(neighbours == -999):
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
        Depth scale for the density profile. [m]
    Returns
    -------
    rho : float
        Density profile of the firn column. [kg m^-3]
    """
    rho = 917 - (917 - rho_sfc) * np.exp(-(1.9 / z_t) * z)
    return rho


def create_model_grid(model_setup, firn_depth, rho, firn_temperature, **overrides):
    """
    Build the initial model grid.

    ``firn_depth``, ``rho`` and ``firn_temperature`` are the required physics
    inputs. Any other grid field may be set by keyword using its catalogue name
    (e.g. ``valid_cell=mask``, ``lat=lats``), or from a runscript via the
    ``initial_conditions`` setting (e.g. ``initial_conditions={'lake_depth':
    0.5}``). See ``monarchs.variables`` for the full list of grid fields.
    """
    # setup the actual grid points
    y, x = np.meshgrid(
        np.arange(0, model_setup.row_amount, 1),
        np.arange(0, model_setup.col_amount, 1),
        indexing="ij",
    )
    inputs = {
        "column": x,
        "row": y,
        "firn_depth": firn_depth,
        "rho": rho,
        "firn_temperature": firn_temperature,
    }
    # fields worked out during setup - valid_cell, size_dx/dy, and lat/lon
    # when there is a DEM
    inputs.update(overrides)
    if model_setup.initial_conditions:
        inputs.update(model_setup.initial_conditions)
    # overrides for settings that need to be passed into physics kernels
    # set here so that any default values get overwritten
    substep = getattr(model_setup, "turbulent_mixing_substep", None)
    if substep is not None:
        inputs["turbulent_mixing_substep"] = substep
    # make_grid reads the variable catalogue and populates the model grid
    return make_grid(
        model_setup.row_amount,
        model_setup.col_amount,
        model_setup.vertical_points_firn,
        model_setup.vertical_points_lake,
        model_setup.vertical_points_lid,
        inputs=inputs,
    )
