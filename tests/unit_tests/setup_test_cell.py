import numpy as np


def setup_cell():
    # populate with all the cell fields needed for lake_functions.lake_development
    cell = {}
    cell["lake"] = True
    cell["lid"] = False
    cell["v_lid"] = False
    cell["lake_depth"] = 0.1  # Initial lake depth in meters
    cell["vert_grid_lake"] = 10
    cell["vert_grid_lid"] = 10
    cell["lake_temperature"] = 276.15 * np.ones(
        cell["vert_grid_lake"]
    )  # Initial lake temperature profile
    cell["lid_temperature"] = 273.15 * np.ones(
        cell["vert_grid_lid"]
    )  # Initial lid temperature profile
    cell["firn_depth"] = 30
    cell["exposed_water"] = True
    cell["melt"] = True
    cell["lid_depth"] = 0.0
    cell["v_lid_depth"] = 0.0
    cell["lake_boundary_change"] = 0.0
    cell["firn_boundary_change"] = 0.0
    cell["lid_boundary_change"] = 0.0

    return cell
