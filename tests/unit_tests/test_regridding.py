from setup_test_cell import setup_cell
from monarchs.physics import firn_column
import numpy as np


def test_regridding():
    cell = setup_cell()
    # bespoke setup for this test
    cell["vert_grid"] = 400
    cell["firn_temperature"] = 273.15 * np.ones(
        cell["vert_grid"]
    )  # Initial firn temperature profile
    cell["Sfrac"] = np.ones(cell["vert_grid"]) * 0.75  # Initial solid fraction
    cell["Lfrac"] = (
        np.ones(cell["vert_grid"]) * 0.25
    )  # Initial liquid fraction
    cell["rho"] = cell["Sfrac"] * 917 + cell["Lfrac"] * 1000  # Density profile
    boundary_change = 0.001

    firn_functions.regrid_after_melt(cell, boundary_change)
    firn_depth_1 = cell["firn_depth"]

    cell = setup_cell()
    cell["vert_grid"] = 1000
    cell["firn_temperature"] = 273.15 * np.ones(
        cell["vert_grid"]
    )  # Initial firn temperature profile
    cell["Sfrac"] = np.ones(cell["vert_grid"]) * 0.75  # Initial solid fraction
    cell["Lfrac"] = (
        np.ones(cell["vert_grid"]) * 0.25
    )  # Initial liquid fraction
    cell["rho"] = cell["Sfrac"] * 917 + cell["Lfrac"] * 1000  # Density profile
    boundary_change = 0.001

    firn_column.regrid_after_melt(cell, boundary_change)
    firn_depth_2 = cell["firn_depth"]
    print(firn_depth_1, firn_depth_2)
    assert firn_depth_2 == firn_depth_1
