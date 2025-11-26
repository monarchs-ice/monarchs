from monarchs.physics import lake
import numpy as np
from setup_test_cell import setup_cell
from numpy import testing as npt

def extra_cell_setup(cell):
    cell["firn_temperature"] = 273.15 * np.ones(
        cell["vert_grid"]
    )  # Initial firn temperature profile
    cell["Sfrac"] = np.ones(cell["vert_grid"]) * 0.75  # Initial solid fraction
    cell["Lfrac"] = (
            np.ones(cell["vert_grid"]) * 0.15
    )  # Initial liquid fraction
    cell['Lfrac'][0] = 0.25
    cell["rho"] = cell["Sfrac"] * 917 + cell["Lfrac"] * 1000  # Density profile
    cell["saturation"] = np.ones(cell["vert_grid"])
    cell["meltflag"] = np.zeros(cell["vert_grid"])
    return cell
def test_lake_development():

    LW_surf = 1200
    SW_surf = 1200
    air_temp = 267
    p_air = 1000
    dew_point_temperature = 265
    wind = 5
    dt = 3600
    cell = setup_cell()
    cell['vert_grid'] = 500
    cell = extra_cell_setup(cell)
    lake.lake_development(
        cell, dt, LW_surf, SW_surf, air_temp, p_air, dew_point_temperature, wind
    )
    print(cell["lake_depth"])
    lake_depth_lowres = cell["lake_depth"]
    assert lake_depth_lowres > 0.1  # Check that lake depth has increased

    cell = setup_cell()
    cell['vert_grid'] = 2000
    cell = extra_cell_setup(cell)
    lake.lake_development(
        cell, dt, LW_surf, SW_surf, air_temp, p_air, dew_point_temperature, wind
    )
    print(cell["lake_depth"])
    lake_depth_highres = cell["lake_depth"]
    # Check that lake depth hasn't increased significantly (less than 0.01%
    # between the runs just because the resolution is higher
    npt.assert_almost_equal(lake_depth_highres, lake_depth_lowres,
                            decimal=4)
