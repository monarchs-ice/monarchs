import numpy as np

def setup_cell():
    # populate with all the cell fields needed for lake_functions.lake_development
    cell = {}
    cell["lake"] = True
    cell["lid"] = False
    cell["v_lid"] = False
    cell["lake_depth"] = 0.1  # Initial lake depth in meters
    cell['vert_grid_lake'] = 10
    cell['vert_grid_lid'] = 10
    cell["lake_temperature"] = 273.15 * np.ones(cell['vert_grid_lake'])  # Initial lake temperature profile
    cell["lid_temperature"] = 273.15 * np.ones(cell['vert_grid_lid'])  # Initial lid temperature profile
    cell["rho_ice"] = 917
    cell["rho_water"] = 1000
    cell['firn_depth'] = 30
    cell["k_air"] = 0.024  # Thermal conductivity of air [W m^-1 K^-1]
    cell["k_water"] = 0.5818  # Thermal conductivity of water [W m^-1 K^-1]
    cell["L_ice"] = 334000  # Latent heat of fusion for ice [J kg^-1]
    cell["cp_air"] = 1004  # Specific heat capacity of air [J kg^-1 K^-1]
    cell["cp_water"] = 4217  # Specific heat capacity of water [J kg^-1 K^-1]
    cell["exposed_water"] = True
    cell["melt"] = True
    cell["lid_depth"] = 0.0
    cell["v_lid_depth"] = 0.0

    return cell