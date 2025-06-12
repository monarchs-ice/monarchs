import netCDF4
import numpy as np

ERA5_data = netCDF4.Dataset("../data/Test_ERA5.nc")
print(ERA5_data.variables.keys())
var_dict = {}
var_dict["long"] = ERA5_data.variables["longitude"][:]
var_dict["lat"] = ERA5_data.variables["latitude"][:]
var_dict["time"] = ERA5_data.variables["time"][:]
var_dict["wind"] = np.sqrt(
    ERA5_data.variables["u10"][:] ** 2 + ERA5_data.variables["v10"][:] ** 2
)
var_dict["temperature"] = ERA5_data.variables["t2m"][:]
var_dict["dew_point_temperature"] = ERA5_data.variables["d2m"][:]
var_dict["snowfall"] = ERA5_data.variables["sf"][:]
var_dict["LW_surf"] = ERA5_data.variables["strd"][:] / 3600
var_dict["SW_surf"] = ERA5_data.variables["ssrd"][:] / 3600
var_dict["snow_albedo"] = ERA5_data.variables["asn"][:]
var_dict["snow_dens"] = ERA5_data.variables["rsn"][:]
var_dict["pressure"] = ERA5_data.variables["sp"][:] / 100
