import netCDF4
import numpy as np

ERA5_data = netCDF4.Dataset("../data/Test_ERA5.nc")
print(ERA5_data.variables.keys())

# Longitude -70 to -66.5
# Latitude -73 to -72
# On GVISS?
# Output is: dict_keys(['longitude', 'latitude', 'time', 'u10', 'v10', 'd2m', 't2m', 'cdir', 'asn', 'rsn', 'sf', 'sp', 'strd'])

# From this we need:

var_dict = {}
var_dict["long"] = ERA5_data.variables["longitude"][:]
var_dict["lat"] = ERA5_data.variables["latitude"][:]
var_dict["time"] = ERA5_data.variables["time"][:]
var_dict["wind"] = np.sqrt(
    ERA5_data.variables["u10"][:] ** 2 + ERA5_data.variables["v10"][:] ** 2
)
var_dict["temperature"] = ERA5_data.variables["t2m"][:]  # [K]
var_dict["dew_point_temperature"] = ERA5_data.variables["d2m"][:]  # [K]
var_dict["snowfall"] = ERA5_data.variables["sf"][:]  # [m water equiv.]
var_dict["LW_surf"] = ERA5_data.variables["strd"][:] / 3600  # [J m^-2] -> [W m^-2]
var_dict["SW_surf"] = (
    ERA5_data.variables["ssrd"][:] / 3600
)  # [J m^-2] -> [W m^-2] #'Surface solar radiation downward clear-sky'?


# New
var_dict["snow_albedo"] = ERA5_data.variables["asn"][
    :
]  # [0-1], can use up until melt occurs then we need to calc our own albedo
var_dict["snow_dens"] = ERA5_data.variables["rsn"][:]  # [kgm^-3]
var_dict["pressure"] = (
    ERA5_data.variables["sp"][:] / 100
)  # [Pa] -> [hPa] Surface pressure
