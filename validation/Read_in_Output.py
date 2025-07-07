import netCDF4 as nc

filepath = "sam_output.nc"


output = nc.Dataset(filepath)

print(output.dimensions.keys())

print(output.variables.keys())

print(output.dimensions["x"])


#
# P=dataset['tp'][:,:,:]
#
