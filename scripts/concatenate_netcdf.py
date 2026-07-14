"""
Combine many Climate Data Store variables into one. This is annoying in our case as we have many yearly files
and two files per year to store the different variables. There is probably a much better way to do this using pure
xarray but the approach I've used here works well enough.

As a side note, running this script requires `dask` as an additional dependency (it is used by xarray
when combining the files). You can install it with `pip install dask`.
"""

from netCDF4 import Dataset
import glob
import os
import xarray as xr


# Get a list of all the netCDF files in the output directory
input_dir = r"C:\Users\jdels\Downloads\ERA5_netcdf_files"

folders = glob.glob(input_dir + "/*")
# remove the "zip" folder
folders.remove(input_dir + "\zip")
# also remove the "concatenated" folder which we will first create
output_dir = input_dir + "\concatenated"
if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
    os.makedirs(output_dir)
folders.remove(output_dir)

# We have two types of file, an "accum" and an "instant" file. These have the same time, but
# different variables. First, load in all these and concatenate them into single files for each
# year of data. These files all have the same filename which complicates things, but we can
# denote them by the folder they are in.
for folder in folders:
    files = glob.glob(folder + "/*")
    with Dataset(files[0]) as f1, Dataset(files[1]) as f2:
        # Get the dimensions
        time_dim = f1.dimensions["valid_time"]
        lat_dim = f1.dimensions["latitude"]
        lon_dim = f1.dimensions["longitude"]
        # Create the new file. First we get the year from the time after epoch (1/1/1970)
        year = int(
            f1["valid_time"][len(f1["valid_time"]) / 2] // (60 * 60 * 24 * 365.25)
            + 1970
        )
        print("Year = ", year)
        new_file = Dataset(output_dir + "/era5_" + str(year) + ".nc", "w", clobber=True)
        # Create the dimensions
        new_file.createDimension("valid_time", None)
        new_file.createDimension("latitude", len(lat_dim))
        new_file.createDimension("longitude", len(lon_dim))
        # Get the variables from each file
        all_variables = {}
        # Append all of the keys together to get a list of all the variables
        all_variables = [(name, variable) for name, variable in f1.variables.items()]
        all_variables += [(name, variable) for name, variable in f2.variables.items()]
        # Get the variables from f1 and f2 and write them (with their metadata) into the
        # new file
        for name, variable in all_variables:
            if name in new_file.variables:
                continue
            new_var = new_file.createVariable(
                name, variable.datatype, variable.dimensions
            )
            new_var.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
        # Write the actual data
        for name, variable in all_variables:
            new_file[name][:] = variable[:]
        new_file.close()


# Now take all of these yearly files we have created and combine them to create one megafile
ds = xr.open_mfdataset(output_dir + "/*.nc", combine="by_coords")
ds.to_netcdf(output_dir + "/era5_megafile.nc")
