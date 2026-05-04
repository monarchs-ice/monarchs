Formatting input meteorological data
====================================
MONARCHS supported formats
---------------------------
By default, `MONARCHS only supports data input in` `ERA5 format <https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation>`_.
Support for other formats is not planned for the immediate future. It is therefore recommended that users intending to use other
datasets convert their data to this format, or make the necessary changes to MONARCHS to support such data.

Required variables
--------------------

+-------------------------------+-------------------+---------------------------------------------------------------+
| Long name                     | Short name        | Description                                                   |
+===============================+===================+===============================================================+
| 10 metre U wind component     | 10m_u_component_of_wind | Zonal (west-east) wind at 10m above surface (m/s)           |
+-------------------------------+-------------------+---------------------------------------------------------------+
| 10 metre V wind component     | 10m_v_component_of_wind | Meridional (south-north) wind at 10m above surface (m/s)    |
+-------------------------------+-------------------+---------------------------------------------------------------+
| 2 metre dewpoint temperature  | 2m_dewpoint_temperature | Dewpoint temperature at 2m above surface (K)                |
+-------------------------------+-------------------+---------------------------------------------------------------+
| 2 metre temperature           | 2m_temperature    | Air temperature at 2m above surface (K)                       |
+-------------------------------+-------------------+---------------------------------------------------------------+
| Surface pressure              | surface_pressure  | Atmospheric pressure at surface (Pa)                          |
+-------------------------------+-------------------+---------------------------------------------------------------+
| Surface solar radiation down  | surface_solar_radiation_downwards | Downwelling shortwave radiation at surface (W/m^2)      |
+-------------------------------+-------------------+---------------------------------------------------------------+
| Surface thermal radiation down| surface_thermal_radiation_downwards | Downwelling longwave radiation at surface (W/m^2)      |
+-------------------------------+-------------------+---------------------------------------------------------------+
| Snow albedo                   | snow_albedo       | Surface snow albedo (fraction, 0-1)                            |
+-------------------------------+-------------------+---------------------------------------------------------------+
| Snow density                  | snow_density      | Density of surface snow layer (kg/m^3)                         |
+-------------------------------+-------------------+---------------------------------------------------------------+
| Snowfall                      | snowfall          | Snowfall water equivalent (m)                                  |
+-------------------------------+-------------------+---------------------------------------------------------------+

Scripts
---------------------
In the `monarchs` root folder, the `data` directory has a sample ERA5 dataset, and an example script for
downloading ERA5 data for the George VI ice shelf over 10 years using `cdsapi`.