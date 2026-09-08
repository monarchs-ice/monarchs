"""
Meteorological forcing variable definitions.

This currently handles ERA5 or user-defined inputs. We define several needed
inputs - the name as defined in the ERA5 catalogue (or a fallback if there are
two related variables one could use), functions to derivve variables from other
variables (e.g. wind from u and v components), and conversion factors (e.g.
pressure from Pa to hPa).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True, kw_only=True)
class MetVariable:
    """
    Meteorological forcing data variable definition.
    This currently just handles ERA5 or user-defined inputs. An extension
    to use e.g. RACMO would need to change the definition here to add the
    relevant fields needed.
    """

    name: str
    units: str = "1"
    long_name: str = ""
    description: str = ""
    dtype: type = np.float64
    era5_name: str = ""
    era5_fallback: str = ""
    # function to compute the field from other ERA5 vars
    # e.g. wind speed = sqrt(u^2 + v^2)
    derived_from: Callable = None
    # function to apply a unit conversion after reading it in
    # (e.g. snowfall from MWE -> m)
    convert: Callable = None


MET_CATALOGUE = [
    MetVariable(name="snowfall", units="m", long_name="Snowfall", era5_name="sf"),
    MetVariable(
        name="snow_dens", units="kg m-3", long_name="Snow density", era5_name="rsn"
    ),
    MetVariable(
        name="temperature",
        units="K",
        long_name="Surface air temperature",
        era5_name="t2m",
    ),
    MetVariable(
        name="wind",
        units="m s-1",
        long_name="Wind speed",
        era5_name="u10, v10",
        derived_from=lambda read: np.sqrt(read("u10") ** 2 + read("v10") ** 2),
    ),
    MetVariable(
        name="surf_pressure",
        units="hPa",
        long_name="Surface air pressure",
        era5_name="sp",
        era5_fallback="msl",
        convert=lambda v, seconds_per_step: v / 100,  # Pa -> hPa
    ),
    MetVariable(
        name="dew_point_temperature",
        units="K",
        long_name="Dew-point temperature",
        era5_name="d2m",
    ),
    MetVariable(
        name="LW_down",
        units="W m-2",
        long_name="Downwelling longwave radiation",
        era5_name="strd",
        era5_fallback="strdc",
        convert=lambda v, seconds_per_step: v / seconds_per_step,  # J m^-2 -> W m^-2
    ),
    MetVariable(
        name="SW_down",
        units="W m-2",
        long_name="Downwelling shortwave radiation",
        era5_name="ssrd",
        era5_fallback="ssrdc",
        convert=lambda v, seconds_per_step: v / seconds_per_step,  # J m^-2 -> W m^-2
    ),
    MetVariable(
        name="lat", units="degrees_north", long_name="Latitude", era5_name="latitude"
    ),
    MetVariable(
        name="lon", units="degrees_east", long_name="Longitude", era5_name="longitude"
    ),
]


def met_dtype():
    """Structured-array dtype for the met data grid."""
    return np.dtype([(var.name, var.dtype) for var in MET_CATALOGUE])
