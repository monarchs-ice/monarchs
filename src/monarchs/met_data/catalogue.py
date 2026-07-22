"""
The MONARCHS met-forcing catalogue.

The fields of the met data grid (one record per timestep per cell), defined as
``MetVariable`` rows - a grid ``Variable`` (see
``monarchs.variables.definitions``) plus how the field is obtained from an ERA5
input file. It lives in ``monarchs.met_data`` because it is a met concern; the
importer (``import_ERA5``) and the grid builder (``met_data_grid``) both read
from it, and ``variables.docs`` renders it into the reference.

How a field is obtained from ERA5:

* ``era5_name``     - the ERA5 short name to read (e.g. ``t2m``).
* ``era5_fallback`` - an alternative read if the primary is absent (e.g.
                      clear-sky ``ssrdc`` when ``ssrd`` is missing).
* ``derived_from``  - ``f(read) -> array`` for fields that are computed rather
                      than read straight (``wind`` = |u10, v10|). ``read(name)``
                      returns the time-sliced ERA5 variable.
* ``convert``       - ``f(value, seconds_per_step) -> value``, a unit conversion
                      applied after reading (Pa -> hPa; J m^-2 -> W m^-2, which
                      uses ``seconds_per_step`` to de-accumulate).

A couple of fields (snow density's fallback value, the mwe -> depth snowfall
conversion) depend on *other* fields and stay explicit in ``import_ERA5``.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True, kw_only=True)
class MetVariable:
    """
    A met-forcing field: its name/units/metadata plus how to obtain it from an
    ERA5 input file. Standalone (not a grid ``Variable``) so ``met_data`` does
    not depend on ``monarchs.variables``; met fields are always per-cell floats.

    Extend ``derived_from``/``convert`` to add fields that are computed or need a unit
    conversion; extend ``era5_name``/``era5_fallback`` for straight reads. This
    could grow to support other sources (AWS, RACMO) via more read strategies.
    """

    name: str
    units: str = "1"
    long_name: str = ""
    description: str = ""
    dtype: type = np.float64
    era5_name: str = ""
    era5_fallback: str = ""
    # f(read) -> array: compute the field from other ERA5 vars
    derived_from: Callable = None
    # f(value, seconds_per_step) -> value: unit conversion applied after reading
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


def met_dtype(catalogue=MET_CATALOGUE):
    """Structured-array dtype for the met data grid."""
    return np.dtype([(var.name, var.dtype) for var in catalogue])
