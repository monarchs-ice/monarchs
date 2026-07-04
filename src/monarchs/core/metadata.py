"""
Run metadata for MONARCHS output files.

Two jobs:

- ``global_attrs(model_setup)``: run-provenance attributes (code version,
  git state, dependency versions, the fully-resolved configuration) written
  as netCDF *global* attributes on every output and checkpoint file, so any
  result file records exactly what produced it (``ncdump -h file.nc``).

- ``VARIABLE_METADATA`` / ``apply_variable_metadata``: units and long_name
  for model grid fields, applied to each netCDF variable written, so output
  files are self-describing.
"""

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np


def global_attrs(model_setup=None):
    """
    Build a dict of provenance attributes for a netCDF file, suitable for
    ``Dataset.setncatts``. All values are strings.
    """
    attrs = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
    }
    for package in ("monarchs-ice", "numpy", "numba", "scipy", "netCDF4"):
        try:
            attrs[f"{package}_version"] = version(package)
        except PackageNotFoundError:
            pass

    # Exact git state when running from a checkout (editable install);
    # silently absent for wheel installs, where the version attribute
    # already pins the release.
    try:
        attrs["git_describe"] = subprocess.check_output(
            ["git", "describe", "--tags", "--dirty", "--always"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass

    if model_setup is not None:
        attrs["model_setup"] = json.dumps(_summarise_config(model_setup))
    return attrs


def _summarise_config(model_setup):
    """
    The fully-resolved configuration (post-defaults) as a JSON-safe dict.
    Arrays are summarised by shape and content hash rather than dumped.
    """
    config = {}
    for key, value in sorted(vars(model_setup).items()):
        if isinstance(value, np.ndarray):
            digest = hashlib.sha256(np.ascontiguousarray(value)).hexdigest()[:12]
            config[key] = f"<array shape={value.shape} sha256:{digest}>"
        else:
            config[key] = repr(value)
    return config


# units / long_name for model grid fields, keyed by field name.
# Fractions and flags are dimensionless ("1" per CF convention).
VARIABLE_METADATA = {
    "firn_depth": ("m", "Firn column depth"),
    "firn_temperature": ("K", "Firn column temperature"),
    "vertical_profile": ("m", "Depth of each firn layer below the surface"),
    "rho": ("kg m-3", "Firn bulk density"),
    "rho_lid": ("kg m-3", "Frozen lid bulk density"),
    "Sfrac": ("1", "Solid (ice) volume fraction"),
    "Lfrac": ("1", "Liquid (water) volume fraction"),
    "albedo": ("1", "Surface albedo"),
    "meltflag": ("1", "Meltwater present at layer (flag)"),
    "saturation": ("1", "Layer saturated (flag)"),
    "water": ("m", "Liquid water depth per layer (lateral-flow working field)"),
    "water_level": ("m", "Hydraulic head used for lateral flow"),
    "lake_depth": ("m", "Surface lake depth"),
    "lake_temperature": ("K", "Lake temperature profile"),
    "lake": ("1", "Lake present (flag)"),
    "lid_depth": ("m", "Frozen lid depth"),
    "lid_temperature": ("K", "Frozen lid temperature profile"),
    "lid": ("1", "Frozen lid present (flag)"),
    "v_lid": ("1", "Virtual (embryonic) lid present (flag)"),
    "v_lid_depth": ("m", "Virtual lid depth"),
    "virtual_lid_temperature": ("K", "Virtual lid temperature"),
    "ice_lens": ("1", "Ice lens present (flag)"),
    "ice_lens_depth": ("1", "Layer index of the highest ice lens"),
    "melt": ("1", "Surface melt occurred this step (flag)"),
    "melt_hours": ("h", "Cumulative hours of surface melt"),
    "exposed_water": ("1", "Exposed water at surface (flag)"),
    "total_melt": ("m", "Cumulative melt (ice thickness equivalent)"),
    "lid_sfc_melt": ("m", "Tracked (unremoved) lid surface melt"),
    "lid_snow_depth": ("m", "Snow depth on the frozen lid"),
    "snow_added": ("m", "Snow depth added"),
    "firn_boundary_change": ("m", "Firn boundary change this day"),
    "lake_boundary_change": ("m", "Lake boundary change this day"),
    "lid_boundary_change": ("m", "Lid boundary change this day"),
    "water_direction": ("1", "Lateral outflow direction flags (0=NW..7=W)"),
    "lat": ("degrees_north", "Latitude of grid cell"),
    "lon": ("degrees_east", "Longitude of grid cell"),
    "size_dx": ("m", "Grid cell size, east-west"),
    "size_dy": ("m", "Grid cell size, north-south"),
    "valid_cell": ("1", "Cell participates in the model physics (flag)"),
}


def apply_variable_metadata(var_write, key):
    """Attach units/long_name to a netCDF variable, where known."""
    if key in VARIABLE_METADATA:
        units, long_name = VARIABLE_METADATA[key]
        var_write.units = units
        var_write.long_name = long_name
