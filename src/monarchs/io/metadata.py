"""
Run metadata for MONARCHS output files.

This is written as netCDF global attributes on each output and
checkpoint file. You can check it with ``ncdump -h``. This
includes the code/dependency versions and git state, plus a ``model_setup``
attribute holding the resolved configuration as a JSON string. The intent is
to allow a run to be reconstructed from its output - e.g. if the model setup
script is changed after running. A future tool will read this back and re-run
the model with the relevant parameters.
TODO - actually write this tool

Note: metadata is only written when an output/checkpoint file is produced,
i.e. with ``save_output`` and/or ``dump_data`` enabled.

Additionally, we handle variable metadata here for the output netCDFs.
This is a WIP - not all fields are covered at time of writing, but the idea
is to aid reproducibility and introduce a kind of convention for MONARCHS
output data, somewhat inspired by the CF conventions.
"""

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np


# only run this once, then cache rather than regenerate each time
_cache = {}


def global_attrs(model_setup=None):
    """
    Build a dict of metadata attributes for a netCDF file, suitable for
    ``Dataset.setncatts``. All values are strings.
    """
    attrs = {"created_utc": datetime.now(timezone.utc).isoformat()}
    attrs.update(_run_metadata(model_setup))
    return attrs


def _run_metadata(model_setup):
    """Run the metadata attach process, loading from cache"""
    cached = _cache.get(id(model_setup))
    if cached is not None:
        return cached

    attrs = {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
    }
    # if the package isnt installed, attach a placeholder rather than nothing
    for package in ("monarchs-ice", "numpy", "numba", "scipy", "netCDF4"):
        try:
            attrs[f"{package}_version"] = version(package)
        except PackageNotFoundError:
            attrs[f"{package}_version"] = "not installed"

    # git description of the model version. From a pypi install this is the
    # exact release, from a checkout it reflects the branch/commit state.
    try:
        attrs["git_describe"] = subprocess.check_output(
            ["git", "describe", "--tags", "--dirty", "--always"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    # if no git, then don't fail silently, just say git isnt a thing
    except (OSError, subprocess.CalledProcessError):
        attrs["git_describe"] = "unavailable"

    if model_setup is not None:
        attrs["model_setup"] = json.dumps(_summarise_config(model_setup))

    _cache[id(model_setup)] = attrs
    return attrs


def _summarise_config(model_setup):
    """
    Generate MONARCHS model setup config as a JSON.

    Array values are given as strigns here rather than being dumped -
    the idea is that we *can* reconstruct this data, not that we necessarily
    need to save/cache it
    """
    config = {}
    for key, value in sorted(vars(model_setup).items()):
        if isinstance(value, np.ndarray):
            # digest - we only need a short hash here, just to identify the array in a reproducible way
            digest = hashlib.sha256(np.ascontiguousarray(value)).hexdigest()[:12]
            config[key] = f"<array shape={value.shape} sha256:{digest}>"
        else:
            config[key] = repr(value)
    return config


# units / long_name for model grid fields, keyed by field name.
# Fractions and flags are dimensionless ("1" per CF convention).
VARIABLE_METADATA = {
    "firn_depth": ("m", "Firn column total depth"),
    "firn_temperature": ("K", "Firn column temperature"),
    "vertical_profile": ("m", "Depth of each firn layer below the surface"),
    "rho": ("kg m-3", "Firn density"),
    "rho_lid": ("kg m-3", "Frozen lid density"),
    "Sfrac": ("1", "Solid (ice) volume fraction"),
    "Lfrac": ("1", "Liquid (water) volume fraction"),
    "albedo": ("1", "Surface albedo, determined by current model state"),
    "meltflag": ("1", "Meltwater present at layer (flag)"),
    "saturation": ("1", "Layer saturated (flag)"),
    "water": ("m", "Liquid water depth per layer, derived from LFrac for firn water"),
    "water_level": ("m", "Water height for lateral flow"),
    "lake_depth": ("m", "Melt lake depth"),
    "lake_temperature": ("K", "Lake temperature profile"),
    "lake": ("1", "Lake present (flag)"),
    "lid_depth": ("m", "Frozen lid depth"),
    "lid_temperature": ("K", "Frozen lid temperature profile"),
    "lid": ("1", "Frozen lid present (flag)"),
    "v_lid": ("1", "Virtual lid present (flag)"),
    "v_lid_depth": ("m", "Virtual lid depth"),
    "virtual_lid_temperature": ("K", "Virtual lid temperature"),
    "ice_lens": ("1", "Ice lens present (flag)"),
    "ice_lens_depth": ("1", "Layer index of the highest ice lens"),
    "melt": ("1", "Surface melt occurred this step (flag)"),
    "melt_hours": ("h", "Cumulative hours of surface melt"),
    "exposed_water": ("1", "Exposed water at surface (flag)"),
    "total_melt": ("m", "Cumulative melt depth"),
    "lid_sfc_melt": ("m", "Tracked lid surface melt"),
    "lid_snow_depth": ("m", "Snow depth on the frozen lid, tracked for albedo changes"),
    "snow_added": ("m", "Snow depth added"),
    "firn_boundary_change": ("m", "Firn boundary change this day"),
    "lake_boundary_change": ("m", "Lake boundary change this day"),
    "lid_boundary_change": ("m", "Lid boundary change this day"),
    "water_direction": ("1", "Lateral outflow direction (0=NW..7=W)"),
    "lat": ("degrees_north", "Latitude of grid cell"),
    "lon": ("degrees_east", "Longitude of grid cell"),
    "size_dx": ("m", "Grid cell size, east-west"),
    "size_dy": ("m", "Grid cell size, north-south"),
    "valid_cell": (
        "1",
        "Cell is included in the set of cells running model physics (flag)",
    ),
}


def apply_variable_metadata(var_write, key):
    """Attach units/long_name to a netCDF variable, where known."""
    if key in VARIABLE_METADATA:
        units, long_name = VARIABLE_METADATA[key]
        var_write.units = units
        var_write.long_name = long_name
