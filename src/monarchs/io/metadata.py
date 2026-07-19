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

from monarchs.variables import variable_metadata


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


# netCDF attributes for model grid fields, keyed by field name. Sourced from
# the single variable catalogue in monarchs.variables (output-eligible
# variables only).
VARIABLE_METADATA = variable_metadata()


def apply_variable_metadata(var_write, key):
    """Attach the catalogue's metadata attributes to a netCDF variable."""
    for attr, value in VARIABLE_METADATA.get(key, {}).items():
        setattr(var_write, attr, value)
