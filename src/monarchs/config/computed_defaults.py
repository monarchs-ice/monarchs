"""
Defaults that are computed from other settings rather than fixed values.
"""

from monarchs.config.definitions import UNSET
from monarchs.config.rules import MET_SOURCE_INPUTS


def output_grid_size(model_setup):
    # single-column firn variable (e.g. `firn_temperature` output size
    # defaults to the number of points in the firn column - smaller
    # numbers = interpolation
    return model_setup.vertical_points_firn


def lateral_timestep(model_setup):
    # t_steps_per_day is implicitly in hours,
    # need to convert it to seconds for the lateral timestep
    return model_setup.t_steps_per_day * 3600


def lat_grid_size(model_setup):
    """'dem' when a DEM is provided; otherwise left unset."""
    if hasattr(model_setup, "DEM_path"):
        return "dem"
    return UNSET


def met_data_source(model_setup):
    """
    Choose the source of the meteorological forcing data based on what the
    user has specified. If this returns UNSET (i.e. no source is specified),
    a later Rule will raise an error since the model requires forcing data!
    """
    for name, setting in MET_SOURCE_INPUTS.items():
        if hasattr(model_setup, setting):
            return name
    return UNSET
