"""
Defaults that are computed from other settings rather than fixed values.
"""

from monarchs.config.definitions import UNSET


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
    Infer the met data source from the inputs. That one is present is enforced
    by a rule (see rules.RULES), so this only has to pick which.
    """
    if hasattr(model_setup, "met_input_filepath"):
        return "ERA5"
    return "user_defined"
