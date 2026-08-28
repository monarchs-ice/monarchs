"""
Sets up the NumPy structured array used to hold meteorological data for the
model run. The fields are defined in the met catalogue
(``monarchs.met_data.catalogue``).
"""

import numpy as np

from monarchs.met_data.catalogue import met_dtype


def initialise_met_data(inputs, num_rows, num_cols, t_steps_per_day):
    """
    From the meteorological data variable catalogue, create a structured array
    containing our met data, with each element associated with an element of the
    model grid.

    This is done at each iteration, so we only store one day's worth of
    met data in memory at any given time.

    Parameters
    ----------
    inputs : dict
        {field name: array} for the met fields, keyed by the names in the
        met catalogue. Each array is dimension(time, num_rows, num_cols),
        except lat/lon which are time-invariant (num_rows, num_cols).
    num_rows, num_cols : int
        Model grid dimensions.
    t_steps_per_day : int
        Number of timesteps in a model day.

    Returns
    -------
    met_data : numpy structured array
        Grid containing meteorological data for the model run.
    """
    met_data = np.zeros((t_steps_per_day, num_rows, num_cols), dtype=met_dtype())
    for key, value in inputs.items():
        met_data[key] = value
    return met_data
