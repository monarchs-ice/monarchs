import numpy as np
from monarchs.met_data import met_data_grid
from monarchs.physics import surface_fluxes


def test_basic_zero_wind():
    """With zero wind, fluxes should be zero."""
    met = np.zeros(1, dtype=met_data_grid.get_spec())[0]
    met["temperature"] = 280.0
    met["surf_pressure"] = 1013.25
    met["dew_point_temperature"] = 275.0
    met["wind"] = 0.0
    cell = {"lake": False, "lid": False}
    Flat, Fsens = surface_fluxes.bulk_fluxes(cell, met, 279.0)
    assert np.isclose(Flat, 0.0), "Latent heat should be zero with no wind"
    assert np.isclose(Fsens, 0.0), "Sensible heat should be zero with no wind"
