import numpy as np
import pytest
from monarchs.physics import surface_fluxes

def test_basic_zero_wind():
    """With zero wind, fluxes should be zero."""
    Flat, Fsens = surface_fluxes.bulk_fluxes(
        wind=0.0,
        air_temp=280.0,
        T_sfc=279.0,
        p_air=1013.25,
        dew_point_temperature=275.0,
        lake=False,
        lid=False
    )
    assert np.isclose(Flat, 0.0), "Latent heat should be zero with no wind"
    assert np.isclose(Fsens, 0.0), "Sensible heat should be zero with no wind"


