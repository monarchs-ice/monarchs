# test_regrid_and_merge.py
import numpy as np
import pytest

from monarchs.physics import regrid_column as rg
from monarchs.core.utils import calc_mass_sum



def make_cell(nz=4, firn_depth=4.0, rho_i=917.0, rho_w=1000.0,
              lake=True, lake_depth=0.0):
    """Create a minimal 'cell' dict-like for tests."""
    cell = {
        "firn_depth": float(firn_depth),
        "vert_grid": int(nz),
        "rho_ice": float(rho_i),
        "rho_water": float(rho_w),
        "lake_depth": lake_depth,
        "Sfrac": np.zeros(nz, dtype=float),
        "Lfrac": np.zeros(nz, dtype=float),
        "firn_temperature": np.zeros(nz, dtype=float),
        "vertical_profile": np.linspace(0.0, firn_depth, nz),
        "lid_depth": 0.0,
        "v_lid_depth": 0.0,
        "v_lid": False,
        "lid": False,
        "lake": lake,
        "saturation": np.zeros(nz, dtype=int),
        "meltflag": np.zeros(nz, dtype=int),
    }
    return cell


def test_lfrac_equals_one_into_lake():
    """
    Check the implementation that if the top layer(s) are fully liquid,
    they get merged into the lake.
    """
    cell = make_cell(nz=4, firn_depth=4.0, lake=True, lake_depth=0.0)
    # Two full-liquid layers at the top
    cell["Sfrac"][:] = np.array([0.0, 0.0, 0.10, 0.10])
    cell["Lfrac"][:] = np.array([1.0, 1.0, 0.0, 0.0])
    cell["firn_temperature"][:] = np.array([273.15, 272.0, 266.0, 261.0])
    height_change = 0  # no melt, just testing the merge
    height_change = rg.merge_cells_into_lake(cell, height_change)
    assert height_change == 2



def test_lfrac_equals_one_into_lake_after_melt():
    """
    As above, but melt a single layer first.
    """
    nz = 4
    dz = 1.0
    cell = make_cell(nz=nz, firn_depth=nz * dz)

    cell["Sfrac"][:] = np.array([1.0, 0.0, 0.0, 1])
    cell["Lfrac"][:] = np.array([0, 1.0, 1.0, 0.0])
    cell["firn_temperature"][:] = np.array([268.0, 273.15, 266.0, 261.0])

    mass_before = calc_mass_sum(cell)
    # melt one layer and do the regrid
    # melt more than one layer for this test - the code should recognise
    # that this is melting into the liquid part and not add extra
    rg.regrid_after_melt(cell, height_change=dz + 0.01, lake=True)

    # check we have neatly merged everything.
    # we melt one layer of solid (1m), and merge two layers of liquid (2m)
    # but as the solid part melts the density changes, so the new lake depth
    # needs to be rho_ice/rho_water * 1m + 2m = 0.917 + 2 = 2.917m
    # In a real case, some of the meltwater might percolate into the top
    # cell, but here we have Sfrac = 1 everywhere so it can't.
    lake_kgo = 2 + (917.0 / 1000.0)

    assert np.isclose(cell["firn_depth"], 1)
    assert np.isclose(cell["lake_depth"], lake_kgo)

    mass_after = calc_mass_sum(cell)
    assert np.isclose(mass_after, mass_before, atol=1e-8)


def test_regrid_saturation():
    """
    When lake=False, all deposit goes to top Lfrac (may exceed saturation).
    """
    nz = 4
    dz = 1.0
    cell = make_cell(nz=nz, firn_depth=nz * dz)

    # Simple firn with some liquid in the first cell
    cell["Sfrac"][:] = np.array([0.50, 0.10, 0.10, 0.10])
    cell["Lfrac"][:] = np.array([0.50, 0.05, 0.05, 0.05])

    mass_before = calc_mass_sum(cell)

    # melt half a layer
    rg.regrid_after_melt(cell, height_change=0.5 * dz, lake=False)

    # we should now have a supersaturated top cell
    assert cell["Lfrac"][0] > 0.50

    # check for mass conservation
    mass_after = calc_mass_sum(cell)
    assert np.isclose(mass_after, mass_before, atol=1e-8)