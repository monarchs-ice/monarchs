from monarchs.physics.firn import percolation
import numpy as np


def make_cell(sfrac=0.5, lfrac=0.1, firn_depth=35, vert_grid=500):
    cell = {}
    cell["Sfrac"] = np.array([sfrac])
    cell["Lfrac"] = np.array([lfrac])
    cell["firn_depth"] = firn_depth
    cell["vert_grid"] = vert_grid
    return cell


def test_perc_time_scaling_with_grid():
    # more layers -> thinner layers -> shorter transit time per layer
    v_lev = 0
    cell = make_cell(vert_grid=500)
    test_1 = percolation.perc_time(cell, v_lev)
    cell["vert_grid"] = 1000
    test_2 = percolation.perc_time(cell, v_lev)
    assert np.isclose(test_2, 0.5 * test_1)


def test_perc_time_positive():
    # The transit time must be a positive, finite number of seconds. (A sign
    # error previously made this negative, so the percolation-time budget in
    # `percolate` never depleted and perc_time_toggle did nothing.)
    p_time = percolation.perc_time(make_cell(), 0)
    assert p_time > 0
    assert np.isfinite(p_time)


def test_perc_time_physically_sensible_magnitude():
    # For typical melting firn (Sfrac 0.5, Lfrac 0.1, 7 cm layers), gravity-
    # driven Darcy flow through Shimizu-permeability firn takes seconds to
    # minutes per layer - not microseconds, not days.
    p_time = percolation.perc_time(make_cell(), 0)
    assert 1e-2 < p_time < 3600


def test_perc_time_denser_firn_is_slower():
    # higher solid fraction -> lower permeability -> longer transit time
    p_loose = percolation.perc_time(make_cell(sfrac=0.4), 0)
    p_dense = percolation.perc_time(make_cell(sfrac=0.8), 0)
    assert p_dense > p_loose


def test_perc_time_tiny_lfrac_short_circuits():
    # near-zero liquid fraction returns 0 (all of it percolates immediately)
    assert percolation.perc_time(make_cell(lfrac=1e-12), 0) == 0
