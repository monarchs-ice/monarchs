from monarchs.physics.firn import percolation
import numpy as np
import pytest


_perc_time_deferred = pytest.mark.xfail(
    reason="perc_time fix deferred to later results-changing branch",
    strict=True,
)


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


@_perc_time_deferred
def test_perc_time_positive():
    # Ensure that percolation time is a positie, finite number
    p_time = percolation.perc_time(make_cell(), 0)
    assert p_time > 0
    assert np.isfinite(p_time)
