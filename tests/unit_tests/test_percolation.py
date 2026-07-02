from monarchs.physics.firn import percolation
import numpy as np


def test_perc_time():
    # define cell parameters
    v_lev = 0
    cell = {}
    cell["Sfrac"] = np.array([0.5])
    cell["Lfrac"] = np.array([0.1])
    cell["firn_depth"] = 35
    cell["vert_grid"] = 500
    test_1 = percolation.perc_time(cell, v_lev)
    cell["vert_grid"] = 1000
    test_2 = percolation.perc_time(cell, v_lev)
    assert test_2 == 0.5 * test_1  # more layers, shorter time per layer
