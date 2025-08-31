from monarchs.physics import percolation_functions
import numpy as np
from numpy import testing as npt

blah = 2
def test_perc_time():
    # define cell parameters
    v_lev = 0
    cell = {}
    cell['Sfrac'] = np.array([0.5])
    cell['Lfrac'] = np.array([0.1])
    cell['rho_ice'] = 917
    cell['rho_water'] = 1000
    cell['firn_depth'] = 35
    cell['vert_grid'] = 500
    test_1 = percolation_functions.perc_time(cell, v_lev)
    cell['vert_grid'] = 1000
    test_2 = percolation_functions.perc_time(cell, v_lev)
    assert test_2 == 0.5 * test_1  # more layers, shorter time per layer