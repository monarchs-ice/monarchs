import numpy as np

"""
Create an instance of an IceShelf object for testing with.
"""


def frozen_testcase():
    from monarchs.core.iceshelf_class import IceShelf
    from monarchs.core.initial_conditions import rho_init_emp

    firn_depth = 39.99973571
    vert_grid = 400
    vert_grid_other = 20
    rho_init = np.ones(vert_grid) * rho_init_emp(0, 500, 37)
    T_init = np.linspace(250, 272, vert_grid)[::-1]

    testcell = IceShelf(
        0,
        0,
        firn_depth,
        vert_grid,
        vert_grid_other,
        vert_grid_other,
        rho_init,
        T_init,
    )
    return testcell


a = frozen_testcase()
