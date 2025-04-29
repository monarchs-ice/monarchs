import numpy as np


def test_ice_lens():
    from iceshelf_testclass import IceShelf
    from monarchs.physics.lateral_functions import move_water
    from monarchs.core.utils import get_2d_grid

    lake_depths = np.array([[0, 0, 0], [0, 0, 0]]).T
    firn_depths = np.array([[20, 30, 20], [20, 25, 20]]).T
    ice_lens = np.array([[False, True, False], [False, True, False]]).T
    ice_lens_depths = np.array([[999, 10, 999], [999, 10, 999]]).T
    Lfrac = np.array([[0, 0.2, 0], [0, 0.2, 0]]).T
    saturation = np.array(
        [
            [np.zeros(40), np.ones(40), np.zeros(40)],
            [np.zeros(40), np.ones(40), np.zeros(40)],
        ]
    ).T
    grid = []
    for i in range(len(lake_depths.T)):
        _l = []
        for j in range(len(lake_depths)):
            _l.append(
                IceShelf(
                    firn_depth=firn_depths[j, i],
                    lake_depth=lake_depths[j, i],
                    vert_grid=40,
                    vertical_profile=np.linspace(0, firn_depths[j, i], 40),
                    saturation=saturation[:, j, i],
                    lake=False,
                    lid=False,
                    ice_lens=ice_lens[j, i],
                    ice_lens_depth=ice_lens_depths[j, i],
                    rho_water=1000,
                    rho=np.ones(40) * 500,
                    Lfrac=np.ones(40) * Lfrac[j, i],
                    Sfrac=np.ones(40) * 0.5,
                    valid_cell=True,
                    firn_temperature=np.linspace(240, 273.15, 40)[::-1],
                    col=i,
                    row=j,
                    meltflag=np.zeros(40),
                    size_dx=20,
                    size_dy=20,
                )
            )

        grid.append(_l)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            grid[i][j].rho_ice = 917

    print(get_2d_grid(grid, "Lfrac"))
    move_water(
        grid,
        3,
        2,
        3600 * 24,
        catchment_outflow=False,
        lateral_movement_percolation_toggle=False,
    )
    print(get_2d_grid(grid, "Lfrac"))
    move_water(
        grid,
        3,
        2,
        3600 * 24,
        catchment_outflow=True,
        lateral_movement_percolation_toggle=False,
    )
    print(get_2d_grid(grid, "Lfrac"))

    lake_depths = np.array([[0, 0, 0], [0, 0, 0]]).T
    firn_depths = np.array([[20, 30, 20], [20, 25, 20]]).T
    ice_lens = np.array([[True, True, True], [True, True, True]]).T
    ice_lens_depths = np.array([[15, 10, 15], [15, 10, 15]]).T
    Lfrac = np.array([[0.01, 0.2, 0.01], [0.01, 0.2, 0.01]]).T
    saturation = np.array(
        [
            [np.zeros(40), np.ones(40), np.zeros(40)],
            [np.zeros(40), np.ones(40), np.zeros(40)],
        ]
    ).T
    grid = []
    for i in range(len(lake_depths.T)):
        _l = []
        for j in range(len(lake_depths)):
            _l.append(
                IceShelf(
                    firn_depth=firn_depths[j, i],
                    lake_depth=lake_depths[j, i],
                    vert_grid=40,
                    vertical_profile=np.linspace(0, firn_depths[j, i], 40),
                    saturation=saturation[:, j, i],
                    lake=False,
                    lid=False,
                    ice_lens=ice_lens[j, i],
                    ice_lens_depth=ice_lens_depths[j, i],
                    rho_water=1000,
                    rho=np.ones(40) * 500,
                    Lfrac=np.ones(40) * Lfrac[j, i],
                    Sfrac=np.ones(40) * 0.5,
                    valid_cell=True,
                    firn_temperature=np.linspace(240, 273.15, 40)[::-1],
                    col=i,
                    row=j,
                    meltflag=np.zeros(40),
                    size_dx=20,
                    size_dy=20,
                )
            )

        grid.append(_l)

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            grid[i][j].rho_ice = 917

    print(get_2d_grid(grid, "Lfrac"))
    move_water(
        grid,
        3,
        2,
        3600 * 24,
        catchment_outflow=False,
        lateral_movement_percolation_toggle=False,
    )
    print(get_2d_grid(grid, "Lfrac"))
    move_water(
        grid,
        3,
        2,
        3600 * 24,
        catchment_outflow=True,
        lateral_movement_percolation_toggle=False,
    )
    print(get_2d_grid(grid, "Lfrac"))
