import numpy as np


def test_catchment_outflow():
    from iceshelf_testclass import IceShelf
    from monarchs.physics.lateral_movement import move_water
    from monarchs.core.utils import get_2d_grid

    lake_depths = np.array([[0.8, 10, 0.8], [0.8, 5, 0.8]]).T
    firn_depths = np.array([[20, 30, 20], [20, 25, 20]]).T
    grid = []
    for i in range(len(lake_depths.T)):
        _l = []
        for j in range(len(lake_depths)):
            _l.append(
                IceShelf(
                    firn_depth=firn_depths[j, i],
                    lake_depth=lake_depths[j, i],
                    vert_grid=40,
                    vertical_profile=np.linspace(0, 20, 40),
                    saturation=np.zeros(40),
                    lake=True,
                    lid=False,
                    ice_lens=False,
                    ice_lens_depth=999,
                    rho_water=1000,
                    rho=np.ones(40) * 917,
                    Lfrac=np.zeros(40),
                    Sfrac=np.ones(40) * 0.5,
                    valid_cell=True,
                    firn_temperature=np.linspace(240, 273.15, 40)[::-1],
                    col=i,
                    row=j,
                    meltflag=np.zeros(40),
                    size_dx=2000,
                    size_dy=2000,
                )
            )

        grid.append(_l)
    print(get_2d_grid(grid, "lake_depth"))
    move_water(grid, 3, 2, 3600, catchment_outflow=False)
    print(get_2d_grid(grid, "lake_depth"))
    move_water(grid, 3, 2, 3600, catchment_outflow=True)
    print(get_2d_grid(grid, "lake_depth"))
