from dataclasses import dataclass

import numpy as np


@dataclass
class IceShelf:
    """
    Test version of an IceShelf class instance with only a few attributes for lateral movement.
    e.g.
    a = IceShelf(firn_depth=20,
             lake_depth=0,
             vert_grid=40,
             vertical_profile=np.linspace(0, 20, 40),
             saturation=np.zeros(40),
             lake=False,
             lid=False,
             ice_lens=False,
             ice_lens_depth=999,
             water=np.zeros(40),
             rho_water=1000,
             rho=np.ones(40) * 913,
             Lfrac=np.zeros(40))
    """

    firn_depth: float
    lake_depth: float
    saturation: np.ndarray
    vert_grid: int
    lake: bool
    lid: bool
    ice_lens: bool
    ice_lens_depth: int
    Lfrac: np.ndarray
    Sfrac: np.ndarray
    rho_water: int
    rho: np.ndarray
    vertical_profile: np.ndarray
    valid_cell: bool
    row: int
    col: int
    firn_temperature: np.ndarray
    meltflag: np.ndarray
    size_dx: float
    size_dy: float
