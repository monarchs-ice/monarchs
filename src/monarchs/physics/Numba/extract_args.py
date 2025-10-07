"""
TODO - module-level docstring
"""

import numpy as np
from numba import jit


def extract_args_firn(args):
    """
    NumbaMinpack gives large performance boosts, but has very crude
    syntax which requires us to place all of our arguments to heateqn in
    a single vector. This function retrieves the relevant
    arrays from the "args" vector and separates them into
    physically-meaningful variables.

    Parameters
    ----------
    args : float64[:]
     Input vector of arguments. Generated using args_array from
     physics.solver.

    Returns
    -------
    T : float64[:]
     Temperature of each vertical point. [K].
    Sfrac : float64[:]
     Solid fraction of each vertical point. [unitless]
    Lfrac : float64[:]
     Liquid fraction in each vertical point. [unitless]
    k_air : float64
     Thermal conductivity of air.
    k_water : float64
     Thermal conductivity of water.
    cp_air : float64
     Heat capacity of air.
    cp_water : float64
     Heat capacity of air.
    dt : float64
     Timestep. [s]
    dz : float64
     Change in firn height with respect to a step in vertical point [m]
    melt : Bool
     Flag which indicates whether melting has occurred.
    exposed_water : Bool
     Flag which describes whether there is exposed water at the surface.
    lid : Bool
     Flag which indicates whether there is a lid at the surface due to
     refreezing.
    lake : Bool
     Flag which indicates whether there is a lake present.
    lake_depth : float64
     Depth of the lake (in vertical points?)
    LW_in : float64
     Incoming longwave radiation. [W m^-2].
    SW_in : float64
     Incoming shortwave (solar) radiation. [W m^-2].
    T_air : float64
     Surface-layer air temperature. [K].
    p_air : float64
     Surface-layer air pressure. [hPa].
    T_dp : float64
     Dew-point temperature. [K].
    wind : float64
     Wind speed. [m s^-1].

    """

    # -1 doesnt work so use 0 and convert to int as we use it for indexing
    vert_grid = np.int32(args[0])

    # initialise vectors and write into them - Numba doesn't like it if
    # we assign them directly for some reason, but this has no impact on
    # performance
    T = np.zeros(vert_grid)
    for i in range(vert_grid):
        T[i] = args[i + 1]
    Sfrac = np.zeros(vert_grid)
    for i in range(vert_grid):
        Sfrac[i] = args[vert_grid + i + 1]
    Lfrac = np.zeros(vert_grid)
    for i in range(vert_grid):
        Lfrac[i] = args[(vert_grid * 2) + i + 1]

    # find index corresponding to where our vertically-resolved variables
    # are all read in - the following are all length-1
    arrind = 3 * vert_grid + 1
    k_air = args[arrind]
    k_water = args[arrind + 1]
    cp_air = args[arrind + 2]
    cp_water = args[arrind + 3]
    dt = args[arrind + 4]
    dz = args[arrind + 5]

    # we had to convert these to floats for the purposes of
    # reading them in as args needed a unified datatype - now recast them
    # to the original dtypes.
    melt = bool(np.round(args[arrind + 6]))
    exposed_water = bool(np.round(args[arrind + 7]))
    lid = bool(np.round(args[arrind + 8]))
    lake = bool(np.round(args[arrind + 9]))
    lake_depth = args[arrind + 10]
    LW_in = args[arrind + 11]
    SW_in = args[arrind + 12]
    T_air = args[arrind + 13]
    p_air = args[arrind + 14]
    T_dp = args[arrind + 15]
    wind = args[arrind + 16]

    return (
        T,
        Sfrac,
        Lfrac,
        k_air,
        k_water,
        cp_air,
        cp_water,
        dt,
        dz,
        melt,
        exposed_water,
        lid,
        lake,
        lake_depth,
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
    )


def extract_args_lid(args):
    """
    NumbaMinpack gives large performance boosts, but has very crude
    syntax which requires us to place all of our arguments to heateqn in
    a single vector. This function retrieves the relevant
    arrays from the "args" vector and separates them into
    physically-meaningful variables.

    Parameters
    ----------
    args : float64[:]
        Input vector of arguments. Generated using args_array from
        <module_name>.

    Returns
    -------
    T : float64[:]
        Temperature of each vertical point. [K].
    Sfrac : float64[:]
        Solid fraction of each vertical point. [unitless]
    k_air : float64
        Thermal conductivity of air.
    k_water : float64
        Thermal conductivity of water.
    cp_air : float64
        Heat capacity of air.
    cp_water : float64
        Heat capacity of air.
    dt : float64
        Timestep. [s]
    dz : float64
        Change in firn height with respect to a step in vertical point?
    melt : Bool
        Flag which indicates whether melting has occurred.
    exposed_water : Bool
        Flag which describes whether there is exposed water at the surface.
    lid : Bool
        Flag which indicates whether there is a lid at the surface due to
        refreezing.
    lake : Bool
        Flag which indicates whether there is a lake present.
    lake_depth : float64
        Depth of the lake (in vertical points?)
    LW_in : float64
        Incoming longwave radiation. [W m^-2].
    SW_in : float64
        Incoming shortwave (solar) radiation. [W m^-2].
    T_air : float64
        Surface-layer air temperature. [K].
    p_air : float64
        Surface-layer air pressure. [hPa].
    T_dp : float64
        Dew-point temperature. [K].
    wind : float64
        Wind speed. [m s^-1].

    """

    # -1 doesnt work so use 0 and convert to int as we use it for indexing
    vert_grid = np.int32(args[0])

    # initialise vectors and write into them - Numba doesn't like it if
    # we assign them directly for some reason, but this has no impact on
    # performance
    T = np.zeros(vert_grid)
    for i in range(vert_grid):
        T[i] = args[i + 1]
    Sfrac = np.zeros(vert_grid)
    for i in range(vert_grid):
        Sfrac[i] = args[vert_grid + i + 1]

    # find index corresponding to where our vertically-resolved variables
    # are all read in - the following are all length-1
    arrind = 2 * vert_grid + 1
    k_lid = args[arrind]
    cp_air = args[arrind + 1]
    cp_water = args[arrind + 2]
    dt = args[arrind + 3]
    dz = args[arrind + 4]

    # we had to convert these to floats for the purposes of
    # reading them in as args needed a unified datatype - now recast them
    # to the original dtypes.
    melt = bool(np.round(args[arrind + 5]))
    exposed_water = bool(np.round(args[arrind + 6]))
    lid = bool(np.round(args[arrind + 7]))
    lake = bool(np.round(args[arrind + 8]))
    lake_depth = args[arrind + 9]
    LW_in = args[arrind + 10]
    SW_in = args[arrind + 11]
    T_air = args[arrind + 12]
    p_air = args[arrind + 13]
    T_dp = args[arrind + 14]
    wind = args[arrind + 15]

    return (
        T,
        Sfrac,
        k_lid,
        cp_air,
        cp_water,
        dt,
        dz,
        melt,
        exposed_water,
        lid,
        lake,
        lake_depth,
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
    )


extract_args_firn = jit(extract_args_firn, nopython=True, fastmath=False)
extract_args_lid = jit(extract_args_lid, nopython=True, fastmath=False)
