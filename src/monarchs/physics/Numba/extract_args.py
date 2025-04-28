import numpy as np
from numba import jit


def extract_args(args):
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
    vert_grid = np.int32(args[0])
    T = np.zeros(vert_grid)
    for i in range(vert_grid):
        T[i] = args[i + 1]
    Sfrac = np.zeros(vert_grid)
    for i in range(vert_grid):
        Sfrac[i] = args[vert_grid + i + 1]
    Lfrac = np.zeros(vert_grid)
    for i in range(vert_grid):
        Lfrac[i] = args[vert_grid * 2 + i + 1]
    arrind = 3 * vert_grid + 1
    k_air = args[arrind]
    k_water = args[arrind + 1]
    cp_air = args[arrind + 2]
    cp_water = args[arrind + 3]
    dt = args[arrind + 4]
    dz = args[arrind + 5]
    melt = bool(args[arrind + 6])
    exposed_water = bool(args[arrind + 7])
    lid = bool(args[arrind + 8])
    lake = bool(args[arrind + 9])
    lake_depth = args[arrind + 10]
    LW_in = args[arrind + 11]
    SW_in = args[arrind + 12]
    T_air = args[arrind + 13]
    p_air = args[arrind + 14]
    T_dp = args[arrind + 15]
    wind = args[arrind + 16]
    return (T, Sfrac, Lfrac, k_air, k_water, cp_air, cp_water, dt, dz, melt,
        exposed_water, lid, lake, lake_depth, LW_in, SW_in, T_air, p_air,
        T_dp, wind)


def extract_args_fixedsfc(args):
    """
    NumbaMinpack gives large performance boosts, but has very crude
    syntax which requires us to place all of our arguments to heateqn in
    a single vector. This function retrieves the relevant
    arrays from the "args" vector and separates them into
    physically-meaningful variables. This is split into a separate
    function to extract_args, since using conditionals causes issues
    with Numba (since the code is pre-compiled, any shape or dtype differences
    as is the case here cause issues).

    Parameters
    ----------
    args : float64[:]
        Input vector of arguments. Generated using args_array from
        <module_name>.

    Returns
    -------
    T : float64[:]
        Temperature of each vertical point in the ice shelf. [K]
    Sfrac : float64[:]
        Solid fraction of each vertical point.
    Lfrac : float64[:]
        Liquid fraction in each vertical point.
    k_air : float64
        Thermal conductivity of air.
    k_water : float64
        Thermal conductivity of water.
    cp_air : float64
        Heat capacity of air.
    cp_water : float64
        Heat capacity of air.
    dt : float64
        Timestep. [s].
    dz : float64
        Change in firn height with respect to a step in vertical point [m]
    Tsfc : float64
        Surface temperature [K]

    """
    vert_grid = int(args[0])
    T = np.zeros(vert_grid)
    for i in range(vert_grid):
        T[i] = args[i + 1]
    Sfrac = np.zeros(vert_grid)
    for i in range(vert_grid):
        Sfrac[i] = args[vert_grid + i + 1]
    Lfrac = np.zeros(vert_grid)
    for i in range(vert_grid):
        Lfrac[i] = args[vert_grid * 2 + i + 1]
    arrind = 3 * vert_grid + 1
    k_air = args[arrind]
    k_water = args[arrind + 1]
    cp_air = args[arrind + 2]
    cp_water = args[arrind + 3]
    dt = args[arrind + 4]
    dz = args[arrind + 5]
    Tsfc = args[arrind + 6]
    return T, Sfrac, Lfrac, k_air, k_water, cp_air, cp_water, dt, dz, Tsfc


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
    vert_grid = np.int32(args[0])
    T = np.zeros(vert_grid)
    for i in range(vert_grid):
        T[i] = args[i + 1]
    Sfrac = np.zeros(vert_grid)
    for i in range(vert_grid):
        Sfrac[i] = args[vert_grid + i + 1]
    arrind = 2 * vert_grid + 1
    k_lid = args[arrind]
    cp_air = args[arrind + 1]
    cp_water = args[arrind + 2]
    dt = args[arrind + 3]
    dz = args[arrind + 4]
    melt = bool(args[arrind + 5])
    exposed_water = bool(args[arrind + 6])
    lid = bool(args[arrind + 7])
    lake = bool(args[arrind + 8])
    lake_depth = args[arrind + 9]
    LW_in = args[arrind + 10]
    SW_in = args[arrind + 11]
    T_air = args[arrind + 12]
    p_air = args[arrind + 13]
    T_dp = args[arrind + 14]
    wind = args[arrind + 15]
    return (T, Sfrac, k_lid, cp_air, cp_water, dt, dz, melt, exposed_water,
        lid, lake, lake_depth, LW_in, SW_in, T_air, p_air, T_dp, wind)


extract_args = jit(extract_args, nopython=True, fastmath=False)
extract_args_fixedsfc = jit(extract_args_fixedsfc, nopython=True, fastmath=
    False)
extract_args_lid = jit(extract_args_lid, nopython=True, fastmath=False)
