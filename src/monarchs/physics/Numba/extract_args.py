"""
NumbaMinpack requires all arguments to be passed as a single flat array.
This module provides functions to pack and unpack model variables
into/from such a flat array, supporting variable grid sizes and optional arrays.

This is done using a mapping of indices for each scalar variable in the packed array,
as well as pointers to the start of variable-length arrays (e.g. firn temperature, lid temperature).

The intent here is to ensure that we just have one set of logic to move data from
the Python structured array format into the flat array format required by NumbaMinpack,
and one set of functions we can use and re-use to extract the relevant parameters from
this packed array.

The indices defined below is the "canonical" mapping of model variables to positions
in the packed args array. If you want to use a new variable inside
the heat equation solver (or equivalently surface fluxes, since this is called inside
the heat equation solver), you need to add it here.

We define functions to extract the specific parameters we need, which is determined by the needs
of the specific physics functions.
"""

import numpy as np
from numba import jit, carray

# Define indices for various parameters in the packed args array.
# Model scalars
IDX_N_GRID = 0
IDX_MELT = 1
IDX_EXPOSED_WATER = 2
IDX_LID = 3
IDX_LID_DEPTH = 4
IDX_LAKE = 5
IDX_LAKE_DEPTH = 6
IDX_SNOW_LID = 7
IDX_ALBEDO = 8
IDX_DT = 9
IDX_DZ = 10
IDX_K_LID_SCALAR = 11
IDX_SFRAC_LID_SCALAR = 12
IDX_V_LID = 13
IDX_V_LID_DEPTH = 14
IDX_FIRN_DEPTH = 15
IDX_V_LID_TEMP = 16
# just makes calcing next indices easier - add your index(es) to the end of the list above
# and update IDX_SCALARS_END accordingly
IDX_SCALARS_END = 17

# Atmospheric data
IDX_LW_IN = IDX_SCALARS_END + 1
IDX_SW_IN = IDX_SCALARS_END + 2
IDX_AIR_TEMP = IDX_SCALARS_END + 3
IDX_P_AIR = IDX_SCALARS_END + 4
IDX_DP_TEMP = IDX_SCALARS_END + 5
IDX_WIND = IDX_SCALARS_END + 6
# Size of the header portion of the args array, i.e. how many scalar
# variables are we packing in. We can increase this if we add more scalars!
HEADER_SIZE = 25


# Array data. The data inside these indices (e.g. arr[22]) will be the start points
# of the corresponding arrays. e.g. arr[PTR_LID_T] will be the start of the lid temperature array.
# Since this is of variable length, we then need to assign PTR_LAKE_T as PTR_LID_T + N_lid, etc.
# So arr[23] i.e. arr[PTR_LAKE_T] will be e.g. 32 if N_lid is 10.
PTR_LID_T = HEADER_SIZE
PTR_LAKE_T = HEADER_SIZE + 1
PTR_FIRN_T = HEADER_SIZE + 2
PTR_SFRAC = HEADER_SIZE + 3
PTR_LFRAC = HEADER_SIZE + 4

PTR_SIZE = HEADER_SIZE + 5


@jit(nopython=True, fastmath=False)
def pack_args(cell, met_data, dt, dz, N=False, k_lid=np.nan, Sfrac_lid=np.nan):
    """
    Convert the variables from the model grid into a unified flat buffer.
    Uses a 'Table of Contents' (pointers) in the header to allow for
    variable grid sizes and optional arrays.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    met_data : dict
        Dictionary containing the meteorological data for the current timestep.
        See firn_column for details.
    dt : float
        Number of seconds in each timestep. [s]
    dz : float
        Change in firn height with respect to a step in vertical point [m]
    N : int, optional
        Number of vertical grid points in the firn column. If not provided,
        uses cell["vert_grid"]. Used for the non-fixed-surface heat
        equation solver. Default is False.
    k_lid : float, optional
        Thermal conductivity of the lid. Default is np.nan.
    Sfrac_lid : float, optional
        Solid fraction of the lid. Default is np.nan.

    """
    if not N:
        N = cell["vert_grid"]

    if not np.isfinite(k_lid):
        k_lid = 0
    # get array lengths of variable-length arrays
    len_lid = len(cell['lid_temperature'])
    len_lake = len(cell['lake_temperature'])
    len_firn = N

    # these are used to create pointers to
    # the start of each array in the args vector
    # which are put into e.g. args[PTR_LID_T]
    DATA_START = PTR_SIZE
    p_lid = DATA_START
    p_lake = p_lid + len_lid
    p_firn = p_lake + len_lake
    p_sfrac = p_firn + len_firn
    p_lfrac = p_sfrac + len_firn
    total_size = p_lfrac + len_firn

    args = np.zeros(total_size)

    # scalars - these are just the actual values
    args[IDX_FIRN_DEPTH] = cell['firn_depth']
    args[IDX_N_GRID] = N
    args[IDX_MELT] = cell['melt']
    args[IDX_EXPOSED_WATER] = cell['exposed_water']
    args[IDX_LID] = cell['lid']
    args[IDX_LAKE] = cell['lake']
    args[IDX_LAKE_DEPTH] = cell['lake_depth']
    args[IDX_SNOW_LID] = cell['snow_on_lid']
    args[IDX_ALBEDO] = cell['albedo']
    args[IDX_DT] = dt
    args[IDX_DZ] = dz
    args[IDX_K_LID_SCALAR] = k_lid
    args[IDX_LID_DEPTH] = cell['lid_depth']
    args[IDX_V_LID_DEPTH] = cell['v_lid_depth']
    args[IDX_V_LID] = cell['v_lid']
    args[IDX_V_LID_TEMP] = cell['virtual_lid_temperature']
    # meteorological/atmospheric data - again, the actual values
    args[IDX_LW_IN] = met_data['LW_down']
    args[IDX_SW_IN] = met_data['SW_down']
    args[IDX_AIR_TEMP] = met_data['temperature']
    args[IDX_P_AIR] = met_data['surf_pressure']
    args[IDX_DP_TEMP] = met_data['dew_point_temperature']
    args[IDX_WIND] = met_data['wind']

    # pointers - indices for the start of each array
    # kind of like a table of contents
    args[PTR_LID_T] = p_lid
    args[PTR_LAKE_T] = p_lake
    args[PTR_FIRN_T] = p_firn
    args[PTR_SFRAC] = p_sfrac
    args[PTR_LFRAC] = p_lfrac

    # fill these arrays now
    args[p_lid: p_lake] = cell['lid_temperature']
    args[p_lake: p_firn] = cell['lake_temperature']
    args[p_firn: p_sfrac] = cell['firn_temperature'][:N]
    args[p_sfrac: p_lfrac] = cell['Sfrac'][:N]
    args[p_lfrac: total_size] = cell['Lfrac'][:N]

    return args

@jit(nopython=True, fastmath=False)
def extract_scalars(args):
    """
    Extract only the scalar parameters from the packed args array.

    Parameters
    ----------
    args: np.ndarray
        Unified flat buffer containing model variables.

    Returns
    -------

    """
    # extract scalar values from header
    vert_grid = int(args[IDX_N_GRID])
    firn_depth = args[IDX_FIRN_DEPTH]
    dt = args[IDX_DT]
    dz = args[IDX_DZ]
    melt = args[IDX_MELT]
    albedo = args[IDX_ALBEDO]
    exposed_water = args[IDX_EXPOSED_WATER]
    lid = args[IDX_LID]
    lid_depth = args[IDX_LID_DEPTH]
    virtual_lid = args[IDX_V_LID]
    virtual_lid_depth = args[IDX_V_LID_DEPTH]
    lake = args[IDX_LAKE]
    lake_depth = args[IDX_LAKE_DEPTH]
    snow_on_lid = args[IDX_SNOW_LID]



    return (
        vert_grid, firn_depth, dt, dz, melt, albedo, exposed_water,
        lid, lid_depth, virtual_lid, virtual_lid_depth, lake, lake_depth, snow_on_lid
    )

@jit(nopython=True, fastmath=False)
def extract_met_data(args):
    """
    Extract only the meteorological data from the packed args array.

    Parameters
    ----------
    args: np.ndarray
        Unified flat buffer containing model variables.

    Returns
    -------

    """

    args = carray(args, (10000,))
    lw_in = args[IDX_LW_IN]
    sw_in = args[IDX_SW_IN]
    air_temp = args[IDX_AIR_TEMP]
    p_air = args[IDX_P_AIR]
    dew_point_temperature = args[IDX_DP_TEMP]
    wind = args[IDX_WIND]
    return lw_in, sw_in, air_temp, p_air, dew_point_temperature, wind

@jit(nopython=True, fastmath=False)
def extract_firn_arrays(args):
    """
    Extract only the variable-length firn arrays from the packed args array.

    Parameters
    ----------
    args: np.ndarray
        Unified flat buffer containing model variables.

    Returns
    -------

    """
    # args here is likely a pointer (float64*)
    # We must wrap it to use slicing.
    # We pick a large enough size (e.g., 10000) or pass the total size in.
    args = carray(args, (2000,))
    # extract pointers for variable-length arrays
    p_firn = int(args[PTR_FIRN_T])
    p_sfrac = int(args[PTR_SFRAC])
    p_lfrac = int(args[PTR_LFRAC])
    N = int(args[IDX_N_GRID])
    # slice using the pointers
    T_firn = args[p_firn: p_firn + N]
    Sfrac = args[p_sfrac: p_sfrac + N]
    # lfrac is last variable so goes to end
    Lfrac = args[p_lfrac:p_lfrac + N]

    return T_firn, Sfrac, Lfrac

@jit(nopython=True, fastmath=False)
def extract_lid_variables(args):
    """
    Extract only the variable-length lid arrays from the packed args array.

    Parameters
    ----------
    args: np.ndarray
        Unified flat buffer containing model variables.

    Returns
    -------

    """
    # args here is likely a pointer (float64*)
    # We must wrap it to use slicing.
    # We pick a large enough size (e.g., 10000) or pass the total size in.
    args = carray(args, (5000,))
    # extract pointers for variable-length arrays
    p_lid = int(args[PTR_LID_T])
    p_lake = int(args[PTR_LAKE_T])

    # slice using the pointers
    T_lid = args[p_lid: p_lake]
    Sfrac_lid = args[IDX_SFRAC_LID_SCALAR]
    k_lid = args[IDX_K_LID_SCALAR]
    v_lid_depth = args[IDX_V_LID_DEPTH]
    v_lid_temperature = args[IDX_V_LID_TEMP]
    vert_grid_lid = len(T_lid)

    return T_lid, Sfrac_lid, k_lid, vert_grid_lid, v_lid_depth, v_lid_temperature

@jit(nopython=True, fastmath=False)
def extract_lake_variables(args):
    """
    Extract only the variable-length lake arrays from the packed args array.

    Parameters
    ----------
    args: np.ndarray
        Unified flat buffer containing model variables.

    Returns
    -------

    """
    # args here is likely a pointer (float64*)
    # We must wrap it to use slicing.
    # We pick a large enough size (e.g., 10000) or pass the total size in.
    args = carray(args, (2000,))
    # extract pointers for variable-length arrays
    p_lake = int(args[int(PTR_LAKE_T)])
    p_firn = int(args[int(PTR_FIRN_T)])

    # slice using the pointers
    T_lake = args[p_lake: p_firn]
    vert_grid_lake = len(T_lake)

    return T_lake, vert_grid_lake,