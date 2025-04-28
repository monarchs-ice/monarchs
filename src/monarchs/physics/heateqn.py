"""
Functions used to solve the heat equation using NumbaMinpack's hybrd function.
This gives significant performance boosts over scipy's fsolve(hybrd = True).

For the equivalent functions for use with Numba (and therefore using fsolve
rather than NumbaMinpack.hybrd), see heateqn in physics/Numba.

"""
import numpy as np
from monarchs.physics.surface_fluxes import sfc_flux
from numba import njit

@njit
def heateqn(x, cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind, k, kappa):
    """
    Solve the heat equation for the firn column with surface energy balance driven surface temperature.
    This is done via the hybrd root-finding algorithm.
    See also /Numba/heateqn for the NumbaMinpack implementation of this.

    Parameters
    ----------
    k
    kappa
    x : array_like, float, dimension(cell.vert_grid)
        initial estimate of the firn column temperature [K]
    cell : core.iceshelf_class.IceShelf
        IceShelf object containing the details for that vertical column
    dt : int
        timestep duration [s]
    dz : float
        size of the vertical grid
    LW_in : float
        Incoming longwave radiation at the current timestep [W m^-2]
    SW_in : float
        Incoming shortwave radiation at the current timestep [W m^-2]
    T_air : float
        Air temperature [K]
    p_air : float
        Surface pressure [Pa]
    T_dp : float
        Dewpoint temperature [K]
    wind : float
        Wind speed [m s^-1]

    Returns
    -------
    output : array_like, float, dimension(cell.vert_grid)
        roots of the function, used by scipy.optimize.fsolve to determine the new firn temperature
    """

    T = cell['firn_temperature']
    Sfrac = cell['Sfrac']
    Lfrac = cell['Lfrac']

    # Prepare the output array
    output = np.zeros_like(x)

    # First grid point (boundary condition)
    Q = sfc_flux(cell['melt'], cell['exposed_water'], cell['lid'], cell['lake'], cell['lake_depth'],
                 LW_in, SW_in, T_air, p_air, T_dp, wind, x[0])
    output[0] = k[0] * ((x[0] - x[1]) / dz) - (Q - 0.98 * 5.670374 * 10 ** -8 * x[0] ** 4)

    # Central grid points (interior points)
    output[1:-1] = T[1:-1] - x[1:-1] + dt * (kappa[1:-1] / dz ** 2) * (x[2:] - 2 * x[1:-1] + x[:-2])

    # Last grid point (boundary condition)
    output[-1] = T[-1] - x[-1] + dt * (kappa[-1] / dz ** 2) * (-x[-1] + x[-2])

    return output

@njit
def surface_temperature_residual(T_sfc, cell, LW_in, SW_in, T_air, p_air, T_dp, wind):
    # Calculate Q for the given T_sfc
    Q = sfc_flux(cell['melt'], cell['exposed_water'], cell['lid'], cell[
        'lake'], cell['lake_depth'], LW_in, SW_in, T_air, p_air, T_dp, wind,
        T_sfc)

    # Surface temperature equation (residual)
    # Assume you already have the correct form for this
    residual = Q - 0.98 * 5.670374 * 10**-8 * T_sfc**4
    return residual

def heateqn_fixedsfc(x, cell, dt, dz, T_sfc, kappa):
    """
    Solve the heat equation for the firn column with surface temperature fixed (nominally to 273.15 K).
    See also /Numba/heateqn_fixedsfc for the NumbaMinpack implementation of this.

    Parameters
    ----------
    kappa
    x : array_like, float, dimension(cell.vert_grid)
        initial estimate of the firn column temperature [K]
    cell : core.iceshelf_class.IceShelf
        IceShelf object containing the details for that vertical column
    dt : int
        timestep duration [s]
    dz : int
        size of the vertical grid
    T_sfc : float
        Surface temperature used to force the firn column [K]

    Returns
    -------
    output : roots of the function, used by scipy.optimize.fsolve to determine the new firn temperature
    """

    output = np.zeros(len(x))

    output[0] = x[0] - T_sfc
    idx = np.arange(1, len(x) - 1)
    output[idx] = cell['firn_temperature'][idx] - x[idx] + dt * (kappa[idx] / dz ** 2) * (x[idx +
        1] - 2 * x[idx] + x[idx - 1])
    output[-1] = cell['firn_temperature'][len(x) - 1] - x[len(x) - 1] + dt * (kappa[len(x) - 1] /
        dz ** 2) * (-x[len(x) - 1] + x[len(x) - 2])
    return output


def heateqn_lid(x, cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind,
    k_lid, Sfrac_lid):
    """
    Solve the heat equation for the frozen lid, similarly to the calculation for the firn column.

    Parameters
    ----------
    x : array_like, float, dimension(cell.vert_grid)
        initial estimate of the firn column temperature [K]
    cell : core.iceshelf_class.IceShelf
        IceShelf object containing the details for that vertical column
    dt : int
        timestep duration [s]
    dz : float
        height of a single vertical layer of the frozen lid [m]
    LW_in : float
        Incoming longwave radiation at the current timestep [W m^-2]
    SW_in : float
        Incoming shortwave radiation at the current timestep [W m^-2]
    T_air : float
        Air temperature [K]
    p_air : float
        Surface pressure [Pa]
    T_dp : float
        Dewpoint temperature [K]
    wind : float
        Wind speed [m s^-1]
    k_lid : int
        thermal conductivity of the frozen lid [W m^-1 K^-1]
    Sfrac_lid : array_like, float, dimension(cell.vert_grid_lid)
        Solid fraction of the frozen lid.

    Returns
    -------
    output : array_like, float, dimension(cell.vert_grid)
        roots of the function, used by scipy.optimize.fsolve to determine the new firn temperature
    """
    cp_ice = 1000 * (0.00716 * cell['lid_temperature'] + 0.138)
    cp = Sfrac_lid * cp_ice + (1 - Sfrac_lid) * cell['cp_air']
    kappa = k_lid / (cp * cell['rho_ice'])
    epsilon = 0.98
    sigma = 5.670373 * 10 ** -8
    Q = sfc_flux(cell['melt'], cell['exposed_water'], cell['lid'], cell[
        'lake'], cell['lake_depth'], LW_in, SW_in, T_air, p_air, T_dp, wind,
        x[0])
    output = np.zeros(cell['vert_grid_lid'])
    output[0] = k_lid * (x[0] - x[1]) / dz - (Q - epsilon * sigma * x[0] ** 4)
    idx = np.arange(1, cell['vert_grid_lid'] - 1)
    output[idx] = cell['lid_temperature'][idx] - x[idx] + dt * (kappa[idx] /
        dz ** 2) * (x[idx + 1] - 2 * x[idx] + x[idx - 1])
    output[-1] = x[cell['vert_grid_lid'] - 1] - 273.15
    return output
