"""
Functions used to solve the heat equation using NumbaMinpack's hybrd function.
This gives significant performance boosts over scipy's fsolve(hybrd = True).

For the equivalent functions for use with Numba (and therefore using fsolve
rather than NumbaMinpack.hybrd), see heateqn in physics/Numba.

"""
import numpy as np
from monarchs.physics.surface_fluxes import sfc_flux
from numba import njit
from scipy.linalg import solve_banded
from scipy.optimize import root

def get_k_and_kappa(cell):
    # precompute some values
    rho = cell['Sfrac'] * 913 + cell['Lfrac'] * 1000
    k_ice = np.zeros(np.shape(cell['firn_temperature']))
    k_ice[cell['firn_temperature'] < 273.15] = 1000 * (2.24e-03 + 5.975e-06 *
                                                       ((273.15 - cell['firn_temperature'][
                                                           cell['firn_temperature'] < 273.15]) ** 1.156))
    k_ice[cell['firn_temperature'] >= 273.15] = 2.24
    k = cell['Sfrac'] * k_ice + (1 - cell['Sfrac'] - cell['Lfrac']) * cell['k_air'] + cell['Lfrac'] * cell[
        'k_water']
    cp_ice = 7.16 * cell['firn_temperature'] + 138
    cp = cell['Sfrac'] * cp_ice + (1 - cell['Sfrac'] - cell['Lfrac']) * cell['cp_air'] + cell['Lfrac'] * cell[
        'cp_water']
    kappa = k / (cp * rho)
    return k, kappa



def surface_temperature_residual(x, cell, LW_in, SW_in, T_air, p_air, T_dp, wind,
                                 dz, dt, epsilon=0.98, sigma=5.670374e-8):

    # Calculate Q for the given T_sfc
    Q = sfc_flux(cell['melt'], cell['exposed_water'], cell['lid'], cell[
        'lake'], cell['lake_depth'], LW_in, SW_in, T_air, p_air, T_dp, wind,
                 x[0])

    k, kappa = get_k_and_kappa(cell)
    residual = np.zeros_like(x)
    # Surface temperature equation (residual)
    residual[0] = k[0] * ((x[0] - x[1]) / dz) - (Q - epsilon * sigma * x[0] ** 4)

    # Calculate the temperature profile for the first 10 layers
    idx = np.arange(1, len(x) - 1)

    residual[idx] = (
        cell['firn_temperature'][idx]
        - x[idx]
        + dt * (kappa[idx] / dz**2) * (x[idx + 1] - 2 * x[idx] + x[idx - 1])
    )
    residual[-1] = (
        cell['firn_temperature'][len(x) - 1]
        - x[len(x) - 1]
        + dt * (kappa[len(x) - 1] / dz**2) * (-x[len(x) - 1] + x[len(x) - 2])
    )
    # print(f"Residual for T_sfc = {x}: {residual}")
    return np.array(residual)


def find_surface_temperature(cell, LW_in, SW_in, T_air, p_air, T_dp, wind,
                             dz, dt, solver_method='hybr', N=10):
    """
    The system of equations being solved (the heat equation with a variable boundary condition based on the
    surface fluxes) is highly nonlinear, given the dependence of the surface fluxes to the surface temperature
    (which is being solved for). This approach uses a root-finding algorithm to iteratively calculate the
    temperature. Doing so for the whole firn column is extremely expensive however; a compromise solution is
    to solve for the first N layers of the firn column using this optimisation method, and then use
    a linear solver to propagate the temperature down the column.

    Parameters
    ----------
    cell
    LW_in
    SW_in
    T_air
    p_air
    T_dp
    wind
    dz
    dt
    solver_method
    N

    Returns
    -------

    """
    initial_guess = cell['firn_temperature'][:N]
    # Use root-finding to solve for surface temperature
    result = root(surface_temperature_residual, initial_guess,
                  args=(cell, LW_in, SW_in, T_air, p_air, T_dp, wind,
                        dz, dt,),
                  method=solver_method, options={'maxfev': 1000})

    if not result.success:
        raise ValueError("Root-finding for surface temperature failed.")

    soldict = result  # Surface temperature solution
    return soldict

def propagate_temperature(cell, dz, dt, T_bc_top, N=10):
    n = len(cell['firn_temperature']) - N  # subtract top nonlinear layers
    k, kappa = get_k_and_kappa(cell)
    # Setup diagonals
    A = np.zeros(n)  # Sub-diagonal (lower diag)
    B = np.zeros(n)  # Main diagonal
    C = np.zeros(n)  # Super-diagonal (upper diag)
    D = np.zeros(n)  # RHS vector

    # Surface boundary condition (Dirichlet)
    B[0] = 1.0
    D[0] = T_bc_top  # Enforced surface temperature

    # Interior points
    factor = dt / dz ** 2

    for i in range(1, n - 1):
        A[i] = -factor * kappa[N + i]
        B[i] = 1 + 2 * factor * kappa[N + i]
        C[i] = -factor * kappa[N + i]
        D[i] = cell['firn_temperature'][N + i]

    # Bottom boundary (e.g., Neumann or simple derivative condition)
    B[-1] = 1.0
    D[-1] = cell['firn_temperature'][-1]

    # Assemble into banded form
    ab = np.zeros((3, n))
    ab[0, 1:] = C[:-1]  # Super-diagonal (shifted right by 1)
    ab[1, :] = B  # Main diagonal
    ab[2, :-1] = A[1:]  # Sub-diagonal (shifted left by 1)

    # Solve the system
    temperature_profile = solve_banded((1, 1), ab, D)

    return temperature_profile


# Solve for surface temperature (T_sfc)


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
