"""

"""

import numpy as np
from monarchs.physics.surface_fluxes import sfc_flux
from scipy.linalg import solve_banded
from scipy.optimize import root
import numpy.testing as npt


def get_k_and_kappa(T, sfrac, lfrac, cp_air, cp_water, k_air, k_water):
    # precompute some values
    rho = sfrac * 917 + lfrac * 1000
    k_ice = np.zeros(np.shape(T), dtype=np.float64)
    k_ice[T < 273.15] = 1000 * (
        2.24e-03
        + 5.975e-06
        * ((273.15 - T[T < 273.15]) ** 1.156
        )
    )
    k_ice[T >= 273.15] = 2.24
    k = (
        sfrac * k_ice
        + (1 - sfrac - lfrac) * k_air
        + lfrac * k_water
    )
    cp_ice = 7.16 * T + 138
    cp = (
        sfrac * cp_ice
        + (1 - sfrac - lfrac) * cp_air
        + lfrac * cp_water
    )
    kappa = k / (cp * rho)  # thermal diffusivity [m^2 s^-1]
    return k, kappa


# def heateqn(
#     x,
#     cell,
#     LW_in,
#     SW_in,
#     T_air,
#     p_air,
#     T_dp,
#     wind,
#     dz,
#     dt,
#     epsilon=0.98,
#     sigma=5.670374e-8,
#     half_cell_ends=True,
#     debug=False,
# ):
#     """
#     Flux-form residual for variable k, rho, cp with correct sign and option
#     for half-thickness at top/bottom (node-centered).
#     - x: temperatures at N nodes
#     - dz: grid spacing (in metres!)
#     - dt: timestep (in seconds!)
#     - half_cell_ends: if True, use dz/2 for top and bottom cell thickness
#     """
#
#     N = len(x)
#     if N < 2:
#         raise ValueError("Need at least two nodes")
#
#     # --- surface flux (depends on x[0]) ---
#     Q = sfc_flux(
#         cell["melt"],
#         cell["exposed_water"],
#         cell["lid"],
#         cell["lake"],
#         cell["lake_depth"],
#         LW_in,
#         SW_in,
#         T_air,
#         p_air,
#         T_dp,
#         wind,
#         x[0],
#     )
#     # net non-conductive surface flux (W/m^2), *downward positive*
#     flux_sfc = Q - epsilon * sigma * (x[0] ** 4)
#
#     # --- old temperature and material properties (node-based) ---
#     T_old = cell["firn_temperature"][:N].astype(float)
#     Sfrac = cell["Sfrac"][:N].astype(float)
#     Lfrac = cell["Lfrac"][:N].astype(float)
#     cp_air = float(cell["cp_air"])
#     cp_water = float(cell["cp_water"])
#     k_air = float(cell["k_air"])
#     k_water = float(cell["k_water"])
#
#     # conductivity k and diffusivity (kappa not used directly)
#     k, _ = get_k_and_kappa(x, Sfrac, Lfrac, cp_air, cp_water, k_air, k_water)
#
#     # density and cp at nodes (must match get_k_and_kappa logic)
#     rho = Sfrac * 917.0 + Lfrac * 1000.0
#     cp_ice = 7.16 * x + 138.0
#     cp = Sfrac * cp_ice + (1.0 - Sfrac - Lfrac) * cp_air + Lfrac * cp_water
#     rho_cp = rho * cp  # J m^-3 K^-1
#
#     # --- cell volumes / thicknesses (node-centered -> half-cells on ends) ---
#     cell_thickness = np.full(N, dz, dtype=float)
#     if half_cell_ends:
#         cell_thickness[0] = dz / 2.0
#         cell_thickness[-1] = dz / 2.0
#
#     # --- face conductivities (harmonic mean), length N-1 ---
#     eps_small = 1e-30
#     k_face = 2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:] + eps_small)
#
#     # --- face fluxes (W m^-2), flux positive downward ---
#     # flux_face[i] is flux at face between node i and i+1
#     flux_face = -k_face * (x[1:] - x[:-1]) / dz   # length N-1
#
#     # --- residual array ---
#     residual = np.zeros_like(x, dtype=float)
#
#     # --- top node (i = 0) ---
#     # flux_in is flux from atmosphere (flux_sfc), flux_out is flux_face[0]
#     flux_in_top = flux_sfc
#     flux_out_top = flux_face[0]
#     # T tendency = (flux_in - flux_out) / (rho_cp * cell_thickness)
#     tendency_top = (flux_in_top - flux_out_top) / (rho_cp[0] * cell_thickness[0])
#     residual[0] = T_old[0] - x[0] + dt * tendency_top
#
#     # --- interior nodes (1 .. N-2) ---
#     if N > 2:
#         # flux_interior_in = flux_face[i-1], flux_interior_out = flux_face[i]
#         flux_interior_in = flux_face[:-1]
#         flux_interior_out = flux_face[1:]
#         # tendency at interior i = (flux_in - flux_out) / (rho_cp[i] * dz_i)
#         div_interior = (flux_interior_in - flux_interior_out) / cell_thickness[1:-1]
#         residual[1:-1] = (T_old[1:-1] - x[1:-1]) + dt * div_interior / rho_cp[1:-1]
#
#     # --- bottom node (i = N-1) ---
#     # assume insulated bottom (flux_out_bottom = 0). Replace if you have geothermal flux.
#     flux_in_bottom = flux_face[-1]
#     flux_out_bottom = 0.0
#     tendency_bottom = (flux_in_bottom - flux_out_bottom) / (rho_cp[-1] * cell_thickness[-1])
#     residual[-1] = T_old[-1] - x[-1] + dt * tendency_bottom
#
#     # --- debug prints to inspect scales that cause large changes ---
#     if debug:
#         print("DEBUG heateqn")
#         print(f"  dz (m) = {dz}, dt (s) = {dt}")
#         print(f"  surface T (K) = {x[0]}")
#         print(f"  flux_sfc (W/m2, down+) = {flux_sfc:.3f}")
#         print(f"  flux_face[0] (W/m2, down+) = {flux_face[0]:.3f}")
#         print(f"  cell_thickness[0] = {cell_thickness[0]:.6f} m")
#         print(f"  rho_cp[0] = {rho_cp[0]:.3e} J/m3/K")
#         print(f"  tendency_top = {tendency_top:.3e} K/s  (dt*tendency = {dt*tendency_top:.3e} K)")
#         # show a couple of interior fluxes
#         if len(flux_face) > 1:
#             print(f"  flux_face[1] = {flux_face[1]:.3f}, flux_face[-1] = {flux_face[-1]:.3f}")
#         print("  residual[0:5] =", residual[: min(5, N)])
#     return residual



def heateqn(
    x,
    cell,
    LW_in,
    SW_in,
    T_air,
    p_air,
    T_dp,
    wind,
    dz,
    dt,
    epsilon=0.98,
    sigma=5.670374e-8,
):

    # Calculate Q for the given T_sfc
    Q = sfc_flux(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["lake_depth"],
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
        x[0],
    )
    # Q = 650  # Placeholder value for testing
    N = len(x)
    T_old = cell["firn_temperature"][:N]
    Sfrac = cell['Sfrac'][:N]
    Lfrac = cell['Lfrac'][:N]
    cp_air = cell['cp_air']
    cp_water = cell['cp_water']
    k_air = cell['k_air']
    k_water = cell['k_water']
    k, kappa = get_k_and_kappa(T_old, Sfrac, Lfrac, cp_air, cp_water, k_air, k_water)

    residual = np.zeros_like(x)
    # Surface temperature equation (residual)
    # residual[0] = k[0] *  - (Q - epsilon * sigma * x[0] ** 4)
    residual[0] = k[0] * ((x[0] - x[1]) / dz) - (Q - epsilon * sigma * x[0] ** 4)
    # Calculate the temperature profile for the first 10 layers
    idx = np.arange(1, len(x) - 1)

    residual[idx] = (
        cell["firn_temperature"][idx]
        - x[idx]
        + dt * kappa[idx] * (x[idx + 1] - 2 * x[idx] + x[idx - 1]) / dz**2
    )
    residual[-1] = (
        cell["firn_temperature"][len(x) - 1]
        - x[len(x) - 1]
        + dt * (kappa[len(x) - 1]) * (-x[len(x) - 1] + x[len(x) - 2]) / dz**2
    )
    # print(f"Residual for T_sfc = {x}: {residual}")

    return residual




def propagate_temperature(cell, dz, dt, T_bc_top, N=10):
    """
    The solution of the heat equation involves a highly nonlinear part in the top N layers, which is driven by the
    surface energy balance, and a linear part, which is the diffusion of heat through the rest of the firn column.
    This function handles the linear part of this calculation.

    Parameters
    ----------
    cell
    dz
    dt
    T_bc_top
    N

    Returns
    -------

    """
    T_old = cell["firn_temperature"][N:]
    Sfrac = cell['Sfrac'][N:]
    Lfrac = cell['Lfrac'][N:]
    cp_air = cell['cp_air']
    cp_water = cell['cp_water']
    k_air = cell['k_air']
    k_water = cell['k_water']
    k, kappa = get_k_and_kappa(T_old, Sfrac, Lfrac, cp_air, cp_water, k_air, k_water)

    total_len = np.shape(cell['firn_temperature'])[0]  # Total number of layers in the firn column
    n = total_len - N  # Number of layers below the nonlinear region

    # Initialize diagonals and RHS
    A = np.zeros(n - 1, dtype=np.float64)
    B = np.zeros(n, dtype=np.float64)
    C = np.zeros(n - 1, dtype=np.float64)
    D = np.zeros(n, dtype=np.float64)

    factor = np.float64(dt / dz**2)

    # First row: connect to top nonlinear region
    i = 0
    alpha = factor * kappa[i]
    B[i] = 1 + 2 * alpha
    C[i] = -alpha
    D[i] = T_old[i] + alpha * T_bc_top

    # Interior rows
    for i in np.arange(1, n - 1):
        alpha = factor * kappa[i]
        A[i - 1] = -alpha
        B[i] = 1 + 2 * alpha
        C[i] = -alpha
        D[i] = T_old[i]

    # Last row: Neumann BC using backward difference
    i = n - 1
    alpha = factor * kappa[i]
    A[i - 1] = -alpha
    B[i] = 1 + alpha
    D[i] = T_old[i]

    # Assemble banded matrix
    # ab = np.zeros((3, n))
    # ab[0, 1:] = C
    # ab[1, :] = B
    # ab[2, :-1] = A
    # T_new = solve_banded((1, 1), ab, D)
    T_new = solve_tridiagonal(A, B, C, D)  
    return T_new


def solve_tridiagonal(a, b, c, d):
    """
    a: sub-diagonal (len n-1), A[i, i-1]
    b: main diagonal (len n), A[i, i]
    c: super-diagonal (len n-1), A[i, i+1]
    d: RHS (len n)
    """
    n = np.shape(d)[0]
    # Copy to avoid modifying input arrays
    ac, bc, cc, dc = map(np.copy, (a, b, c, d))
    # Forward elimination
    for i in range(1, n):
        m = ac[i - 1] / bc[i - 1]
        bc[i] -= m * cc[i - 1]
        dc[i] -= m * dc[i - 1]

    # Back substitution
    x = np.zeros(n)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]

    return x
    

def heateqn_lid(
    x, cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind, k_lid, Sfrac_lid
):
    """
    Solve the heat equation for the frozen lid, similarly to the calculation for the firn column.

    Parameters
    ----------
    x : array_like, float, dimension(cell.vert_grid)
        initial estimate of the firn column temperature [K]
    cell : numpy structured array
        Element of the model grid we are operating on.
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
    cp_ice = 1000 * (0.00716 * cell["lid_temperature"] + 0.138)
    cp = Sfrac_lid * cp_ice + (1 - Sfrac_lid) * cell["cp_air"]
    kappa = k_lid / (cp * cell["rho_ice"])
    epsilon = 0.98
    sigma = 5.670374 * 10**-8
    Q = sfc_flux(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["lake_depth"],
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
        x[0],
    )
    output = np.zeros(cell["vert_grid_lid"])
    output[0] = k_lid * (x[0] - x[1]) / dz - (Q - epsilon * sigma * x[0] ** 4)
    idx = np.arange(1, cell["vert_grid_lid"] - 1)
    output[idx] = (
        cell["lid_temperature"][idx]
        - x[idx]
        + dt * (kappa[idx] / dz**2) * (x[idx + 1] - 2 * x[idx] + x[idx - 1])
    )
    output[-1] = x[cell["vert_grid_lid"] - 1] - 273.15
    return output
