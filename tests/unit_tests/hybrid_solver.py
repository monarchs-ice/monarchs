from monarchs.physics.heateqn import propagate_temperature, get_k_and_kappa
from matplotlib import pyplot as plt
import numpy as np
from monarchs.physics.surface_fluxes import sfc_flux
from monarchs.core.initial_conditions import rho_init_emp


def setup_cell(vp=500):
    firn_depth = 35
    T_init = np.linspace(253.15, 263.15, vp)[::-1]
    vertical_profile = np.linspace(0, firn_depth, vp)
    rho_init = rho_init_emp(vertical_profile, 500, 37)
    Sfrac = rho_init / 917
    Lfrac = np.zeros_like(Sfrac)
    dz = firn_depth / vp
    # cell dict with constants
    cell = {
        "firn_temperature": T_init.copy(),
        "Sfrac": Sfrac,
        "Lfrac": Lfrac,
        "cp_air": 1004,
        "cp_water": 4217,
        "k_air": 0.024,
        "k_water": 0.5818,
        "melt": False,
        "exposed_water": False,
        "lid": False,
        "lake": False,
        "lake_depth": 0.0,
        "rho": rho_init,
    }
    return cell, dz, vertical_profile


# --- test harness ---
def run_test(vp):
    """
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
    epsilon = 0.98,
    sigma = 5.670374e-8,
    """

    dt = 3600
    cell, dz, vertical_profile = setup_cell(vp)
    T_init = cell["firn_temperature"].copy()
    # call solver
    sol = hybrid_solver(cell, dz, dt, 800, 800, 267, 1000, 265, 5)

    # Find the point 0.5m below the surface
    below_val = np.where(vertical_profile >= 0.1)[0][0]
    print("Vertical point at 10cm down is", below_val)
    print("Solution is", sol[0:10])


def hybrid_solver(
    cell,
    dz,
    dt,
    LW_in,
    SW_in,
    T_air,
    p_air,
    T_dp,
    wind,
    N_top=10,
    max_iter=50,
    tol=1e-6,
):
    """
    Hybrid solver for firn heat equation.

    Top N layers: nonlinear (surface BC)
    Bottom layers: linear implicit diffusion

    Returns
    -------
    T_full : np.array
        Temperature profile for the entire firn column
    """

    # Extract variables
    T_old = cell["firn_temperature"].copy()
    cp_air = cell["cp_air"]
    cp_water = cell["cp_water"]
    k_air = cell["k_air"]
    k_water = cell["k_water"]
    cp_ice = 2009  # J/kg/K

    rho_cp = (
        cell["rho"] * cell["Sfrac"] * cp_ice
        + (1.0 - cell["Sfrac"] - cell["Lfrac"]) * cp_air
        + cell["Lfrac"] * cp_water
    )
    # assumed array of length len(T_old)
    Sfrac = cell["Sfrac"]
    Lfrac = cell["Lfrac"]
    cp_air = cell["cp_air"]
    cp_water = cell["cp_water"]
    k_air = cell["k_air"]
    k_water = cell["k_water"]

    N = len(T_old)
    T_top = T_old[:N_top].copy()  # initial guess for top N layers
    T_bottom_old = T_old[N_top:]
    n_bottom = N - N_top

    k, kappa = get_k_and_kappa(
        T_old, Sfrac, Lfrac, cp_air, cp_water, k_air, k_water
    )
    k_top = k[:N_top]
    kappa_bottom = kappa[N_top:]

    # --- Solve top N layers with Newton iteration ---
    for it in range(max_iter):
        residual = np.zeros(N_top)
        jac_diag = np.zeros(N_top)

        # Compute residuals and Jacobian for top layers
        for i in range(N_top):
            if i == 0:
                # Surface layer with nonlinear BC
                flux_down = k_top[i] * (T_top[i] - T_top[i + 1]) / dz
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
                    T_top[0],
                )
                residual[i] = (
                    rho_cp[i] / dt * (T_top[i] - T_old[i])
                    - (Q - 5.670374e-8 * T_top[i] ** 4)
                    - flux_down
                )
                jac_diag[i] = (
                    rho_cp[i] / dt
                    + 4 * 5.670374e-8 * T_top[i] ** 3
                    + k_top[i] / dz
                )
            elif i < N_top - 1:
                # Interior top layers
                flux_up = k_top[i - 1] * (T_top[i] - T_top[i - 1]) / dz
                flux_down = k_top[i] * (T_top[i] - T_top[i + 1]) / dz
                residual[i] = rho_cp[i] / dt * (T_top[i] - T_old[i]) - (
                    -flux_up - flux_down
                )
                jac_diag[i] = (
                    rho_cp[i] / dt + k_top[i - 1] / dz + k_top[i] / dz
                )
            else:
                # Bottom of top-N region
                flux_up = k_top[i - 1] * (T_top[i] - T_top[i - 1]) / dz
                flux_down = (
                    kappa_bottom[0] * (T_bottom_old[0] - T_top[i]) / dz
                )  # link to bottom
                residual[i] = rho_cp[i] / dt * (T_top[i] - T_old[i]) - (
                    -flux_up - flux_down
                )
                jac_diag[i] = (
                    rho_cp[i] / dt + k_top[i - 1] / dz + kappa_bottom[0] / dz
                )

        # Newton update
        delta = residual / jac_diag
        T_top -= delta

        if np.max(np.abs(delta)) < tol:
            break
        if it == max_iter - 1:
            print("Warning: Newton did not converge for top layers")

    # --- Propagate bottom layers with implicit solver ---
    T_bottom_new = propagate_temperature(cell, dz, dt, T_top[-1])

    # Combine
    T_full = np.zeros(N)
    T_full[:N_top] = T_top
    T_full[N_top:] = T_bottom_new

    return T_full


if __name__ == "__main__":
    for dz in [500, 2500]:
        run_test(dz)
