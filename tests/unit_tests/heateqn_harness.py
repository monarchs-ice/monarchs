from matplotlib import pyplot as plt
import numpy as np
from monarchs.physics.heateqn import heateqn
from monarchs.physics.surface_fluxes import sfc_flux
from scipy.optimize import root
from monarchs.core.initial_conditions import rho_init_emp

def setup_cell(vp=500, scaling_factor=0.05):
    firn_depth = 35
    T_init = np.linspace(253.15, 263.15, vp)[::-1]
    vertical_profile = np.linspace(0, firn_depth, vp)
    rho_init = rho_init_emp(vertical_profile, 500, 37)
    Sfrac = rho_init/917
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

    }
    # However, in practice, we only actually want the top ~5% of the profile.
    # So slice the cell attributes to this length.
    N = int(np.floor(vp * scaling_factor))
    for key in ['firn_temperature', 'Sfrac', 'Lfrac']:
        cell[key] = cell[key][:N]
    return cell, dz, vertical_profile
# --- test harness ---
def run_test(vp, scaling_factor=0.05):
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
    cell, dz, vertical_profile = setup_cell(vp, scaling_factor)


    T_init = cell['firn_temperature'].copy()
    # call solver
    sol = root(
        heateqn,
        T_init,
        args=(cell, 800, 800, 267, 1000, 265, 5, dz, dt),
        method="hybr", tol=1e-6
    )

    # Find the point 0.5m below the surface
    below_val = np.where(vertical_profile >= 0.1)[0][0]
    print('Vertical point at 10cm down is', below_val)
    if sol.success:
        T_sol = sol.x
    #     print(f"dz={dz:5.3f} m -> surface T = {T_sol[0]:.2f} K, 0.1m T = {T_sol[below_val]:.2f} K")
    # else:
    #     print(f"dz={dz:5.3f} m -> solver failed: {sol.message}")
    return T_sol, vertical_profile

if __name__ == "__main__":
    scaling_factor = 0.01
    spacing = np.arange(500, 10500, 500)
    colours = plt.cm.viridis(np.linspace(0, 1, len(spacing)))
    for k, dz in enumerate(spacing):
        T_sol, vertical_profile = run_test(dz, scaling_factor=scaling_factor)
        plt.plot(T_sol, vertical_profile[:int(np.floor(dz*scaling_factor))],
                 label=f"dz={3500/dz:.2f} cm", color=colours[k])
    plt.gca().invert_yaxis()
    plt.xlabel("Temperature (K)")
    plt.ylabel("Depth (m)")
    plt.ylim(0.05, -0.05)
    plt.xlim(282, 290)
    plt.legend()
    plt.grid(True)
