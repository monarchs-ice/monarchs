"""
Lake surface energy balance equations, and scaffolding around
the solver so that we can call it in lake.formation and lake.development.
"""

import numpy as np
from monarchs.core.kernels import kernel
from monarchs.physics import surface_fluxes, material_properties
from monarchs.physics.solver import newton_scalar
from monarchs.physics.constants import (
    emissivity,
    stefan_boltzmann,
    rho_water,
    cp_water,
    J,
)


@kernel()
def lake_formation_eqn(x, args):
    """
    Residual function used to solve the surface energy balance when
    we have a lake forming on the surface.

    Solved by lake_seb_solver with formation=False.

    Parameters
    ----------
    x : float
        Current estimate of the lake surface temperature [K].
    args : tuple
        ``(cell, met_data, k, T1, dz_firn)``, where k is the firn thermal
        conductivity, T1 the second-layer firn temperature, and dz_firn the
        firn layer thickness.
    """
    cell, met_data, k, T1, dz_firn = args

    Q = surface_fluxes.sfc_flux(cell, met_data, x)

    return -emissivity * stefan_boltzmann * x**4 + Q - k * (x - T1) / dz_firn


@kernel()
def lake_development_eqn(x, args):
    """
    Residual function used to solve the surface energy balance when
    we have a lake on top of the firn.

    Solved by lake_seb_solver with formation=True.

    Parameters
    ----------
    x : float
        Current estimate of the lake surface temperature [K].
    args : tuple
        ``(cell, met_data, T_core)``, where T_core is the mid-column lake
        temperature [K].
    """
    cell, met_data, T_core = args

    Q = surface_fluxes.sfc_flux(cell, met_data, x)

    return (
        -emissivity * stefan_boltzmann * x**4
        + Q
        + np.sign(T_core - x) * rho_water * cp_water * J * abs(T_core - x) ** (4 / 3)
    )


@kernel()
def lake_seb_solver(cell, met_data, formation=False):
    """
    Solve the surface energy balance for conditions where we have
    either exposed water at the surface, or a full lake.

    Called in lake.formation (formation=True) and lake.development.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    met_data : numpy structured array
        Meteorological data for the current timestep.
    formation : bool, optional
        If True, solve the lake *formation* equation (conduction into the
        firn), else the lake *development* equation (turbulent mixing
        with the lake core). Default False.

    Returns
    -------
    sol : float
        Calculated lake surface temperature, rounded to 8 decimal places [K].
    success : bool
        True if the iteration converged.
    n_iter : int
        Number of Newton-Raphson iterations.
    """
    if formation:
        k = material_properties.k_mixture(
            cell["firn_temperature"][0], cell["Sfrac"][0], cell["Lfrac"][0]
        )
        args = (
            cell,
            met_data,
            k,
            cell["firn_temperature"][1],
            cell["firn_depth"] / cell["vert_grid"],
        )
        x, success, n_iter = newton_scalar(
            lake_formation_eqn, met_data["temperature"], args
        )
    else:
        T_core = cell["lake_temperature"][cell["vert_grid_lake"] // 2]
        args = (cell, met_data, T_core)
        x, success, n_iter = newton_scalar(
            lake_development_eqn, cell["lake_temperature"][0], args
        )

    return np.around(x, decimals=8), success, n_iter
