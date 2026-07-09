"""
Lid surface energy balance equations, and scaffolding around
the solver so that we can call it in the virtual lid and lid
development stages.
"""

import numpy as np
from monarchs.core.kernels import kernel
from monarchs.physics import surface_fluxes
from monarchs.physics.solver import newton_scalar
from monarchs.physics.constants import emissivity, stefan_boltzmann


@kernel()
def sfc_energy_lid(x, args):
    """
    Residual of the full-lid surface energy balance [W m^-2]: surface flux
    and emission balanced against conduction into the first lid layer.
    Solved by lid_seb_solver when ``cell["v_lid"]`` is False.

    Parameters
    ----------
    x : float
        Current estimate of the lid surface temperature [K].
    args : tuple
        ``(cell, met_data, k_lid, sub_T, dz_lid)``, where sub_T is the first
        lid layer temperature [K] and dz_lid the lid layer thickness [m].
    """
    cell, met_data, k_lid, sub_T, dz_lid = args

    Q = surface_fluxes.sfc_flux(cell, met_data, x)

    return -emissivity * stefan_boltzmann * x**4 + Q - k_lid * (x - sub_T) / dz_lid


@kernel()
def sfc_energy_virtual_lid(x, args):
    """
    Residual of the virtual-lid surface energy balance [W m^-2]: surface flux
    and emission balanced against conduction through the combined thermal
    resistance of the virtual-lid ice plus the upper half-layer of lake
    water. Solved by lid_seb_solver when ``cell["v_lid"]`` is True.

    Parameters
    ----------
    x : float
        Current estimate of the virtual-lid surface temperature [K].
    args : tuple
        ``(cell, met_data, k_v_lid, total_thickness)``, where
        total_thickness = v_lid_depth + lake_depth / (vert_grid_lake / 2).
    """
    cell, met_data, k_v_lid, total_thickness = args

    Q = surface_fluxes.sfc_flux(cell, met_data, x)

    conduction = k_v_lid * (x - 273.15) / total_thickness
    return Q - emissivity * stefan_boltzmann * x**4 - conduction


@kernel()
def lid_seb_solver(cell, met_data, k_lid):
    """
    Solve the lid surface energy balance (full or virtual lid form) with the
    scalar Newton solver. Runs identically on both backends.

    Called in lid.initialise_lid and lid.virtual_lid.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    met_data : numpy structured array
        Meteorological data for the current timestep.
    k_lid : float
        Thermal conductivity of the (virtual) lid [W m^-1 K^-1].

    Returns
    -------
    sol : float
        Calculated lid surface temperature, rounded to 8 decimal places [K].
    success : bool
        True if the iteration converged.
    n_iter : int
        Number of Newton iterations used.
    """
    # Initial guess: current virtual-lid surface temperature (as before,
    # for both the full-lid and virtual-lid branches).
    x0 = cell["virtual_lid_temperature"]

    if cell["v_lid"]:
        # Combined thermal resistance: virtual-lid ice + half-lake water
        total_thickness = cell["v_lid_depth"] + cell["lake_depth"] / (
            cell["vert_grid_lake"] / 2
        )
        args = (cell, met_data, k_lid, total_thickness)
        x, success, n_iter = newton_scalar(sfc_energy_virtual_lid, x0, args)
    else:
        # sub_T: first lid layer, used as the sub-surface boundary condition
        args = (
            cell,
            met_data,
            k_lid,
            cell["lid_temperature"][0],
            cell["lid_depth"] / cell["vert_grid_lid"],
        )
        x, success, n_iter = newton_scalar(sfc_energy_lid, x0, args)

    return np.around(x, decimals=8), success, n_iter
