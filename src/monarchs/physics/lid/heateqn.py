"""
Frozen-lid heat equation: residual, Jacobian, and Newton driver.

The lid heat equation has the same structure as the firn one
(``firn/heateqn.py``) - implicit diffusion with lagged properties, coupled to
the surface energy balance - with two differences: an absorbed-solar source
term in the interior, and a *Dirichlet* bottom row pinning the lake-lid
boundary to 273.15 K. So the Jacobian is again exactly tridiagonal and each
Newton step is one Thomas solve.

- ``heateqn_lid``: the residual F(T), single definition of the lid system.
- ``newton_jacobian_lid``: the tridiagonal Jacobian J = dF/dT only.
- ``lid_heateqn_solver``: builds the lid residual/Jacobian and hands them to
  the shared ``solver.newton_tridiagonal`` driver (the same loop the firn
  solver uses).
"""

import numpy as np
from monarchs.core.kernels import kernel
from monarchs.physics.surface_fluxes import sfc_flux
from monarchs.physics import material_properties
from monarchs.physics.solver import newton_tridiagonal, DQ_DT_STEP
from monarchs.physics.constants import (
    cp_air,
    emissivity,
    rho_ice,
    stefan_boltzmann,
    sfc_absorbed_frac,
    tau_ice,
)


@kernel()
def heateqn_lid(x, cell, met_data, dt, dz, Sfrac_lid):
    """
    Residual of the frozen-lid heat equation, F(T) = 0.

    Parameters
    ----------
    x : array_like, float, dimension(cell.vert_grid_lid)
        Current estimate of the lid column temperature [K].
    cell : numpy structured array
        Element of the model grid we are operating on.
    met_data : numpy structured array
        Meteorological data for the current timestep.
    dt : int
        timestep duration [s]
    dz : float
        height of a single vertical layer of the frozen lid [m]
    Sfrac_lid : array_like, float, dimension(cell.vert_grid_lid)
        Solid fraction of the frozen lid.

    Returns
    -------
    output : array_like, float, dimension(cell.vert_grid_lid)
        Residual of the lid heat equation at ``x``.
    """
    cp_ice = material_properties.cp_ice(cell["lid_temperature"])
    cp = Sfrac_lid * cp_ice + (1 - Sfrac_lid) * cp_air
    k_lid = material_properties.k_ice(cell["lid_temperature"])
    kappa = k_lid / (cp * rho_ice)
    epsilon = emissivity
    sigma = stefan_boltzmann
    rho = rho_ice

    Q = sfc_flux(cell, met_data, x[0])
    sw_in = met_data["SW_down"]

    output = np.zeros(cell["vert_grid_lid"])
    output[0] = k_lid[0] * ((x[0] - x[1]) / dz) - (Q - epsilon * sigma * x[0] ** 4)
    idx = np.arange(1, cell["vert_grid_lid"] - 1)
    z_depth = idx * dz
    flux_in = (
        sw_in
        * (1 - cell["albedo"])
        * (1 - sfc_absorbed_frac)
        * np.exp(-tau_ice * z_depth)
    )
    flux_out = (
        sw_in
        * (1 - cell["albedo"])
        * (1 - sfc_absorbed_frac)
        * np.exp(-tau_ice * (z_depth + dz))
    )
    sw_absorbed_in_layer = flux_in - flux_out
    dT_solar = sw_absorbed_in_layer / (rho * cp[idx] * dz)

    output[idx] = (
        cell["lid_temperature"][idx]
        - x[idx]
        + dt * ((kappa[idx]) * (x[idx + 1] - 2 * x[idx] + x[idx - 1]) / dz**2)
        + dt * dT_solar
    )
    output[-1] = x[cell["vert_grid_lid"] - 1] - 273.15
    return output


@kernel()
def newton_jacobian_lid(x, kappa, k0, dz, dt, dq_val):
    """
    Assemble the tridiagonal Jacobian J = dF/dT of the lid heat equation,
    returning the sub/main/super diagonals (a, b, c). The residual F comes
    from ``heateqn_lid``. The absorbed-solar source is additive and
    state-independent, so it does not appear in the Jacobian.

    Rows match ``heateqn_lid``: surface energy balance (as the firn system),
    interior implicit diffusion, and a *Dirichlet* bottom row pinning the
    lake-lid boundary to 273.15 K. Parameters as ``firn.heateqn.newton_jacobian_firn``.
    """
    n = x.shape[0]
    a = np.empty(n - 1)
    b = np.empty(n)
    c = np.empty(n - 1)
    fac = dt / (dz * dz)

    # Surface row: identical form to the firn system.
    b[0] = k0 / dz - (dq_val - 4.0 * emissivity * stefan_boltzmann * x[0] ** 3)
    c[0] = -k0 / dz

    # Interior rows: implicit diffusion stencil (constant in x).
    for i in range(1, n - 1):
        alpha = fac * kappa[i]
        a[i - 1] = alpha
        b[i] = -1.0 - 2.0 * alpha
        c[i] = alpha

    # Bottom row: Dirichlet, lake-lid boundary at the melting point.
    a[n - 2] = 0.0
    b[n - 1] = 1.0
    return a, b, c


@kernel()
def _lid_residual(x, args):
    """Adapter binding the lid residual to the ``newton_tridiagonal`` driver."""
    cell, met_data, dz, dt, kappa, k0, Sfrac_lid = args
    return heateqn_lid(x, cell, met_data, dt, dz, Sfrac_lid)


@kernel()
def _lid_jacobian(x, args):
    """Adapter returning the lid tridiagonal Jacobian for the driver, taking
    the surface flux derivative dQ/dT by finite difference."""
    cell, met_data, dz, dt, kappa, k0, Sfrac_lid = args
    q = sfc_flux(cell, met_data, x[0])
    q_p = sfc_flux(cell, met_data, x[0] + DQ_DT_STEP)
    dq = (q_p - q) / DQ_DT_STEP
    return newton_jacobian_lid(x, kappa, k0, dz, dt, dq)


@kernel()
def lid_heateqn_solver(cell, met_data, dt, dz):
    """
    Solve the full-column frozen-lid heat equation via the ``newton_tridiagonal``
    driver: it supplies the lid residual and Jacobian, the driver runs the
    Newton loop (one Thomas solve per iteration). Runs identically on both
    backends.

    Called in lid.lid_development. The residual F(T) = 0 is ``heateqn_lid``.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    met_data : numpy structured array
        Meteorological data for the current timestep.
    dt : float
        Timestep [s].
    dz : float
        Lid layer thickness [m].

    Returns
    -------
    T : array_like, float, dimension(vert_grid_lid)
        Lid temperature profile, rounded to 8 decimal places [K].
    success : bool
        True if the iteration converged.
    n_iter : int
        Number of Newton iterations used.
    """
    T_old = cell["lid_temperature"]

    # Lagged thermal properties (solid ice); computed once and passed to the
    # Jacobian via args. Frozen lid is solid ice, so Sfrac_lid is all ones.
    k_lid = material_properties.k_ice(T_old)
    cp = material_properties.cp_ice(T_old)
    kappa = k_lid / (cp * rho_ice)
    Sfrac_lid = np.ones(len(T_old))

    args = (cell, met_data, dz, dt, kappa, k_lid[0], Sfrac_lid)
    return newton_tridiagonal(_lid_residual, _lid_jacobian, T_old, args)
