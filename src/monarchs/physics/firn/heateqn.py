"""
Firn column heat equation functions. This contains the residual function,
a function to calculate the Jacobian, and the scaffolding needed to
actually solve the system.

The firn column is basically a 1D diffusion problem with a fixed bottom
boundary and a nonlinear surface boundary condition, driven by the atmosphere.

We use the previous timestep state as the initial guess and for the
material properties in the column. This means that our system is entirely
linear except at the surface, which means we can get away with a tridiagonal
solver since we only have a single nonlinear row and therefore a tridiagonal
Jacobian. We can determine this analytically for all terms except dQ/dT,
which we obtain by finite difference.

Previously, we used MINPACK (via either Scipy or a bespoke Numba-compatible
library) to solve this. However, this was bad architecturally (required
two incompatible paths to make it actually work with the two versions)
and slower than this version since we calculate a tridiagonal Jacobian
rather than a dense one, and exploit a fast tridiagonal solver.
"""

import numpy as np
from monarchs.core.kernels import kernel
from monarchs.physics.surface_fluxes import sfc_flux
from monarchs.physics import material_properties
from monarchs.physics.solver import newton_tridiagonal, DQ_DT_STEP
from monarchs.physics.constants import emissivity, stefan_boltzmann


@kernel()
def heateqn_firn(x, cell, met_data, dz, dt, fixed_sfc=False):
    """
    Function called by the Newtwon-Raphson solver.

    This evolves the firn column temperature for a given timestep dt,
    typically 3600 seconds (1 hour).

    Parameters
    ----------
    x: array_like, float
        Initial estimate from the previous timestep of the firn temperature
    cell: array_like, float
        NumPy structured array containing this grid cell's properties.
    met_data: array_like, float
        NumPy structured array containing met data for this grid cell.
    dz: float
        Layer thickness [m].
    dt: int
        Timestep [s].
    fixed_sfc: bool, optional
        If True, the surface is melting: replace the atmosphere-forced surface
        row with a Dirichlet constraint pinning x[0] to 273.15 K. The interior
        is unchanged, so the whole system becomes linear. Default False.

    Returns
    -------
    residual: array_like, float
        Residual of the firn heat equation at ``x``. From this we can
        determine the temperature column.
    """
    T_old = cell["firn_temperature"]
    Sfrac = cell["Sfrac"]
    Lfrac = cell["Lfrac"]

    k, kappa = material_properties.k_and_kappa(T_old, Sfrac, Lfrac)

    residual = np.zeros_like(x)
    # fixed surface - always 273.15 so just subtract this for the residual
    if fixed_sfc:
        residual[0] = x[0] - 273.15
    else:
        # This is the part forced by the atmosphere, so is a nonlinear
        # upper boundary condition
        Q = sfc_flux(cell, met_data, x[0])
        residual[0] = k[0] * ((x[0] - x[1]) / dz) - (
            Q - emissivity * stefan_boltzmann * x[0] ** 4
        )
    idx = np.arange(1, len(x) - 1)

    # inner columns, linear in x, diffusion
    residual[idx] = (
        cell["firn_temperature"][idx]
        - x[idx]
        + dt * kappa[idx] * (x[idx + 1] - 2 * x[idx] + x[idx - 1]) / dz**2
    )

    # ghost-cell lower boundary condition
    residual[-1] = (
        cell["firn_temperature"][len(x) - 1]
        - x[len(x) - 1]
        + dt * (kappa[len(x) - 1]) * (-x[len(x) - 1] + x[len(x) - 2]) / dz**2
    )
    return residual


@kernel()
def newton_jacobian_firn(x, kappa, k0, dz, dt, dq_val, fixed_sfc=False):
    """
    Assemble the Jacobian J = dF/dT of the firn heat equation. Since
    the temperature of a given point x only depends on x-1 and x+1,
    (with the exception of the surface) we can use a tridiagonal matrix and solver.

    For the surface, the x-1 dependency is removed, but we have forcing
    due to the surface energy balance. This is determined by
    d(Q - epsilon * sigma * T^4)/dT. The dQ/dT part of this is determined
    via finite-difference, similarly to how the *entire* Jacobian is calculated
    when using MINPACK.

    Parameters
    ----------
    x : array_like, float
        Current Newton iterate (length n) [K]; only x[0] is used.
    kappa : array_like, float
        Thermal diffusivity at T_old (length n) [m^2 s^-1].
    k0 : float
        Surface thermal conductivity at T_old [W m^-1 K^-1].
    dz : float
        Layer thickness [m].
    dt : float
        Timestep [s].
    dq_val : float
        dQ/dT at x[0] [W m^-2 K^-1].
    fixed_sfc : bool, optional
        If True, match ``heateqn_firn``'s Dirichlet surface row (x[0] pinned):
        the surface diagonal is 1 with no coupling to x[1]. Default False.

    Returns
    -------
    a, b, c : array_like, float
        Sub/main/super diagonals of the Jacobian.
    """
    n = x.shape[0]
    a = np.empty(n - 1)
    b = np.empty(n)
    c = np.empty(n - 1)
    fac = dt / (dz * dz)  # dt/dz^2

    # surface - no ``a`` term as top of column so nothing above
    # fixed surface so the row is just 1.0 * x[0] = 273.15
    if fixed_sfc:
        b[0] = 1.0
        c[0] = 0.0
    # driven surface - need to calculate via derivative of Q - epsilon sigma T^4
    # w.r.t. T.
    # = dQ/dT - 4 epsilon sigma T^3
    else:
        b[0] = k0 / dz - (dq_val - 4.0 * emissivity * stefan_boltzmann * x[0] ** 3)
        c[0] = -k0 / dz

    # interior rows - linear in x, diffusion calculation
    for i in range(1, n - 1):
        alpha = fac * kappa[i]
        a[i - 1] = alpha
        b[i] = -1.0 - 2.0 * alpha
        c[i] = alpha

    # bottom row - ghost cell boundary
    i = n - 1
    alpha = fac * kappa[i]
    a[i - 1] = alpha
    b[i] = -1.0 - alpha
    return a, b, c


@kernel()
def _firn_residual(x, args):
    """
    Argument packer for the residual calculation. Packing it like this
    lets us make the solver itself generic by just taking a tuple
    of arguments ``args`` as input, so we can use it for both
    firn and lid (which have different argument lengths).
    """
    cell, met_data, dz, dt, kappa, k0, fixed_sfc = args
    return heateqn_firn(x, cell, met_data, dz, dt, fixed_sfc)


@kernel()
def _firn_jacobian(x, args):
    """
    Argument packer for the Jacobian calculation. As with ``_firn_residual``,
    constructing it like this with a wrapper function lets the solever take
    arbitrary number of arguments packed into a single tuple ``args``.
    """
    cell, met_data, dz, dt, kappa, k0, fixed_sfc = args
    if fixed_sfc:
        dq = 0.0
    else:
        q = sfc_flux(cell, met_data, x[0])
        q_p = sfc_flux(cell, met_data, x[0] + DQ_DT_STEP)
        dq = (q_p - q) / DQ_DT_STEP
    return newton_jacobian_firn(x, kappa, k0, dz, dt, dq, fixed_sfc)


@kernel()
def firn_heateqn_solver(cell, met_data, dt, dz, fixed_sfc=False):
    """
    Solve the full-column firn heat equation via the ``newton_tridiagonal``
    driver: it supplies the firn residual and Jacobian, the driver runs the
    Newton loop (one Thomas solve per iteration).

    Called in <firn_column>, <timestep>, <lake.formation>. Runs identically
    on the pure-Python and Numba backends.

    The residual F(T) = 0 is ``heateqn_firn`` (implicit diffusion with
    properties lagged at T_old, coupled to the surface energy balance through
    Q(T0)); its Jacobian is exact and tridiagonal apart from dQ/dT, taken by
    finite difference. Convergence is declared when max |F| falls below a
    tolerance scaled from SOLVER_TOL. Because the system is near-linear (only
    the scalar surface term is nonlinear), the full Newton step converges from
    the previous timestep's profile without a line search - verified over a
    multi-year run with lake/lid/combine events.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    met_data : numpy structured array
        Meteorological data for the current timestep (LW_down, SW_down,
        temperature, surf_pressure, dew_point_temperature, wind).
    dt : float
        Timestep [s].
    dz : float
        Layer thickness [m].
    fixed_sfc : bool, optional
        If True, the surface is melting: the surface row becomes a Dirichlet
        constraint pinning x[0] to 273.15 K, which makes the whole system
        linear so Newton converges in a single step. Default False.

    Returns
    -------
    T : array_like, float, dimension(vert_grid)
        Temperature profile, rounded to 8 decimal places [K].
    success : bool
        True if the iteration converged.
    n_iter : int
        Number of Newton iterations used (a single step for the fixed-surface
        path, since that system is linear).
    """
    T_old = cell["firn_temperature"]
    # k and kappa are lagged at T_old (see module docstring), so compute once
    # and pass through to the Jacobian via args.
    k, kappa = material_properties.k_and_kappa(T_old, cell["Sfrac"], cell["Lfrac"])
    args = (cell, met_data, dz, dt, kappa, k[0], fixed_sfc)
    return newton_tridiagonal(_firn_residual, _firn_jacobian, T_old, args)
