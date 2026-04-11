"""
solver_newton.py
================
Newton–tridiagonal solver for the firn heat equation and surface energy balance.

This solver exploits the 1-D stencil structure of the heat equation:
the Jacobian for the top nonlinear block is tridiagonal, so each Newton
iteration is O(N) rather than O(N²) dense work. It is compatible with Numba
@njit for fast execution and requires no flat-array packing machinery.

Algorithm
---------
1. Scalar or nonlinear Newton iteration on the top N layers (N=50 by default).
   - Build tridiagonal Jacobian.
   - Solve with Thomas algorithm.
   - Optional backtracking line search.
2. Solve the remaining deep column with implicit tridiagonal propagation
   using the converged surface temperature as boundary condition.

References
----------
See benchmarks/solvers/solver_newton_tridiag.py for performance evaluation.
"""

import numpy as np
from numba import njit

from monarchs.physics.constants import (
    cp_air,
    cp_water,
    emissivity,
    k_air,
    k_water,
    rho_air,
    rho_ice,
    rho_water,
    stefan_boltzmann,
)

_N_NONLINEAR = 50

# Module-level reference to surface_fluxes (will be set at import or warmup time)
_surface_fluxes = None


@njit(cache=False)
def _get_k_and_kappa(T, sfrac, lfrac):
    """
    Compute thermal conductivity k and diffusivity kappa.

    Uses volumetric heat capacity C_vol [J m^-3 K^-1] to match the
    production heateqn_nb implementation exactly.

    Parameters
    ----------
    T : array_like, float
        Temperature profile [K]
    sfrac : array_like, float
        Solid fraction [0, 1]
    lfrac : array_like, float
        Liquid fraction [0, 1]

    Returns
    -------
    k : array_like, float
        Thermal conductivity [W m^-1 K^-1]
    kappa : array_like, float
        Thermal diffusivity [m^2 s^-1]
    """
    air_frac = 1.0 - sfrac - lfrac
    k_ice = np.empty(T.shape[0], dtype=np.float64)
    for i in range(T.shape[0]):
        if T[i] < 273.15:
            k_ice[i] = 1000.0 * (2.24e-03 + 5.975e-06 * ((273.15 - T[i]) ** 1.156))
        else:
            k_ice[i] = 2.24

    k = sfrac * k_ice + air_frac * k_air + lfrac * k_water
    cp_ice = 7.16 * T + 138.0
    cv_ice = sfrac * rho_ice * cp_ice
    cv_water = lfrac * rho_water * cp_water
    cv_air = air_frac * rho_air * cp_air
    c_vol = cv_ice + cv_water + cv_air
    kappa = k / c_vol
    return k, kappa


@njit(cache=False)
def _thomas_inplace(a, b, c, d, b_work, d_work, x_out):
    """
    Solve tridiagonal system Ax = d using the Thomas algorithm.

    Uses caller-provided work arrays to avoid allocations in Newton iteration loop.

    Parameters
    ----------
    a : array_like, float
        Sub-diagonal (length n-1)
    b : array_like, float
        Main diagonal (length n)
    c : array_like, float
        Super-diagonal (length n-1)
    d : array_like, float
        RHS (length n)
    b_work : array_like, float
        Work array for modified main diagonal (length n, modified in-place)
    d_work : array_like, float
        Work array for modified RHS (length n, modified in-place)
    x_out : array_like, float
        Solution array (length n, filled in-place)
    """
    n = d.shape[0]

    for i in range(n):
        b_work[i] = b[i]
        d_work[i] = d[i]

    for i in range(1, n):
        m = a[i - 1] / b_work[i - 1]
        b_work[i] -= m * c[i - 1]
        d_work[i] -= m * d_work[i - 1]

    x_out[n - 1] = d_work[n - 1] / b_work[n - 1]
    for i in range(n - 2, -1, -1):
        x_out[i] = (d_work[i] - c[i] * x_out[i + 1]) / b_work[i]


@njit(cache=False)
def _assemble_newton_system_inplace(x, T_old, kappa, k0, dz, dt, q_val, dq_val, a, b, c, rhs):
    """
    Assemble Newton linearized system for the top nonlinear block.

    Residual at layer i:
    - i=0 (surface): k0*(x[0]-x[1])/dz - (Q(x[0]) - ε*σ*x[0]^4)
    - i>0: T_old[i] - x[i] + (dt/dz^2)*κ[i]*(x[i+1] - 2*x[i] + x[i-1])

    Jacobian is tridiagonal (except surface row which couples to i=1 only).

    Parameters
    ----------
    x : array_like, float
        Current Newton iterate (length n)
    T_old : array_like, float
        Temperature from previous timestep (length n)
    kappa : array_like, float
        Thermal diffusivity (length n)
    k0 : float
        Thermal conductivity at surface [W m^-1 K^-1]
    dz : float
        Layer thickness [m]
    dt : float
        Timestep [s]
    q_val : float
        Surface heat flux Q(x[0]) [W m^-2]
    dq_val : float
        dQ/dT|(x[0]) [W m^-2 K^-1]
    a, b, c, rhs : arrays, float
        Tridiagonal system components (modified in-place)

    Returns
    -------
    max_abs_resid : float
        Maximum absolute residual, used for convergence check
    """
    n = x.shape[0]
    fac = dt / (dz * dz)

    eps = emissivity
    sigma = stefan_boltzmann

    # Surface row: nonlinear BC through q(T0) and sigma*T0^4.
    f0 = k0 * (x[0] - x[1]) / dz - (q_val - eps * sigma * x[0] ** 4)
    b[0] = k0 / dz - (dq_val - 4.0 * eps * sigma * x[0] ** 3)
    c[0] = -k0 / dz
    rhs[0] = -f0

    max_abs_resid = abs(f0)

    # Interior rows: standard implicit diffusion stencil
    for i in range(1, n - 1):
        alpha = fac * kappa[i]
        fi = T_old[i] - x[i] + alpha * (x[i + 1] - 2.0 * x[i] + x[i - 1])
        a[i - 1] = alpha
        b[i] = -1.0 - 2.0 * alpha
        c[i] = alpha
        rhs[i] = -fi
        if abs(fi) > max_abs_resid:
            max_abs_resid = abs(fi)

    # Bottom row: Neumann zero-flux boundary
    i = n - 1
    alpha = fac * kappa[i]
    fi = T_old[i] - x[i] + alpha * (-x[i] + x[i - 1])
    a[i - 1] = alpha
    b[i] = -1.0 - alpha
    rhs[i] = -fi
    if abs(fi) > max_abs_resid:
        max_abs_resid = abs(fi)

    return max_abs_resid


@njit(cache=False)
def _max_abs_residual(x, T_old, kappa, k0, dz, dt, q_val):
    """
    Evaluate maximum absolute residual for convergence check and line search.

    Parameters
    ----------
    x : array_like, float
        Temperature iterate (length n)
    T_old : array_like, float
        Previous temperature (length n)
    kappa : array_like, float
        Thermal diffusivity (length n)
    k0 : float
        Surface thermal conductivity [W m^-1 K^-1]
    dz : float
        Layer thickness [m]
    dt : float
        Timestep [s]
    q_val : float
        Surface heat flux Q(x[0]) [W m^-2]

    Returns
    -------
    max_abs_resid : float
        Maximum absolute residual across all layers
    """
    n = x.shape[0]
    fac = dt / (dz * dz)

    eps = emissivity
    sigma = stefan_boltzmann

    max_abs_resid = abs(
        k0 * (x[0] - x[1]) / dz - (q_val - eps * sigma * x[0] ** 4)
    )

    for i in range(1, n - 1):
        alpha = fac * kappa[i]
        fi = T_old[i] - x[i] + alpha * (x[i + 1] - 2.0 * x[i] + x[i - 1])
        if abs(fi) > max_abs_resid:
            max_abs_resid = abs(fi)

    i = n - 1
    alpha = fac * kappa[i]
    fi = T_old[i] - x[i] + alpha * (-x[i] + x[i - 1])
    if abs(fi) > max_abs_resid:
        max_abs_resid = abs(fi)

    return max_abs_resid


def solve_firn_heateqn(
    cell,
    met_data,
    dt,
    dz,
    fixed_sfc=False,
    solver_method=None,
    n_nonlinear=None,
    tol=1e-10,
    maxiter=12,
    line_search=True,
    line_search_maxiter=6,
    line_search_shrink=0.5,
):
    """
    Solve the 1-D firn heat equation using full-column Newton iterations with
    tridiagonal Jacobian and backtracking line search.

    Uses Newton's method directly on all layers (not splitting into nonlinear/linear
    regions) to exploit the tridiagonal structure of the Jacobian.

    Designed to be a drop-in replacement for the scipy/minpack solvers,
    with identical call signature (solver_method and n_nonlinear are ignored).

    Parameters
    ----------
    cell : structured array
        Grid cell containing firn_temperature, Sfrac, Lfrac, albedo, lid, lake
    met_data : structured array
        Meteorological data: LW_down, SW_down, temperature, surf_pressure,
        dew_point_temperature, wind
    dt : float
        Timestep [s]
    dz : float
        Layer thickness [m]
    fixed_sfc : bool, optional
        If True, use fixed surface temperature (273.15 K) instead of solving
        the nonlinear surface energy balance. Default False.
    solver_method : str, optional
        Ignored (accepted for API compatibility). Default None.
    n_nonlinear : int, optional
        Ignored (accepted for API compatibility; solver uses full column).
        Default None.
    tol : float, optional
        Newton convergence tolerance [K]. Default 1e-10.
    maxiter : int, optional
        Maximum Newton iterations. Default 12.
    line_search : bool, optional
        Enable backtracking line search. Default True.
    line_search_maxiter : int, optional
        Max line search subdivisions. Default 6.
    line_search_shrink : float, optional
        Line search shrink factor. Default 0.5.

    Returns
    -------
    T : array_like, float
        Temperature profile [K]
    infodict : dict
        Diagnostic info (always empty dict for API compatibility)
    ier : int
        Success flag (1 if converged, 0 otherwise)
    mesg : str
        Status message
    """
    # For API compatibility with scipy path
    infodict = {}

    # Ensure surface_fluxes is initialized
    if _surface_fluxes is None:
        raise RuntimeError(
            "surface_fluxes not initialized. Call warmup() before using solver."
        )

    # Handle fixed surface temperature case
    if fixed_sfc:
        # Even for fixed surface, use full-column Newton with fixed BC
        N = int(cell["vert_grid"])
        T_old = np.asarray(cell["firn_temperature"][:], dtype=np.float64)
        sfrac = np.asarray(cell["Sfrac"][:], dtype=np.float64)
        lfrac = np.asarray(cell["Lfrac"][:], dtype=np.float64)

        k, kappa = _get_k_and_kappa(T_old, sfrac, lfrac)
        x = T_old.copy()
        x[0] = 273.15  # Fix surface temperature

        # Preallocate Newton buffers
        a = np.empty(N - 1, dtype=np.float64)
        b = np.empty(N, dtype=np.float64)
        c = np.empty(N - 1, dtype=np.float64)
        rhs = np.empty(N, dtype=np.float64)
        b_work = np.empty(N, dtype=np.float64)
        d_work = np.empty(N, dtype=np.float64)
        delta = np.empty(N, dtype=np.float64)

        converged = False
        dT_fd = 1e-3

        # Scale tolerance
        resid_tol = max(float(tol), float(tol) * max(1.0, float(np.max(np.abs(T_old)))))

        # For fixed surface, we still run Newton but with q = 0 (no surface flux)
        # since T[0] is fixed
        for it in range(maxiter):
            # Build tridiagonal system for full column with T[0] fixed
            n = N
            fac = float(dt) / (dz * dz)
            eps = emissivity
            sigma = stefan_boltzmann

            # Surface row: T[0] = 273.15 (Dirichlet)
            f0 = x[0] - 273.15
            b[0] = 1.0
            if n > 1:
                c[0] = 0.0
            rhs[0] = -f0

            max_abs_resid = abs(f0)

            # Interior rows
            for i in range(1, n - 1):
                alpha = fac * kappa[i]
                fi = T_old[i] - x[i] + alpha * (x[i + 1] - 2.0 * x[i] + x[i - 1])
                a[i - 1] = alpha
                b[i] = -1.0 - 2.0 * alpha
                c[i] = alpha
                rhs[i] = -fi
                if abs(fi) > max_abs_resid:
                    max_abs_resid = abs(fi)

            # Bottom row: Neumann
            i = n - 1
            alpha = fac * kappa[i]
            fi = T_old[i] - x[i] + alpha * (-x[i] + x[i - 1])
            a[i - 1] = alpha
            b[i] = -1.0 - alpha
            rhs[i] = -fi
            if abs(fi) > max_abs_resid:
                max_abs_resid = abs(fi)

            # Solve for Newton step
            _thomas_inplace(a, b, c, rhs, b_work, d_work, delta)

            # Check convergence
            max_abs_step = float(np.max(np.abs(delta)))
            if max_abs_resid < resid_tol:
                converged = True
                break

            # Accept step
            x += delta

            # Convergence check
            if max_abs_step < tol or max_abs_resid < resid_tol:
                converged = True
                break

        # Ensure surface stays fixed
        x[0] = 273.15
        T = np.around(x, decimals=8)
        return T, infodict, (1 if converged else 0), "Fixed surface temperature (full-column Newton)"

    # Full-column Newton solve
    N = int(cell["vert_grid"])
    T_old = np.asarray(cell["firn_temperature"][:], dtype=np.float64)
    sfrac = np.asarray(cell["Sfrac"][:], dtype=np.float64)
    lfrac = np.asarray(cell["Lfrac"][:], dtype=np.float64)

    k, kappa = _get_k_and_kappa(T_old, sfrac, lfrac)
    x = T_old.copy()

    # Preallocate Newton buffers
    a = np.empty(N - 1, dtype=np.float64)
    b = np.empty(N, dtype=np.float64)
    c = np.empty(N - 1, dtype=np.float64)
    rhs = np.empty(N, dtype=np.float64)
    b_work = np.empty(N, dtype=np.float64)
    d_work = np.empty(N, dtype=np.float64)
    delta = np.empty(N, dtype=np.float64)

    # Scale tolerance by state magnitude
    resid_tol = max(float(tol), float(tol) * max(1.0, float(np.max(np.abs(T_old)))))

    converged = False
    used_iter = 0
    dT_fd = 1e-3
    failure_reason = "maxiter"

    for it in range(maxiter):
        used_iter = it + 1

        # Evaluate surface flux Q and finite-difference dQ/dT
        q = _surface_fluxes.sfc_flux(
            float(cell["albedo"]),
            bool(cell["lid"]),
            bool(cell["lake"]),
            float(met_data["LW_down"]),
            float(met_data["SW_down"]),
            float(met_data["temperature"]),
            float(met_data["surf_pressure"]),
            float(met_data["dew_point_temperature"]),
            float(met_data["wind"]),
            float(x[0]),
        )
        q_p = _surface_fluxes.sfc_flux(
            float(cell["albedo"]),
            bool(cell["lid"]),
            bool(cell["lake"]),
            float(met_data["LW_down"]),
            float(met_data["SW_down"]),
            float(met_data["temperature"]),
            float(met_data["surf_pressure"]),
            float(met_data["dew_point_temperature"]),
            float(met_data["wind"]),
            float(x[0] + dT_fd),
        )
        dq = (q_p - q) / dT_fd

        # Assemble tridiagonal Newton system for full column
        max_abs_resid = _assemble_newton_system_inplace(
            x, T_old, kappa, float(k[0]), float(dz), float(dt),
            float(q), float(dq), a, b, c, rhs,
        )
        _thomas_inplace(a, b, c, rhs, b_work, d_work, delta)

        # Check convergence before line search
        max_abs_step = float(np.max(np.abs(delta)))
        if max_abs_resid < resid_tol:
            converged = True
            failure_reason = "converged"
            break

        # Trial full Newton step
        x_trial = x + delta
        q_trial = _surface_fluxes.sfc_flux(
            float(cell["albedo"]),
            bool(cell["lid"]),
            bool(cell["lake"]),
            float(met_data["LW_down"]),
            float(met_data["SW_down"]),
            float(met_data["temperature"]),
            float(met_data["surf_pressure"]),
            float(met_data["dew_point_temperature"]),
            float(met_data["wind"]),
            float(x_trial[0]),
        )
        trial_resid = _max_abs_residual(
            x_trial, T_old, kappa, float(k[0]), float(dz), float(dt), float(q_trial),
        )
        best_alpha = 1.0
        best_resid = trial_resid

        # Backtracking line search if needed
        if line_search and trial_resid > max_abs_resid:
            alpha = 1.0
            for _ in range(line_search_maxiter):
                alpha *= line_search_shrink
                x_trial = x + alpha * delta
                q_trial = _surface_fluxes.sfc_flux(
                    float(cell["albedo"]),
                    bool(cell["lid"]),
                    bool(cell["lake"]),
                    float(met_data["LW_down"]),
                    float(met_data["SW_down"]),
                    float(met_data["temperature"]),
                    float(met_data["surf_pressure"]),
                    float(met_data["dew_point_temperature"]),
                    float(met_data["wind"]),
                    float(x_trial[0]),
                )
                trial_resid = _max_abs_residual(
                    x_trial, T_old, kappa, float(k[0]), float(dz), float(dt), float(q_trial),
                )
                if trial_resid < best_resid:
                    best_resid = trial_resid
                    best_alpha = alpha
                if trial_resid <= max_abs_resid:
                    break

        # Accept best step
        x += best_alpha * delta

        # Convergence checks
        accepted_step = best_alpha * max_abs_step
        tiny_step_resid_tol = max(1.0e-6, 1.0e3 * resid_tol)
        if best_resid < resid_tol:
            converged = True
            failure_reason = "converged"
            break
        if accepted_step < tol and best_resid < tiny_step_resid_tol:
            converged = True
            failure_reason = "tiny_step"
            break

    T = np.around(x, decimals=8)
    return T, infodict, (1 if converged else 0), failure_reason


def warmup():
    """
    Initialize module-level surface_fluxes reference.

    This must be called before solve_firn_heateqn() is used, to ensure that
    the surface_fluxes module is available for use by the njit-compiled solver.

    Typically called during model initialization or setup.
    """
    global _surface_fluxes
    if _surface_fluxes is None:
        from monarchs.physics import surface_fluxes
        _surface_fluxes = surface_fluxes

