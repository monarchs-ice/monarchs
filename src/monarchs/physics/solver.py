"""
Solver architecture. This is basically a pure-Python
replacement for scipy.optimize.root, which is not available in
Numba mode, and for NumbaMinpack, which requires a very bespoke
architecture to make work. For MONARCHS, we can get away with
a tridiagonal solve for the vast majority of the column, with
Newton-Raphson at the surface.

The actual equations and the scaffolding needed to solve these live
in the relevant sections in physics.firn.heateqn, physics.lid.heateqn,
physics.lake.seb and physics.lid.seb.
"""

import numpy as np
from monarchs.core.kernels import kernel

# tolerances/max newton-raphson iterations
# TODO - could put these in a numerics section of the setup file?
SOLVER_TOL = 1e-10
SOLVER_MAXITER = 60
# finite-differencing step size for surface flux jacobians
DQ_DT_STEP = 1e-3
# tolerances/parameters for the Newton-Raphson solver used for the
# SEB
FTOL = 1e-11
XTOL = 1e-13
FD_STEP = 1e-6


@kernel()
def solve_tridiagonal(a, b, c, d):
    """
    Solve the tridiagonal system Ax = d with the Thomas algorithm.

    a: sub-diagonal (len n-1), A[i, i-1]
    b: main diagonal (len n), A[i, i]
    c: super-diagonal (len n-1), A[i, i+1]
    d: RHS (len n)

    Used both for the fixed-surface propagation and for
    each Newton-Raphson step in the surface-driven
    firn/lid heat equation.
    """

    n = np.shape(d)[0]
    # Copy to avoid modifying input arrays
    bc = b.copy()
    cc = c.copy()
    dc = d.copy()
    # Forward elimination
    for i in range(1, n):
        m = a[i - 1] / bc[i - 1]
        bc[i] -= m * cc[i - 1]
        dc[i] -= m * dc[i - 1]

    # Back substitution
    x = np.zeros(n)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]

    return x


@kernel()
def newton_scalar(f, x0, args):
    """
    Scalar Newton-Raphson iteration for the surface energy balances.


    We need to calculate the temperature of lakes and lids based on the
    surface energy balance. However, the surface energy balance is dependent
    on that surface temperature, meaning we have an optimisation problem.

    Instead of using `scipy.optimise.root`, which is designed for large multivariate
    problems, we instead implement a simple Newton-Raphson solver here.

    This lets us jit-compile, and ensure that we get the same results for both Numba
    and pure-Python backends.

    TODO - could instead overload this and call scipy.optimize.newton
    directly when use_numba = False?

    Parameters
    ----------
    f : function
        Residual function - e.g. physics.lake.seb.lake_development_eqn
    x0 : float
        Initial guess, almost certainly in [K]
    args : tuple
        Arguments to pass to the residual function.

    Returns
    -------
    x : float
        Solution to the system of equations, probably in [K].
        If ``success`` != ``True``, then returns the last
        result before the iteration stopped.
    success : bool
        True if the iteration converged, as in ``MINPACK``.
    n_iter : int
        Number of iterations used.
    """
    x = x0
    n_iter = 0
    for _ in range(SOLVER_MAXITER):
        n_iter += 1
        # calculate the residual at x
        fx = f(x, args)
        # residual small enough we can return True
        if abs(fx) < FTOL:
            return x, True, n_iter
        # calculate the derivative
        dfdx = (f(x + FD_STEP, args) - fx) / FD_STEP
        # derivative is zero, so can't continue
        if dfdx == 0.0:
            return x, False, n_iter
        # modify the guess
        x -= fx / dfdx
        # step size now so small that we have a solution
        if abs(fx / dfdx) < XTOL:
            return x, True, n_iter

    # MAXITER reached, so didnt succeed
    return x, False, n_iter


@kernel()
def newton_tridiagonal(residual, jacobian, x0, args):
    """
    Newton-Raphson driver for a column heat equation whose Jacobian is
    tridiagonal (the firn and lid heat equations).

    Newton-Raphson as defined above finds the root(s) of a function f via
    x1 = x0 - f(x0)/f'(x0). This is the multivariate form of that - so instead
    of solving for one x, we have a matrix of equations, where we can solve
    X1 = X0 - J(X0)^-1 * F(X0), where J is the Jacobian of F.

    In our case, the Jacobian is tridiagonal, since each point only depends on
    its nearest-neighbours; we can solve this efficiently with the Thomas algorithm.

    This is faster than using an optimised but more generic solver like those available
    in ``MINPACK``, since we know the Jacobian is tridiagonal and can be calculated
    analytically for all but the surface layer.

    Parameters
    ----------
    ``residual`` and ``jacobian`` are functions that are specific to the
    system being solved - in MONARCHS' case either the firn or lid heat equations
    and their surface-driven forcing.

    x0 : array_like, float
        Initial guess - the previous timestep's profile.
        This is firn/lid temperature in [K] for MONARCHS' heat equation routines.
    args : tuple
        Input arguments to the Jacobian/residual functions being called. The exact
        arguments are defined in the definitions of those specific functions, e.g.
        args will be different for ``lid.heateqn`` vs ``firn.heateqn``.

    Returns
    -------
    T : array_like, float
        Solution rounded to 8 decimal places [K].
    success : bool
        True if the iteration converged, as in ``MINPACK``.
    n_iter : int
        Number of Newton-Raphson iterations used.
    """
    n = len(x0)
    x = x0.copy()
    a = np.empty(n - 1)
    b = np.empty(n)
    c = np.empty(n - 1)
    resid_tol = max(SOLVER_TOL, SOLVER_TOL * np.max(np.abs(x0)))
    success = False
    n_iter = 0
    for _ in range(SOLVER_MAXITER):
        n_iter += 1
        fvec = residual(x, args)
        if np.max(np.abs(fvec)) < resid_tol:
            success = True
            break
        jacobian(x, a, b, c, args)
        x += solve_tridiagonal(a, b, c, -fvec)
    return np.around(x, decimals=8), success, n_iter
