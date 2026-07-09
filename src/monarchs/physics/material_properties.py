"""
Bulk properties of ice, and of ice/water/air mixtures (firn).

Previously, these were scattered and redefined throughout the model.
Now, all physics kernels refer to these functions to define these
parameters.

See monarchs.physics.constants for definitions of e.g. densities.

Ice thermal conductivity and heat capacity formulae are defined in
 Alexiades & Solomon (1993),
  "Mathematical Modelling of Melting and Freezing Processes", pg. 8.
"""

import numpy as np
from monarchs.core.kernels import kernel
from monarchs.physics.constants import (
    rho_ice,
    rho_water,
    rho_air,
    cp_air,
    cp_water,
    k_air,
    k_water,
)


@kernel()
def k_ice(T):
    """
    Thermal conductivity of solid ice [W m^-1 K^-1].

    k = 2.24 + 5.975e-3 * (273.15 - T)^1.156 (Alexiades & Solomon, pg. 8),
    clamped at the melting point (k = 2.24 for T >= 273.15 K).

    Works elementwise on scalars or arrays.
    """
    dT = np.maximum(273.15 - T, 0.0)
    return 1000.0 * (2.24e-3 + 5.975e-6 * dT**1.156)


@kernel()
def cp_ice(T):
    """
    Specific heat capacity of solid ice [J kg^-1 K^-1].

    cp = 7.16 * T + 138 (Alexiades & Solomon, pg. 8), clamped at the melting
    point (cp = 2093.6 for T >= 273.15 K).
    """
    T_eff = np.minimum(T, 273.15)
    return 7.16 * T_eff + 138.0


@kernel()
def k_mixture(T, sfrac, lfrac):
    """
    Effective thermal conductivity of an ice/water/air mixture
    [W m^-1 K^-1], volume-fraction weighted.
    """
    air_frac = 1.0 - sfrac - lfrac
    return sfrac * k_ice(T) + lfrac * k_water + air_frac * k_air


@kernel()
def cv_mixture(T, sfrac, lfrac):
    """
    Volumetric heat capacity of an ice/water/air mixture [J m^-3 K^-1],
    volume-fraction weighted.
    """
    air_frac = 1.0 - sfrac - lfrac
    return (
        sfrac * rho_ice * cp_ice(T)
        + lfrac * rho_water * cp_water
        + air_frac * rho_air * cp_air
    )


@kernel()
def k_and_kappa(T, sfrac, lfrac):
    """
    Effective conductivity k [W m^-1 K^-1] and thermal diffusivity
    kappa = k / C_vol [m^2 s^-1] of the firn mixture.
    """
    k = k_mixture(T, sfrac, lfrac)
    kappa = k / cv_mixture(T, sfrac, lfrac)
    return k, kappa
