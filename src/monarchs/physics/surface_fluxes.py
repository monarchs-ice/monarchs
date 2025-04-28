import numpy as np
from numba import njit


def sfc_flux(melt, exposed_water, lid, lake, lake_depth, LW_in, SW_in,
    T_air, p_air, T_dp, wind, xsurf):
    """
    Calculate the surface heat flux from the input shortwave and longwave fluxes
    and latent/sensible heat fluxes.

    Parameters
    ----------
    melt : bool
        Flag which indicates whether melting has occurred.
        Used here to determine surface albedo.
    exposed_water : bool
        Flag which describes whether there is exposed water at the surface.
        Used here to determine surface albedo.
    lid : bool
        Flag which indicates whether there is a lid at the surface due to
        refreezing. Used here to determine surface albedo.
    lake : bool
        Flag which indicates whether there is a lake present. Used here to
        determine surface albedo.
    lake_depth : float
        Depth of the lake (in vertical points?)
    LW_in : float
        Incoming longwave radiation. [W m^-2].
    SW_in : float
        Incoming shortwave (solar) radiation. [W m^-2].
    T_air : float
        Surface-layer air temperature. [K].
    p_air : float
        Surface-layer air pressure. [hPa].
    T_dp : float
        Dew-point temperature at the surface. [K]
    wind : float
        Wind speed. [m s^-1].
    xsurf : float
        Surface temperature. Taken from our initial guess x (i.e. x[0]) [K].

    Returns
    -------
    Q : float64
        Surface energy flux. [W m^-2].

    """
    alpha = sfc_albedo(melt, exposed_water, lid, lake, lake_depth)
    Flat, Fsens = bulk_fluxes(wind, T_air, xsurf, p_air, T_dp)
    epsilon = 0.98
    Q = epsilon * LW_in + (1 - alpha) * SW_in + Flat + Fsens
    return Q


def sfc_albedo(melt, exposed_water, lid, lake, lake_depth):
    """
    Determine the effective surface albedo depending on the situation at the
    top of the ice shelf (i.e. is there exposed water, firn or snow etc.)

    TODO - snow albedo, add later

    Parameters
    ----------
    melt : bool
        Flag which indicates whether melting has occurred.
    exposed_water : bool
        Flag which describes whether there is exposed water at the surface.
    lid : bool
        Flag which indicates whether there is a lid at the surface due to
        refreezing.
    lake : bool
        Flag which indicates whether there is a lake present.
    lake_depth : float
        Depth of the lake [m]

    Returns
    -------
    alpha : float
        Effective surface albedo for shortwave radiation.

    """
    if melt:
        if exposed_water:
            if lid:
                alpha = 0.413
            elif lake:
                h = lake_depth
                alpha = (9702 + 1000 * np.exp(3.6 * h)) / (-539 + 20000 *
                    np.exp(3.6 * h))
            else:
                alpha = 0.6
        else:
            alpha = 0.6
    else:
        alpha = 0.867
    return alpha


def bulk_fluxes(wind, T_air, T_sfc, p_air, T_dp):
    """
    Calculate the latent and sensible heat fluxes given the wind speed and
    surface meteorology.

    Parameters
    ----------
    wind : float
        Wind speed. [m s^-1].
    T_air : float
        Surface-layer air temperature. [K].
    T_sfc : float
        Surface temperature. Taken from our initial guess x (i.e. x[0]) [K].
    p_air : float
        Surface air pressure. [hPa]
    T_dp : float
        2m dewpoint temperature. [K]

    Returns
    -------
    Flat : float
         Latent heat flux. [W m^-2].
    Fsens : float
         Sensible heat flux.   [W m^-2].

    Documentation on how the saturation vapour pressure is calculated can be found in Section 12 of
    https://www.ecmwf.int/sites/default/files/elibrary/2021/20198-ifs-documentation-cy47r3-part-vi-physical-processes.pdf

    =======
    """
    g = 9.8
    b = 20
    dz = 10
    CT0 = 1.3 * 10 ** -3
    c = 1961 * b * CT0
    R_dry = 287.0597
    R_sat = 461.525
    a1 = 611.21
    T_0 = 273.16
    a3 = 17.502
    a4 = 32.19
    e_sat = a1 * np.exp(a3 * (T_dp - T_0) / (T_dp - a4))
    s_hum = R_dry / R_sat * e_sat / (p_air - e_sat * (1 - R_dry / R_sat))
    if wind == 0:
        Ri = 0
    else:
        Ri = g * (T_air - T_sfc) * dz / (T_air * wind ** 2)
    if Ri < 0:
        CT = CT0 * (1 - 2 * b * Ri / (1 + c * abs(Ri) ** 0.5))
    else:
        CT = CT0 * (1 + b * Ri) ** -2
    L = 2.501 * 10 ** 6
    p_v = 2.53 * 10 ** 8 * np.exp(-5420 / T_sfc)
    q_0 = 0.622 * p_v / (p_air - 0.378 * p_v)
    Fsens = 1.275 * 1005 * CT * wind * (T_air - T_sfc)
    Flat = 1.275 * L * CT * wind * (s_hum / 1000 - q_0)
    return Flat, Fsens
