""" """

# TODO - module-level docstring
import numpy as np


def sfc_flux(
    melt,
    exposed_water,
    lid,
    lake,
    lake_depth,
    lw_in,
    sw_in,
    air_temp,
    p_air,
    dew_point_temperature,
    wind,
    xsurf,
):
    """
    Calculate the surface heat flux from the input shortwave and longwave
    fluxes and latent/sensible heat fluxes.

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
    lw_in : float
        Incoming longwave radiation. [W m^-2].
    sw_in : float
        Incoming shortwave (solar) radiation. [W m^-2].
    air_temp : float
        Surface-layer air temperature. [K].
    p_air : float
        Surface-layer air pressure. [hPa].
    dew_point_temperature : float
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
    # Positive going into ice shelf
    alpha = sfc_albedo(melt, exposed_water, lid, lake, lake_depth)
    Flat, Fsens = bulk_fluxes(
        wind, air_temp, xsurf, p_air, dew_point_temperature
    )
    epsilon = 0.98  # emissivity
    Q = epsilon * lw_in + (1 - alpha) * sw_in + Flat + Fsens
    return Q


def sfc_albedo(melt, exposed_water, lid, lake, lake_depth):
    """
    Determine the effective surface albedo depending on the situation at the
    top of the ice shelf (i.e. is there exposed water, firn or snow etc.)


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
    #     TODO - snow albedo, add later
    if melt:
        if exposed_water:
            if lid:
                alpha = 0.413  # ice lid albedo
            elif lake:
                h = lake_depth
                alpha = (9702 + 1000 * np.exp(3.6 * h)) / (
                    -539 + 20000 * np.exp(3.6 * h)
                )  # lake albedo
            else:
                alpha = 0.6  # saturated firn albedo
        else:
            alpha = 0.6  # wet snow albedo
    else:
        alpha = 0.867  # dry snow albedo
    return alpha


def bulk_fluxes(wind, air_temp, T_sfc, p_air, dew_point_temperature):
    """
    Calculate the latent and sensible heat fluxes given the wind speed and
    surface meteorology.

    Parameters
    ----------
    wind : float
        Wind speed. [m s^-1].
    air_temp : float
        Surface-layer air temperature. [K].
    T_sfc : float
        Surface temperature. Taken from our initial guess x (i.e. x[0]) [K].
    p_air : float
        Surface air pressure. [hPa]
    dew_point_temperature : float
        2m dewpoint temperature. [K]

    Returns
    -------
    Flat : float
         Latent heat flux. [W m^-2].
    Fsens : float
         Sensible heat flux.   [W m^-2].

    Documentation on how the saturation vapour pressure is calculated can be
    found in Section 12 of
    https://www.ecmwf.int/sites/default/files/elibrary/2021/20198-ifs-
    documentation-cy47r3-part-vi-physical-processes.pdf

    =======
    """
    # Gravity
    g = 9.8
    b = 20
    # Height windspeed is measured at
    dz = 10
    CT0 = 1.3 * 10**-3
    c = 1961 * b * CT0
    # J kg−1 K−1 From section 12 of documentation in docstring
    R_dry = 287.0597
    # J kg−1 K−1
    R_sat = 461.5250
    # Pa
    a1 = 611.21
    # K
    T_0 = 273.16
    # This and a4 set to over water values as dewpoint temp being
    # used (following ERA-5 documentation)
    a3 = 17.502
    # K
    a4 = 32.19
    # Calculate the saturation vapour pressure at the dewpoint temperature
    # (i.e. the vapour pressure at the real temp)
    e_sat = a1 * np.exp(
        a3 * (dew_point_temperature - T_0) / (dew_point_temperature - a4)
    )
    # Alternative form for testing - Clausius-Clapeyron over ice
    # e_sat = a1 * np.exp(22.587 * (dew_point_temperature - T_0) / (dew_point_temperature + 0.7))
    # this is in kg/kg I think?
    s_hum = (R_dry / R_sat) * e_sat / (p_air - (e_sat * (1 - (R_dry / R_sat))))
    # Richardson number
    if wind == 0:
        Ri = 0
    else:
        Ri = g * (air_temp - T_sfc) * dz / (air_temp * wind**2)
    if Ri < 0:
        CT = CT0 * (1 - 2 * b * Ri / (1 + c * abs(Ri) ** 0.5))
    else:
        CT = CT0 * (1 + b * Ri) ** -2
    L = 2.501 * 10**6
    p_v = 2.53 * 10**8 * np.exp(-5420 / T_sfc)
    q_0 = 0.622 * p_v / (p_air - 0.378 * p_v)
    Fsens = 1.275 * 1005 * CT * wind * (air_temp - T_sfc)
    Flat = 1.275 * L * CT * wind * (s_hum / 1000 - q_0)
    return Flat, Fsens
#
#
# def bulk_fluxes(
#     wind,                   # m s^-1 at 10 m
#     air_temp,               # K (2 m or surface-layer air temperature)
#     T_sfc,                  # K (skin/surface temperature)
#     p_air_hPa,              # hPa (surface pressure)
#     dew_point_temperature,  # K (2 m dew point)
#     is_ice_surface=True,    # choose saturation at surface over ice or water
#     z_ref=10.0              # m (measurement height)
# ):
#     """
#     Compute bulk turbulent sensible and latent heat fluxes using a bulk-Ri stability correction.
#     SIGN CONVENTION: Positive flux = downward (from air to surface).
#
#     Returns:
#         Flat (W m^-2): Latent heat flux (downward positive; evaporation makes this typically negative).
#         Fsens (W m^-2): Sensible heat flux (downward positive if air > surface).
#     """
#
#     # --- Constants ---
#     g      = 9.80665          # m s^-2
#     Rd     = 287.05           # J kg^-1 K^-1
#     Rv     = 461.5            # J kg^-1 K^-1
#     cp     = 1005.0           # J kg^-1 K^-1 (dry air)
#     Lv     = 2.5e6            # J kg^-1 (vaporization over liquid)
#     Ls     = 2.834e6          # J kg^-1 (sublimation over ice)
#     L      = Ls if is_ice_surface else Lv
#
#     # Neutral transfer coefficient (heat/moisture); can tune 1.1e-3–1.5e-3
#     C0     = 1.3e-3
#
#     # Bulk-Richardson stability parameters (as in your code)
#     b      = 20.0
#     c_coef = 1961.0 * b * C0
#
#     # --- Pressure & density ---
#     p_Pa   = 100.0 * float(p_air_hPa)                 # convert hPa -> Pa
#     # virtual temperature (simple): Tv ≈ T * (1 + 0.61 q). Start with q from dew point
#     # Saturation vapor pressure at dew point (over water; dewpoint is over water by definition)
#     # Magnus (Alduchov & Eskridge-like) form:
#     a1, a3, a4 = 611.21, 17.502, 32.19  # Pa, -, K  (over water)
#     e_air = a1 * np.exp(a3 * (dew_point_temperature - 273.16) /
#                         (dew_point_temperature - a4))  # Pa
#
#     # Specific humidity of air (kg/kg)
#     q_air = (Rd / Rv) * e_air / (p_Pa - (1.0 - Rd/Rv) * e_air)
#
#     # Virtual temperature and density
#     Tv_air = air_temp * (1.0 + 0.61 * q_air)
#     rho    = p_Pa / (Rd * Tv_air)                     # kg m^-3
#
#     # --- Surface saturation specific humidity q_sfc ---
#     # Choose saturation over ice or over water for the skin temperature
#     if is_ice_surface:
#         # Saturation over ice (e.g., Murphy & Koop 2005 or a common approximation)
#         # Here: a practical approximation
#         e_sfc = 611.15 * np.exp(22.587 * (T_sfc - 273.16) / (T_sfc + 0.7))  # Pa (over ice)
#     else:
#         # Saturation over water (Magnus)
#         e_sfc = a1 * np.exp(a3 * (T_sfc - 273.16) / (T_sfc - a4))          # Pa (over water)
#
#     q_sfc = 0.622 * e_sfc / (p_Pa - 0.378 * e_sfc)    # kg/kg
#
#     # --- Stability: bulk Richardson number at z_ref ---
#     # Avoid division by zero
#     U = max(1e-6, float(wind))
#     Ri = g * (air_temp - T_sfc) * z_ref / (air_temp * U**2)
#
#     # Bulk stability correction to transfer coefficient
#     if Ri < 0.0:
#         C_bulk = C0 * (1.0 - 2.0 * b * Ri / (1.0 + c_coef * abs(Ri)**0.5))
#     else:
#         C_bulk = C0 * (1.0 + b * Ri)**(-2.0)
#
#     # Use same coefficient for heat/moisture (common in bulk schemes)
#     CH = CE = C_bulk
#
#     # --- Fluxes (downward positive) ---
#     Fsens =  rho * cp * CH * U * (air_temp - T_sfc)          # W m^-2
#     Flat  =  rho * L  * CE * U * (q_air    - q_sfc)          # W m^-2
#
#     return float(Flat), float(Fsens)
