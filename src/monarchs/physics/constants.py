"""
Physical constants used in MONARCHS calculations.

Previously, these were either scattered throughout the codebase, or
defined in the model grid.
"""

rho_ice = 917.0
rho_water = 1000.0
rho_air = 1.29 # kg m^-3
L_ice = 334000  # J kg^-1
k_air = 0.022  # W m^-1 K^-1
cp_air = 1004  # J kg^-1 K^-1
k_water = 0.5818  # W m^-1 K^-1
cp_water = 4217  # J kg^-1 K^-1
stefan_boltzmann = 5.670374e-8  # W m^-2 K^-4
emissivity = 0.98  # emissivity of ice/water
pore_closure = 830 # density in kg m^-3 where firn pores close
ice_extinction_coefficient = 1.5  # m^-1, for solar radiation in ice
sfc_absorbed_frac = 0.5  # fraction of solar radiation absorbed in surface layer of water or ice
tau_water = 0.36  # Table 1, panchromatic absorption coefficient,
# Pope et al. (2016), The Cryosphere,
# doi:10.5194/tc-10-15-2016
tau_ice = 1.5  # black ice extinction coefficient
# rough value taking into account:
# https://tc.copernicus.org/articles/15/1931/2021/#section3
# Cooper, M.G., et al., 2021. Spectral attenuation coefficients from
# measurements of light transmission in bare ice on the Greenland Ice Sheet.
# The Cryosphere, 15(4), pp.1931-1953.
v_lid_min_thickness = 1e-3