# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:04:04 2023

@author: sglsb6
"""

import numpy as np
from monarchs.core.utils import calc_mass_sum


def snowfall(cell, snow_depth, snow_rho, snow_T):
    """
    Adds snowfall to surface of model and regrids model to incorporate it. This snow is added to the firn or lake
    depth, depending on the current state of the IceShelf (cell).
    Called in <timestep_loop>.
    TODO - what happens if we have a lid? Gets added to top and we just regrid as normal.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf
        IceShelf object to add snow onto.
    snow_depth : float
        Depth of the snow, read in from the meteorological data input [m]
    snow_rho : float
        Snow density, either read in from the meteorological data input, or assumed 300 [kg m^-3]
    snow_T : float
        Snow temperature, either read in from meterological data input, or assumed 260 [K]

    Returns
    -------
    None.
    """

    if snow_depth != 0:
        # if lake - add it to lake depth
        if cell.lake:
            cell.lake_depth += snow_depth * snow_rho / cell.rho_water
            # TODO - lake temperature change due to snow falling
        else:
            original_mass = calc_mass_sum(cell)
            orig_liquid = np.sum((cell.Lfrac * cell.rho_water * (cell.firn_depth / cell.vert_grid)))
            orig_solid = np.sum((cell.Sfrac * cell.rho_ice * (cell.firn_depth / cell.vert_grid)))
            old_sfrac = np.copy(cell.Sfrac)
            old_sfrac_0 = cell.Sfrac[0]
            # Set temperature just below freezing if it is somehow above 0 degrees, which it shouldn't be
            # if we don't do this, we can run into errors when solving the heat equation later
            if snow_T > 273.15:
                snow_T = 273.149
            # track pre-snow dz and firn depth
            dz_old = cell.firn_depth / cell.vert_grid
            old_firn_depth = cell.firn_depth + 0

            # add height to the firn
            cell.firn_depth += snow_depth
            dz_new = cell.firn_depth / cell.vert_grid
            # create a new layer for the firn depth

            # Write some code to interpolate the new snow into the existing firn profile. This should conserve mass,
            # so that the sum of the mass of the firn and the snow is equal to the mass of the firn before the snow was
            # added. This is done by weighting the mass of the new snow and the mass of the existing firn in each layer
            # by the proportion of the new cell that is made up by the old cells.

            # First interpolate the existing profiles
            sfrac_hold = np.zeros(np.shape(cell.Sfrac))
            lfrac_hold = np.zeros(np.shape(cell.Lfrac))
            T_hold = np.zeros(np.shape(cell.firn_temperature))

            # Scale temperature in the top layer to account for the fact that we have added snow
            weight1 = dz_old * (
                    (cell.Sfrac[0] * cell.rho_ice) + (cell.Lfrac[0] * cell.rho_water)
            )
            weight2 = snow_depth * snow_rho

            cell.firn_temperature[0] = (
                                               weight1 * cell.firn_temperature[0] + weight2 * snow_T
                                       ) / (weight1 + weight2)
            # Scale Sfrac and Lfrac in the top layer also
            weight1 = snow_depth
            weight2 = dz_old
            cell.Sfrac[0] = (
                                    weight1 * (snow_rho / cell.rho_ice) + (weight2 * cell.Sfrac[0])
                            ) / (weight1 + weight2)

            # error handling
            if np.any(cell.Lfrac < -0.001):
                print(np.where(lfrac_hold < -0.001))
                raise ValueError("Lfrac before snow < 0")

            cell.Lfrac[0] = (weight1 * 0 + weight2 * cell.Lfrac[0]) / (
                    weight1 + weight2
            )

            if np.any(cell.Lfrac < -0.001):
                print(np.where(lfrac_hold < -0.001))
                print(cell.Lfrac)
                print(lfrac_hold)
                raise ValueError("Lfrac before hold < 0")
            if np.any(cell.Sfrac < -0.001):
                print(np.where(cell.Sfrac < -0.001))
                raise ValueError("Sfrac before snow > 0")
            if np.any(cell.Sfrac > 1.001):
                print(np.where(cell.Sfrac > 1.001))
                raise ValueError("Sfrac before snow > 1")

            # Calculate the new profile as a weighted average of the proportion of the new cell that was made up by the old
            # cells.
            for i in range(1, len(cell.Sfrac)):
                weight_1 = (
                        cell.firn_depth - (i * dz_new) - (old_firn_depth - (i * dz_old))
                )  # new top of upper layer - bottom of old upper layer
                weight_2 = (old_firn_depth - (i * dz_old)) - (cell.firn_depth - ((i + 1) * dz_new))
                # old bottom of upper layer - new bottom of upper layer

                lfrac_hold[i] = (
                                        (cell.Lfrac[i - 1] * weight_1) + cell.Lfrac[i] * weight_2
                                ) / (weight_1 + weight_2)
                sfrac_hold[i] = (
                                        (cell.Sfrac[i - 1] * weight_1) + cell.Sfrac[i] * weight_2
                                ) / (weight_1 + weight_2)
                T_hold[i] = (
                                    cell.firn_temperature[i - 1] * weight_1
                                    + cell.firn_temperature[i] * weight_2
                            ) / (weight_1 + weight_2)

            lfrac_hold[0] = cell.Lfrac[0]
            sfrac_hold[0] = cell.Sfrac[0]
            T_hold[0] = cell.firn_temperature[0]
            # further error handling
            if np.any(lfrac_hold < -0.001):
                w = np.where(lfrac_hold < -0.001)
                print(w[0][0])
                print("New Lfrac = ", lfrac_hold[: w[0][0] + 2])
                print("Old Lfrac = ", cell.Lfrac[: w[0][0] + 2])
                raise ValueError("Lfrac hold > 0")
            if np.any(sfrac_hold < -0.001):
                w = np.where(sfrac_hold < 0)
                print(w[0][0])
                print("New Sfrac = ", sfrac_hold[: w[0][0] + 2])
                print("Old Sfrac = ", cell.Sfrac[: w[0][0] + 2])
                raise ValueError("Sfrac hold > 0")
            if np.any(sfrac_hold > 1.001):
                print(np.where(sfrac_hold > 1))
                print(sfrac_hold)
                raise ValueError("Sfrac before snow > 1")
            cell.Sfrac = sfrac_hold
            cell.Lfrac = lfrac_hold
            cell.firn_temperature = T_hold
            assert abs(calc_mass_sum(cell) - (original_mass + (snow_depth * snow_rho))) < (1.5 * 10 ** -7)


def densification(cell, t_steps_per_day):
    """
    TODO - not important now, but implement for later.

    Parameters
    ----------
    cell : core.iceshelf_class.IceShelf

    t_steps_per_day : int

    Returns
    -------

    """
    dt = 1 / (8 * t_steps_per_day * 365)  # change per timestep
    g = 9.81  # acceleration due to gravity [m s^-2]
    ec = 60  # values used in Arthern et al.
    eg = 42.4  # values used in Arthern et al.
    R = 8.3144598  # gas constant
    T_av = 264.56010894609415
    e = np.exp((-ec / (R * cell.firn_temperature[0])) + (eg / (R * T_av)))
    cell.rho = cell.Sfrac * cell.rho_ice + cell.Lfrac * cell.rho_water

    b = 0.4953 * (
            1000 / 350
    )  # total annual accumulation - TODO taken direct from MATLAB, need to calc
    for i in range(cell.vert_grid):
        if cell.rho[i] > cell.rho_ice:
            cell.rho[i] = cell.rho_ice
        if cell.rho[i] < 550:
            c = 0.07
        else:
            c = 0.03
        d_rho = c * b * g * (cell.rho_ice - cell.rho[i]) * e * dt
        cell.rho[i] = cell.rho[i] + d_rho
        # before = cell.Sfrac[i]
        cell.Sfrac[i] = cell.rho[i] / cell.rho_ice
        if cell.Sfrac[i] > 1:
            print("Snow accumulation has caused Sfrac to be > 1")

# function[rho]=SnowDens(rho, M, b, T_Av, T, tsp_frac)
# %1D firn densification
# %T_Av is the average annual surface temperature
# Dt=1/(8*tsp_frac*365);%Change per timestep (as formula calculates in years)
# g=9.8;%Acceleration due to gravity
# EC=60; %Values used in Arthern et al.
# EG=42.4;
# R=8.3144598;%Gas constant
# e=exp((-EC/(R*T))+(EG/(R*T_Av)));%This doesn't change with depth
# rho_ice= 917; %Ice density
# %Calculate densfication for each layer
# for i=1:M
#     if rho(i)>rho_ice
#         rho(i)=rho_ice;
#     end
#     if rho(i)<550
#         C=0.07;
#     else
#         C=0.03;
#     end
#     D_rho=C*b*g*(rho_ice-rho(i))*e*Dt;%Density change
#     rho(i)=rho(i)+D_rho;%New density
# end
