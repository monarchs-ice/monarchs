# -*- coding: utf-8 -*-
"""
Import functions for various automatic weather station (ASW) data sets

S Buzzard created, last updated 06/05/2021
"""

import numpy as np


def import_AWS_LCIS(filepath):
    # Peter KM provided
    # T is in Celsius
    # RH in %
    # Pressure in hPa
    # Wind speed in m/s
    # Wind direction in degrees
    # Z_wind and Z_temp are measurement heights of the wind and temperature observations.
    # SW and LW are in W m^-2
    # Column order 0.Year 1.DayOfYear 2.Hour 3.DecimalDay 4.Tair 5.RelHum 6.Pres 7.WindSpeed 8.WindDir 9.Zwind 10.Ztemp 11.SWin 12.SWout 13.LWin 14.LWout 15.Hsnow 16.RimeYesNo

    data = np.loadtxt(filepath, skiprows=1)
    T_air = data[:, 4] + 273.15  # Convert to Kelvin
    r_hum = data[:, 5] / data[:, 5]  # Convert to specific humidity...
    p_air = data[:, 6] / 10  # Convert to kPa
    wind = data[:, 7]
    SW_in = data[:, 11]
    LW_in = data[:, 13]

    return T_air, p_air, r_hum, wind, SW_in, LW_in


def import_AWS_Wisc(aws, start_year, start_month, end_year, end_month):
    # data is daily, UNITS

    ###BPT###
    # No Feb 2008 for have created new file with 2nd half Jan + 1st half Mar repleated)
    # No May/June 2011, have doubled April and July
    # Big chunk missing from Oct 2012 to Jan 2014
    # Starts Sep 2001, Ends Apr 2017

    ###BRP###
    # Starts Jan 2013, Ends Nov 2019
    # No Jan 2014 to Jan 2016 and no Feb 2016 to Jan 2017
    # No Jul 2017 or Jul 2019

    ###THI###
    # Starts Jan 2011, Ends Nov 2019
    # No missing data 🎉

    temp = []
    p = []
    r_hum = []
    wind = []

    for file_month in range(start_month, 13):
        data = np.loadtxt(
            "Wisc_AWS/" + aws + str(start_year) + str("%02d" % file_month) + "q3h.txt",
            skiprows=2,
        )
        temp = np.append(temp, data[:, 5])
        p = np.append(p, data[:, 6])
        r_hum = np.append(r_hum, data[:, 9])
        wind = np.append(wind, data[:, 7])
    for file_year in range(start_year + 1, end_year):
        for file_month in range(1, 13):
            data = np.loadtxt(
                "Wisc_AWS/"
                + aws
                + str(file_year)
                + str("%02d" % file_month)
                + "q3h.txt",
                skiprows=2,
            )
            temp = np.append(temp, data[:, 5])
            p = np.append(p, data[:, 6])
            r_hum = np.append(r_hum, data[:, 9])
            wind = np.append(wind, data[:, 7])
    for file_month in range(1, end_month + 1):
        data = np.loadtxt(
            "Wisc_AWS/" + aws + str(end_year) + str("%02d" % file_month) + "q3h.txt",
            skiprows=2,
        )
        temp = np.append(temp, data[:, 5])
        p = np.append(p, data[:, 6])
        r_hum = np.append(r_hum, data[:, 9])
        wind = np.append(wind, data[:, 7])

    return temp, p, r_hum, wind

    # 1     Year
    # 2     Julian day
    # 3     Month
    # 4     Day
    # 5     Three-hourly observation time
    # 6     Temperature (C)
    # 7     Pressure (hPa)
    # 8     Wind Speed (m/s)
    # 9     Wind Direction
    # 10    Relative Humidity (%)
    # 11    Delta-T (C)


def import_AWS_LCIS_BAS():
    data = np.loadtxt(
        "/Users/samanthabuzzard/Documents/Work/MATLAB/BAS_Rad_2009to10_hourly_edited_final.txt",
        skiprows=1,
    )
    # T_air=data[:,4]+273.15 #Convert to Kelvin
    # r_hum=data[:,5]/data[:,5]#Convert to specific humidity...
    # p_air=data[:,6]/10#Convert to kPa
    # wind=data[:,7]
    SW_in = data[:, 4]
    LW = data[:, 6] - data[:, 8]

    return SW_in, LW
