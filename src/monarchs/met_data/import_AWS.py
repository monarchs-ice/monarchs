"""
Import functions for various automatic weather station (ASW) data sets

S Buzzard created, last updated 06/05/2021
"""

import numpy as np


def import_AWS_LCIS(filepath):
    data = np.loadtxt(filepath, skiprows=1)
    T_air = data[:, 4] + 273.15
    r_hum = data[:, 5] / data[:, 5]
    p_air = data[:, 6] / 10
    wind = data[:, 7]
    SW_in = data[:, 11]
    LW_in = data[:, 13]
    return T_air, p_air, r_hum, wind, SW_in, LW_in


def import_AWS_Wisc(aws, start_year, start_month, end_year, end_month):
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


def import_AWS_LCIS_BAS():
    data = np.loadtxt(
        "/Users/samanthabuzzard/Documents/Work/MATLAB/BAS_Rad_2009to10_hourly_edited_final.txt",
        skiprows=1,
    )
    SW_in = data[:, 4]
    LW = data[:, 6] - data[:, 8]
    return SW_in, LW
