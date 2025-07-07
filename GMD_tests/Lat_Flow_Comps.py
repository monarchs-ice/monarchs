from netCDF4 import Dataset
from matplotlib import pyplot as plt
import numpy.ma as ma
import matplotlib
matplotlib.use('TkAgg')

dumppath_0 = '/media/dmwq2/New Volume/monarchs_120625/full/model_output.nc'
dumppath_m10 = '/media/dmwq2/New Volume/monarchs_120625/experiments/flow_minus10/model_output.nc'
dumppath_p10 = '/media/dmwq2/New Volume/monarchs_120625/experiments/flow_plus10/model_output.nc'
dumppath_m50 = '/media/dmwq2/New Volume/monarchs_120625/experiments/flow_minus50/model_output.nc'
dumppath_p50 = '/media/dmwq2/New Volume/monarchs_120625/experiments/flow_plus50/model_output.nc'


flowdata_0 = Dataset(dumppath_0)
flowdata_m10 = Dataset(dumppath_m10)
flowdata_p10 = Dataset(dumppath_p10)
flowdata_m50 = Dataset(dumppath_m50)
flowdata_p50 = Dataset(dumppath_p50)


#Closest output to Jan 1 for final 5 years
#Saves every 5 days

#1st Jan 2015 365
#2016 438
#2017 511
#2018 584
#2019 657
#2020 731

#Average these and compare individually in case any were high or low melt years. Do mean lake depth, max lake depth, percent lake coverage



