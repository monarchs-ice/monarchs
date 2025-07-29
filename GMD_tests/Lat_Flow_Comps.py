from netCDF4 import Dataset
import numpy as np
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
#2020 730
#Average these and compare individually in case any were high or low melt years. Do mean lake depth, max lake depth, percent lake coverage
av_lakes = np.zeros([5,6])
av_coverage = np.zeros([5,6])

model_runs = ['0', 'm10', 'm50', 'p50'] #p10 rerunning
years = [365, 438, 511, 584, 657, 730]

for i in range(0,4): #model run
    for j in range(0,6): #year
        av_lakes[i,j] = np.average(np.ravel(eval('flowdata_'+model_runs[i])['lake_depth'][years[j]]))
        av_coverage[i,j] = np.count_nonzero(np.ravel(eval('flowdata_'+model_runs[i])['lake_depth'][years[j]]))
print(av_lakes)
print(av_coverage)


