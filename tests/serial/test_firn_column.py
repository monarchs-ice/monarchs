import numpy as np

from monarchs.met_data.import_ERA5 import (
    ERA5_to_variables,
    grid_subset,
    interpolate_grid,
)


# TODO - make sure tests test for mass conservation


def setup_test():
    from create_test_IceShelf import frozen_testcase

    cell = frozen_testcase()
    dz = cell.firn_depth / cell.vert_grid
    dt = 3600  # seconds
    ERA5_input = "data/ERA5_data_subset.nc"
    ERA5_vars = ERA5_to_variables(ERA5_input)
    # Select some bounds - lat upper/lower, long upper/lower
    ERA5_grid = grid_subset(ERA5_vars, -70.5, -73, 19, 16.5)
    # Interpolate to our grid size
    ERA5_grid = interpolate_grid(ERA5_grid, 5, 5)
    # set up random humidity as not specified in test met data I downloaded
    ERA5_grid["T_dp"] = np.ones(np.shape(ERA5_grid["snowfall"])) * 250
    return cell, dt, dz, ERA5_grid


def test_firn():
    """Test to ensure that running firn_column matches known good output."""


def test_percolation():
    """Simple test to ensure that percolation is working correctly"""


def test_refreezing():
    """Test to ensure that the correct amount of refreezing occurs"""
