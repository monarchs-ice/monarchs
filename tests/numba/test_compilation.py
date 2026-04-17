import numpy as np

"""
Test to ensure that the model compiles and runs for a very simple
test case. This also tests for whether we can import ERA5 data from netCDF.
"""
def run(model_setup):
    from monarchs.core import driver
    from monarchs.core import initial_conditions
    from monarchs.met_data import setup_met_data

    T_firn, rho, firn_depth, valid_cells, dx, dy, _, _ = (
        initial_conditions.initialise_firn_profile(
            model_setup, diagnostic_plots=False
        )
    )

    lat_array = (
        np.zeros((model_setup.row_amount, model_setup.col_amount)) * np.nan
    )
    lon_array = (
        np.zeros((model_setup.row_amount, model_setup.col_amount)) * np.nan
    )
    # Set up meteorological data and return the path to the grid actually used by MONARCHS
    setup_met_data.met_data_from_era5(model_setup, lat_array, lon_array)
    # Initialise the model grid.
    grid = initial_conditions.create_model_grid(
        model_setup,
        firn_depth,
        rho,
        T_firn,
        valid_cells=valid_cells,
        lats=lat_array,
        lons=lon_array,
    )

    grid = driver.main(model_setup, grid)

    return grid


def test_numba_compilation():
    """Run a very simple case for 10 days. This is mostly to make sure that the code compiles correctly without
    any Numba-specific errors."""
    from monarchs.core import load_model_setup, configuration

    model_setup = load_model_setup.get_model_setup('model_test_setup_numba.py')
    configuration.handle_incompatible_flags(model_setup)
    configuration.handle_invalid_values(model_setup)
    configuration.create_defaults_for_missing_flags(model_setup)
    configuration.jit_modules()
    run(model_setup)
