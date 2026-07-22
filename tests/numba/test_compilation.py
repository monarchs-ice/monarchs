import os

"""
Test to ensure that the model compiles and runs for a very simple
test case. This also tests for whether we can import ERA5 data from netCDF.
"""

HERE = os.path.dirname(os.path.abspath(__file__))


def run(model_setup):
    from monarchs.core import driver
    from monarchs.core.setup_run import initialise_model_data

    # initialise_model_data runs the whole production setup path (firn profile,
    # met data, model grid) from the frozen config.
    grid = initialise_model_data(model_setup)
    return driver.run_model(model_setup, grid)


def test_numba_compilation():
    """Run a very simple case for 10 days. This mostly checks that the code
    compiles and runs without any Numba-specific errors."""
    from monarchs.core import load_model_setup, kernels
    from monarchs.config import configure

    model_setup = load_model_setup.get_model_setup(
        os.path.join(HERE, "model_test_setup_numba.py")
    )
    model_setup = configure(model_setup)
    kernels.compile_all(model_setup.use_numba)
    run(model_setup)
