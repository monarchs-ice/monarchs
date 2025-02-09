import numpy.testing as npt


def setup_test():
    from create_test_IceShelf import frozen_testcase
    from create_test_MetData import met_data

    cell = frozen_testcase()
    x = cell.firn_temperature
    dz = cell.firn_depth / cell.vert_grid
    dt = 3600  # seconds

    args = (
        cell,
        dt,
        dz,
        met_data["LW_in"],
        met_data["SW_in"],
        met_data["T_air"],
        met_data["p_air"],
        met_data["T_dp"],
        met_data["wind"],
    )

    return cell, x, args


def test_solver():
    """Test to ensure solver function gives the same result in Numba and non-Numba cases"""


def test_solver_fixedsfc():
    """Test to ensure solver function gives the same result in Numba and non-Numba cases, assuming fixed surface"""


def test_surface_fluxes():
    """Test to ensure solver function gives the same result in Numba and non-Numba cases, assuming fixed surface"""
    cell, x, args = setup_test()
    from create_test_MetData import met_data
    from src.monarchs.physics.surface_fluxes import sfc_flux

    Q = sfc_flux(
        cell.melt,
        cell.exposed_water,
        cell.lid,
        cell.lake,
        cell.lake_depth,
        met_data["LW_in"],
        met_data["SW_in"],
        met_data["T_air"],
        met_data["p_air"],
        met_data["T_dp"],
        met_data["wind"],
        x[0],
    )

    known_good_output = 145.23303524011789
    npt.assert_almost_equal(Q, known_good_output, decimal=8)
