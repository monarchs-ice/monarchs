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
    #
    # cell, x, args = setup_test()
    #
    #
    #
    # root, infodict, ier, mesg = solver(x, args)
    # root_numba, fvec, success, info = solver_numba(x, args)
    # if ier == info and success == True:
    #     print("\nSame result for hybrd and fsolve")
    #     print(mesg)
    # else:
    #     print(f"ier = {ier}")
    #     print(f"info = {info}")
    #     print(
    #         "fsolve and hybrd returned different failure messages..."
    #         "This is likely due to the initial conditions of the test setup"
    #     )
    # npt.assert_almost_equal(root, root_numba, decimal=8)
    # print(root)


def test_solver_fixedsfc():
    """Test to ensure solver function gives the same result in Numba and non-Numba cases, assuming fixed surface"""
    # cell, x, args = setup_test()
    # from monarchs.core.choose_solver import get_firn_heateqn_solver
    #
    # solver_numba = get_firn_heateqn_solver(True)
    # root_numba, fvec, success, info = solver_numba(x, args, fixed_sfc=True)
    # print(success)
    # solver = get_firn_heateqn_solver(False)
    # root, infodict, ier, mesg = solver(x, args, fixed_sfc=True)
    # print(ier)
    # if ier == info and success == True:
    #     print("\nSame result for hybrd and fsolve")
    #     print(mesg)
    # else:
    #     print(f"ier = {ier}")
    #     print(f"info = {info}")
    #     print(
    #         "fsolve and hybrd returned different failure messages..."
    #         "This is likely due to the initial conditions of the test setup"
    #     )
    # npt.assert_almost_equal(root, root_numba, decimal=8)
    # print(root)


def test_fluxes():
    """Test to ensure solver function gives the same result in Numba and non-Numba cases, assuming fixed surface"""
    cell, x, args = setup_test()
    from create_test_MetData import met_data
    from monarchs.physics.surface_fluxes import sfc_flux

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

    known_good_output = 125.64994697591283
    npt.assert_almost_equal(Q, known_good_output, decimal=8)


def test_spinup():
    pass
    # cell, x, args = setup_test()
    # from monarchs.core.choose_solver import get_firn_heateqn_solver
    #
    # # set the temperature grid to something which should definitely not work
    # x = np.linspace(231, 270, cell.vert_grid)[::-1]
    # xn = np.array(x)
    # numba_args = list(args)  # add an empty tuple to make a copy
    # args = list(args)
    # args[0].T = x
    # numba_args[0].T = x
    # print("\n")
    # solver_numba = get_firn_heateqn_solver(True)
    # solver = get_firn_heateqn_solver(False)
    # success = False
    # num_days = 0
    #
    # while success is False and days < 100:
    #     print("T = ", args[0].T[:5])
    #     print("T_numba = ", numba_args[0].T[:5])
    #     root_numba, fvec, success, info = solver_numba(xn, numba_args)
    #     root, infodict, ier, mesg = solver(x, args)
    #     xn = root_numba
    #     x = root
    #     numba_args[0].T = xn
    #     args[0].T = x
    #     # reduce SW/LW/air temp a bit to simulate cooling
    #     for i in range(3, 5):
    #         numba_args[i] -= 1
    #         args[i] -= 1
    #     numba_args[5] -= 0.1
    #     args[5] -= 0.1
    #
    #     days += 1
    #     print("Numba solution: ", root_numba[:5])
    #     print("Scipy solution: ", root[:5])

    # if ier == info:
    #     print('\nSame result for hybrd and fsolve')
    #     print(mesg)
    # else:
    #     print(f'ier = {ier}')
    #     print(f'info = {info}')
    #     print('fsolve and hybrd returned different failure messages...'
    #           'This is likely due to the initial conditions of the test setup')
    # npt.assert_almost_equal(root, root_numba, decimal=8)


if __name__ == "__main__":
    # test_solver()
    # test_solver_fixedsfc()
    test_fluxes()
