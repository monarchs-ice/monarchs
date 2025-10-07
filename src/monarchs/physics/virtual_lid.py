"""
TODO - module-level docstring, other docstrings
"""

import numpy as np
from monarchs.physics import surface_fluxes
from monarchs.physics import solver
from monarchs.core import utils


def virtual_lid_development(cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind):
    """
    When a lake undergoes freezing from the top due to the surface conditions,
    a lid forms. However, the depth of this lid can oscillate significantly
    during the initial stages of formation, including disappearing entirely.
    To model this, we assume that the model is in a "virtual lid" state while
    the lid depth is less than 0.1 m.
    This function tracks the development of this virtual lid, and switches
    the model into a non-lid state if it melts entirely, or into a true lid
    state if the lid depth exceeds the 0.1 m depth threshold.

    This virtual lid, unlike the "true" lid, has a single value for its
    temperature (cell.v_lid_temperature), rather than a vertical profile.
    The function handles freezing and melting of this virtual lid from the
    top and bottom.

    As with many other functions, this operates on a single column.

    Called in timestep_loop.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    dt : int
        Number of seconds in the timestep, very like 3600 (i.e. 1h) [s]
    LW_in : float
        Incoming longwave radiation. [W m^-2].
    SW_in : float
        Incoming shortwave (solar) radiation. [W m^-2].
    T_air : float
        Surface-layer air temperature. [K].
    p_air : float
        Surface-layer air pressure. [hPa].
    T_dp : float
        Dew-point temperature at the surface. [K]
    wind : float
        Wind speed. [m s^-1].

    Returns
    -------
    None (amends cell inplace)
    """
    original_mass = utils.calc_mass_sum(cell)
    x = np.array([cell["virtual_lid_temperature"]])

    Q = surface_fluxes.sfc_flux(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["lake_depth"],
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
        x[0],
    )
    k_v_lid = 1000 * (
        2.24 * 10**-3
        + 5.975 * 10**-6 * (273.15 - cell["virtual_lid_temperature"]) ** 1.156
    )

    # Surface energy balance calculation to determine the virtual lid
    # temperature
    args = np.array(
        [
            Q,
            k_v_lid,
            cell["lake_depth"],
            cell["vert_grid_lake"],
            cell["v_lid_depth"],
        ]
    )
    args = np.append(args, cell["lake_temperature"])

    # want only root, not fvec etc, and root is an array of one element so
    # extract out first elem
    cell["virtual_lid_temperature"] = solver.lid_seb_solver(
        x, args, v_lid=True
    )[0][0]

    # JE TODO - are we missing Fu from here also?
    kdTdz = (
        (cell["virtual_lid_temperature"] - cell["lake_temperature"][1])
        * abs(cell["k_water"])
        / (
            cell["lake_depth"] / (cell["vert_grid_lake"] / 2)
            + cell["v_lid_depth"]
        )
    )

    new_boundary_change = kdTdz / (cell["L_ice"] * cell["rho_ice"]) * dt
    # further freezing of the virtual lid
    if cell["virtual_lid_temperature"] < 273.15:
        if new_boundary_change < 0:
            freeze_virtual_lid(cell, new_boundary_change)
    # melting of the virtual lid
    else:
        melt_virtual_lid(cell, Q, dt)

    # If the lid is greater than 10 cm - now have a permanent lid.
    if cell["v_lid_depth"] > 0.1 or cell["lake_depth"] == 0:
        cell["lid"] = True
        cell["v_lid"] = False
        cell["lid_depth"] = cell["v_lid_depth"]
        cell["v_lid_depth"] = 0

    if cell["v_lid_depth"] <= 0 and cell["lid_depth"] <= 0:
        cell["lid"] = False
        cell["v_lid"] = False

    # check we've not gained or lost too much mass
    new_mass = utils.calc_mass_sum(cell)
    try:
        assert abs(new_mass - original_mass) < 1.5 * 10**-7
    except Exception:
        print("Original mass = ", original_mass)
        print("New mass = ", new_mass)
        print("Difference = ", abs(new_mass - original_mass))
        raise Exception


def update_virtual_lid_temperature(
    cell, LW_in, SW_in, T_air, p_air, T_dp, wind
):
    x = np.array([cell["virtual_lid_temperature"]])
    Q = surface_fluxes.sfc_flux(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["lake_depth"],
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        wind,
        x[0],
    )
    k_v_lid = 1000 * (
        2.24e-3
        + 5.975e-6 * (273.15 - cell["virtual_lid_temperature"]) ** 1.156
    )
    args = np.array(
        [
            Q,
            k_v_lid,
            cell["lake_depth"],
            cell["vert_grid_lake"],
            cell["v_lid_depth"],
        ]
    )
    args = np.append(args, cell["lake_temperature"])
    cell["virtual_lid_temperature"] = solver.lid_seb_solver(
        x, args, v_lid=True
    )[0][0]
    return Q


def freeze_virtual_lid(cell, new_boundary_change):
    if (new_boundary_change * cell["rho_ice"] / cell["rho_water"]) < cell[
        "lake_depth"
    ]:
        old_depth_grid = np.linspace(
            0, cell["lake_depth"], cell["vert_grid_lake"]
        )
        # now reduce the size of the lake due to freezing
        cell["lake_depth"] += new_boundary_change * (
            cell["rho_ice"] / cell["rho_water"]
        )

        # new grid - same # of vertical points, but with a reduced value
        new_depth_grid = np.linspace(
            0, cell["lake_depth"], cell["vert_grid_lake"]
        )
        cell["lake_temperature"] = np.interp(
            new_depth_grid, old_depth_grid, cell["lake_temperature"]
        )
        cell["v_lid_depth"] -= new_boundary_change
        cell["lake_temperature"][-1] = 273.15
    # whole lake freezes
    else:
        cell["v_lid_depth"] += cell["lake_depth"] * (
            cell["rho_water"] / cell["rho_ice"]
        )
        orig_lake_depth = cell["lake_depth"] + 0
        cell["lake_depth"] = 0
        cell["lake"] = False


def melt_virtual_lid(cell, Q, dt):
    cell["virtual_lid_temperature"] = 273.15
    k_im_lid = 1000 * (
        1.017 * 10**-4 + 1.695 * 10**-6 * cell["virtual_lid_temperature"]
    )
    kdTdz = (
        (cell["virtual_lid_temperature"] - 273.15)
        * abs(k_im_lid)
        / (
            cell["lake_depth"]
            / (cell["vert_grid_lake"] / 2 + cell["v_lid_depth"])
        )
    )
    new_boundary_change = (Q - kdTdz) / (cell["L_ice"] * cell["rho_ice"]) * dt

    if new_boundary_change > 0:
        # whole virtual lid melts
        if new_boundary_change > cell["v_lid_depth"]:
            cell["lake_depth"] += (
                cell["v_lid_depth"] * cell["rho_ice"] / cell["rho_water"]
            )
            cell["v_lid_depth"] = 0
            cell["total_melt"] = cell["total_melt"] + cell["v_lid_depth"]
        # some of the lid melts
        else:
            cell["lake_depth"] = (
                cell["lake_depth"]
                + new_boundary_change * cell["rho_ice"] / cell["rho_water"]
            )
            cell["v_lid_depth"] = cell["v_lid_depth"] - new_boundary_change
            cell["total_melt"] = cell["total_melt"] + new_boundary_change
    # check if we still have a virtual lid
    if cell["v_lid_depth"] <= 0 and cell["v_lid"]:
        print("Virtual lid no longer present")
        cell["v_lid"] = False
        cell["v_lid_depth"] = 0
