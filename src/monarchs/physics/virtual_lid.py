""" """

# disable pylint warnings for broad exceptions as they are needed with Numba
# pylint: disable=broad-exception-raised
# TODO - module-level docstring, other docstrings
import numpy as np
from monarchs.core import utils
from monarchs.core.error_handling import check_for_mass_conservation
from monarchs.physics import surface_fluxes, solver
from monarchs.physics.constants import L_ice, rho_ice, rho_water
MODULE_NAME = "monarchs.physics.virtual_lid"


def virtual_lid_development(
    cell, dt, met_data, turbulent_flux_upper
):
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
    lw_in : float
        Incoming longwave radiation. [W m^-2].
    sw_in : float
        Incoming shortwave (solar) radiation. [W m^-2].
    air_temp : float
        Surface-layer air temperature. [K].
    p_air : float
        Surface-layer air pressure. [hPa].
    dew_point_temperature : float
        Dew-point temperature at the surface. [K]
    wind : float
        Wind speed. [m s^-1].

    Returns
    -------
    None (amends cell inplace)
    """
    routine_name = f"{MODULE_NAME}.virtual_lid_development"
    original_mass = utils.calc_mass_sum(cell)

    # JE - As with the lake, we need to calculate the surface energy based
    # on how much radiation is actually absorbed at the surface, which is
    # not going to be *all* of it, even it it may be a bit higher than
    # for a lake.
    # Update cell albedo
    cell["albedo"] = surface_fluxes.sfc_albedo(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["v_lid"],
        cell["lake_depth"],
        cell["snow_on_lid"]
    )

    # Surface energy balance calculation to determine the virtual lid
    # temperature
    # want only root, not fvec etc, and root is an array of one element so
    # extract out first elem
    update_virtual_lid_temperature(cell, met_data, dt)
    k_ice = 2.24
    # a conductive approach across the virtual lid rather than using the fluxes
    # until a true lid forms (at which point the real fluxes are used).
    # this stops a blowout due to low virtual lid depth

    # for freezing - negative *net* flux = freezing, positive = melting
    # as convention is +ve downwards; upward flux means energy leaving the
    # interface
    effective_thickness = max(cell["v_lid_depth"], 1e-5)
    delta_T = cell["virtual_lid_temperature"] - 273.15

    # conductive flux - via our positive downwards convention, this is negative
    # if we have a cold virtual lid (i.e. heat is flowing up, out of the interface)
    # since interface loses energy to the lid, this causes freezing
    conductive_flux_down = (k_ice * delta_T) / effective_thickness
    # conversely, the turbulent flux from the lake (again, positive downwards) will
    # inhibit freezing if it is negative (i.e. heat flows from the lake into the
    # interface).
    # the net flux at the interface is therefore (flux entering - flux leaving)
    # with a positive downwards convention, we have to subtract the two as they are coming
    # from different sides
    q_net = conductive_flux_down - turbulent_flux_upper
    new_boundary_change = - q_net * dt / (L_ice * rho_ice)

    # further freezing of the virtual lid
    if new_boundary_change > 0:
        freeze_virtual_lid(cell, new_boundary_change)
    # melting of the virtual lid
    else:
        melt_virtual_lid(cell, new_boundary_change)

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
    check_for_mass_conservation(cell, original_mass, new_mass, routine_name)


def update_virtual_lid_temperature(
    cell, met_data, dt
):
    # Update cell albedo
    cell["albedo"] = surface_fluxes.sfc_albedo(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["v_lid"],
        cell["lake_depth"],
        cell["snow_on_lid"]
    )

    k_v_lid = 1000 * (
        2.24e-3
        + 5.975e-6 * (273.15 - cell["virtual_lid_temperature"]) ** 1.156
    )

    #print('Temp before virtual lid update:', cell["virtual_lid_temperature"])
    cell["virtual_lid_temperature"] = solver.lid_seb_solver(
        cell, met_data, dt, 0, k_v_lid
    )[0][0]

    # debug
    # cell["virtual_lid_temperature"] = min(
    #     cell["virtual_lid_temperature"], 275.15
    # )
    #print('Temp after virtual lid update:', cell["virtual_lid_temperature"])


def freeze_virtual_lid(cell, ice_added):
    if (ice_added * rho_ice / rho_water) < cell[
        "lake_depth"
    ]:
        old_depth_grid = np.linspace(
            0, cell["lake_depth"], cell["vert_grid_lake"]
        )
        # now reduce the size of the lake due to freezing
        cell["lake_depth"] -= ice_added * (
            rho_ice / rho_water
        )

        # new grid - same # of vertical points, but with a reduced value
        new_depth_grid = np.linspace(
            0, cell["lake_depth"], cell["vert_grid_lake"]
        )
        cell["lake_temperature"] = np.interp(
            new_depth_grid, old_depth_grid, cell["lake_temperature"]
        )
        cell["v_lid_depth"] += ice_added
        cell["lake_temperature"][0] = 273.15
        #print('Partial freezing of virtual lid')
        #print('ice_added:', ice_added)
        #print('v_lid_depth:', cell["v_lid_depth"])
    # whole lake freezes
    else:
        cell["v_lid_depth"] += cell["lake_depth"] * (
            rho_water / rho_ice
        )
        cell["lake_depth"] = 0
        #print('Whole lake frozen into virtual lid')
        #print('ice_added:', ice_added)


def melt_virtual_lid(cell, ice_added):
    cell["virtual_lid_temperature"] = 273.15
    ice_removed = - ice_added #  negative ice_added means melting
    if ice_removed > 0:
        # whole virtual lid melts
        if ice_removed > cell["v_lid_depth"]:
            cell["lake_depth"] += (
                cell["v_lid_depth"] * rho_ice / rho_water
            )
            #print('Virtual lid depth before melting:', cell["v_lid_depth"])
            #print('ice_removed:', ice_removed)
            cell["total_melt"] = cell["total_melt"] + cell["v_lid_depth"]
            cell["v_lid_depth"] = 0
            #print('Whole virtual lid melted')
        # some of the lid melts
        else:
            cell["lake_depth"] = (
                cell["lake_depth"]
                + ice_removed * rho_ice / rho_water
            )
            cell["v_lid_depth"] = cell["v_lid_depth"] - ice_removed
            cell["total_melt"] = cell["total_melt"] + ice_removed
            cell["lake_temperature"][0] = 273.15

            #print('Partial melting of virtual lid')
    # check if we still have a virtual lid
    if cell["v_lid_depth"] <= 0 and cell["v_lid"]:
        #print('Whole virtual lid melted - step 2 ')

        cell["v_lid"] = False
        cell["v_lid_depth"] = 0
    cell["snow_on_lid"] = 0
