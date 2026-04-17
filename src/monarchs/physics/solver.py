"""
This module contains functions used to solve the various heat/surface energy
balance equations, using the Scipy.optimize.fsolve implementation.
This is rather unwieldy, but allows us to select a solver conditionally,
and do so both in the main model code using the relevant flag in
``model_setup.py``, but also in our test suite.
This is in part a tradeoff between usability of the model, and code clarity.
This approach was chosen to maximise usability, so that the different solvers
can be generated according to the value of a single Boolean.
"""

import numpy as np
from scipy.optimize import fsolve, root
from monarchs.physics import heateqn, surface_fluxes
from monarchs.physics.constants import (
    emissivity,
    stefan_boltzmann,
    k_air,
    k_water,
)


def solve_firn_heateqn(
    cell, met_data, dt, dz, fixed_sfc=False, solver_method="hybr", toggle_dict=None
):
    """
    Dispatcher for firn heat equation solver.

    Called in <firn_column>, <timestep>, <lake>.

    Parameters
    ----------
    cell : structured array
        Element of the model grid we are operating on
    met_data : structured array
        Meteorological data for current timestep (LW_down, SW_down, temperature,
        surf_pressure, dew_point_temperature, wind)
    dt : float
        Timestep [s]
    dz : float
        Layer thickness [m]
    fixed_sfc : bool, optional
        If True, use fixed surface temperature (273.15 K). Default False.
    solver_method : str, optional
        Solver method (ignored, accepted for backwards compatibility). Default "hybr".


    Returns
    -------
    T : array_like, float
        Temperature profile [K]
    infodict : dict
        Diagnostic info dict
    ier : int
        Success flag (1 if converged)
    mesg : str
        Status message
    """

    lw_in = met_data["LW_down"]
    sw_in = met_data["SW_down"]
    air_temp = met_data["temperature"]
    p_air = met_data["surf_pressure"]
    dew_point_temperature = met_data["dew_point_temperature"]
    wind = met_data["wind"]

    if fixed_sfc:
        infodict = {}
        ier = 1
        mesg = "Fixed surface temperature"
        T_tri = heateqn.propagate_temperature(cell, dz, dt, 273.15, N=1)
        T = np.concatenate((np.array([273.15]), T_tri))
    else:

        N = cell["vert_grid"]
        x = cell["firn_temperature"][:N]
        x = np.asarray(x)

        soldict = root(
            heateqn.heateqn,
            x,
            args=(
                cell,
                lw_in,
                sw_in,
                air_temp,
                p_air,
                dew_point_temperature,
                wind,
                dz,
                dt,
            ),
            method=solver_method,
        )

        if not soldict.success:
            print(
                "Root-finding for surface temperature failed - returning"
                f" original guess. row = {cell['row']}, col = {cell['column']}"
            )

        if N == cell["vert_grid"]:
            return soldict.x, soldict, soldict.success, soldict.message
        else:
            sol = soldict.x

        ier = soldict.success
        mesg = soldict.message
        infodict = soldict.success
        # Take our root-finding algorithm output (from first N layers),
        # use it as the top boundary condition to the tridiagonal solver,
        # then concatenate the two - use tridiagonal solver to solve
        # the heat equation once we have the surface temp

        T_tri = heateqn.propagate_temperature(cell, dz, dt, sol[-1], N=N)
        T = np.concatenate((sol[:], T_tri))
    T = np.around(T, decimals=8)

    return T, infodict, ier, mesg


######################
# EQUATIONS TO SOLVE #
######################
def lake_formation_eqn(x, args):
    """
    Refactored lake formation eqn with dynamic surface flux.
    """
    (
        albedo,
        lid,
        lake,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        k,
        T1,
        dz,
    ) = args

    # Calculate Q dynamically based on the solver's current guess x[0]
    Q = surface_fluxes.sfc_flux(
        albedo,
        lid,
        lake,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        x[0],
    )

    # Now the residual accounts for changes in sensible/latent heat
    output = np.array(
        [-emissivity * stefan_boltzmann * x[0] ** 4 + Q - k * (x[0] - T1) / dz]
    )
    return output


def lake_development_eqn(x, args):
    """
    Scipy-compatible form of the lake development version of the surface
    temperature equation.  Called in lake_seb_solver.

    Parameters
    ----------
    x : array_like, float, shape (1,)
        Current estimate of the lake surface temperature [K].
    args : tuple
        ``(albedo, lid, lake, lw_in, sw_in, air_temp, p_air,
        dew_point_temperature, wind, lake_temperature, vert_grid_lake)``

    Returns
    -------
    output : array_like, float, shape (1,)
        Residual of the surface energy balance equation.
    """
    J = 0.1 * (9.8 * 5e-5 * (1.19e-7) ** 2 / 1e-6) ** (1 / 3)

    (
        albedo,
        lid,
        lake,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        lake_temperature,
        vert_grid_lake,
    ) = args

    Q = surface_fluxes.sfc_flux(
        albedo,
        lid,
        lake,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        x[0],
    )

    T_core = lake_temperature[int(vert_grid_lake / 2)]
    output = np.array(
        [
            -emissivity * stefan_boltzmann * x[0] ** 4
            + Q
            + np.sign(T_core - x[0]) * 1000 * 4181 * J * abs(T_core - x[0]) ** (4 / 3)
        ]
    )
    return output


def sfc_energy_virtual_lid(x, args):
    """
    Surface energy balance for the virtual lid.
    Called in lid_seb_solver when ``cell["v_lid"]`` is True.

    Parameters
    ----------
    x : array_like, float, shape (1,)
        Current estimate of the virtual-lid surface temperature [K].
    args : tuple
        ``(albedo, lid, lake, lw_in, sw_in, air_temp, p_air,
        dew_point_temperature, wind, k_v_lid, lake_depth,
        vert_grid_lake, v_lid_depth)``

    Returns
    -------
    output : array_like, float, shape (1,)
        Residual of the surface energy balance equation.
    """
    (
        albedo,
        lid,
        lake,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        k_v_lid,
        lake_depth,
        vert_grid_lake,
        v_lid_depth,
    ) = args

    Q = surface_fluxes.sfc_flux(
        albedo,
        lid,
        lake,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        x[0],
    )

    # Combined thermal resistance: virtual-lid ice + half-lake water column
    total_thickness = v_lid_depth + lake_depth / (vert_grid_lake / 2)
    conduction = k_v_lid * (x[0] - 273.15) / total_thickness

    output = np.zeros(1)
    output[0] = Q - emissivity * stefan_boltzmann * x[0] ** 4 - conduction
    return output


def sfc_energy_lid(x, args):
    """
    Surface energy balance for the frozen lid.
    Called in lid_seb_solver when ``cell["v_lid"]`` is False.

    Parameters
    ----------
    x : array_like, float, shape (1,)
        Current estimate of the lid surface temperature [K].
    args : tuple
        ``(albedo, lid, lake, lw_in, sw_in, air_temp, p_air,
        dew_point_temperature, wind, k_lid, lid_depth,
        vert_grid_lid, sub_T)``

        sub_T : float
            Temperature of the first lid layer used as the sub-surface
            boundary condition [K].

    Returns
    -------
    output : array_like, float, shape (1,)
        Residual of the surface energy balance equation.
    """
    (
        albedo,
        lid,
        lake,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        k_lid,
        lid_depth,
        vert_grid_lid,
        sub_T,
    ) = args

    Q = surface_fluxes.sfc_flux(
        albedo,
        lid,
        lake,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        x[0],
    )

    output = np.zeros(1)
    output[0] = (
        -emissivity * stefan_boltzmann * x[0] ** 4
        + Q
        - k_lid * (x[0] - sub_T) / (lid_depth / vert_grid_lid)
    )
    return output


######################
# DISPATCH FUNCTIONS #
######################


def lake_seb_solver(cell, met_data, dt, dz, formation=False):
    """
    Scipy version of the lake solver.
    Solves lake_development_eqn or lake_formation_eqn.

    Values are read directly from ``cell`` and ``met_data``, consistent with
    the Numba solver in solver_nb.py.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    met_data : dict
        Meteorological data for the current timestep.
    dt : float
        Timestep [s].
    dz : float
        Layer thickness [m].
    formation : bool, optional
        If True, solve the lake *formation* equation; otherwise solve the lake
        *development* equation.  Default False.

    Returns
    -------
    sol : array_like, float
        Calculated lake surface temperature [K].
    infodict : dict
        Diagnostic output from fsolve.
    ier : int
        1 if a solution was found, otherwise see mesg.
    mesg : str
        Status message from fsolve.
    """
    lw_in = met_data["LW_down"]
    sw_in = met_data["SW_down"]
    air_temp = met_data["temperature"]
    p_air = met_data["surf_pressure"]
    dew_point_temperature = met_data["dew_point_temperature"]
    wind = met_data["wind"]

    if formation:
        eqn = lake_formation_eqn
        # initial guess
        x = np.array([air_temp])

        # firn properties
        T0 = float(cell["firn_temperature"][0])
        sfrac0 = float(cell["Sfrac"][0])
        lfrac0 = float(cell["Lfrac"][0])
        k_ice = 1000.0 * (2.24e-3 + 5.975e-6 * (273.15 - T0) ** 1.156)
        k = sfrac0 * k_ice + lfrac0 * k_water + (1.0 - sfrac0 - lfrac0) * k_air
        T1 = float(cell["firn_temperature"][1])
        dz_firn = float(cell["firn_depth"]) / float(cell["vert_grid"])

        # pack args to make consistent with Numba implementation
        args = (
            cell["albedo"],
            cell["lid"],
            cell["lake"],
            lw_in,
            sw_in,
            air_temp,
            p_air,
            dew_point_temperature,
            wind,
            k,
            T1,
            dz_firn,
        )
    else:
        eqn = lake_development_eqn
        x = np.array([cell["lake_temperature"][0]])
        args = (
            cell["albedo"],
            cell["lid"],
            cell["lake"],
            lw_in,
            sw_in,
            air_temp,
            p_air,
            dew_point_temperature,
            wind,
            cell["lake_temperature"],
            int(cell["vert_grid_lake"]),
        )

    # solve the equation
    sol, infodict, ier, mesg = fsolve(eqn, x, args=(args,), full_output=True)
    sol = np.around(sol, decimals=8)
    return sol, infodict, ier, mesg


def lid_seb_solver(cell, met_data, dt, dz, k_lid, Sfrac_lid=None):
    """
    Scipy version of the lid surface energy balance solver.
    Solves sfc_energy_lid or sfc_energy_virtual_lid.

    Values are read directly from ``cell`` and ``met_data``, consistent with
    the Numba solver in solver_nb.py.

    Called in virtual_lid and lid.lid_development.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    met_data : dict
        Meteorological data for the current timestep.
    dt : float
        Timestep [s].
    dz : float
        Layer thickness [m].
    k_lid : float
        Thermal conductivity of the lid [W m^-1 K^-1].
    Sfrac_lid : array_like, optional
        Solid fraction of the lid.  Unused in the scipy path but accepted for
        API parity with solver_nb.py.

    Returns
    -------
    sol : array_like, float
        Calculated lid surface temperature [K].
    infodict : dict
        Diagnostic output from fsolve.
    ier : int
        1 if a solution was found, otherwise see mesg.
    mesg : str
        Status message from fsolve.
    """
    lw_in = met_data["LW_down"]
    sw_in = met_data["SW_down"]
    air_temp = met_data["temperature"]
    p_air = met_data["surf_pressure"]
    dew_point_temperature = met_data["dew_point_temperature"]
    wind = met_data["wind"]

    # Initial guess: current virtual-lid surface temperature (matches solver_nb.py)
    x = np.array([float(cell["virtual_lid_temperature"])])

    if cell["v_lid"]:
        eqn = sfc_energy_virtual_lid
        args = (
            cell["albedo"],
            cell["lid"],
            cell["lake"],
            lw_in,
            sw_in,
            air_temp,
            p_air,
            dew_point_temperature,
            wind,
            float(k_lid),
            float(cell["lake_depth"]),
            int(cell["vert_grid_lake"]),
            float(cell["v_lid_depth"]),
        )
    else:
        eqn = sfc_energy_lid
        # sub_T: first element of the lid temperature array, used as the
        # sub-surface BC (matches Numba sfc_energy_lid which reads
        # lid_temperature[0] from the packed args).
        sub_T = float(cell["lid_temperature"][0])
        args = (
            cell["albedo"],
            cell["lid"],
            cell["lake"],
            lw_in,
            sw_in,
            air_temp,
            p_air,
            dew_point_temperature,
            wind,
            float(k_lid),
            float(cell["lid_depth"]),
            int(cell["vert_grid_lid"]),
            sub_T,
        )

    # fsolve calls func(x, *args), so wrap args tuple to preserve (x, args) signature.
    sol, infodict, ier, mesg = fsolve(eqn, x, args=(args,), full_output=True)
    sol = np.around(sol, decimals=8)
    return sol, infodict, ier, mesg


def lid_heateqn_solver(cell, met_data, dt, dz):
    """
    Scipy version of the lid heat equation solver.
    Solves heateqn.heateqn_lid.

    Values are read directly from ``cell`` and ``met_data``, consistent with
    the Numba solver in solver_nb.py.

    Called in lid.lid_development.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    met_data : dict
        Meteorological data for the current timestep.
    dt : float
        Timestep [s].
    dz : float
        Layer thickness [m].

    Returns
    -------
    sol : array_like, float, dimension(cell.vert_grid_lid)
        Calculated lid column temperature [K].
    infodict : dict
        Diagnostic output from fsolve.
    ier : int
        1 if a solution was found, otherwise see mesg.
    mesg : str
        Status message from fsolve.
    """
    lw_in = met_data["LW_down"]
    sw_in = met_data["SW_down"]
    air_temp = met_data["temperature"]
    p_air = met_data["surf_pressure"]
    dew_point_temperature = met_data["dew_point_temperature"]
    wind = met_data["wind"]

    eqn = heateqn.heateqn_lid
    x = cell["lid_temperature"].copy()

    n_lid = int(cell["vert_grid_lid"])
    Sfrac_lid = np.ones(n_lid, dtype=np.float64)

    args = (
        cell,
        dt,
        dz,
        lw_in,
        sw_in,
        air_temp,
        p_air,
        dew_point_temperature,
        wind,
        Sfrac_lid,
    )

    sol, infodict, ier, mesg = fsolve(eqn, x, args=args, full_output=True)
    sol = np.around(sol, decimals=8)
    return sol, infodict, ier, mesg
