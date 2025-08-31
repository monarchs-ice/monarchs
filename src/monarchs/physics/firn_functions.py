"""
Module containing functions relating to the firn column. Some physics is contained in percolation.py.
"""

import numpy as np
from monarchs.physics import percolation_functions
from monarchs.physics import surface_fluxes
from monarchs.physics import solver
from monarchs.core import utils


def firn_column(
    cell,
    dt,
    dz,
    LW_in,
    SW_in,
    T_air,
    p_air,
    T_dp,
    wind,
    toggle_dict,
    prescribed_height_change=False,
):
    """
    Perform the various processes applied to the firn each timestep where we don't have exposed water at the surface.

    The logic works as follows:
    Solve heat equation for firn

    If surface temperature is above melting point of water
    (i,e melting takes place):

    - Determine the height change as a result of the melting
    - Regrid everything to take account for this deformation
    - Re-solve heat equation, now with fixed surface temperature
    - Calculate the amount of water added to the surface as a result of melt
    - Percolate that water down, taking into account any lids or lakes that may have formed.

    Otherwise:
    - Update cell temperature and continue.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    dt : int
        Number of seconds in the current timestep [s]
    dz : float
        Height of each vertical point in the cell. [m]
    LW_in : float
        Downwelling longwave radiation at the current timestep. [W m^-2]
    SW_in : float
        Downwelling shortwave radiation at the current timestep. [W m^-2]
    T_air : float
        Surface air temperature at the current timestep. [K]
    p_air : float
        Surface air pressure at the current timestep. [Pa]
    T_dp : float
        Dewpoint temperature of the air at the surface at the current timestep. [K]
    wind : float
        Wind speed at the surface at the current timestep. [m s^-1]
    toggle_dict : dict
        Dictionary containing some switches that affect the running of the model.

    Returns
    -------
    None (amends cell inplace)
    """
    # calculate mass before any melting occurs to use for a diagnostic later
    original_mass = utils.calc_mass_sum(cell)
    percolation_toggle = toggle_dict["percolation_toggle"]
    perc_time_toggle = toggle_dict["perc_time_toggle"]
    # initial guess for the heat equation solver
    x = cell["firn_temperature"]

    # Solve the 1D heat equation. This needs to account for the top boundary condition, which is driven
    # by the meteorology/surface fluxes, so we need to pass these variables into the solver.
    # Since we are using a Numba cfunc to solve the heat equation, we need to pack the arguments
    # into a single vector.
    args = [cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind]
    # Hard-coding 'hybr' as the MINPACK solver method, as other methods are not supported by the Numba cfunc.
    # If you are running with Scipy, you could consider changing this or adding it as a namelist parameter.
    root, fvec, success, info = solver.firn_heateqn_solver(
        x, args, fixed_sfc=False, solver_method='hybr'
    )

    # If the solver didn't fail (e.g. due to too many iterations), and we have a surface temperature above
    # the freezing point, then melt will occur.
    # Since the firn column has a fixed boundary condition (273.15 K), recalculate the firn column temperature
    # and regrid the column to account for the height change that occurs due to melting.
    if (root[0] > 273.15) and success:
        cell["meltflag"][0] = 1
        cell["melt"] = True
        height_change = calc_height_change(
            cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind, root
        )
        print('Height change = ', height_change)

        if np.isnan(height_change):
            raise ValueError("Height change is NaN - likely due to unrealistic meteorological data.")

        dz = cell["firn_depth"] / cell["vert_grid"]
        args = cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind
        root, fvec, success_fixedsfc, info = solver.firn_heateqn_solver(
            x, args, fixed_sfc=True, solver_method='hybr'
        )
        # if *this* solver works, then update the firn temperature and regrid the firn column, accounting
        # for the melt.
        if success_fixedsfc:
            cell["firn_temperature"] = root
            regrid_after_melt(cell, height_change)

    # If the surface temperature is below the melting point, then we simply update the firn temperature provided
    # the solver didn't fail.
    elif success:
        cell["firn_temperature"] = root
        cell["melt"] = False

    # Run the percolation algorithm to ensure that any meltwater moves down the column (refreezing as it goes).
    if percolation_toggle:
        percolation_functions.percolation(cell, dt, perc_time_toggle=perc_time_toggle)

    # Update density to reflect newly calculated solid/liquid fractions
    cell["rho"] = cell["Sfrac"] * cell["rho_ice"] + cell["Lfrac"] * cell["rho_water"]

    # Test for mass conservation - if we lose mass, then there is an issue with the regridding.
    new_mass = utils.calc_mass_sum(cell)
    assert abs(original_mass - new_mass) < 1.5 * 10**-7

def regrid_after_melt(cell, height_change, lake=False):
    """
    After melting occurs, subtract the amount of melting from the firn height, convert it into meltwater,
    and interpolate the entire column to the new vertical profile accounting for this height change.
    This meltwater is either converted into surface liquid water fraction, or if there is a lake, into lake height.


    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    height_change : float
        Change in the firn height as a result of melting. [m]
    lake : bool, optional
        Flag to determine whether a lake is present or not. This is contained here so that we can re-use the bulk
        of this algorithm, but with some small changes to reflect the different situation that occurs when a lake is
        present.

    Returns
    -------
    None
    """
    original_mass = utils.calc_mass_sum(cell)

    old_depth = cell["firn_depth"]
    dz_old = old_depth / cell["vert_grid"]

    new_depth = old_depth - height_change
    cell["firn_depth"] = new_depth
    original_sfrac = np.copy(cell["Sfrac"])
    original_lfrac = np.copy(cell["Lfrac"])
    # we want to conserve the *bottom* of the grid, not the top. So the correct
    # way to interpolate is to use depth as a coordinate not height, with the
    # new top boundary being the lowered surface and the final depth being the same.
    # This is crucially different to interpolating from 0 to the new depth!
    # Define old and new vertical edges for layers
    old_edges = np.linspace(0, old_depth, cell["vert_grid"] + 1)
    new_edges = np.linspace(height_change, old_depth, cell["vert_grid"] + 1)  # shift down by melt height


    # a subtlety. if the height change is greater than dz_old, then
    # the Sfrac we need to use is not just the top layer Sfrac, but a weighted average
    # of the melted layers. This may be more than one layer.
    if height_change > dz_old:
        # the weights are just the fraction of each layer that is melted
        weights = np.ones(int(height_change / dz_old) + 1)
        weights[-1] = (height_change % dz_old) / dz_old  # partial layer at the bottom
        w_sum = np.sum(weights)
        sfrac_weight = np.sum(original_sfrac[: int(height_change / dz_old) + 1] * weights) / w_sum
    else:
        sfrac_weight = original_sfrac[0]

    meltwater_mass = height_change * cell['rho_ice'] * sfrac_weight
    # Determine the original column mass for later.
    original_column_mass = original_mass - meltwater_mass

    # to conserve mass we need meltwater_mass = new_height * cell['rho_water']
    meltwater_height = meltwater_mass / cell["rho_water"]
    # but we now need to convert that into a liquid fraction. we do this by
    # fitting this new meltwater height into the top level of the column
    # meltwater = meltwater_height / dz_old
    meltwater = meltwater_height / dz_old
    cell["Lfrac"][0] += meltwater

    new_sfrac = conservative_regrid(old_edges, cell["Sfrac"], new_edges)
    new_lfrac = conservative_regrid(old_edges, cell["Lfrac"], new_edges)
    cell['firn_temperature'] = conservative_regrid(old_edges, cell["firn_temperature"], new_edges)
    cell['Sfrac'] = new_sfrac
    # Lfrac is handled a bit differently. Scaling it by the volume ratio ensures that it is
    # conserved. This can be done since we are not removing water, we are adding it, unlike
    # with the solid fraction which is removed (and therefore has to be specifically regridded)
    volume_ratio = old_depth / new_depth
    scaled_lfrac = cell["Lfrac"] * volume_ratio
    cell['Lfrac'] = scaled_lfrac
    dz_new = new_depth / cell["vert_grid"]
    new_mass = utils.calc_mass_sum(cell)
    # We need our interpolation to conserve mass.
    # Therefore, we may need to scale Sfrac and LFrac to compensate.



    if lake:
        print('Lfrac[0] = ', cell['Lfrac'][0])

        if cell["Lfrac"][0] + cell["Sfrac"][0] > 1:
            excess_water = cell["Lfrac"][0] + cell["Sfrac"][0] - 1
            print('excess water = ', excess_water)
            cell["lake_depth"] += excess_water * (
                cell["firn_depth"] / cell["vert_grid"]
            )
            cell["Lfrac"][0] = 1 - cell["Sfrac"][0]

    cell["vertical_profile"] = np.linspace(0, cell["firn_depth"], cell["vert_grid"])
    # don't clip top layer as may have meltwater we want to percolate later
    cell['Lfrac'][1:] = np.clip(cell['Lfrac'][1:], 0, 1)

    try:
        assert abs(new_mass - original_mass) < 1 * 10**-2
    except Exception:
        print(new_mass)
        print(original_mass)
        raise AssertionError


def calc_height_change(cell, timestep, LW_in, SW_in, T_air, p_air, T_dp, wind, surf_T):
    """
    Determine the amount of firn height change that arises due to melting.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    timestep : float
        Number of seconds in each timestep. [s]
    LW_in : float
        Downwelling longwave radiation at the current timestep. [W m^-2]
    SW_in : float
        Downwelling shortwave radiation at the current timestep. [W m^-2]
    T_air : float
        Surface air temperature at the current timestep. [K]
    p_air : float
        Surface air pressure at the current timestep. [Pa]
    T_dp : float
        Dewpoint temperature of the air at the surface at the current timestep. [K]
    wind : float
        Wind speed at the surface at the current timestep. [m s^-1]
    surf_T : float
        Calculated surface temperature of the firn from the initial (non-fixed surface) implementation of the heat
        equation. [K]

    Returns
    -------

    """
    epsilon = 0.98
    sigma = 5.670373 * 10**-8
    dz = cell["firn_depth"] / cell["vert_grid"]
    L_fus = 334000
    if cell["firn_temperature"][0] > 273.14999999 and cell["firn_temperature"][0] < 273.151:
        cell["firn_temperature"][0] = 273.15
    if cell["firn_temperature"][1] > 273.14999999 and cell["firn_temperature"][1] < 273.151:
        cell["firn_temperature"][1] = 273.15
    k_sfc = (
        1000 * 2.24 * 10**-3
        + 5.975
        * 10**-6
        * (273.15 - (cell["firn_temperature"][0] + cell["firn_temperature"][1]) / 2)
        ** 1.156
    )
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
        surf_T[0],
    )
    dHdt = (Q - epsilon * sigma * (
                1.5 * (cell['firn_temperature'][0] - 0.5 * cell['firn_temperature'][1])) ** 4 -
            k_sfc * ((cell['firn_temperature'][0] - cell['firn_temperature'][1]) / dz)) / (
                       cell['rho_ice'] * (cell['Sfrac'][0] * L_fus)) * timestep
    print('dHdt = ', dHdt)
    if 0 > dHdt > -0.01:
        dHdt = 0
    elif dHdt < -0.01:
        raise ValueError(
            "Height change during melt is negative, and outside the bounds of a numerical error"
        )
    elif np.isnan(dHdt):
        pass
        print("...")
        raise ValueError("Height change during melt is NaN")
    return dHdt


def interp_nb(x_vals, x, y):
    """
    Wrapper function for np.interp. This function exists purely so that an alternative interpolation algorithm can be
    used throughout the code without needing to change every instance. Named interp_nb as this function also has
    Numba compatibility in its default form.

    Parameters
    ----------
    x_vals : array_like
        New coordinates that we want to interpolate our input y values onto.
    x : array_like
        Original coordinates of our y values.
    y : array_like
        Values we want to interpolate.

    Returns
    -------
    res : array_like
        values from y, interpolated onto our new grid of x_vals
    """
    res = np.interp(x_vals, x, y)
    return res

def conservative_regrid(old_edges, old_values, new_edges):
    """
    Mass-conserving remapping of column values from old grid to new grid.

    Parameters
    ----------
    old_edges : 1D numpy array of floats, length M+1
        The edges of old vertical layers (depth boundaries).
    old_values : 1D numpy array of floats, length M
        Values in old layers (assumed piecewise-constant).
    new_edges : 1D numpy array of floats, length N+1
        The edges of new vertical layers (depth boundaries).

    Returns
    -------
    new_values : 1D numpy array of floats, length N
        Values remapped onto the new grid conservatively.
    """
    old_dz = np.diff(old_edges)
    old_mass_cum = np.zeros(len(old_edges))
    old_mass_cum[1:] = np.cumsum(old_values * old_dz)

    new_values = np.zeros(len(new_edges) - 1)

    for i in range(len(new_values)):
        z_start = new_edges[i]
        z_end = new_edges[i + 1]

        # Integrate cumulative mass at the start and end of new layer
        mass_start = np.interp(z_start, old_edges, old_mass_cum)
        mass_end = np.interp(z_end, old_edges, old_mass_cum)

        layer_mass = mass_end - mass_start
        layer_dz = z_end - z_start
        if layer_dz > 0:
            new_values[i] = layer_mass / layer_dz
        else:
            new_values[i] = 0.0
    return new_values

