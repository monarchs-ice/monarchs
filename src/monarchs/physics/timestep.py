"""
Model timestepping module.
core.timestep_loop handles all the 1D calculations, by first checking the current
state of the column, then running firn_column and lake/lid functions accordingly.
"""

import numpy as np
from monarchs.physics import snow_accumulation
from monarchs.physics import firn_functions, lake_functions, solver, lid_functions
from monarchs.core import utils

def timestep_loop(cell, dt, met_data, t_steps_per_day, toggle_dict):
    """
    Main timestepping loop applied to an instance of the model grid.
    Called by loop_over_grid to work in parallel over multiple instances.

    Parameters
    ----------
    cell : numpy structured array
        Element of our model grid.
    dt : int32
        Time for each timestep <t_steps_per_day>. [s]


    met_data : numpy structured array
        Element of the met data grid. Contains thermodynamic variables, which
        are listed as follows:

        LW_in : int32 or float64
            Incoming broadband longwave radiation [W m^-2].
        SW_in : int32 or float64
             Incoming broadband shortwave radiation [W m^-2].
        T_air : int32 or float64
            Air temperature [K].
        p_air : int32 or float64
            Surface pressure [hPa].
        T_dp : int32 or float64
            Dew-point temperature [K]
        wind : int32 or float64
            Surface wind speed. [m s^-1]

    toggle_dict : dict
        A dictionary containing toggles that determine whether certain features are enabled.
        Values are the following:

    Returns
    -------
    None. The function amends the instance of <cell> passed to it.

    """
    parallel = toggle_dict["parallel"]
    use_numba = toggle_dict["use_numba"]
    snowfall_toggle = toggle_dict["snowfall_toggle"]
    firn_column_toggle = toggle_dict["firn_column_toggle"]
    firn_heat_toggle = toggle_dict["firn_heat_toggle"]
    lake_development_toggle = toggle_dict["lake_development_toggle"]
    lid_development_toggle = toggle_dict["lid_development_toggle"]
    ignore_errors = toggle_dict["ignore_errors"]

    if not cell["valid_cell"]:
        return cell


    if np.isnan(cell["firn_temperature"]).any():
        raise ValueError("NaN in firn temperature")
    cell["t_step"] = 1
    for t_step in range(t_steps_per_day):
        if cell["lake_depth"] == 0:
            cell["lake"] = False
        if cell["lid_depth"] == 0:
            cell["lid"] = False
        if cell["v_lid_depth"] == 0:
            cell["v_lid"] = False
        dz = cell["firn_depth"] / cell["vert_grid"]
        if snowfall_toggle:
            cell["rho"] = (
                cell["Sfrac"] * cell["rho_ice"] + cell["Lfrac"] * cell["rho_water"]
            )

            snow_accumulation.snowfall(
                cell, met_data['snowfall'][t_step], met_data['snow_dens'][t_step], 260
            )
        SW_in = met_data['SW_down'][t_step]
        LW_in = met_data['LW_down'][t_step]
        wind = met_data['wind'][t_step]
        T_dp = met_data['dew_point_temperature'][t_step]
        T_air = met_data['temperature'][t_step]
        p_air = met_data['surf_pressure'][t_step]

        cell["daily_melt"]=0.0

        """
        # Two main paths - either no exposed water, in which case the dry firn evolves, or we have exposed water,
        # in which case there are further branches depending on whether lakes or lids have formed yet.
        """

        if not cell["exposed_water"] and not cell["saturation"][0]:
            if firn_column_toggle:
                root0 = firn_functions.firn_column(
                    cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind, toggle_dict
                )

        elif cell["exposed_water"]:
            args = cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind
            x = cell["firn_temperature"]

            if firn_heat_toggle:
                sol, fvec, success, info = solver.firn_heateqn_solver(
                    x, args, fixed_sfc=True, solver_method='hybr'
                )
                if success:
                    cell["firn_temperature"] = sol
                else:
                    print("Warning - solver failed to converge - lake development")
                    print(x)
                    print(cell["lake"])
                    print(cell["lid"])
                    print(cell["lake_depth"])
                    print(success)
                    print(cell["firn_depth"])
                    print(cell["firn_temperature"])

            cell["rho"] = (
                cell["Sfrac"] * cell["rho_ice"] + cell["Lfrac"] * cell["rho_water"]
            )
            if (
                cell["lake"]
                and not cell["lid"]
                and not cell["v_lid"]
                and cell["lake_depth"] < 0.1
                and cell["v_lid_depth"] <= 0
            ):
                cell["lake"] = False

            if cell["lake_depth"] == 0:
                cell["exposed_water"] = False

            if not cell["lake"]:
                if lake_development_toggle:
                    lake_functions.lake_formation(
                        cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind, toggle_dict
                    )

            elif cell["lake"] and not cell["lid"]:
                if lake_development_toggle:
                    lake_functions.lake_development(
                        cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind, toggle_dict
                    )

                if cell["v_lid"]:
                    if lid_development_toggle:
                        lid_functions.virtual_lid(
                            cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind
                        )

            elif cell["lake"] and cell["lid"]:
                if lid_development_toggle:
                    cell["v_lid"] = False  # turn virtual lid off if full lid present
                    if lake_development_toggle:
                        lake_functions.lake_development(
                            cell,
                            dt,
                            LW_in,
                            SW_in,
                            T_air,
                            p_air,
                            T_dp,
                            wind,
                            toggle_dict,
                        )
                    lid_functions.lid_development(
                        cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind
                    )

                if (
                    cell["lake_depth"] <= 1E-5 or
                        np.any(cell['lake_temperature'][cell['lake_temperature'] > 273.15])
                        is False
                ):
                    lid_functions.combine_lid_firn(cell)

        if not ignore_errors:
            utils.check_correct(cell)

        cell["rho"] = (
            cell["Sfrac"] * cell["rho_ice"] + cell["Lfrac"] * cell["rho_water"]
        )

        cell["t_step"] += 1



    cell["day"] += 1

    if parallel and not use_numba:
        return cell
