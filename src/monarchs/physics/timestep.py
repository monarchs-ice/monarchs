"""
Model timestepping module.
core.timestep_loop handles all the 1D calculations, by first checking the
current state of the column, then running firn_column and lake/lid functions
accordingly.
"""

import numpy as np
from monarchs.physics import (
    firn_column,
    lake,
    solver,
    lid,
    percolation,
    virtual_lid,
    snow_accumulation,
    reset_column,
)
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
        A dictionary containing toggles that determine whether certain features
        are enabled.
        Values are the following:
        TODO - fill

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

    # track evolution of firn, lake and lid over time. We do this for each day,
    # so it resets at the start of the model day.
    cell["firn_boundary_change"] = 0
    cell["lake_boundary_change"] = 0
    cell["lid_boundary_change"] = 0

    if np.isnan(cell["firn_temperature"]).any():
        raise ValueError("NaN in firn temperature")
    cell["t_step"] = 1
    for t_step in range(t_steps_per_day):
        if cell["lake_depth"] <= 0 and cell["lake"]:
            cell["lake"] = False
            cell["lake_depth"] = 0
        if cell["lid_depth"] <= 0 and cell["lid"]:
            cell["lid"] = False
            cell["lid_depth"] = 0
        if cell["v_lid_depth"] <= 0 and cell["v_lid"]:
            print("Setting virtual lid to False at start of timestep")
            cell["v_lid"] = False
            cell["v_lid_depth"] = 0

        dz = cell["firn_depth"] / cell["vert_grid"]
        if snowfall_toggle:
            cell["rho"] = (
                cell["Sfrac"] * cell["rho_ice"]
                + cell["Lfrac"] * cell["rho_water"]
            )

            snow_accumulation.snowfall(
                cell,
                met_data["snowfall"][t_step],
                met_data["snow_dens"][t_step],
                273.15,
            )
        SW_in = met_data["SW_down"][t_step]
        LW_in = met_data["LW_down"][t_step]
        wind = met_data["wind"][t_step]
        T_dp = met_data["dew_point_temperature"][t_step]
        T_air = met_data["temperature"][t_step]
        p_air = met_data["surf_pressure"][t_step]

        """
        Two main paths - either no exposed water, in which case the dry firn
        evolves, or we have exposed water,
        in which case there are further branches depending on whether lakes
        or lids have formed yet.
        """

        if not cell["exposed_water"]:
            if firn_column_toggle:
                firn_column.firn_column(
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
                )

        elif cell["exposed_water"]:
            # print('Exposed water present')
            args = cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind
            x = cell["firn_temperature"]

            if firn_heat_toggle:
                sol, fvec, success, info = solver.firn_heateqn_solver(
                    x, args, fixed_sfc=True, solver_method="hybr"
                )
                if success:
                    cell["firn_temperature"] = sol
                else:
                    print(
                        "Warning - solver failed to converge - lake"
                        " development"
                    )
                    print(x)
                    print(cell["lake"])
                    print(cell["lid"])
                    print(cell["lake_depth"])
                    print(success)
                    print(cell["firn_depth"])
                    print(cell["firn_temperature"])

            cell["rho"] = (
                cell["Sfrac"] * cell["rho_ice"]
                + cell["Lfrac"] * cell["rho_water"]
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
                    lake.lake_formation(
                        cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind
                    )

            elif cell["lake"] and not cell["lid"]:
                if lake_development_toggle:
                    lake.lake_development(
                        cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind
                    )

                if cell["v_lid"]:
                    if lid_development_toggle:
                        virtual_lid.virtual_lid_development(
                            cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind
                        )


                    # if virtual lid freezes the entire lake (possible if
                    # lateral movement), then combine (virtual) lid and
                    # lake profiles.
                    if (
                        cell["lake_depth"] <= 1e-5
                        or np.any(
                            cell["lake_temperature"][
                                cell["lake_temperature"] > 273.15
                            ]
                        )
                        is False
                    ):
                        reset_column.combine_lid_firn(cell)

            elif cell["lake"] and cell["lid"]:
                if lid_development_toggle:
                    cell["v_lid"] = (
                        False  # turn virtual lid off if full lid present
                    )
                    lake.lake_development(
                        cell,
                        dt,
                        LW_in,
                        SW_in,
                        T_air,
                        p_air,
                        T_dp,
                        wind,
                    )
                    lid.lid_development(
                        cell,
                        dt,
                        LW_in,
                        SW_in,
                        T_air,
                        p_air,
                        T_dp,
                        wind,
                    )
                # If we have any of the following:
                #    very small lake depth
                #    any points in the lake below freezing
                #    The whole lake is below a threshold temperature
                # then we say that the lake is frozen, and combine it
                # with the lid and firn to create a singular profile again.
                # first determine if the lake temperature is below freezing.
                # If so, count.
                # If the count goes above 48 (48h of freezing), then freeze
                # the whole lake.
                if (cell["lake_temperature"] < 273.155).all():
                    cell["lake_refreeze_counter"] += 1
                else:
                    cell["lake_refreeze_counter"] = 0
                if cell["lake_refreeze_counter"] > 47:
                    print('Lake has been very cold for two full diurnal cycles'
                          '- freezing...')
                    reset_column.combine_lid_firn(cell)

                if (
                    cell["lake_depth"] <= 1e-5
                    or (
                        np.any(
                            cell["lake_temperature"][
                                cell["lake_temperature"] > 273.15
                            ]
                        )
                        is False
                    )
                    or (cell["lid_melt_count"] > 24)
                ):
                    reset_column.combine_lid_firn(cell)

        # If we have Sfrac + Lfrac > 1, we need to ensure that Lfrac is
        # adjusted accordingly.
        # We can do this via calc_saturation, which is used to similar effect
        # in the percolation step.
        if np.any(cell["Lfrac"] + cell["Sfrac"] > 1):
            # Get the lowest point at which the column is saturated, and use
            # this as the starting point for the saturation algorithm.
            # Then find the other points that are saturated and perform the
            # same calculation here also.
            saturation_points = np.where(cell["Lfrac"] + cell["Sfrac"] > 1)
            for saturation_point in saturation_points[::-1][0]:
                percolation.calc_saturation(cell, saturation_point, end=True)
            # Check again. We however will tolerate any instances where the
            # solid + liquid fraction goes above 1 (unless Sfrac is above 1).
            # This is because for small grid spacings, a layer may become
            # saturated as a result of the regridding.
            # Instead of hacking it into an adjacent cell, we just let it
            # percolate in the next timestep.
            if (
                np.any(cell["Lfrac"][1:] + cell["Sfrac"][1:] > 1)
                and cell["Sfrac"][0] <= 1
            ):
                raise ValueError(
                    "Lfrac + Sfrac > 1 after regridding, after saturation"
                    " calculation."
                )

        if not ignore_errors:
            utils.check_correct(cell)

        cell["rho"] = (
            cell["Sfrac"] * cell["rho_ice"] + cell["Lfrac"] * cell["rho_water"]
        )

        cell["t_step"] += 1
        # Update vertical profile to ensure we account for any firn depth
        # changes
        cell["vertical_profile"] = np.linspace(
            0, cell["firn_depth"], cell["vert_grid"]
        )

    # If firn depth goes below 10, then we now consider this cell to be
    # invalid.
    if cell["firn_depth"] < 10:
        print("Firn depth below 10 m - setting cell to invalid")
        cell["valid_cell"] = False

    cell["day"] += 1

    if parallel and not use_numba:
        return cell
