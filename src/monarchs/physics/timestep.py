"""
Model timestepping module.
core.timestep_loop handles all the 1D calculations, by first checking the current
state of the column, then running firn_column and lake/lid functions accordingly.
"""
import numpy as np

from monarchs.core.utils import check_correct
from monarchs.physics import (
    snow_accumulation
)
from monarchs.physics import firn_functions, lake_functions, solver, lid_functions


def timestep_loop(cell, dt, met_data, t_steps_per_day, toggle_dict):
    """
    Main timestepping loop applied to an instance of the IceShelf class.
    Called by loop_over_grid to work in parallel over multiple instances.

    Parameters
    ----------
    cell : numba.jitclass
        Instance of the IceShelf class. For more info, see class definition
        docstring in iceshelf_class.py.
    dt : int32
        Time for each timestep <t_steps_per_day>. [s]


    met_data : numba.jitclass
        Instance of the MetData class. Contains thermodynamic variables, which
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
    # densification_toggle = toggle_dict['densification_toggle']
    ignore_errors = toggle_dict["ignore_errors"]
    # if the cell is not "valid" (e.g. filtered out as land by our firn threshold), then we don't want to run any
    # of the physics here, so we just return and move onto the next cell

    if not cell.valid_cell:
        # print(f'Skipping over invalid cell at x = {cell.column}, y = {cell.row}')
        if parallel and not use_numba:
            return cell
        else:
            return

    if np.isnan(cell.firn_temperature).any():
        raise ValueError('NaN in firn temperature')
    # TODO:
    # We can have a situation where lateral movement of water can completely
    # drain a lake. If this is the case, we need to turn exposed_water to False.
    # The alternative is also true.

    cell.t_step = 1  # track timestep for bugfixing
    # cell.log += f'Iteration = {cell.iteration} \n'

    # Main timestepping loop
    for t_step in range(t_steps_per_day):  # int(timestep/dt)):
        if cell.lake_depth == 0:
            cell.lake = False

        if cell.lid_depth == 0:
            cell.lid = False
            cell.v_lid = False

        dz = cell.firn_depth / cell.vert_grid
        # dz = 0.1

        # Add snowfall
        if snowfall_toggle:
            cell.rho = cell.Sfrac * cell.rho_ice + cell.Lfrac * cell.rho_water
            snow_accumulation.snowfall(
                cell,
                met_data.snowfall[t_step],
                met_data.snow_dens[t_step],  # density
                260,  # temperature # TODO - update to use real values not assume 260
            )

        # Unpack met_data into its constituent variables for the desired timestep
        SW_in = met_data.SW_down[t_step]
        LW_in = met_data.LW_down[t_step]
        wind = met_data.wind[t_step]
        T_dp = met_data.dew_point_temperature[t_step]
        T_air = met_data.temperature[t_step]
        p_air = met_data.surf_pressure[t_step]

        """
        # Two main paths - either no exposed water, in which case the dry firn evolves, or we have exposed water,
        # in which case there are further branches depending on whether lakes or lids have formed yet.
        """
        # Firn column only
        if not cell.exposed_water and not cell.saturation[0]:  # Initial state, melting and ice lens formation
            # TODO - added clause that saturation[0] must also be False. The intent behind this is
            # TODO - to ensure that excess water at the surface percolates if it can do so.

            if firn_column_toggle:
                firn_functions.firn_column(
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
            # Commented densification out as this is future work once we can handle how it works with mass conservation.
            # if densification_toggle:
            #     snow_accumulation.densification(cell, t_steps_per_day)

        # Firn column, and form/develop lakes and lids if applicable
        elif cell.exposed_water and cell.saturation[0]:  # We have exposed water - therefore we need to
            # just calculate the heat equation through the firn
            # and then do the lake/lid formation and development.
            # TODO - testing - adding requirement that surface cell is saturated also.
            # TODO - I think this will prevent some weirdness due to lateral movement causing lakes to form
            # TODO - when they really shouldn't (the water should percolate instead)
            args = (cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, wind)
            x = cell.firn_temperature

            if firn_heat_toggle:
                sol, fvec, success, info = solver.firn_heateqn_solver(x, args, fixed_sfc=True)
                if success:
                    cell.firn_temperature = sol
                else:
                    print('Warning - solver failed to converge - lake development')
                    print(x)
                    print(cell.lake)
                    print(cell.lid)
                    print(cell.lake_depth)
                    print(success)
                    print(cell.firn_depth)
                    print(cell.firn_temperature)


            cell.rho = cell.Sfrac * cell.rho_ice + cell.Lfrac * cell.rho_water

            # Lake and lid formation/development logic follows from here if there is
            # exposed water.

            if cell.lake and not cell.lid and not cell.v_lid and cell.lake_depth < 0.1 and cell.v_lid_depth <= 0:
                cell.lake = False
                # cell.Lfrac[0] += cell.lake_depth / (cell.firn_depth/cell.vert_grid)

            # # TODO - added this to test case where water moves laterally, meaning that we no longer have exposed water
            if cell.lake_depth == 0:
                cell.exposed_water = False

            if cell.lake is False:  # Firn saturated but no lake
                # print('Saturated firn - no lake')
                # cell.log = cell.log + "lake formation \n"

                if lake_development_toggle:
                    lake_functions.lake_formation(
                        cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind
                    )
                # if cell.lake is False:
                # cell.log += 'Meltwater has not reached critical threshold to yet form a lake'
                # cell.log += f'Total amount of meltwater = {cell.lake_depth}'

            elif (
                cell.lake is True and cell.lid is False
            ):  # Lake present, but no frozen lid
                # print('lake development', 'x = ', cell.column, 'y = ', cell.row)
                if lake_development_toggle:
                    lake_functions.lake_development(
                        cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind
                    )

                # If the temperature is below freezing at the top of the lake then a
                # virtual lid can form - triggering the virtual lid evolution function
                # This is obtained via lake_development
                if cell.v_lid is True:
                    if lid_development_toggle:
                        # cell.log = cell.log + f'Virtual lid formation/development \n'
                        lid_functions.virtual_lid(
                            cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind
                        )
            # Lake and lid present - so do evolution of lake and lid. The lake will freeze from the top if conditions
            # are cold enough.
            elif cell.lake is True and cell.lid is True:
                if lid_development_toggle:
                    # cell.log = cell.log + "lid development \n"
                    cell.v_lid = False
                    # Lake and lid both present
                    if lake_development_toggle:
                        lake_functions.lake_development(
                            cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind
                        )
                    lid_functions.lid_development(
                        cell, dt, LW_in, SW_in, T_air, p_air, T_dp, wind
                    )

                # if lake frozen - combine frozen lake + firn profiles to
                # get a single profile as everything is now solid
                if (
                    cell.lake_depth <= 0
                    or np.any(cell.lake_temperature[cell.lake_temperature > 273.15])
                    is False
                ):
                    # or cell.lid_melt_count > 11:
                    # cell.log = cell.log + "Combining lid and firn to make one profile \n"

                    lid_functions.combine_lid_firn(cell)
            # else:
            # cell.log = cell.log + "Exposed water - but no lake or lid - error \n"

        # print('Before checking - Lfrac = ', cell.Lfrac[:20])
        if not ignore_errors:
            check_correct(cell)

        # Hourly output - currently commented out as this does not work with Numba. I am uncommenting this block
        # whenever I want to run with hourly output until I can find a better solution - likely using numba.overload
        # to create a "dummy" function
        # if hourly_output:
        #     update_model_output(model_setup.output_filepath, [[cell]], cell.iteration,
        #                         vars_to_save=model_setup.vars_to_save,
        #                         hourly=True, t_step=t_step)

        # update density - not used for any physics, so we do it at the end of the timestep. Mostly included
        # as a convenience variable for plotting/data analysis.
        cell.rho = cell.Sfrac * cell.rho_ice + cell.Lfrac * cell.rho_water
        cell.t_step += 1

    # Move to the next iteration as we are done with our loop over t_steps_per_day (24 h most likely)
    cell.day += 1

    # If we are using pathos.Pool as our multithreading solution, we need to return the cell to loop_over_grid so that
    # we can write it to the grid, as it doesn't amend things in-place
    if parallel and not use_numba:
        return cell
