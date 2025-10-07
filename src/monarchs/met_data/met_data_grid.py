import numpy as np


def initialise_met_data(
    snowfall,
    snow_dens,
    temperature,
    wind,
    surf_pressure,
    dew_point_temperature,
    LW_down,
    SW_down,
    latitude,
    longitude,
    num_rows,
    num_cols,
    dtype,
    t_steps_per_day,
):
    """
    Create a structured array containing our met data, with each element
    associated with an element of the model grid.
    The data is loaded in as NumPy arrays, obtained from the input netCDF
    file of meterological data.
    This is done at each iteration, so we only store one day's worth of met
    data in memory at any given time.
    Called in <main>.

    Parameters
    ----------
    row_amount : int
        Number of rows in the grid
    col_amount : int
        Number of columns in the grid
    snowfall : array_like, float, dimension(time)
        Array of snowfall, as a function of row index, column index
        and time [m]
    snow_dens :  array_like, float, dimension(time)
        Array of snow density as a function of row index, column index and
        time [kg m^-3]
    temperature :  array_like, float, dimension(time)
        Array of surface air temperature as a function of row index, column
        index and time [K]
    wind : array_like, float, dimension(time)
        Array of wind speed as a function of row index, column index and
        time [m s^-1]
    surf_pressure :  array_like, float, dimension(time)
        Array of surface pressure as a function of row index, column index
        and time [Pa]
    dewpoint_temperature :  array_like, float, dimension(time)
        Array of dewpoint temperature as a function of row index, column
        index and time [K]
    LW_down : array_like, float, dimension(time)
        Array of downwelling longwave radiation as a function of row index,
        column index and time [W m^-2]
    SW_down : array_like, float, dimension(time)
        Array of downwelling shortwave radiation as a function of row index,
        column index and time [W m^-2]

    Returns
    -------
    metdata - numpy structured array
        Grid containing meteorological data for the model run.
    """

    metdata = np.zeros((t_steps_per_day, num_rows, num_cols), dtype=dtype)
    metdata["snowfall"] = snowfall
    metdata["temperature"] = temperature
    metdata["wind"] = wind
    metdata["surf_pressure"] = surf_pressure
    metdata["dew_point_temperature"] = dew_point_temperature
    metdata["LW_down"] = LW_down
    metdata["SW_down"] = SW_down
    metdata["snow_dens"] = snow_dens
    metdata["lat"] = latitude
    metdata["lon"] = longitude
    return metdata


def get_spec(t_steps_per_day):
    dtype = np.dtype(
        [
            ("snowfall", np.float64),
            ("snow_dens", np.float64),
            ("temperature", np.float64),
            ("wind", np.float64),
            ("surf_pressure", np.float64),
            ("dew_point_temperature", np.float64),
            ("LW_down", np.float64),
            ("SW_down", np.float64),
            ("lat", np.float64),
            ("lon", np.float64),
        ]
    )
    return dtype
