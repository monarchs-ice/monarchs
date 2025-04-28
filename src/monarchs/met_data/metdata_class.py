"""
Created on Mon Aug 28 15:08:04 2023

@author: jdels
"""


def initialise_met_data_grid(row_amount, col_amount, snowfall, snow_dens,
    temperature, wind, surf_pressure, dewpoint_temperature, LW_down,
    SW_down, latitude, longitude, use_numba=False):
    """
    Create a grid of MetData objects, each one associated with an IceShelf object.
    This is done using a Numba typed list if possible, to ensure compatibility.
    The data is loaded in as NumPy arrays, obtained from the input netCDF file of meterological data.
    This is done at each iteration, so we only store one day's worth of met data in memory at any given time.
    Called in <main>.

    Parameters
    ----------
    row_amount : int
        Number of rows in the grid
    col_amount : int
        Number of columns in the grid
    snowfall : array_like, float, dimension(time)
        Array of snowfall, as a function of row index, column index and time [m]
    snow_dens :  array_like, float, dimension(time)
        Array of snow density as a function of row index, column index and time [kg m^-3]
    temperature :  array_like, float, dimension(time)
        Array of surface air temperature as a function of row index, column index and time [K]
    wind : array_like, float, dimension(time)
        Array of wind speed as a function of row index, column index and time [m s^-1]
    surf_pressure :  array_like, float, dimension(time)
        Array of surface pressure as a function of row index, column index and time [Pa]
    dewpoint_temperature :  array_like, float, dimension(time)
        Array of dewpoint temperature as a function of row index, column index and time [K]
    LW_down : array_like, float, dimension(time)
        Array of downwelling longwave radiation as a function of row index, column index and time [W m^-2]
    SW_down : array_like, float, dimension(time)
        Array of downwelling shortwave radiation as a function of row index, column index and time [W m^-2]

    Returns
    -------
    grid : List, or numba.typed.List
        Model grid of IceShelf objects. This is the main data structure of MONARCHS.
    """
    if use_numba:
        from numba.typed import List
        grid = List()
    else:
        grid = []
    for i in range(row_amount):
        if use_numba:
            _l = List()
        else:
            _l = []
        for j in range(col_amount):
            _l.append(MetData(snowfall[:, i, j], snow_dens[:, i, j],
                temperature[:, i, j], wind[:, i, j], surf_pressure[:, i, j],
                dewpoint_temperature[:, i, j], LW_down[:, i, j], SW_down[:,
                i, j], latitude[i, j], longitude[i, j]))
        grid.append(_l)
    return grid


class MetData:
    """
    Class for storing meteorological conditions used to drive MONARCHS. Each point in the model grid has an IceShelf,
    and a corresponding MetData object.
    Called in <initialise_met_data_grid>.
    Attributes
    ----------
    snowfall : array_like, float, dimension(time)
        snowfall, as a function of time [m]
    snow_dens : array_like, float, dimension(time)
        snow density as a function of time [kg m^-3]
    temperature : array_like, float, dimension(time)
        surface air temperature as a function time [K]
    wind : array_like, float, dimension(time)
       wind speed as a function of time [m s^-1]
    surf_pressure : array_like, float, dimension(time)
       surface pressure as a function time [Pa]
    dew_point_temperature : array_like, float, dimension(time)
        dewpoint temperature as a function of time [K]
    LW_down : array_like, float, dimension(time)
        downwelling longwave radiation as a function of time [W m^-2]
    SW_down : array_like, float, dimension(time)
        downwelling shortwave radiation as a function of time [W m^-2]
    """

    def __init__(self, snowfall, snow_dens, temperature, wind,
        surf_pressure, dew_point_temperature, LW_down, SW_down, latitude,
        longitude):
        self.snowfall = snowfall
        self.temperature = temperature
        self.wind = wind
        self.surf_pressure = surf_pressure
        self.dew_point_temperature = dew_point_temperature
        self.LW_down = LW_down
        self.SW_down = SW_down
        self.snow_dens = snow_dens
        self.lat = latitude
        self.lon = longitude


def get_spec():
    from numba import float64
    spec = [('snowfall', float64[:]), ('snow_dens', float64[:]), (
        'temperature', float64[:]), ('wind', float64[:]), ('surf_pressure',
        float64[:]), ('dew_point_temperature', float64[:]), ('LW_down',
        float64[:]), ('SW_down', float64[:]), ('lat', float64), ('lon',
        float64)]
    return spec
