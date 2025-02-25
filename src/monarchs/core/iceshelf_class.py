# -*- coding: utf-8 -*-
"""
Define IceShelf class and initialise grid

S Buzzard created, last updated 06/07/2021

"""

import numpy as np


class IceShelf:
    """
    Class containing column information about the state of the ice shelf at coordinates x and y.
    This class is the primary building block of MONARCHS, and the object in which all information is stored.
    A single IceShelf just contains information for a single column - a grid of IceShelf objects is required to
    construct information for a real ice shelf in practice.

    If adding additional variables to IceShelf, e.g. to track certain things, ensure that they are also added to the
    variable <spec>, found in <iceshelf_class.py>, else you will run into issues if trying to run with Numba.

    Several of these attributes are constants, and are labelled as such.
    All Boolean flags are True if their documented condition is met, and False if not met.
    All array_like variables are dimension(vert_grid).

    A long-term goal is to refactor IceShelf to utilise inheritance and split things up somewhat.
    However, Numba jitclasses do not currently support inheritance, and this is required for use of NumbaMinpack.hybrd,
    in such a way that we retain the ability to select whether to use that or scipy.optimize.fsolve.

    Attributes
    ----------
    x : int
         x (row) coordinate of the current IceShelf on the model grid. int
    y : int
        y (column) coordinate of the current IceShelf on the model grid. int
    firn_depth : float
        Height of the firn column [m]
    lake_depth : float
        Height of the surface lake, 0 if not present [m]
    lid_depth : float
        Height of the frozen lid on top of the surface lake, 0 if not present [m]
    v_lid_depth : float
        Height of the virtual lid on top of the surface lake, 0 if not present, converts to lid if > 0.1 [m]
    vert_grid : int
        Number of vertical grid points in the firn column.
    vertical_profile : array_like, float, dimension(vert_grid)
        Array containing the vertical profile (top = 0, bottom = firn_depth, len = vert_grid) of the IceShelf. [m],
    vert_grid_lake : int
        Number of vertical grid points in the lake. int
    vert_grid_lid : int
        Number of vertical grid points in the frozen lid. int
    rho : array_like, float, dimension(vert_grid)
        Firn density profile. Mostly used for convenience as all calculations use Sfrac and Lfrac. [kg m^-3]
    rho_lid : array_like, float, dimension(vert_grid_lid)
        Lid density [kg m^-3]
    firn_temperature : array_like, float, dimension(vert_grid)
        Temperature profile of t
    Sfrac : array_like, float, dimension(vert_grid)
        Solid fraction of the firn.
    Lfrac : array_like, float, dimension(vert_grid)
        Liquid fraction of the firn.
    meltflag : array_like, bool, dimension(vert_grid)
        Boolean flag to determine if there is melt at a given vertical point
    saturation : array_like, bool, dimension(vert_grid)
        Boolean flag to determine whether a vertical layer is saturated.
    lake_temperature : array_like, float, dimension(vert_grid_lake)
        Temperature of the lake as a function of vertical level [K]
    lid_temperature : array_like, float, dimension(vert_grid_lid)
        Temperature of the lid as a function of vertical level [K]
    water_level : float
        Water level of the cell. Can be either at the height of the lake, infinite (in the case of a lid), or
        at the height of the highest vertical level with saturated firn.
    water : array_like, float, dimension(vert_grid)
        Water content of each grid cell. Used only in lateral_functions.move_water, where it is necessary to combine
        water from the lake and from the firn (which is determined by Lfrac).
    melt_hours : int
        Tracks the number of hours of melt that have occurred. Currently only tracks melting of the firn due to
        the temperature of the lake above it.
    exposed_water_refreeze_counter : int
        Tracks the number of times that exposed water at the surface freezes due to surface conditions.
    lid_sfc_melt : float
        Tracks the amount of meltwater resulting from melting of the frozen lid. Used to
    melt : bool
        Flag to determine whether the model is in a melting state or not. This affects the surface albedo for the
        surface flux calculation, used variously but mostly in the surface energy balance for the lake and lid, and for
        the calculation of the heat equation.
    exposed_water : bool
        Flag to track whether there is exposed water due to surface melting.
    lake : bool
        Flag to track whether the model is in a state that includes a lake or not. Is True even if there is a
        frozen lid, until the lid freezes enough to create a single firn profile.
    v_lid : bool
        Flag to track whether a virtual lid is present. Set to False if a true lid is present.
    virtual_lid_temperature : float
        Temperature of the virtual lid, if there is one. The virtual lid is so small that this can be a single number
        rather than a vertical profile. [K]
    lid : bool
        Flag to track whether a frozen lid has formed.
    ice_lens: bool
        Flag to track whether there is pore closure and the formation of a lens of solid ice. Necessary for
        saturation to occur and lakes to form.
    ice_lens_depth : int
        Vertical layer (not physical depth) of the ice lens if there is one. Default 999, i.e. no lens present.
    has_had_lid : bool
        Flag to determine whether the model has undergone lid development.
        Set True if lid depth exceeds 0.1.
        Resets to False if lid melts below 0.1 m depth.
    lid_melt_count : int
        Track the number of times that the lid has undergone melting. Used as a tracker.
    total_melt : float
        Total amount of melting that has occurred. Not used for any physics, but as a tracker.
    rho_ice : int
        Density of ice. Constant. [kg m^-3]
    rho_water : int
        Density of water. Constant. [kg m^-3]
    L_ice : int
        Latent heat of fusion of ice, i.e. change in enthalpy due to state change. Constant. [J kg^-1]
    pore_closure : int
        Density at which pores close and water can no longer percolate. Constant. [kg m^-3]
    k_air : float
        Thermal conductivity of air. Constant. [W m^-1 K^-1]
    cp_air : int
        Heat capacity of air. Constant. [J K^-1]
    k_water : float
        Thermal conductivity of water. Constant. [W m^-1 K^-1]
    cp_water : int
        Heat capacity of water. Constant. [J K^-1]
    t_step : int
        Current hour. Almost certainly between 1 and 24.
    iteration : int
        Current iteration (i.e. day). Updated at each timestep.
    snow_added : float
        Track how much snow mass has been added to the model via snowfall, in arbitrary units. [a.u.]
    log : string
        Logging information about the state of the model, to track which paths the model takes in the logic, and to
          inform the user of warnings.
    reset_combine : bool
        Flag to track whether a frozen lid has completely frozen and been combined with the firn column
        to make one profile.
    valid_cell : bool
        Flag that determines whether a cell is "valid" or not. If not, then we don't care about the physics of that
        cell, and it is ignored accordingly.
        Default True, i.e. the cell is valid.
        For example, if a DEM has an ice shelf, with land either side, we are unlikely to care about that land, and
        the ice shelf physics would not apply there anyway, so we can safely ignore it.
    lat: float
        Latitude of the cell in degrees. Taken from the input DEM.
    lon: float
        Longitude of the cell in degrees. Taken from the input DEM.
    size_dx: float
        Size of the grid cell in the x (along rows) direction [m]
    size_dy: float
        Size of the grid cell in the y (along columns) direction [m]
    """

    def __init__(
            self,
            y,
            x,
            firn_depth,
            vert_grid,
            vert_grid_lake,
            vert_grid_lid,
            rho,
            firn_temperature,
            Sfrac_in=np.array([np.nan]),
            Lfrac_in=np.array([np.nan]),
            meltflag_in=np.array([np.nan]),
            saturation_in=np.array([np.nan]),
            lake_depth=0,
            lake_temperature_in=np.array([np.nan]),
            lid_depth=0,
            lid_temperature_in=np.array([np.nan]),
            melt=False,
            exposed_water=False,
            lake=False,
            v_lid=False,
            lid=False,
            water_level=0,
            water=np.array([np.nan]),
            ice_lens=False,
            ice_lens_depth=999,
            has_had_lid=False,
            lid_sfc_melt=0,
            lid_melt_count=0,
            melt_hours=0,
            exposed_water_refreeze_counter=0,
            virtual_lid_temperature=273.15,
            total_melt=0.0,
            valid_cell=True,
            lat=0,
            lon=0,
            size_dx=1000,
            size_dy=1000,
    ):
        """
        **__init__**

        Initialise our IceShelf class. For most variables, see class-level documentation. The parameters below are only
        those which differ between the initialiser and the class attributes.

        Parameters
        ----------
        Sfrac_in : array_like, float, optional
            Input solid fraction. Can be set to np.array([np.nan]) (the default) to use the default profile as
            calculated by rho_init_emp.
            Required to be an array even if setting to nan if using Numba as Numba expects an array.
        Lfrac_in : array_like, float, optional
            Input liquid fraction. Can be set to np.array([np.nan]) (the default) to use the default profile as
            calculated by rho_init_emp.
            Required to be an array even if setting to nan if using Numba as Numba expects an array.
        meltflag_in : array_like, float, optional
            Array of booleans determining whether meltwater is present at that vertical layer. As above, can be set
            to np.array([np.nan]) to use a default profile (all zeros).
        saturation_in : array_like, float, optional
            Array of booleans determining whether each vertical layer is saturated or not. As above, can be set
                to np.array([np.nan]) to use a default profile (all zeros).
        lake_temperature_in : array_like, float, optional
            Vertical profile of lake temperature.
            As above, can be set to np.array([np.nan]) to use a default profile (all zeros), as lakes are not present
            by default (so it doesn't matter).
        lid_temperature_in : array_like, float, optional
            As above, can be set to np.array([np.nan]) to use a default profile (all zeros), as lids are not present
            by default (so it doesn't matter).
        valid_cell : bool, optional
            Flag that determines whether we actually run any of the physics or not for this cell.
            See documentation in the class docstring for details.
        """
        # y is 'i' in matrix (column), x is 'j' (row)
        # Variables
        self.column = x  # x (column) coordinate of cell on grid
        self.row = y  # y (row) coordinate of cell on grid, i.e. in grid[row][col]
        self.firn_depth = firn_depth  # Firn column height
        self.vert_grid = vert_grid  # Number of vertical grid cells
        self.vertical_profile = np.linspace(0, self.firn_depth, self.vert_grid)
        self.vert_grid_lake = (
            vert_grid_lake  # Number of vertical grid cells in lake/ ice lid
        )
        self.vert_grid_lid = vert_grid_lid  # Number of vertical grid cells in ice lid
        self.rho = rho  # Initial density only, after this use only Sfrac
        self.rho_lid = 917 * np.ones(vert_grid_lid)
        self.firn_temperature = firn_temperature
        self.v_lid_depth = 0
        # JE - changed these conditionals from string "init" to one based on
        # whether the input is entirely NaNs. This stops Numba from throwing
        # an error due to incorrect dtypes.
        if np.isnan(Sfrac_in).all():  # Solid fraction (0 to 1)
            self.Sfrac = np.ones(vert_grid) * rho / 917
        else:
            self.Sfrac = Sfrac_in

        if np.isnan(Lfrac_in).all():  # Liquid fraction (0 to 1)
            self.Lfrac = np.zeros(vert_grid)
        else:
            self.Lfrac = Lfrac_in

        if np.isnan(meltflag_in).all():  # Melt has reached here (0 or 1)
            self.meltflag = np.zeros(vert_grid)
        else:
            self.meltflag = meltflag_in

        if np.isnan(
                saturation_in
        ).all():  # all pore space saturated with meltwater (0 or 1)
            self.saturation = np.zeros(vert_grid)
        else:
            self.saturation = saturation_in

        # Lake and lid initial temperature profiles
        if np.isnan(lake_temperature_in).any():
            self.lake_temperature = (
                    np.ones(vert_grid_lake) * 273.15
            )  # lake temperature (well mixed), 0 is top of lake

        else:
            self.lake_temperature = (
                lake_temperature_in  # ensure this is an array if loading in your
            )
            # own initial temperature

        if np.isnan(lid_temperature_in).any():
            self.lid_temperature = np.ones(vert_grid_lid) * 273.15

        else:
            self.lid_temperature = (
                lid_temperature_in  # ensure this is an array if loading in your
            )
            # own initial temperature

        self.lake_depth = lake_depth
        self.lid_depth = lid_depth
        # variable, possibly merge into
        # self.water or self.water_level
        self.water_level = water_level  # Set to be 0 if no value is given
        if np.isnan(water).any():
            self.water = np.zeros(
                self.vert_grid
            )  # Volume of water available for lateral transport
        else:
            self.water = water
        # lake formation. Set to be 0 if no value is given
        self.melt_hours = 0  # hours of melting there have been

        # Counter to switch model back to a state of no exposed water
        self.exposed_water_refreeze_counter = exposed_water_refreeze_counter
        self.lid_sfc_melt = lid_sfc_melt  # default 0

        # States model can be in
        self.melt = melt  # melt has occurred but not saturated
        self.exposed_water = exposed_water  # Firn is saturated to surface

        self.lake = lake  # Lake present
        self.v_lid = v_lid  # Virtual frozen lid on lake
        self.virtual_lid_temperature = (
            virtual_lid_temperature  # Virtual frozen lid temperature
        )
        self.lid = lid  # True frozen lid on lake
        self.ice_lens = ice_lens  # Frozen water has caused ice lens formation
        self.ice_lens_depth = ice_lens_depth
        # (no percolation below this)
        self.has_had_lid = has_had_lid
        self.lid_melt_count = lid_melt_count
        self.total_melt = total_melt  # total amount of melting that has occurred
        # Constants
        self.rho_ice = 917
        self.rho_water = 1000
        self.L_ice = 334000
        self.pore_closure = 830
        self.k_air = 0.022
        self.cp_air = 1004
        self.k_water = 0.5818
        self.cp_water = 4217
        self.t_step = 0
        self.day = 0
        self.snow_added = 0
        self.log = ""
        self.reset_combine = False
        self.valid_cell = valid_cell
        self.lat = lat
        self.lon = lon
        self.size_dx = size_dx
        self.size_dy = size_dy



def get_spec():
    """
    Define class datatypes for the IceShelf class.
    This is necessary for all the optimisations provided by Numba to work.
    When/if adding new variables to the IceShelf class, ensure that you add
    the respective variable and its datatype to this list.
    e.g. if it is a single number use int32 or float64, if an array use
    float64[:] if a float array, int32[:] if integer, boolean[:] if Bool etc.
    Be *very* careful about making sure that you get the right dtype between
    float and int, If in doubt use float; if you specify an integer then the
    code will give you integers for any calculation without throwing an error
    or being intelligent (e.g. float - int will give an int, which will just be
    a rounded up or down version of the number you'd expect).
    """
    from numba import int32, float64, boolean
    from numba.types import string
    spec = [
        ("column", int32),
        ("row", int32),
        ("firn_depth", float64),
        ("vert_grid", int32),
        ("vertical_profile", float64[:]),
        ("vert_grid_lake", int32),
        ("vert_grid_lid", int32),
        ("rho", float64[:]),
        ("rho_lid", float64[:]),
        ("firn_temperature", float64[:]),
        ("Sfrac", float64[:]),
        ("Lfrac", float64[:]),
        ("meltflag", float64[:]),
        ("saturation", float64[:]),
        ("lake_temperature", float64[:]),
        ("lid_temperature", float64[:]),
        ("water_level", float64),
        ("water", float64[:]),
        ("melt", boolean),
        ("exposed_water", boolean),
        ("lake", boolean),
        ("lake_depth", float64),
        ("v_lid", boolean),
        ("virtual_lid_temperature", float64),
        ("lid", boolean),
        ("lid_depth", float64),
        ("ice_lens", boolean),
        ("ice_lens_depth", int32),
        ("rho_ice", float64),
        ("rho_water", float64),
        ("L_ice", float64),
        ("pore_closure", float64),
        ("k_air", float64),
        ("cp_air", float64),
        ("k_water", float64),
        ("cp_water", float64),
        ("v_lid_depth", float64),
        ("has_had_lid", boolean),
        ("melt_hours", int32),
        ("exposed_water_refreeze_counter", int32),
        ("lid_sfc_melt", float64),
        ("lid_melt_count", int32),
        ("total_melt", float64),
        ("t_step", int32),
        ("day", int32),
        ("log", string),
        ("snow_added", float64),
        ("reset_combine", boolean),
        ("valid_cell", boolean),
        ("lat", float64),
        ("lon", float64),
        ("size_dx", float64),
        ("size_dy", float64),
    ]
    return spec
