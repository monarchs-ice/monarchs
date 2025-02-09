``model_setup`` variable reference
**********************************

This document is effectively an API reference for the MONARCHS runscript (by default named ``model_setup.py``).
If adding new MONARCHS setup parameters, please document them here.

Optional parameters may be optional either because they are not needed for the code to run, or because they have a default value.
These default values are specified in ``monarchs.core.configuration.create_defaults_for_missing_flags``, in case you
want to look these up or change them.

Spatial resolution parameters
------------------------------------------------------
Parameters controlling the spatial resolution of MONARCHS, in terms of the number of grid points and the vertical resolution.

    row_amount : ``int``, required
        Number of rows (i.e. y-points) you want to run MONARCHS with.
        This has only been tested with the same # of points as ``col_amount``, so use caution
        if using different values for each of these.

    col_amount : ``int``, required
        Number of columns (i.e. x-points) you want to run MONARCHS with.
        This has only been tested with the same number of points as ``col_amount``, so use caution
        if using different values for each of these.

    lat_grid_size : ``float``, required
        Size of each gridcell in metres. This determines how long it takes for water to move laterally. WIP is to automate this process, and possibly

    vertical_points_firn : ``int``
        Number of vertical grid cells in the firn profile. Default is 400. This determines the resolution of the model vertically.
        For best results ensure that your vertical resolution is on the order of ~10 cm (i.e. your maximum firn height divided by
        ``vertical_points_firn`` is approximately 0.1)

    vertical_points_lake : ``int``
        Number of vertical grid cells in the lake profile. Default 20, as lakes are both much smaller than the firn
        column and less sensitive to vertical resolution since they are turbulently mixed.

    vertical_points_lid : ``int``
        Number of vertical grid cells in the lid profile. Default 20.


Timestepping parameters
------------------------------------------------------
Parameters controlling the temporal resolution of MONARCHS.

    num_days : ``int``, required
        Number of days (i.e. full model iterations) to run for. A day is ``t_steps_per_day`` steps of the single-column physics,
        followed by one pass of the lateral flow algorithm.
    t_steps_per_day : ``int``, optional
        Default ``24``.
        Number of timesteps to run for each day. 24 = 1h resolution, 8 = 3h resolution, etc.
    lateral_timestep : ``int``, optional
        Default ``t_steps_per_day * 3600``

        Timestep for each iteration of lateral water flow calculation (in s)
        It is highly unlikely this should be anything other than 3600 * 24. Included here for testing only.

Initial conditions - DEM or manually-specified firn height
----------------------------------------------------------
Parameters determining the height of the firn column. This can be specified manually, or read in
automatically from a Digital Elevation Model (DEM). If in doubt, write your own DEM retrieval script and
load the result in as an array.

    DEM_path : str, optional
        Path to a digital elevation map (DEM) to be read in by MONARCHS.
        This will be read in by MONARCHS according to its filetype, and
        interpolated to shape ``(row_amount, col_amount)``.

        If using a relative import, it is a relative import from the folder you are running
        MONARCHS from, not the folder that the code repository is included in. For example, if your
        DEM is in ``/home/data/DEM/test_dem.tif``, and you run MONARCHS from ``/home/model_runs/test_run``, then the DEM path
        would be ``'../../data/DEM/test_dem.tif'``

    firn_depth : ``float``, or ``array_like``, ``float``, dimension(``row_amount, col_amount``), required if ``DEM_path`` not set, else optional

        Initial depth of the firn columns making up the MONARCHS model grid.
        **If a valid DEM path is specified, then this is overridden by the DEM**. Use this only if you want to manually
        specify your own firn heights. Specify as either a number or an array.

        For example, to generate a test Gaussian three-lake profile, one can import ``gaussian_testcase.export_gaussian_DEM``
        from the ``10x10_gaussian_threelake`` example, and call it to generate a usable firn profile.

        If a number is specified, this number is assumed as the firn depth across the whole grid.
        If an array is specified, this should be an array of dimension(``row_amount``, ``col_amount``),
        i.e. the firn depth is user-specified across the whole grid. This is likely the safest option if you want to
        pre-process your firn profile, or don't trust MONARCHS to interpolate it to your desired model grid for you.

    firn_max_height : ``float``, optional
        Default ``150``.

        Maximum height that your firn column can be at. Use this if you're loading in a DEM which has large height
        ranges.

    firn_min_height : ``float``, optional
        Default ``20``.

        Minimum height that we consider to be "firn". Anything below this we consider to be solid ice, which affects
        some of the physics. Notably, we will see a lot more surface water in these cells.

    max_height_handler : str, optional
        How to handle cells that exceed ``firn_max_height``. This is designed to help us filter out land cells.
        These are still part of the overall grid, since they occupy geographic space, but are not useful in terms of
        the physics. This variable can be one of the following:
            ``'filter'`` - Set the variable ``cell.valid_cell = False``, which prevents MONARCHS from running any of the physics
            to these cells. This means they effectively stay the same throughout the whole model.

            ``'clip'`` - Set all cells above the max firn height to ``firn_max_height``. This will not prevent MONARCHS
            from running physics on these cells.

    min_height_handler : str, optional
        How to handle cells where below ``firn_min_height``. This is designed to help us filter out land cells.
        These are still part of the overall grid, since they occupy geographic space, but are not useful in terms of
        the physics. This variable can be one of the following:
            ``'filter'`` - Set the variable cell.valid_cell = False, which prevents MONARCHS from running any of the physics
            to these cells. This means they effectively stay the same throughout the whole model.

            ``'clip'`` - Set all cells below the min firn height to ``firn_min_height``. This will not prevent MONARCHS
            from running physics on these cells.

            ``'extend'`` - Add some metres of firn to the column everywhere to ensure that everywhere is at least
            ``firn_min_height`` metres in height. We do this for every cell in the model to retain the correct relative water level.
            This will give you more realistic firn columns for low-height cells, at the cost of lower resolution for larger
            height cells. Useful if e.g. reading in data from a DEM, where you know that some of the firn is below sea level.

Initial conditions - firn column profiles
------------------------------------------------------
This section determines what parameters you want to use for your initial firn density and temperature profiles.

    rho_init : str, or ``array_like``, ``float``, optional

        Initial density profile.

        This follows Paterson, W. (2000). The Physics of Glaciers. Butterworth-Heinemann,
        using the formula of *Schytt, V. (1958). Glaciology. A: Snow studies at Maudheim. Glaciology. B: Snow studies
        inland. Glaciology. C: The inner structure of the ice shelf at Maudheim as shown by
        core drilling. Norwegian- British- Swedish Antarctic Expedition, 1949-5, IV).*

        Defaults to 'default', in which case MONARCHS will calculate an empirical density profile with ``rho_sfc`` = ``500``
        and ``z_t`` = ``37``.

        Alternatively, specify as either a) a pair of points in the form ``[rho_sfc, zt]`` to use this equation and specify
        ``rho_sfc`` and ``z_t`` yourself, b) a 1D array of length ``vertical_points_firn`` to specify a user-specified
        uniform density profile across the whole grid, or c) an array of
        dimension(``row_amount``, ``col_amount``, ``vertical_points_firn``) to specify different density profiles across your
        model grid.

    T_init : str, or ``array_like``, ``float``, optional
        Initial temperature profile.

        Defaults to 'default', which MONARCHS reads in and uses an assumed firn top temperature of 260 K and
        bottom temperature of 240 K, linearly interpolated between these points.

        Alternatively, specify as either a) a pair of points in the form [top, bottom] to assume a linear
        temperature profile across the whole grid, b) a 1D array of length ``vertical_points_firn`` to specify a user-specified
        uniform temperature profile across the whole grid, or c) an array of
        dimension(``row_amount``, ``col_amount``, ``vertical_points_firn``) to specify different temperature profiles across
        your model grid.

    rho_sfc: ``float``, optional
        Initial surface density used to calculate the profile if using ``rho_init`` = 'default'. Defaults to 500.

Initial conditions - meteorology and surface
------------------------------------------------------
This section defines parameters relating to the input meteorological data, typically from ERA5.

    met_input_filepath : str, required

        Path to a file of meteorological data to be used as a driver to MONARCHS.
        At the moment, only ERA5 format (in netCDF) is supported.
        If this is a relative filepath, then you should ensure that is relative to the folder in which
        you are running MONARCHS from, not the source code directory.

    met_start_index : ``int``, optional
        Default ``0``.

        If specified, start reading the data from ``met_input`` at this index. Useful if you e.g. have a met data file
        that starts at a point sooner than you want to run MONARCHS from.
        This only affects runs starting at iteration 0, i.e. runs that have not been reloaded from a dump.
        Such runs will continue from the index it would have run next were the code not to have stopped regardless
        of this parameter.

    met_timestep : str, or ``int``, optional
        Default ``'hourly'``.

        Temporal resolution of your input meteorological data.
        Ideally, MONARCHS would read in hourly gridded data. However, it is possible that the user may want
        to run long climate simulation runs, which may necessitate lower temporal resolution. This flag tells
        MONARCHS how often the meteorological input data should be run for.
        If str - the value should be 'hourly', 'three-hourly' or 'daily'. For other resolutions, please
        specify an integer, corresponding to how many hours each point in your data corresponds to.
        In this integer form, 'hourly' corresponds to met_timestep = 1, 'three_hourly' to met_timestep = 3, and
        'daily' to met_timestep = 24.

    met_output_filepath : str, optional
        Default ``interpolated_met_data.nc``.

        Filepath for the interpolated grid used by MONARCHS to be saved.
        This is used to save memory, and prevent us from having to repeatedly interpolate our input data.
        This file can be large if running for large domains and timescales. Therefore,this setting is useful
        for those who e.g. want to save this file into scratch space rather than locally.

Geospatial parameters
---------------------
Parameters controlling how MONARCHS brings together DEM and met data inputs and ensures that they are consistent spatially.

    lat_bounds : str, optional
        Default ``False``.

        Toggle whether to constrain the input met data file to lat/long bounds specified by a Digital Elevation Map (DEM) or not. If set to ``'dem'``, then the
        model grid and input meteorological data are constrained to the lat/long of the DEM, i.e. the data from the
        met data netCDF is matched/regridded to the DEM, accounting for changes in e.g. the coordinate reference systems between the two.

        See ``examples/50x50_numba_parallel`` for an example of this; this example run has ``met_dem_diagnostic_plots == True``, so a plot will be generated to show what
        this does visually when running with the appropriate DEM, see ``examples/50x50_numba_parallel/README.md`` for details.

    bbox_top_right, bbox_bottom_left, bbox_top_left, bbox_bottom_right : ``array_like``, ``float``, dimension(lat, long), optional

        Default ``False``.

        Arrays defining a bounding box that we want to constrain the model to.
        If you want to use bounding boxes, they should each be in the form ``[lat, long]``.
        This is useful for e.g. running with
        a DEM that has a large area, but we want to run on a subset of it. Since it is a bounding box where the
        corners are specified, you can define this on any square or rectangular area without being constrained
        by a Cartesian grid (which is useful for e.g. DEMs in polar stereographic projection).
        If defined with ``lat_bounds == 'dem'``, then this will also constrain the input met data to this grid.
        The met data will be regridded to this bounding box, so that the final model grid and met data grid are
        co-located.

    met_dem_diagnostic_plots : bool, optional
        Default ``False``.

        If ``True``, generate some plots to show the regridding of the meteorological data onto the DEM lat/long grid.
        Useful as a sanity check to make sure that this has worked as intended. Typically you might run a test
        (in serial, on a local machine) where you cancel the run during the first model day to check these plots,
        then re-run (in parallel, possibly on HPC) with this set to ``False``.


Output settings - time series (i.e. scientific output)
------------------------------------------------------
This section controls how the model outputs information over time. It does this by appending to a netCDF file
every ``output_timestep`` days.

    save_output : bool, optional
        Default ``True``.

        Flag to determine whether you want to save the output of MONARCHS to netCDF. If True, save the variables
        defined in ``vars_to_save`` into a netCDF file at ``output_filepath`` every timestep (i.e. save spatial and temporal
        data for the selected variables). File sizes can get rather large for large model grids and long
        runs, so you may want to change this from the defaults.

        Note that this is separate from dumping, where only a snapshot of the current iteration is saved. It is not
        possible to restart MONARCHS from the output defined here. See ``Output settings - dumping and reloading model state`` for information on how to enable restarting MONARCHS.

    vars_to_save : tuple, str, optional
        Default ('firn_temperature', 'Sfrac', 'Lfrac', 'firn_depth', 'lake_depth', 'lid_depth', 'lake', 'lid', 'v_lid').

        Tuple containing the names of the variables that we wish to save during the evolution of MONARCHS over time.
        If you want to save a particular diagnostic, then you should add it here.
        See ``monarchs.core.iceshelf_class`` for details on the full list of variables that ``vars_to_save`` accepts.

    output_filepath : str, optional (required if ``save_output`` is ``True``)
        Path to the file that you want to save output into, including file extension.
        MONARCHS uses netCDF for saving output data, so this may be e.g. ``"/work/monarchs/monarchs_run1.nc"``.

    output_grid_size : ``int``, optional
        Defaults to the value set for ``vertical_points_firn`` (i.e. no interpolation occurs).

        Size of the vertical grid that you want to write to. This can be different from the size of the grid used in the
        actual model calculations, in which case the results are interpolated to this grid size. Useful to reduce the
        size of output files, which can be large.

    output_timestep : ``int``, optional
        Default ``1``. (i.e. at every model timestep (``day``))
        Write model output every ``output_timestep`` model days. Useful if you want to save data less regularly than
        every timestep, e.g. if filesizes are getting too large and you don't need daily resolution.

Output settings - dumping and reloading model state
------------------------------------------------------

    dump_data : bool, optional
        Default ``False``.

        Flag that determines whether to dump the current model state at the end of each iteration (day). Doing so
        will allow the user to restart MONARCHS in the event of a crash. Set True to enable this behaviour.
        If this is ``True``, then you also need to specify ``dump_filepath``.

        Note that dumping the model state is separate
        to setting model output - this only dumps a snapshot of the model in its current state, needed to restart the
        model. If you desire output over time, see ``Output settings - time series``.

    dump_filepath : str, optional (required if ``dump_data`` is True)
        File path to dump the current model state into at the end of each timestep,
        for use if ``dump_data`` or ``reload_state`` are True.

    reload_state : bool, optional
        Default ``False`` (i.e. model will start from the initial conditions specified by ``firn_depth`` or the DEM input file by default).

        Flag to determine whether we want to reload from a dump (see ``dump_data`` for details). If ``True``, reload model
        state from file at the path determined by ``dump_filepath``.

Computational and numerical settings
------------------------------------------------------
These parameters mostly control whether the code runs in parallel, which flavour of parallelism to use if so,
how many CPU cores to use if running in parallel, and whether to use Numba to jit-compile the code
(resulting in performance boosts).

    use_numba : bool, optional
        Default ``False``.

        Toggle whether to jit-compile the code using Numba or not. Gives a performance boost, but may not always work and
        adds a few complications. See :docs:``numba`` for more details.
    parallel : bool, optional
        Default ``False``.

        Determines whether or not to run in parallel, or serially. If running in parallel, then performance is improved
        since the model will many of the single-column gridpoints at the same time.

        The exact flavour of parallelism is determined by other flags - if ``use_numba`` and ``use_mpi`` are False, then
        parallelism is via ``pathos.Pool``, a more powerful version of the default ``multiprocessing`` module. If ``use_numba``
        is enabled, then this comes via Numba's ``prange`` function, which works similarly to an OpenMP parallel do loop.
        If ``use_mpi`` is enabled, then ``mpi4py`` is used.

    use_mpi : bool, optional
        Default ``False``.

        Toggle whether to use MPI parallelism to run across multiple nodes. This is an experimental WIP feature.
        This should give large performance boosts if you have the HPC architecture to use it, as it allows for running
        MONARCHS on more than one compute node. However, it is not yet compatible with Numba, so there is also
        some opportunity cost. Not recommended unless you can run on multiple nodes.

    cores : str, bool or ``int``, optional
        Default ``'all'``.

        Number of processing cores to use. 'all' or ``False`` will instruct MONARCHS to use all available CPU cores,
        else it will use however many you specify. You may want to manually specify this to something lower than the number
        of cores on your system if e.g. running on a laptop which you are using for other purposes,
        or if running on HPC and you are experiencing memory bottleneck issues.

Lateral flow settings
------------------------------------------------------
These parameters determine the behaviour of the lateral flow algorithm, i.e. how water is moved around between
grid cells.

    catchment_outflow : bool, optional
        Default ``True``.

        If ``True``, then water that a) reaches the edge of the grid and b) is at a local minimum in terms of the cell's water level
        will disappear from the model, i.e. it moves outside of the model domain. This may or may not be a good assumption
        depending on location.
    flow_into_land: bool, optional
        Default ``True``.

        If ``True``, then similarly to ``catchment_outflow``, water that reaches the edge of the grid and is at a local minimum
        will flow out of the model if it is adjacent to a land cell. This is motivated by the presence of large lakes at the edge
        of the ice shelf in the validation runs, which are not seen in observational datasets. This occurs since the water has
        nowhere else to go, and thus a positive feedback loop occurs where the lake grows, melts the firn underneath,
        and more water flows in.

Debug settings
------------------------------------------------------
These can be safely ignored unless you are actively developing the model.

### Physics toggles
These parameters control the physics that is applied to either the single-column vertical processes, or the
lateral processes. By default these should all be on unless specified, but you may want to switch some off for testing purposes.

    snowfall_toggle : bool, optional
        Default ``True``.

        Determines whether to add height to the firn column via snowfall over time, or not.
        e.g. can be turned off if you don't have a source of snowfall data and don't want to make assumptions.
    firn_column_toggle : bool, optional
        Default ``True``.

        Determines whether the firn column is allowed to evolve or not, i.e. if ``physics.firn_column`` is ever invoked.
    firn_heat_toggle : bool, optional
        Default ``True``.

        Determines whether the temperature of the firn is allowed to evolve, i.e. if ``physics.heateqn`` is ever invoked.
    lake_development_toggle : bool, optional
        Default ``True``.

        Determines whether lakes are allowed to form, i.e. if ``physics.lake_development`` is ever invoked.
    lake_development_toggle : bool, optional
        Default ``True``.

        Determines whether frozen lids are allowed to form, i.e. if ``physics.lid_development`` is ever invoked.
    lateral_movement_toggle : bool, optional
        Default ``True``.

        Determines whether water can move between grid points laterally, or if we treat each column as entirely independent.
    lateral_movement_percolation_toggle : bool, optional
        Default ``True``.

        Determines whether water can percolate during the lateral movement step, assuming that ``lateral_movement_toggle`` is ``True``.
    densification_toggle : bool, optional
        Default ``False``.

        Determines whether snow densification is enabled.
        This is currently always False since our implementation of snow densification is WIP.
    percolation_toggle : bool, optional
        Default ``True``.

        Determines whether water can percolate during the firn column evolution step.
    perc_time_toggle : bool, optional
        Default ``True``.

        Determines if percolation occurs over timescales (if ``True``), or all water can percolate forever until
        it can no longer move.


Other flags - mostly for testing
------------------------------------------------------

This section includes miscellaneous flags that have been used during the development of MONARCHS to test certain things, but have been
and retained as possible configuration flags for testing purposes for other users. These can be entirely ignored.

    simulated_water_toggle : bool, or ``float``
        If False or not present, then nothing happens. If a ``float``, then add that many units of water to each grid cell
        at every timestep. This is to simulate water from outside the catchment area moving in, as in Buzzard (2017).

        This may be useful if running 1D test cases.

    ignore_errors : bool, optional
        Default ``False``.
        If ``True``, then ``monarchs.core.utils.check_correct`` will never be invoked, i.e. the model may be free to
        evolve into an unphysical state. Errors may still occur, but these will be Python errors rather than MONARCHS
        errors if so. May be useful for debugging.

    heateqn_res_toggle : bool, optional
        Defaults to ``False``, i.e. nothing changes.

        An experimental feature whereby, in an attempt to improve performance, the heat equation step
        (which takes up most of the model runtime) is performed with a lower-resolution version of the
        vertical profile, then re-interpolated back to the model grid.

        The thinking is that this may have improved performance without drastically affecting the results
        since the temperature profile should be smooth, and the vertical resolution is mostly required for
        percolation purposes. Initial testing showed large differences between the two versions,
        so this was abandoned for now.

    radiation_forcing_factor : ``float``, optional
        Undefined by default, equivalent to setting it to ``1``.

        Multiply the shortwave/longwave radiation variables by this factor for testing purposes, e.g. if running
        a 1D case and you want to ensure that lake formation occurs. This is left in mostly as an example;
        see ``monarchs.core.initial_conditions`` and search for this variable for more details.

    spinup : bool, optional
        Default ``False``.

        Experimental feature - if True, then try and force the heat equation to converge at the start of the run,
        similar to spinning up a climate model. It may be better to just run the model for longer than using this function
        however, or starting the model from a dump with a pre-spun up state.
    verbose_logging : bool
        Default ``False``.

        Experimental feature - if True, then output data every hour rather than every day. This will override the
        ``output_timestep`` defined in ``Output settings - dumping and reloading model state``. This will generate very
        large files, and doesn't work properly yet (particularly with Numba) so likely best left alone for now unless you have a strong need for
        hourly output.