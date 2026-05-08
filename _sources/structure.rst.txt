
Code structure
************
The MONARCHS source code is split into the following folders:
    ``core`` - Code containing the things that make the model tick. Things like definitions of the model grid datatype,
           data pipelines (particularly output and data flows within the model,
           and determining which parts of the physics need to be called at what time.

    ``DEM`` - Code that controls the loading of any digital elevation model (DEM) used in MONARCHS.

    ``met_data`` - Code that handles loading in meteorological data, particularly netCDF format data from ERA5 reanalyses.

    ``physics`` - Code that handles the physics of the model. This includes the "single-column" physics
                (e.g. the heat equation, surface energy balance, lake and lid formation) and lateral movement
                of water between these single columns.

    ``plots`` - Utilities and code to handle generation of plots using model output data.

    There are flowcharts located in ``docs/flowcharts`` which may help visualise which parts of the model are called where.
    These are:

        ``monarchs_main.svg`` - The main flowchart of the model, showing things from a top-level perspective.

        ``firn_column.svg`` - Illustrates the evolution of the dry firn over time within the model.

        ``lake_development.svg`` - Shows how formation of a lake occurs on saturated firn, and the further
                            development of a lake within the model.

        ``lid_development.svg`` - Shows how a lid forms on the surface of a lake, and how this evolves over time, including
                                the combination of the lid and firn column into a new firn column.

        ``lateral_flow.svg`` - Shows the algorithms used to flow water laterally between firn columns in the model.

