
Structure
************
The MONARCHS source code is split into the following folders:
    `core` - Code containing the things that make the model tick. Things like definitions of the IceShelf class that
           contains the data, data pipelines (particularly output and data flows within the model,
           and determining which parts of the physics need to be called at what time.
    `DEM` - Code that controls the loading of any digital elevation model (DEM) used in MONARCHS.
    `met_data` - Code that handles loading in meteorological data, particularly netCDF format data from ERA5 reanalyses.
    `physics` - Code that handles the physics of the model. This includes the "single-column" physics
                (e.g. the heat equation, surface energy balance, lake and lid formation) and lateral movement
                of water between these single columns.
    `plots` - Utilities and code to handle generation of plots using model output data.

    There are flowcharts located in `docs/flowcharts` which may help visualise which parts of the model are called where.
    These are:
        `monarchs_main.svg` - The main flowchart of the model, showing things from a top-level perspective.
        `firn_column.svg` - Illustrates the evolution of the dry firn over time within the model.
        `lake_development.svg` - Shows how formation of a lake occurs on saturated firn, and the further
                            development of a lake within the model.
        `lid_development.svg` - Shows how a lid forms on the surface of a lake, and how this evolves over time, including
                                the combination of the lid and firn column into a new firn column.
        `lateral_flow.svg` - Shows the algorithms used to flow water laterally between firn columns in the model.

The ``IceShelf`` class
======================
The ``IceShelf`` class is the core of the MONARCHS model. It contains all of the data used to run one single column
of MONARCHS. The model grid is effectively a 2D grid of these ``IceShelf`` objects. The code in ``monarchs.physics``
often refers to an object called ``cell``, each of which is an ``IceShelf`` instance. All of the physics functions in
``firn_functions``, ``lake_functions`` and ``lid_functions`` are set up in such a way that they are called on these
cells in 1D.