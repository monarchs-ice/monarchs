
Structure
************
The MONARCHS code is split into the following folders:
    core - Code containing the things that make the model tick. Things like definitions of the IceShelf class that
           contains the data, data pipelines (particularly output and data flows within the model,
           and determining which parts of the physics need to be called at what time.
    DEM - Code that controls the loading of any Digital Elevation Map (DEM) used in MONARCHS.
    inputs - Code that handles loading in data, particularly netCDF atmospheric data.
    plots - Utilities and code to handle generation of plots using model output data.
# TODO - update this