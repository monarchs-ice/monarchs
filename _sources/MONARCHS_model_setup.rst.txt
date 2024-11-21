

Setting up a model run
*************************

MONARCHS uses a Python file (by default called ``model_setup.py``) to handle user-defined input into the model.
This file contains all of the necessary parameters required to run the model. You need to at minimum specify the following:
    - An initial profile of firn column height (referred to in the model as "firn depth", since arrays in MONARCHS
      are ordered with the surface at index 0). This can be in the form of a Numpy array
    - The initial conditions of what is referred to in this documentation as the "model grid", i.e. the columns
      of firn that you start with. You must at least provide an initial density and temperature profile. Alternatively,
      default profiles can be used for both of these.
    - A set of meteorological conditions to drive the model. This should ideally be in the form of a netCDF file in
      `ERA5 format <https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation>`_. A full list of allowed and
      required variables is in the :ref:`Inputting meteorological data` section. This section also details how to use
      a dictionary if preferred over using ERA5 format netCDF.

You can then run MONARCHS using python run_MONARCHS.py, if ``model_setup.py`` is located in the same folder as MONARCHS.
Alternatively, you can pass a path to a model setup file as an argument, e.g. python run_MONARCHS.py

Setting up ``model_setup``
============================
The MONARCHS runscript (default name ``model_setup.py``) is the core input to MONARCHS. There are a large number of
possible input parameters, many of which are optional. MONARCHS will flag when there are incompatible parameters
or where a parameter needs a second parameter to be specified to work properly.

For more details on the runscript, see :doc:`model_setup_reference`.

Inputting meteorological data
=============================
This input data should be specified hourly, if possible. If using data at a lower temporal resolution, specify
``met_timestep`` in your runscript. See the detailed ``model_setup.py`` documentation for details. # TODO - link to this
Alternatively, a python Dictionary object can be passed in, with the dict values being a set of Numpy arrays and the
keys in the same format as ERA5. The Numpy arrays should be of dimension(time) or
dimension(<time, row_amount>, <col_amount>). In the former case, at each timestep, the value will be used over the
whole model grid. In the latter, the data is gridded to the model grid.

Making user-defined changes to the inputs
=========================================
It is possible to amend the input pipeline to force changes to certain variables if you want to run some tests.
For example, to force the model temperature to a higher value, one could amend ``interpolate_met_data``
in ``initial_conditions.py``, or to change the default temperature profile one could edit
``initialise_firn_profile`` in the same file.

An example of this is shown in ``interpolate_met_data``, where a toggle ``radiation_forcing_factor`` may be specified in the
runscript to adjust SW downwelling radiation by a multiplicative factor.

Outputting data
===============
MONARCHS has two ways of outputting data. The main way which is scientifically useful is using the `save_output`
flag in `model_setup`. This reads in a variable `vars_to_save`, which determines which IceShelf variables
the user wants to output after each model day (iteration). This is used to generate time series of the model evolution.
By default it will save firn variables at the model vertical resolution, but if output filesizes are an issue you can
interpolate these to a resolution set using the `output_grid_size` parameter.

The other way MONARCHS saves data is in the form of a "dump". This is effectively MONARCHS' way of saving the model
state after every model day (iteration). This allows the user to restart a run that has failed for whatever reason
(PC crashing, batch job timing out, etc). Additionally, it provides the user with a fully detailed view of the model
state at the end of a run, in case more information is required than was saved to the time series output, or you
want to look at some diagnostics after a failed run.