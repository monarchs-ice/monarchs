Running MONARCHS (a quickstart guide)
------------------------------------------

This section is a tutorial/quickstart on how to run MONARCHS.

Running a basic (1D) test case
==============================
First, follow the (non-advanced) instructions in the :doc:`installation` section to install the model.

To run MONARCHS, the best way is to use the command line. From a terminal
(you can open this in an IDE like PyCharm, Spyder, VSCode if you prefer to use one of these),
make sure you activate the virtual environment MONARCHS is installed in.

Then, navigate to the ``examples/1D_test_case`` folder in the MONARCHS repository, and run the model using:

``monarchs``

.. note::
    If you are uncomfortable using the command line, you can run ``1D_test_case/model_setup.py`` as a Python script
    directly to run MONARCHS. This works since it has the following code at the end of the file.

    .. code-block:: python

        if __name__ == "__main__":
            from monarchs.core.driver import monarchs
            monarchs()

    The other examples in the ``examples`` folder can be run in the same way.

This will run a 1D column test case. It should run in a few seconds. The code will write a lot of output into the
console. The first few lines will be from ``monarchs.core.configuration``. You can ignore these for now, since this
is the model telling you that it is using default values for several parameters that are not included in our simple
1D runscript (``model_setup.py``).


The model will run for 105 days and then stop. It will print out some simple diagnostics - the firn depth, lake depth
and lid depth, and how much water was detected during the lateral movement step (which doesn't occur here of course
as we are running a 1D case!).

Model outputs
=============
You should notice a new folder in the ``1D_test_case`` directory called ``output``. This contains three files. In ``model_setup.py``,
there are variables ``met_output_filepath``, ``output_filepath``  and ``dump_filepath``. The model will write these files to the
location set in these variables, and create any folders necessary to do this. In this case, we are running from the
``1D_test_case`` folder, and we have specified in ``model_setup.py`` that we want to write these outputs to ``output/*.nc``, so the ``output`` folder
is created here and the files written to this path.

The first of the created files is a file called ``met_data_1D_test_case.nc``. This is a netCDF file containing the meteorological data that
is used in the model, and is not useful to us aside from for debugging.

The other files are ``1d_testcase_dump.nc``, and
``1d_testcase_output.nc``. The first of these is a dump file, which contains the full state of the model at the end of the run.
This can be useful to get a quick view of the state of the model at the end of the run, but more importantly can be loaded
into the model to restart a run from the end of the previous one (or from a run that crashed). We will do this later in this tutorial.

The second file, ``1d_testcase_output.nc``, contains the output of the model. This contains information collected
over the course of the model run. By default, it will output the firn depth, lake depth, lid depth, firn temperature,
and ice lens depths. You can tell MONARCHS exactly which variables you are interested in by changing the ``vars_to_save``
variable in ``model_setup.py``. You can also change the frequency at which the model save the output, and the
vertical resolution, as for larger model sizes this file can get very big.

A more complex test case
========================
Let's ignore the outputs for now, and move onto a more complex test case. In our previous case, the firn profile was
specified as a single value to the ``firn_depth`` variable in ``model_setup.py``. In this case, we now have a 2D grid
to run on. We therefore need to specify our firn profile on a 2D grid. This example uses a Gaussian profile, with one
large lake in the centre and two smaller lakes in the top-left and bottom-right corners.

Change directory to ``examples/10x10_gaussian_threelake`` (or open up ``model_setup.py`` from this directory in your IDE
if not using the command line), and run it in the same way as before. You will notice that this takes significantly longer to run
than the 1D case. Wait for it to complete, and make a note of the time taken displayed at the end of the model run.
We can make it faster by delving into ``model_setup.py``.

In the ``model_setup.py`` file, you will see a variable called ``parallel``. This is set to ``False`` by default.
Set this to ``True``, and re-run the model. You can specify how many cores you want to use by changing the ``cores``
variable. By default, it is set to ``'all'``, so if you are doing other things on your machine it may be best to set this
to e.g. ``4`` for now.

Since we are now running in parallel, the model should run significantly faster. Let's take advantage of this and run
the model at higher resolution. You can control this  by changing ``row_amount`` and ``col_amount`` from 10 to 20. Note that this increase in resolution
will make the model take at least ~4x longer to run!

We could instead increase the time the model runs for via the ``num_days`` variable. Currently is set to ``105``.
However, if we increase this, we also need to give the model a larger array of meteorological data as input.
Our current data (see the ``met_data`` dictionary) covers 2520 timesteps, with 800 "warm" timesteps,
where the longwave radiation, shortwave radiation and temperature are set to quite high values and 1720 "cold" timesteps
where they are set to lower values - these add up to 105 days at hourly resolution.

If we increase the number of days, we could e.g. increase the number of warm and cold timesteps to match.
We can do this by changing the ``warm_timesteps`` and ``cold_timesteps`` in this specific example.
For example, if we set ``num_days`` to 110 from 105, we need to increase the value of ``cold_timesteps`` or ``warm_timesteps`` by an additional 120 (``5 * 24``),
or extend the data in some other way (e.g. appending another array with a different set of values for a different number of timesteps, or splitting the 120 extra timesteps required between ``cold_timesteps`` and ``warm_timesteps``).

.. note::
    Note that ``warm_timesteps`` and ``cold_timesteps`` are not values used by MONARCHS itself, they are just used to control the
    size of the meteorological data fed into MONARCHS in *this particular case*. The ``met_data`` dictionary (or a path to a netCDF file in ERA5 format, see :doc:`met_data`)
    is what is actually used by MONARCHS.

    We are merely exploiting the fact that our model setup script is a piece of Python code to generate an arbitrary set of values
    to use as input for this example. You could put anything you like here, using this example as a guideline - by e.g. changing the values of the LW/SW that correspond to
    the "warm" and "cold" timesteps, or changing the arrays from being constant to ramping up over time, etc.

    For a full list of variables that *are* used by MONARCHS, see :doc:`model_setup_reference`. Many of these you do not need to worry about
    until running more advanced cases, to have more control over exactly how the model runs.

Since our model is quite large, and we are running for a longer time, our output files can become quite large also. We can reduce the temporal frequency of the output
by adding the ``output_timestep`` variable  into ``model_setup.py`` anywhere before the ``if __name__ == '__main__'`` section  - if you look at the output of the start of a model run
without this variable included in the runscript you will see the line

.. code-block:: python
    ``monarchs.core.configuration.create_defaults for missing flags: Setting missing model_setup attribute <output_timestep> to default value 1``

i.e. that MONARCHS has detected that it is missing from ``model_setup.py`` and set a "sensible" default value.
Adding ``output_timestep`` into ``model_setup.py`` will override this default value. Sensible values might be e.g.
``7`` for weekly output, or ``30`` for daily output.
You can also reduce the vertical resolution of the output by adding or changing ``output_grid_size`` from e.g. ``400`` to ``200``.
Both of these steps will give you less vertical/temporal information, but decrease the size of the output file.

The size of the output file can also be significantly decreased by removing variables from the ``vars_to_save`` variable. For example,
if we are only interested in the amount/depth of lakes in the model, and not the firn column properties, then we can remove
``firn_temperature``, ``Sfrac`` and ``Lfrac`` from ``vars_to_save``. By default, these variables take up
400 times as much space as e.g. the firn depth, lake depth and lid depth, since they are saved at the model vertical resolution (400 in this case)
rather than being single values.

You can see that the model setup script has a few additional parameters compared to the 1D case. As mentioned earlier,
MONARCHS will set "sensible" default values for any parameters that are not specified in the model setup script, aside
from those that the model will not be able to run without - i.e. an initial firn profile, and meteorological data.

You will notice that the firn profile is determined by an imported Python function from ``monarchs.dem_utils.create_DEM_GaussianTestCase``.
A neat feature of our model setup file being a Python script is that you can freely generate any input firn distribution
you want using Python code, as long as it is passed in as a 2D ``numpy`` array.
This can be useful for testing, but also for generating realistic initial conditions using
meshes that aren't supported by default (see the :doc:`dem` section of the documentation for more on this).

A more detailed introduction to ``model_setup.py`` can be found in :doc:`MONARCHS_model_setup`.
You can see all of the possible ``model_setup`` variables in the :doc:`model_setup_reference` section of the documentation.
This tutorial will not cover all of these, as many of them are for testing and debugging purposes. Many of these
are related to the use of a digital elevation model (DEM) to set the initial firn profile, and synchronising this
to the input meterorological data. This is covered more in the :doc:`dem` section of the documentation.

Restarting a model run
======================

If you have a model that has crashed, or you want to restart a model from the end of a previous run, you can use the
``reload_from_dump`` variable in the model setup script. This will load in the state from the dump file specified in the
``dump_filepath`` variable, and restart the model from this point. This allows for finishing of crashed runs, or to
use the initial conditions of a previous run as a starting point for a subsequent one.

If your model run was not successful, then re-running will run it until your initially-intended
finishing point.

If it `was` successful, then attempting to re-run with no changes to the setup script will result in
nothing happening (as the model will try and start from the same day that it is supposed to finish at!).
However, you can extend the run further by increasing ``num_days``.

If you do this, remember that you will need to extend your
meteorological dataset. MONARCHS will read from the index corresponding to the start day of the run,
which in this case is not 0 - e.g. if restarting from day 5 it will read from ``met_data`` at index (5 * 24) = 120 - the
prior indices have already been used in the model.


Having more control over output directories
===========================================

You can call your run scripts anything you want, rather than just ``model_setup.py``, and they can be in any folder, not
just the folder you are running in. This is useful if you want to e.g. keep several test cases in the same folder, and write
the outputs elsewhere. You can tell MONARCHS exactly which setup script to run from by using the ``-i`` flag on the command line.
For example, from anywhere on your machine, assuming MONARCHS is installed in ``/home/users/username/monarchs``, you can do:

``monarchs -i /home/users/username/monarchs/examples/10x10_gaussian_threelake/model_setup.py``

You could rename ``model_setup.py`` to ``model_setup_threelake.py`` and pass this as the name, and it would work the same
were you to pass this as the argument.




