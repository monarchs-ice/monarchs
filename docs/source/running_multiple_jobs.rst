Running multiple jobs with (slightly) different configurations
----------------------------------------------------------------
If you have several jobs where you want to change only a few parameters (e.g. for benchmarking purposes), you can
exploit the fact that `model_setup` is a Python script, and use the `argparse` module to set up your own
command line interface.

For example, say you wanted to run a single-column job, with different vertical resolutions as a benchmarking test.
Using the `1d_testcase` example as a template, we can amend it with the following:

.. code-block:: python
   print(f"Loading runscript from {os.getcwd()}/model_setup.py")
   """
   Spatial parameters
   """
   ### new code goes here ###
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--vertical_points_firn', type=int, default=2500, help='Number of vertical grid cells in firn')
   vertical_points_firn = parser.parse_args().vertical_points_firn

   ### original code from here ###
   # vertical_points_firn = 2500  # original, commented out
   row_amount = 1  # Number of rows in your model grid, looking from top-down.
   col_amount = 1  # Number of columns in your model grid, looking from top-down.
   lat_grid_size = 1000  # size of each lateral grid cell in m - possible to automate
   vertical_points_lake = 20  # Number of vertical grid cells in lake
   vertical_points_lid = 20  # Number of vertical grid cells in ice lid


Now, you can run the code with

`python model_setup.py --vertical_points_firn 500`

to run with at 500 point resolution. You can exploit this by writing a script to loop over different values, e.g. in
Windows:

.. code-block:: batch
    @echo off
    for %%v in (200 300 400 500 700 1000 2500 5000 10000 35000) do (
        start "" python model_setup.py --vp %%v
        timeout /t 5 /nobreak >nul
    )

or Linux:

.. code-block:: bash
    #!/bin/bash
    for v in 200 300 400 500 700 1000 2500 5000 10000 35000
    do
       python model_setup.py --vertical_points_firn $v &
       sleep 5
    done

or even using a one-liner with GNU Parallel:

.. code-block:: bash
    parallel -j 4 --delay 5 "python model_setup.py --param {}" ::: 200 300 400 500 700 1000 2500 5000 10000 35000

or via an array job using a job scheduler if running larger jobs on HPC.

By default all of these approaches run in parallel, which is suitable for small jobs. For larger jobs, it may be more
prudent to do them in series (i.e. removing the `start ""` in Windows or the `&` in Linux).
A delay (via `timeout`, `sleep` or `delay` respectively) is included
so that the jobs don't all start at exactly the same time, in case there are issues with all the jobs trying to read from
the same file. You can generalise this to multiple variables - add an option via `parser.add_argument` as before, and
then use a nested loop in your batch/bash script (or separating the variables via `:::` in GNU Parallel)