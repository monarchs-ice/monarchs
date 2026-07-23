Running MONARCHS with a Digital Elevation Model (DEM)
=======================================================

In the previous examples (see :doc:`quickstart`), we have been running MONARCHS using a user-generated firn profile -
either a single value for our 1D case, or a profile generated from a Python function in our 2D Gaussian lakes case.
In the previous cases, we use the ``firn_depth`` variable, and supply it either a single value or a 2D ``numpy`` array.

In this example, we instead use the ``dem_path`` variable to specify a path to a Digital Elevation Model (DEM) file.
The presence of this variable will override the ``firn_depth`` variable if present, and the model will use the DEM to generate
the initial firn height distribution.

MONARCHS supports DEMs in ``.tif`` format, and uses `rasterio <https://rasterio.readthedocs.io/en/latest/>`_ to process
these. An example DEM for the George VI ice shelf is available at
https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/32m/38_12/. It is this DEM we will use for this example.
Download the folder and extract the contents to a folder of your choosing. Specify the path to the ``.tif`` file in the
``dem_path`` variable in ``model_setup.py``.

We also need to use some meteorological data to drive the model. An example dataset can be found in the ``data`` folder
of the MONARCHS repository. This is a one-year ERA5 dataset, encompassing the area used in the DEM.
Specify the path to this file in ``met_input_path``. For more information on the input meteorological data, see
:doc:`met_data`.

This is all of the information you need to run the model with the George VI ice shelf.
It will take some time for the initial conditions to be set up, as the DEM is at very high resolution and requires
interpolation to fit the model domain. The actual running of the model will also be slow. However, you may notice that
it is faster than the Gaussian test case when running with the same model size. This is because of a few parameters
that exist within this version of the model setup script that are used to handle DEM data. These are:

.. code-block:: python

    firn_max_height = 100
    firn_min_height = 35
    max_height_handler = "filter"
    min_height_handler = "extend"


A DEM of a real ice shelf will have a large range of values. MONARCHS is designed for modelling ice shelves, but the
DEM we have loaded in also contains lots of actual land in addition to the ice shelf. These parameters are used to handle
this. The land is much higher than the ice shelf - therefore we set the ``firn_max_height`` parameter to 100, alongside
the ``max_height_handler`` parameter to ``'filter'`` to tell MONARCHS
that anything in the initial firn profile above 100 metres should be filtered out. This sets a flag in the ``IceShelf``
class used to contain the model data (see :doc:`structure` for more information on this, or see the API reference).
This flag is called ``valid_cell``. If this flag is ``False``, then MONARCHS ignores the cell entirely, no single-column
physics can run on it, and no water can flow to or from it.

We also have a situation where the data is say in metres above sea level, which can cause some values to be very low.
In order to run the single-column physics effectively, we need to ensure that the firn profile is at least 35 metres deep.
The ``extend`` option in the ``min_height_handler`` parameter tells MONARCHS to increase the lowest point in the DEM
up to 35m (if it is below this value), and increasing every other cell by the same amount.

Other options to this can be found in the :doc:`model_setup_reference`.

Selecting a subset of the DEM
------------------------------
