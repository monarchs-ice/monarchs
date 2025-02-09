
Installation
************

The best way to get MONARCHS if you are planning on developing/changing the code is to clone the github repository using

.. code-block:: console

    git clone https://github.com/monarchs-ice/MONARCHS
and doing (ideally in a new virtual environment)

.. code-block:: console

    python -m pip install -e .
from the top level MONARCHS folder.

To install the required modules to get MONARCHS to work (if it doesn't work after doing `pip install -e .`,
you can install the required modules using:
.. code-block:: console

    python -m pip install -r requirements.txt
    conda install --yes --file requirements.txt

Alternatively, you can install MONARCHS with its dependencies in one line using (from the top level MONARCHS folder):

.. code-block:: console

    pip install -e .[mpi,numba]

.. note::
    Not all of the modules in requirements.txt are required to make MONARCHS work, but are required to enable certain features.
    If ``pip install -r requirements.txt`` fails on either of the following, MONARCHS will still work, but only without the
    relevant ``model_setup`` flags enabled. MPI is suggested only for running on HPC systems.

.. warning::
    The install will fail on ``NumbaMinpack`` if you don't have a Fortran/C++ compiler.
    To solve, on Windows, you need to get a Fortran compiler. See https://fortran-lang.org/compilers/
    On linux, doing ``apt-get install gfortran build-essential`` will work.
    On HPC, you may need to use ``module load <name>`` to load in whichever compiler setup your HPC has.
    Alternatively, do
    ``pip install -e .[netcdf,mpi,multithreading,dem]``, i.e. omitting ``numba``, and run with ``use_numba = False`` in all
    of your runscripts.

.. warning::
    The installation will fail on ``mpi4py`` if you don't have a working MPI installation. On Windows, you need to install
    https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi.
    On Linux, ``apt-get install mpich`` should suffice.
    If you don't want to run with MPI at any point, you can do
    ``pip install -e .[netcdf, multithreading, dem, numba]`` to install MONARCHS with all the dependencies except
    ``mpi4py``.
.. note::
    It should be re-iterated that neither NumbaMinpack nor Pyina are required to be installed for the code to run,
    provided that you set ``use_numba = False`` and ``use_mpi = False`` in ``model_setup.py``.

Singularity/Docker image
========================
Instead of cloning the repo and installing the requirements yourself, you can get MONARCHS via a Docker image. This image
contains a barebones Linux distribution, with all of the required libraries (including a Fortran compiler and MPI
implementation) pre-installed.

To obtain this, # TODO - instructions here

On HPC systems, you likely won't have Docker installed, but may have Singularity or Apptainer.
You can obtain the repo with Singularity (replace ``singularity`` with ``apptainer`` if required) via

``singularity pull docker://jelsey92/monarchs``

If you already have a copy of the repository, then you may need to delete it first.
A shell script to do this is provided in the `/scripts` folder of the MONARCHS repository.
Additionally provided is an example runscript for running a batch job using this Singularity container.

The MONARCHS source is included in the ``/MONARCHS`` folder of the container. If you are running the container
interactively (e.g. using ``singularity shell`` or Docker Desktop), you can run MONARCHS using the ``monarchs`` CLI
command as normal.