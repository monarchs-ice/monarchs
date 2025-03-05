
Installing MONARCHS
************

Requirements
------------

MONARCHS requires Python 3.9 or later, and is specifically tested with versions 3.9-3.12.
It is highly recommended that before installing MONARCHS, you set up and activate a
fresh virtual environment. You can do this either via e.g.

``conda``

.. code-block:: console

    conda create -n monarchs_env python=3.9
    conda activate monarchs_env

or using ``venv`` by first creating the environment with

.. code-block:: console

    python -m venv monarchs_env

and then activating (on Windows) via:

.. code-block:: console

    ./monarchs_env/Scripts/activate

or on Linux via:

.. code-block:: console

    source ./monarchs_env/bin/activate

Installation
-----------
The best way to get MONARCHS is to clone the GitHub repository using

.. code-block:: console

    git clone https://github.com/monarchs-ice/MONARCHS

and doing (ideally in a virtual environment, and including the ``.``)

.. code-block:: console

    python -m pip install -e .

from the top level MONARCHS folder.

This will install MONARCHS with all of its required dependencies, except those required for specific optimisation flags
designed for use on HPC systems (see below). If when trying to run MONARCHS you get an ``ImportError``, please submit
an Issue on GitHub.

Installation for use on HPC
-------------------------------
You can install MONARCHS with its advanced dependencies using (from the top level MONARCHS folder):

.. code-block:: console
    pip install -e .[mpi,numba]

.. note::
    Not all of the modules in requirements.txt are required to make MONARCHS work, but are required to enable certain features.
    If the install fails on either of the following, MONARCHS will still work, but only without the
    relevant ``model_setup`` flags enabled. MPI especially is suggested only for running on HPC systems.

.. warning::
    The install will fail on ``NumbaMinpack`` if you don't have a Fortran/C++ compiler.
    To solve, on Windows, you need to get a Fortran compiler. See https://fortran-lang.org/compilers/
    On linux, doing ``apt-get install gfortran build-essential`` will work.
    On HPC, you may need to use ``module load <name>`` to load in whichever compiler setup your HPC has.
    You can get around the need for ``NumbaMinpack`` by setting ``use_numba = False`` in ``model_setup.py``.

.. warning::
    The installation will fail on ``mpi4py`` if you don't have a working MPI installation. On Windows, you need to install
    https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi.
    On Linux, ``apt-get install mpich`` should suffice.
    If you don't want to run with MPI at any point, you can do
    ``pip install -e .[numba]`` to install MONARCHS with all the dependencies except
    ``mpi4py``.
.. note::
    It should be re-iterated that neither NumbaMinpack nor Pyina are required to be installed for the code to run,
    provided that you set ``use_numba = False`` and ``use_mpi = False`` in ``model_setup.py``.

Singularity/Docker image
========================
Instead of cloning the repo and installing the requirements yourself, you can get MONARCHS via a Docker image. This image
contains a barebones Linux distribution, with all of the required libraries (including a Fortran compiler and MPI
implementation) pre-installed.

To obtain this, please get in touch with the model maintainers for access as the image is currently private.

On HPC systems, you likely won't have Docker installed, but may have Singularity or Apptainer.
You can obtain the repo with Singularity (replace ``singularity`` with ``apptainer`` if required) via

``singularity pull docker://jelsey92/monarchs``

If you already have a copy of the container, then you may need to delete it first.
A shell script to do this is provided in the ``/scripts`` folder of the MONARCHS repository.
Additionally provided is an example runscript for running a batch job using this Singularity container.

The MONARCHS source is included in the ``/MONARCHS`` folder of the container. If you are running the container
interactively (e.g. using ``singularity shell`` or Docker Desktop), you can run MONARCHS using the ``monarchs`` CLI
command as normal.