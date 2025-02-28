
Installation
************

The best way to get MONARCHS is to clone the github repository using

.. code-block:: console

    git clone https://github.com/monarchs-ice/MONARCHS
and doing (ideally in a new virtual environment created via e.g. ``conda create --name monarchs_env`` or ``python -m venv monarchs_env``)

.. code-block:: console

    python -m pip install -e .
from the top level MONARCHS folder.

.. note::
    If the model gives an ``ImportError`` for any reason,
    you can install the required modules using:
    .. code-block:: console

        python -m pip install -r requirements.txt
        conda install --yes --file requirements.txt
    This will install all the required modules for MONARCHS to run.

Installation for use on HPC
-------------------------------
You can install MONARCHS with its advanced dependencies using (from the top level MONARCHS folder):

.. code-block:: console
    pip install -e .[mpi,numba]

.. note::
    Not all of the modules in requirements.txt are required to make MONARCHS work, but are required to enable certain features.
    If ``python -m pip install -r requirements.txt`` fails on either of the following, MONARCHS will still work, but only without the
    relevant ``model_setup`` flags enabled. MPI is suggested only for running on HPC systems.

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

To obtain this, please email j.d.elsey@leeds.ac.uk for access as the image is currently private.

On HPC systems, you likely won't have Docker installed, but may have Singularity or Apptainer.
You can obtain the repo with Singularity (replace ``singularity`` with ``apptainer`` if required) via

``singularity pull docker://jelsey92/monarchs``

If you already have a copy of the repository, then you may need to delete it first.
A shell script to do this is provided in the `/scripts` folder of the MONARCHS repository.
Additionally provided is an example runscript for running a batch job using this Singularity container.

The MONARCHS source is included in the ``/MONARCHS`` folder of the container. If you are running the container
interactively (e.g. using ``singularity shell`` or Docker Desktop), you can run MONARCHS using the ``monarchs`` CLI
command as normal.