How to run MONARCHS
===================
If MONARCHS has been installed via ``pip`` or ``conda``, you can run the model using

``monarchs``

or by navigating to the MONARCHS root folder and doing

``python run_MONARCHS.py``

This assumes that your runscript (:doc:`MONARCHS_model_setup`) is in the folder you are currently in.  If your model
setup script has a different name or is in a different folder, you can pass the name of that folder to the code
by using:

``monarchs -i <path_to_runscript>``

or ``python run_MONARCHS.py -i <path_to_runscript>``

Running in an IDE
*****************
In PyCharm or VSCode, you can debug or run the model when ``model_setup.py`` is in a different folder.
To do this, make a run configuration, specifying the working directory to be the folder you have your runscript in,
and the "script" to run as ``run_MONARCHS.py``.

Advanced users
**************
Running with MPI
----------------

It is possible to run MONARCHS across multiple nodes on HPC systems using MPI. This is achieved without the need for
significant MPI code using an ``mpi4py.futures`` ``Pool``. This allows us to spawn MPI processes as and when they are needed, in
a similar way to how a ``multiprocessing`` ``Pool`` works, and allows for the code to be mostly free of complex MPI directives
and boilerplate. This does not have a significant overhead compared to using ``multiprocessing`` even on single-node
systems.

Currently, it is not possible to use MPI with Numba. This is current WIP but will not be ready on release.

Running MONARCHS with MPI may be slightly different to how you may have used MPI in the past, since it uses this pool/spawning
approach. You should run with only a single MPI process, i.e. do:

``mpirun -n 1 python run_MONARCHS.py -i <model_setup_path>``

The number of MPI processes is controlled by the model setup variable ``cores``. If running on HPC, you likely want
to use ``cores = 'all'`` to ensure that you use all of the cores you have available in your job.

Attempting to do mpirun with more than 1 process will result in several processes running the whole code, which will result
in an attempt to spawn ``<cores>`` processes for each of the N processes you create with the call to ``mpirun``.