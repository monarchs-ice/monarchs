# MONARCHS

[![DOI](https://zenodo.org/badge/890500319.svg)](https://doi.org/10.5281/zenodo.14217406)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

MOdel of aNtARtic iCe shelf Hydrology and Stability 

Code associated with the ice shelf surface hydrology model MONARCHS. Please check out the documentation at the
[MONARCHS website](https://monarchs-ice.github.io/monarchs) for more info on setting up and running the model, and how best to contribute.

MONARCHS was developed by [Sammie Buzzard](https://www.northumbria.ac.uk/about-us/our-staff/b/sammie-buzzard/) at Northumbria University and [Jon Elsey](https://environment.leeds.ac.uk/see/staff/12908/dr-jon-elsey) at the University of 
Leeds (both formerly Cardiff University), in collaboration with Alex Robel and the [Ice and Climate](https://iceclimate.eas.gatech.edu/) group at 
Georgia Tech.

It is available and open source under the GPL v3.0-or-later License. If you are planning on using it, we would very much
appreciate being kept in the loop, or a citation being given to our paper: #TODO - link/doi etc. 

If you make changes to the code, please feel free to submit a pull request to get your features added into the core model repo!
Alternatively, if you have feature requests, please submit an Issue on the GitHub repo, or get in touch with one of the
model developers.

The model source code is located in the `src/monarchs` folder. 

If you have any feedback or issues, please submit via Github Issues if a problem with the code, or email Jon Elsey at 
the University of Leeds for help setting up on HPC systems.

Todos
-----
- Validation tests (Moussavi lakes)
- How to handle DEMs with large vertical changes - e.g. 100m-0m firn.
- Type hints 
- Full unit test suite
- Longer term - a MONARCHS GUI may be really handy for specifying the inputs in model_setup.
- MPI - still WIP to get working on e.g. ARCHER2

Installation
------------
I recommend setting up a new virtual environment and using it for MONARCHS only, to avoid dependency issues. You 
can do this using `conda create -n monarchs python=3.9` or `python -m venv monarchs`.

You can install MONARCHS and its core dependencies on your system by cloning this repository and from the project root folder do

`python -m pip install -e .`

Note that this requires an up-to-date version of `pip` (e.g. if you get an error related to `pyproject.toml`-based projects).
You may first need to do

`python -m pip install --upgrade pip`.

If you want changes you have made to be picked up on your system, ensure that you use the `-e` flag when installing. See `pyproject.toml` for details.

In future, it will be possible to install a stable, version of MONARCHS with all its dependencies by doing 

`python -m pip install monarchs`

or 

`conda install -c conda-forge monarchs`

It is still recommended to clone the repository and install with `pip install -e .` if you are intending to make 
changes to the model source code.

## Installing optional dependencies
The simplest way to install the extra MONARCHS dependencies from an existing build is doing

`python -m pip install -r requirements.txt`

Alternatively, you can install MONARCHS from scratch with all optional dependencies using 

`python -m pip install -e .[mpi,numba]`

Some systems may be incompatible with certain libraries. For example. the install will fail on ```NumbaMinpack```
if you don't have a Fortran/C++ compiler. 

To solve, on Windows, you need to get a Fortran compiler. See https://fortran-lang.org/compilers/
On linux, doing ```apt-get install gfortran build-essential``` will work.

Alternatively, do
`pip install -e .`, and run with `use_numba = False` in all 
of your runscripts.

The installation will fail on ```mpi4py``` if you don't have a working MPI installation. On Windows, you need to install
https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi
On Linux, ```apt-get install mpich``` should suffice. This is only suggested for advanced users running on HPC systems.

It should be re-iterated that neither NumbaMinpack nor Pyina are required to be installed for the code to run, 
provided that you set ```use_numba = False``` and/or ```use_mpi = False``` in your runscript (see below).


Running the model
-----------------
However you install the model, you can run MONARCHS using

`monarchs --input_path <path_to_runscript>`

or 

`python run_monarchs.py --input_path <path_to_runscript>`

where `<path_to_runscript>` is the location of a file with the model running parameters as seen in `model_setup.py`.
You can use `-i` as shorthand for `--input_path`. 

You may omit `--input_path` entirely if your runscript is called `model_setup.py` and is in the same folder you run 
MONARCHS from. In general, it is recommended to use a separate runscript for each run, and point to these directly.



## Plotting output
To plot output, you can do this yourself by adding your relevant plotting functionality to run_monarchs.py, 
or there are some functions provided in `/plots` which can help you with this. This is WIP. Additionally, in `/scripts` 
there is a script `debug_model_state` which is useful for loading in a snapshot of the model state and plotting it out,
e.g. for looking for the presence of melt lakes quickly.

## Issues and pull requests
Please, if possible, submit problems/feature requests via GitHub Issues rather than email.
We welcome pull requests from those who have solved bugs or added features themselves!

## Debugging

A useful workflow for debugging runs is to re-run it from the last iteration, with optimisations (Numba and parallelism) 
turned off. This can be done using ```dump_data = True``` and specifying a ```reload_file``` in ```model_setup.py```. 

## Testing

Tests are located in the `tests` folder at the root of the repository. These are separated by type, as some variables 
are set in model_setup.py that are akin to environment variables, in particular `use_numba`. They therefore require
separate runscripts. See [README.md](tests/README.md) in the `tests` folder for more info.
