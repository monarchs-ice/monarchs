Documentation 
=============

Documentation is generated using Sphinx, and published on the ``monarchs-ice`` ``github.io`` page.
If you want your functions, classes or methods to be included in the documentation automatically, please 
ensure that they have appropriate docstrings in the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html/>).

The source files are located in ``/source``, and can be 
made locally using ``make html`` from inside this folder. Github Actions is configured to build and push the documentation to ``monarchs-ice.github.io``
whenever a commit or PR is made to ``main``.

If adding a new page (i.e. a new ``.rst`` file) to ``/source``, please add a link in the contents page (``index.rst``).