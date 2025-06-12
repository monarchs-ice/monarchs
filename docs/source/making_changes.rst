
Making changes to MONARCHS
====================================

Adding new diagnostics or physics
**********************************

If adding new variables to the code, you will need to do the following:
    - Add the variable into the ``spec``, found in ``monarchs.core.model_grid``, and initialising it if appropriate in ``monarchs.core.initial_conditions.create_model_grid``. This is required since we need to ensure that our code is compilable with Numba, which forces strict typing.
    - Add the variable into the model code itself. This likely involves making the relevant changes to the various files/functions in ``monarchs.physics``.
    - If your new variable is a diagnostic, add the variable to ``vars_to_save`` in your runscript, so that the code knows to track it over time and save it to the output netCDF file.
    - If your new variable relies on a toggle or other ``model_setup`` variable, set a default value for this in ``monarchs.core.configuration.create_defaults_for_missing_flags``.


Merging your changes into the MONARCHS source
*********************************************

If you have added anything to the MONARCHS source code, please let us know! We welcome pull requests from users who have added features or squashed bugs that we have let slip through the net.
Please make any changes you want to make to the code in a new branch of the main repo (i.e. not ``main``), or create a fork. Pushes directly into ``main`` are not allowed even if you are a collaborator in the ``monarchs-ice`` organisation.

For your work to be merged into ``main``, it needs to pass our test suite. The tests are automatically run via Github Actions
when a pull request into ``main`` is made, so it will quickly be apparent if it does not pass.

Additionally, any new functions or physics/diagnostics should have docstrings or comments where possible. If you have written
new physics functions, it is best if these have some kind of unit test. Any new physics should have an associated toggle
defined in ``model_setup.py``, added to ``toggle_dict`` in ``monarchs.core.driver.setup_toggle_dict``, and
a switch to turn it on or off using the value of ``toggle_dict``.

Any changes that require amendments to ``model_setup.py``
should have suitable documentation added to ``docs/source/model_setup_reference.rst``.

Advanced users
------------------------------------
Adding functions with Numba support
***********************************

(first, see :doc:``advanced`` for some background information).
*If* Numba support is useful for your change, consult `the Numba documentation <https://numba.readthedocs.io/en/stable/user/5minguide.html#will-numba-work-for-my-code>`_ to ensure that your code uses only pure Python and ``numpy`` functions.
Using other modules (e.g. ``scipy``) is not supported by Numba, and therefore the code won't work with the ``use_numba`` optimisation flag set in ``model_setup.py``. This will mean that the model as a whole runs slower.

If your function is included within any of the existing ``physics`` modules (with the exception of ``heateqn`` and ``solver``), or within ``utils`` or ``timestep`` in ``core``, then provided that it fits the Numba specifications, Numba support should be
automatic, i.e. MONARCHS will automatically try and jit the function. If you add a function that you specifically do not want to apply Numba decoration to (e.g. the code is not called by other Numba code and contains incompatible code),
you can ensure that this step is avoided using the ``do_not_jit`` decorator in ``core.utils``.