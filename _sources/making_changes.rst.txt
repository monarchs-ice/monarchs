
Making changes to MONARCHS
====================================

Code style guidelines
*******************************
These guidelines are intended to keep the codebase as consistent as possible, and to make it easier for others to read your code.

.. note::
The guidelines are there as a developer aid;
not as a barrier to contribution - so if your code is not compliant please feel free to PR it anyway, and we can help you make it work.

When making changes to the ``monarchs`` source (inside ``src/monarchs``), please try and adhere to the following:
- Follow PEP8 guidelines where possible. We *strongly* recommend using the `black <https://black.readthedocs.io/en/stable/>`_  auto-formatter to ensure that your code is PEP8 compliant.
The GitHub Actions runner checks this, and will fail if the code does not adhere to the standards if trying to PR into ``main``. You can check that your code is compliant by running

    ```bash
    python -m black src/monarchs --check --line-length=79 --preview --enable-unstable-feature string_processing
    ```

from the ``monarchs`` root directory, or do the same thing without ``--check`` to automatically reformat.
Any scripts outside of ``src/monarchs`` do not need to be compliant.

- Add docstrings to any new functions you write, following the format used in existing functions.
- Add comments to your code where appropriate, especially if the code is complex or not self-explanatory.
- Ensure that these comments are above the line you are commenting on, and not at the end of the line, unless the comment
is a) very short and b) is referring to a scalar rather than an array. This is because ``black`` will make this harder to read otherwise.
For example:

    ```python

    """Compliant"""

    # Do some complex function

    def complex_function(x):
        ...

    x = x + 1  # Increment x by 1  --- fine, as commenting a scalar

    # Set x to next array element

    x = array[i + 1]

    """Not compliant"""

    x = array[i + 1]  # Set x to next array element

    # would become

    x = array[
            i
            ] + 1  # Increment x by 1

    ```

- New functions added are in "active" tense, e.g. ``calculate_thing`` rather than ``thing_calculation``.

- New variables in functions should be named either in ``snake_case`` (e.g. ``firn_temperature``, not ``T_firn``) or if corresponding to
a physical quantity, using something close to its mathematical notation if appropriate (e.g. ``dTdz`` for the vertical temperature gradient).

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

If your function is in a new module (e.g. ``monarchs/source/physics/new_physics_routines.py``, you should add your module
name to ``module_list`` in ``jit_modules`` in ``monarchs.core.configuration`` (and import it in the line above).
This ensures that the Numba decoratior is applied to all functions in your module automatically. If any are non-compatible,
you can use the ``do_not_jit`` decorator as above, or add them to ``ignore_list`` in the same file.