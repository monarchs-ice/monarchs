Outstanding work
================
Redo test suite now initial development phase is nearing completion.
Finish adding tests for lake/lid/firn column (previous tests were outdated and removed). 

Testing
=======

Tests are split into several folders. This is done so that each test setup can have a separate runscript, meaning
that we can run tests without having to do a significant refactor to the code by injecting the parameters from this
runscript into the model configuration. 

Tests in `serial` are the main model physics unit tests and comparisons to known good output.

Tests in `numba` are the tests that ensure that the code can be `jit`-compiled, and that the results obtained from 
the `NumbaMinpack` version of the solver are equal to those with the `scipy.optimize` version.

Tests in `parallel` are there to check that MONARCHS runs in parallel and the results obtained are equal to the serial 
version.

Running tests
-------------
Ensure that you `cd` into the relevant directory, _then_ run the test, rather than using a filepath.
For example, do
    cd test/serial
    python -m pytest test_heateqn.py
and not
    python -m pytest tests/serial/test_heateqn.py

Not doing this will cause the tests to fail as MONARCHS uses `configuration.py` to load in a runscript, but when testing
it tries to load in runscripts from specific paths. A workaround is WIP to load in the correct filepath rather than
naively using `os.getcwd()`, but for now please run test cases in the relevant directory. An example can be seen in 
`.github/workflows/python-project.yml`, where the Github Actions runner is instructed to `cd` into `./tests/serial`, 
then run `pytest`, then cd into `/tests/numba`, etc.

Adding tests
------------

When adding tests, ensure that any imports of the MONARCHS physics or utilities (i.e. anything inside `physics` or 
`core`) are done inside the body of the test function. This is to ensure that `configuration.py` recognises that a test
is being run, and therefore passes the relevant `model_test_setup_<runtype>.py` file.

e.g. do:

    import numpy as np
    # works!
    def test_bar():
        from monarchs.physics.foo import bar
        from monarchs.core.utils import get_2d_grid
        <test_code>
    
rather than:

    import numpy as np
    from monarchs.physics.foo import bar
    from monarchs.core.utils import get_2d_grid
    # doesnt work!
    def test_bar():
        <test_code>
    
If you create a new folder for testing, ensure that you create an appropriately named runscript (`model_setup.py`), and
that you amend `monarchs.core.configuration.parse_args` to read it in correctly.
