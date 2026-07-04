# Contributing to MONARCHS

Thanks for your interest in contributing! We welcome contributions from the 
community - feedback, bug reports, feature requests and pull requests are all
massively appreciated!

## Reporting bugs / requesting features

Open a [GitHub issue](https://github.com/monarchs-ice/monarchs/issues).
For bugs, please include your `model_setup.py`, the full console output, and
the MONARCHS version (`pip show monarchs-ice`). Ideally, a link to the input
data used for the run would be included too, so that we can try and reproduce 
the error and work out if it is a bug, an issue with the setup/input data, or 
something machine-dependent.

## Development setup

```bash
git clone https://github.com/monarchs-ice/monarchs
cd monarchs
python -m venv .venv && source .venv/bin/activate
python -m pip install -e .[numba] --group dev
pre-commit install
```
This will install the development dependencies, including `pytest` and `ruff` for testing and linting,
and the `pre-commit` hooks for automatic code formatting and linting.

`NumbaMinpack` (pulled in by the `[numba]` extra) needs a Fortran compiler;
if you don't have one, install without the extra and run with
`use_numba = False`.

## Running the tests

```bash
pytest tests/unit_tests            # fast, pure-Python physics tests
cd tests/numba && pytest .         # compiles and runs a 10-day model (~1 min)
```

Some of the tests are required to pass for a PR to be merged, particularly those 
to ensure that the model compiles and runs. The broader "correctness" tests are more for tracking
whether code changes have impacts on the model output, so are not required to pass
since your change likely alters the physics in some way! 

## Code style

- Formatting and linting are enforced by `ruff` via pre-commit and CI.
  This means that we can make our code styling consistent. You can set up
  the precommit hooks by running `pre-commit install` in the root of the repository.
- Physics functions at `timestep_loop` level and below are decorated
  `@kernel()` (see `monarchs/core/kernels.py`) and must be
  Numba-compilable. This means:
  - no `Exception` or `raise` for errors (use the
    `error_flag`/`generic_error` pattern). This is needed because Numba will 
  silently flag an exception but not actually raise, so the model just continues running
  despite something having gone wrong
  - kernels should be pure Python or Numpy; no `scipy` or other libraries. If you need something that is in
    e.g. scipy for your change, get in touch and we will see if it can be implemented in a compatible way!
  
- New constants should be added to `core/constants.py` and used in the physics, not
  hard-coded. Likewise, formulae to calculate e.g. parameters should be made into
  functions (feel free to create new submodules) so we can re-use them, rather than inlined. 
- Adding literature references is always useful when defining equations!

## Adding a model variable

See the "Making changes" page of the documentation. In short, add it to the
grid spec (`core/model_grid.py`), initialise it, use it in the physics, and
add it to `vars_to_save` if it is a diagnostic. New `model_setup` options
need a default in `core/configuration.py`.

## Pull requests

- Branch from `main` where possible. Smaller PRs are easier to review and merge
so will have a faster turnaround time, but if your change is large, please open a draft PR and we can discuss it!
- PRs need to be approved by at least one reviewer before merging. 
- Please get in touch with one of the model developers if you have questions, or are unsure how to 
implement a change. We are happy to help and provide guidance on how to contribute to the codebase!
