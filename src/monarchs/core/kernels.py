"""
This module defines the ``@kernel`` decorator. This should be applied to all
physics code - i.e. all code at the level of ``timestep_loop`` and below.

The decorator itself registers the function for compilation later, but
compilation is triggered later on,

This approach avoids the need for a manually-compiled list of functions to compile
- instead we just add ``@kernel`` to anything we want to do so,
and then the model infers via context (specifically the ``use_numba``
flag) whether or not to apply the compilation.

We also re-export ``prange`` and ``objmode`` here so that physics modules
do not import numba directly.
"""

# numba.prange behaves like range, and objmode like a no-op passthrough,
# in uncompiled (pure-Python) code - so these are safe in both modes
from numba import prange, objmode  # noqa: F401  (re-exported)


# tracks all functions decorated by @kernel
_KERNELS = []

# a couple of overrides here - these *are* hard-coded but future work will hopefully
# remove them in favour of a unified solver approach
_NUMBA_OVERRIDES = [
    ("monarchs.physics.solver", "monarchs.physics.Numba.solver_nb"),
]


def kernel(**njit_opts):
    """
    Mark a function as a Numba kernel.

    This is a "decorator factory" - i.e. it creates a decorator with the
    desired options (e.g. fastmath, parallelisation) and returns it.
    This pattern is needed so that we can pass options to the decorator,
    e.g. ``@kernel(parallel=True)`` - this ensures the options are passed
    through to the appropriate ``@njit`` call.

    This means we need to invoke with brackets rather than just @kernel.

    e.g.
        @kernel()
        def lake_function(...): ...

        @kernel(parallel=True, fastmath=False)
        def parallel_lake_function(...): ...

    In pure-Python mode the function is returned unchanged. When Numba is enabled,
    ``compile_all`` compiles it
    and rebinds every reference to it to the compiled version.
    """

    # register function/decorator that adds the Python function to the list of
    # kernels to compile alongside the options specified
    def register(func):
        _KERNELS.append((func, njit_opts))
        return func

    return register


def compile_all(use_numba):
    """
    Compile all registered kernels and rebind references to them.

    No-op if ``use_numba`` is False. Steps:

    1. Import every module in the ``monarchs`` package so the registry is
       fully populated (kernels register themselves at import time).
    2. Compile each registered kernel with ``numba.njit`` and its options.
    3. Rebind, across every imported ``monarchs`` module, any attribute that *is*
       one of the compiled kernels (matched by ``id``) to its compiled version.
    4. Apply whole-module Numba variant overrides (e.g. ``solver`` <-
       ``solver_nb``).
    """
    if not use_numba:
        return

    import sys  # pylint: disable=import-outside-toplevel
    import pkgutil  # pylint: disable=import-outside-toplevel
    import importlib  # pylint: disable=import-outside-toplevel
    from numba import njit as njit  # pylint: disable=import-outside-toplevel

    print("\nmonarchs.core.kernels.compile_all: compiling Numba kernels")

    # Step 1
    # Import every module in the package. Skip the
    # .Numba modules since these contain cfuncs that require the
    # kernels it uses as dependencies first as they are compiled
    # eagerly rather than lazily - step 4 handles this
    import monarchs  # pylint: disable=import-outside-toplevel

    for _, modname, _ in pkgutil.walk_packages(
        monarchs.__path__, monarchs.__name__ + "."
    ):
        if ".Numba" in modname:
            continue
        importlib.import_module(modname)

    # Step 2
    # Add the @njit decorator to each kernel.
    jitted = {}
    for func, opts in _KERNELS:
        jitted[id(func)] = njit(**opts)(func)

    # Step 3
    # For each reference to a kernel, modify the reference to point to the
    # njit-decorated kernel.
    for modname, module in list(sys.modules.items()):
        if module is None or not modname.startswith("monarchs"):
            continue
        for attr, val in list(vars(module).items()):
            compiled = jitted.get(id(val))
            if compiled is not None:
                setattr(module, attr, compiled)

    # Step 4
    # Apply the overrides for the heat equation code that relies
    # on cfuncs (see step 1). This code has the appropriate
    # decorators already applied in the source as it is only
    # ever invoked in the Numba path
    for target_name, source_name in _NUMBA_OVERRIDES:
        target = importlib.import_module(target_name)
        source = importlib.import_module(source_name)
        for attr, val in vars(source).items():
            if not attr.startswith("__"):
                setattr(target, attr, val)

    print(f"monarchs.core.kernels.compile_all: compiled {len(jitted)} kernels")
