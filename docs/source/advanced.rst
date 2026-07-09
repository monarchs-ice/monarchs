Advanced users
=============
Optimised performance using Numba (the ``use_numba`` flag)
**********************************

tl;dr - should I set ``use_numba`` to True or False?
===========================================
If performance is important (e.g. if running on HPC), set it to ``True``. Otherwise, you are OK without.

If you need to debug your run, then keep ``use_numba`` ``False``.

Why Numba?
==================
MONARCHS is written in Python, due to its ease of use, portability, suitability for use on Windows, Mac and Linux, and
myriad other reasons. However, one of the drawbacks of Python is that it is slow, compared to low-level languages
such as C and Fortran. A compromise is to make use of Numba, a just-in-time compiler for Python. This significantly
bridges the performance gap between these languages, at the cost of somewhat more complex code.

In many cases, this is "free". The heat equation and surface energy balance solvers are written as Newton iterations
in Numba-compatible Python (see ``monarchs.physics.solver`` and ``monarchs.physics.heateqn``), so the same solver
code runs compiled or uncompiled - no external solver library is needed.

Numba also has great support for parallel Python via OpenMP, which we use in MONARCHS. Since the single-column physics
does not affect other columns, this approach is very efficient; resulting in speedups to the overall runtime of ~50x
or more for long runs.

What are the drawbacks?
========================
Since the code is compiled before running, for small runs
(e.g. single column over fewer than 1000 timesteps), this overhead will dominate the runtime, in which case it may
be prudent to keep ``use_numba`` ``False``.

This makes code development more complex since Numba requires strict static typing, and only supports a subset of
inbuilt Python and ``numpy`` functions. Important libraries such as ``scipy`` are not yet supported. The code has been
written to try and hide away most of this where possible, but some design choices were made during the development
of MONARCHS that are inelegant or un-Pythonic, to accommodate the use of Numba.

In this vein, feedback or suggestions on how to improve the readability of the MONARCHS source code are appreciated.

Additionally, Numba code is significantly harder to debug, since it doesn't use the normal Python stack trace.
A compromise here is to initially run your code with Numba, ensuring that the model :doc:``dumping flags <model_setup_reference>``
are enabled, and then after the code crashes, run the model from this dump with ``parallel = False`` and
``use_numba = False`` in your runscript to debug.

What parts of the code are actually different if using Numba?
=============================================================
None - the same code runs in both modes. Physics functions are marked with the ``@kernel`` decorator
(``monarchs.core.kernels``); when ``use_numba`` is enabled, ``kernels.compile_all`` compiles every registered
kernel with ``numba.njit`` and rebinds references to the compiled versions. With ``use_numba = False`` the
decorator is a no-op and the identical functions run as pure Python.

Are there any differences between the two versions?
===================================================
The model flow and algorithms are exactly the same in the non-Numba and Numba versions. However, there may be some
small differences in the numerics, which can evolve over time. In general, the output of the two versions is extremely
close, with no significant divergence observed over multi-year runs.
