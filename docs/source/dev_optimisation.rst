Optimisation guide for developers
=================================

Why Numba?
------------------------
MONARCHS uses an array-of-structs layout for the grid. This is in part due to the
legacy of the model (originally converted from a 1D MATLAB code), but there are readability
and performance advantages to this layout, since most operations within the model are
per-column, meaning that array-of-structs allows for better cache locality compared to
a struct-of-arrays layout.

This motivates the use of Numba and ``@njit`` rather than
vectorised NumPy expressions, since the latter would require a struct-of-arrays layout
to be efficient for the kind of operations we are doing.


Using explicit loops rather than array expressions
-----------------------------------------------------

Compared to regular Python, Numba prefers the use of explicit loops over
NumPy style array expressions. For example, we might want to do:

```python
```

as opposed to:

```python
```

This is mostly because Numba can then avoid allocating temporaries for the intermediate results,
which saves compute cycles but most importantly aids memory contention, particularly when running in parallel.

.. warning::

    Writing loops in this way *will make the model slower if Numba is not enabled*.
    NumPy's vector operations are significantly faster than Python loops, even if compiled
    Numba loops are faster! Therefore, the debug path (with ``use_numba = False``) will
    be significantly slower. This is expected behaviour and a tradeoff for the sake of
    performance with the optimisations enabled.


Parallel efficiency and memory contention
-------------------------------------------------
The initial findings when it came to parallel scaling were that it was far from ideal.
Even at ~10 threads, we were only getting a ~5x speedup in the single-column physics
compared to the ~9x we might hope for.

Testing indicates that this is due to memory contention between threads. Particularly
in the heat equation and turbulent mixing loops, there are many allocations and
deallocations of memory. This isn't a problem with one thread, but as the number of
threads increases, there are more allocations going on at the same time, and
we end up with threads waiting idle for the allocator to finish its work on other threads.

This was tested by running a) the model with 10 threads, and b) running 10 separate
single-threaded instances of the model at once. The time per-day was lower in the latter
case, despite the same amount of work being done on the machine.

The solution is to use a better memory allocator that is optimised for multithreaded programs.
One example is ``libjemalloc``, which is a drop-in replacement for the system allocator.
It is available on Linux and macOS, and can be installed via Homebrew on macOS or via your package manager on Linux.

To use it::

    # Linux / HPC
    LD_PRELOAD=/path/to/libjemalloc.so  python -m monarchs runscript.py

If using the Docker/Apptainer image, this is already pre-loaded so this optimisation
is pre-enabled.


