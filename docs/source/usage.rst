Tips and tricks
===============

What to do if the model fails
-----------------------------
There may be some situations in which the model fails. Sometimes this
may be a bug with the model itself, but the model can also stop running due to e.g. a computer crash.
Another example may be that you are running on a HPC system with a job scheduler, but have not scheduled enough time
or compute cores for the model to finish.

It is recommended because of this to run with the ``dump_data`` flag in ``model_setup`` set to ``True``. This will
dump the entire model state into a file (called ``dump.nc`` by default, but this is controllable via the ``dump_filepath``
``model_setup`` variable).

With this file, it is then possible to load this into Python to look at the model state (e.g. for debugging,
see ``scripts/debug_lateral_flow`` for an example), or to use this to restart your MONARCHS run
(by setting ``reload_from_dump`` to ``True`` in ``model_setup``.

Data analysis
-------------
A few tools exist to aid in looking at MONARCHS output.

``get_2d_grid`` is useful particularly for looking at the progress (i.e. dump) files.