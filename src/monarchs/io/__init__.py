"""
MONARCHS I/O package.

Reads and writes model data to/from disk. Currently covers the two netCDF
writers that share a common grid<->netCDF engine:

Public API
----------
initialise_output   : Create the time-series output netCDF file.
append_output       : Append the current grid state as a new time slice.
write_checkpoint    : Write the full model state for crash/restart.
read_checkpoint     : Load a full model state back from a checkpoint file.
"""

from monarchs.io.output import initialise_output, append_output
from monarchs.io.checkpoint import write_checkpoint, read_checkpoint

__all__ = [
    "initialise_output",
    "append_output",
    "write_checkpoint",
    "read_checkpoint",
]
