"""
Access to datasets bundled with MONARCHS:

- ``era5_example.nc``: an ERA5-format dataset for the George VI ice shelf.
- ``checkpoints/`` + ``met_data/``: 1D test checkpoints for a run with
real ERA5 data. set ``regime="lake"`` or ``regime="lid"`` to get a
test checkpoint at either a lid or lake state.
"""

from importlib.resources import files


# Example data
def era5_example_path():
    """
    Absolute path to the bundled example ERA5 dataset (``era5_example.nc``).
    """
    return str(files(__package__).joinpath("met_data", "era5_example.nc"))


# Test data
def checkpoint_path(regime):
    """
    Absolute path to a bundled 1-D full-state checkpoint. ``regime`` is
    ``"lake"`` (open lake + thin virtual lid) or ``"lid"`` (established lid).
    Load with ``monarchs.io.checkpoint.read_checkpoint(path, get_spec(500, 20, 20))``.
    """
    return str(files(__package__).joinpath("checkpoints", f"1d_{regime}_checkpoint.nc"))


def met_slice_path(regime):
    """
    Absolute path to one model day (24 timesteps, shape (24, 1, 1)) of
    interpolated met forcing matching ``checkpoint_path(regime)``. Load with
    ``numpy.load``; index ``[t_step, 0, 0]`` for a single-timestep record.
    """
    return str(files(__package__).joinpath("met_data", f"1d_{regime}_met.npy"))
