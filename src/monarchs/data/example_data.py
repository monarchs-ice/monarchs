"""
Access to datasets bundled with MONARCHS. At the moment, this just consists
of one ERA5-format dataset for the George VI ice shelf.
"""

from importlib.resources import files


def era5_example_path():
    """
    Absolute path to the bundled example ERA5 dataset (``era5_example.nc``).
    """
    return str(files("monarchs").joinpath("", "era5_example.nc"))
