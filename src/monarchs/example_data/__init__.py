"""
Access to the small example/test datasets bundled with MONARCHS (an example
ERA5 file, and 1D checkpoints + met slices used by the tests). Named
``example_data`` to make clear these are sample inputs, distinct from the
model's grid ``variables``.
"""

from monarchs.example_data.paths import era5_example_path

__all__ = ["era5_example_path"]
