"""
The meteorological forcing sources MONARCHS can read.

To add a source, add a row here to SOURCES.
"""

import os
from dataclasses import dataclass
from typing import Callable

from monarchs.met_data.setup_met_data import met_data_from_era5, prescribed_met_data

MODULE_NAME = "monarchs.met_data.sources"


@dataclass(frozen=True, kw_only=True)
class MetSource:
    """A source MONARCHS can build its forcing netCDF from."""

    # the model_setup setting this source reads its input from
    input_setting: str
    # f(model_setup, lat_array, lon_array), writes the met netCDF
    build: Callable


SOURCES = {
    "ERA5": MetSource(input_setting="met_input_filepath", build=met_data_from_era5),
    "user_defined": MetSource(input_setting="met_data", build=prescribed_met_data),
}


def prepare_met_data(model_setup, lat_array, lon_array):
    """
    Build the met netCDF for this run from the configured source, or reuse the
    one already on disk if the user asked for that.
    """
    if model_setup.load_precalculated_met_data:
        if os.path.exists(model_setup.met_output_filepath):
            print(
                f"{MODULE_NAME}.prepare_met_data: Loading in pre-calculated"
                f" MONARCHS format met data from {model_setup.met_output_filepath}"
            )
            return
        print(
            f"{MODULE_NAME}.prepare_met_data: Pre-calculated met data file"
            f" {model_setup.met_output_filepath} does not exist. Calculating"
            f" from {model_setup.met_data_source} input instead."
        )
    SOURCES[model_setup.met_data_source].build(model_setup, lat_array, lon_array)
