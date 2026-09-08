""" """

# TODO - module-level docstring, other docstrings
import argparse
import os


def parse_args():
    """
    Parse input. Most things are controlled by `model_setup.py`; the only input
    here is (optionally) the location (as a filepath, so including the
    filename) of that setup file.
    """
    parser = argparse.ArgumentParser(
        prog="MONARCHS",
        description=(
            "A model of ice shelf development, written by Sammie Buzzard, Jon"
            " Elsey and Alex Robel."
        ),
    )
    parser.add_argument(
        "--input_path",
        "-i",
        help=(
            "Absolute or relative path to an input file, in the format of"
            " <model_setup.py>"
        ),
        default="model_setup.py",
        required=False,
    )
    args, _ = parser.parse_known_args()
    model_setup_path = args.input_path
    return model_setup_path


def create_output_folders(model_setup):
    """
    Create the output folders for the model output, meteorological data and
    dump files, if they do not already exist.
    """
    for filepath in (
        model_setup.output_filepath,
        model_setup.dump_filepath,
        model_setup.met_output_filepath,
    ):
        # skip if undefined to avoid later errors
        if not filepath:
            continue
        # os.path.dirname is "" by default so writes to cwd
        folder = os.path.dirname(filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)
