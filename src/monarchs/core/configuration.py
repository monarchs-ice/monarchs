""" """

# TODO - module-level docstring, other docstrings
import argparse
import os


def parse_args():
    """
    Parse input. Most things are controlled by `model_setup.py`; the only input
    here is (optionally) the location (as a filepath, so including the
    filename) of that setup file.

    The ``MONARCHS_INPUT_PATH`` environment variable, if set, takes precedence
    over the command line - handy for tests and batch runs that need to point at
    a specific runscript without passing argv.
    """
    env_path = os.environ.get("MONARCHS_INPUT_PATH")
    if env_path:
        return env_path
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
        # optional paths (e.g. output/dump filepaths when not saving) are None -
        # skip them rather than choking on os.path.dirname(None)
        if not filepath:
            continue
        # os.path.dirname is "" by default so writes to cwd
        folder = os.path.dirname(filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)
