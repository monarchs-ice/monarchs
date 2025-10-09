"""
Main MONARCHS execution script. Invoked via the command line with

`monarchs -i <filepath>

where <filepath> is the path to a model setup script.
"""

from monarchs.core.driver import monarchs


def run_from_cli(return_grid=False):
    """
    Command line entry point for running the MONARCHS model.
    """
    grid = monarchs()
    if return_grid:
        return grid
    return None


if __name__ == "__main__":
    run_from_cli()
