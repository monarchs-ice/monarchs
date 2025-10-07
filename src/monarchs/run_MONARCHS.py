from monarchs.core.driver import monarchs


def run_from_cli(return_grid=False):
    """
    Command line entry point for running the MONARCHS model.
    """
    grid = monarchs()
    if return_grid:
        return grid


if __name__ == "__main__":
    run_from_cli()
