from monarchs.core.driver import monarchs


def cli_entry(return_grid=False):
    """
    Command line entry point for running the MONARCHS model.
    """
    grid = monarchs()
    if return_grid:
        return grid

if __name__ == "__main__":
    cli_entry()
