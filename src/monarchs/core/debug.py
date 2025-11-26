def neighbourhood_check(grid, target_row, target_col):
    """
    A debugging function that prints out the status of the target cell and its
    immediate neighbors. Useful to debug e.g. lateral flow issues.
    Parameters
    ----------
    grid
    target_row
    target_col

    Returns
    -------

    """
    print("\n--- NEIGHBORHOOD CHECK ---")
    for r in range(target_row - 1, target_row + 2):
        for c in range(target_col - 1, target_col + 2):
            try:
                cell = grid[r, c]
                print(f"Cell [{r}, {c}] Status:")
                print(f"  Valid: {cell['valid_cell']}")
                print(f"  Lake depth: {float(cell['lake_depth'])}")
                print(f"  Lid depth: {cell['lid_depth']}")
                print(f"  Firn depth: {float(cell['firn_depth'])}")
                print(f"  Lake status: {cell['lake']}")
                print(f"  Lid status: {cell['lid']}")
                print(
                    f"  Saturation status at surface = ",
                    cell["saturation"][:5],
                )
                print(f"  Meltflag status at surface = ", cell["meltflag"][:5])
                print(f"  Exposed water flag = ", cell["exposed_water"])
                if not cell["valid_cell"]:
                    print(
                        "  WARNING: Cell is marked INVALID before physics starts."
                    )
                print("-------------------------")
            except IndexError:
                print(f"Cell [{r}, {c}] is OUT OF BOUNDS")
                print("-------------------------")
    print("--- END OF NEIGHBORHOOD CHECK ---\n")
