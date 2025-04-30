import matplotlib.pyplot as plt
import numpy as np
from monarchs.core.utils import get_2d_grid

def flow_plot(grid, netcdf=False, index=0, fig=False, ax=False):
    # Direction vectors (clockwise from NW at index 0)

    # NW, N, NE, E, SE, S, SW, W
    directions = np.array([
        (-1, -1),  # NW
        (0, -1),  # N
        (1, -1),  # NE
        (1, 0),  # E
        (1, 1),  # SE
        (0, 1),  # S
        (-1, 1),  # SW
        (-1, 0)  # W
    ])  # mirrored from MONARCHS as dx, dy is the way they are accessed later, whereas MONARCHS works on [row, col]
    if not netcdf:
        data = get_2d_grid(grid, 'water_direction', index='all')
        height = len(grid)
        width = len(grid[0])
    else:
        data = grid.variables['water_direction'][index]
        height = len(data)
        width = len(data[0])
    # Prepare plot
    if not fig:
        fig, ax = plt.subplots(figsize=(6, 6))

    for y in range(height):
        for x in range(width):
            flows = data[y, x]
            for i, val in enumerate(flows):
                if val == 1:
                    dx, dy = directions[i]
                    ax.arrow(x, height - y - 1, dx * 0.3, -dy * 0.3, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

    # Grid formatting
    plt.xlim(-1, width)
    plt.ylim(-1, height)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.title("Flow direction")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks(range(width))
    plt.yticks(range(height))
    plt.gca().invert_yaxis()
    plt.show()