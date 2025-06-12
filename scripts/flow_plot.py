import matplotlib.pyplot as plt
import numpy as np
from monarchs.core.utils import get_2d_grid
from matplotlib.patches import FancyArrow
from matplotlib.animation import ArtistAnimation

def get_arrow_directions(width, height, data):
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

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    dx = np.zeros_like(xx)
    dy = np.zeros_like(yy)

    x_arrows, y_arrows, u_arrows, v_arrows = [], [], [], []
    for i in range(height):
        for j in range(width):
            flows = data[i, j]  # A 1D array of 4 directions (N, S, E, W)
            for k, val in enumerate(flows):
                if val == 1:
                    # If direction k is 1, add that arrow
                    direction = directions[k]
                    x_arrows.append(j)  # x position of the grid point
                    y_arrows.append(i)  # y position of the grid point
                    u_arrows.append(direction[0])  # u = x-component of direction
                    v_arrows.append(direction[1])  # v = y-component of direction

    # Convert lists to numpy arrays
    x_arrows = np.array(x_arrows)
    y_arrows = np.array(y_arrows)
    u_arrows = np.array(u_arrows)
    v_arrows = np.array(v_arrows)
    return x_arrows, y_arrows, u_arrows, v_arrows

def flow_plot(grid, netcdf=False, index=0, fig=False, ax=False):

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

    x_arrows, y_arrows, u_arrows, v_arrows = get_arrow_directions(width, height, data)

    ax.grid(True, which='both', color='black', linestyle='-', linewidth=0.5, zorder=1)
    quiv = ax.quiver(x_arrows, y_arrows, u_arrows, -v_arrows, color='blue', scale=25, width=0.003, headwidth=3,
              zorder=2)
    plt.xlim(-1, width)
    plt.ylim(-1, height)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Flow direction")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks(range(width))
    plt.yticks(range(height))
    plt.gca().invert_yaxis()
    plt.show()
    return quiv

def make_fd_plot(a, idx=0, fig=False, ax=False):
    if not fig and not ax:
        fig, ax = plt.subplots()
    im = ax.imshow(a.variables['water_level'][idx], vmax=65, animated=True)
    return fig, ax, im


def make_both(a, idx=0):
    fig, ax, im = make_fd_plot(a, idx=idx)
    quiv = flow_plot(a, netcdf=True, index=idx, fig=fig, ax=ax)
    return fig, ax, im, quiv

def flow_anim(data):

    fig, ax, im, quiv = make_both(data, idx=0)

    def make_new_frame(frame, fig, ax):
        fig, ax, im = make_fd_plot(data, idx=frame, fig=fig, ax=ax)
        quiv = flow_plot(data, netcdf=True, index=frame, fig=fig, ax=ax)
        return [im, quiv]

    frames = []

    for i in range(len(data.variables['water_level'])):
        frame = make_new_frame(i, fig, ax)
        frames.append(frame)
    ani = ArtistAnimation(fig, frames, interval=200, blit=False)
    ani.save("flow_animation.gif", writer="imagemagick")
    plt.show()