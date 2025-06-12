from monarchs.core.dump_model_state import reload_from_dump
from monarchs.core.utils import get_2d_grid
from matplotlib import pyplot as plt
from monarchs.physics.lateral_functions import move_water
from monarchs.core.model_grid import get_spec
path = r"C:\Users\jdels\Documents\Work\MONARCHS_runs\archer2_lateral/progress.nc"
import numpy.testing as npt


# Set up a dummy IceShelf instance, create a grid of these, then write out our dumpfile into this.
class IceShelf:
    pass


row_amount = 100
col_amount = 100

file_fmt = "netcdf"

dtype = get_spec(500, 20, 20)
grid, _, _, _ = reload_from_dump(path, dtype)

move_water(
    grid,
    row_amount,
    col_amount,
    3600 * 24,
    catchment_outflow=True,
    flow_into_land=True,
    lateral_movement_percolation_toggle=True,
)

