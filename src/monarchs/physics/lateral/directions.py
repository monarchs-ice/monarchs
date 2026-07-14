"""
Schema for mapping water direction to an index.
"""

import numpy as np

#   0=NW  1=N  2=NE  3=E  4=SE  5=S  6=SW  7=W
_DIR_ROW = np.array([-1, -1, -1, 0, 1, 1, 1, 0], dtype=np.int32)
_DIR_COL = np.array([-1, 0, 1, 1, 1, 0, -1, -1], dtype=np.int32)
_N_DIRS = 8
