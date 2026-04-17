import numpy as np


def build_coarse_index_map(
    coarse_lats: np.ndarray,  # shape (n_clat,)
    coarse_lons: np.ndarray,  # shape (n_clon,)
    fine_lats_2d: np.ndarray,  # shape (num_rows, num_cols)  ← full 2-D from DEM
    fine_lons_2d: np.ndarray,  # shape (num_rows, num_cols)  ← full 2-D from DEM
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand a coarse ERA5 array to the model grid using a pre-built index map.

    Every cell (i, j) in the model grid has its own (lat, lon), so the index
    maps must also be 2-D.  The broadcasting trick used in the 1-D version is
    replaced by direct 2-D advanced indexing in apply_index_map_2d.

    Parameters
    ----------
    coarse_lats, coarse_lons : 1-D arrays
        ERA5 coordinate axes (regular lat/lon grid).
    fine_lats_2d, fine_lons_2d : 2-D arrays, shape (num_rows, num_cols)
        Lat/lon of every model cell, as reprojected from the polar
        stereographic DEM (e.g. from your get_met_bounds_from_DEM output).

    Returns
    -------
    lat_idx : ndarray of int32, shape (num_rows, num_cols)
    lon_idx : ndarray of int32, shape (num_rows, num_cols)
    """
    lat_idx = np.argmin(
        np.abs(coarse_lats[:, np.newaxis, np.newaxis] - fine_lats_2d[np.newaxis]),
        axis=0,
    ).astype(np.int32)

    lon_idx = np.argmin(
        np.abs(coarse_lons[:, np.newaxis, np.newaxis] - fine_lons_2d[np.newaxis]),
        axis=0,
    ).astype(np.int32)

    return lat_idx, lon_idx


def apply_index_map_1d(
    coarse_array: np.ndarray,  # shape (time, n_clat, n_clon)
    lat_idx: np.ndarray,  # shape (num_cols,) - index per fine lat
    lon_idx: np.ndarray,  # shape (num_rows,) - index per fine lon
) -> np.ndarray:
    """
    Expand a coarse ERA5 array to a separable (regular) fine grid using
    1-D index maps and broadcasting.

    For regular lat/lon grids where fine lat varies only along one axis
    and fine lon along the other. Returns shape (time, num_cols, num_rows)
    to match the (time, lat, lon) convention with lat_idx indexing the
    lat dimension and lon_idx the lon dimension.

    Returns
    -------
    ndarray, shape (time, num_cols, num_rows)
        With lat_idx shape (num_cols,), lon_idx shape (num_rows,),
        broadcasting gives coarse_array[:, lat_idx[:, None], lon_idx[None, :]].
    """
    return coarse_array[:, lat_idx[:, np.newaxis], lon_idx[np.newaxis, :]]


def apply_index_map(
    coarse_array: np.ndarray,  # shape (time, n_clat, n_clon)
    lat_idx: np.ndarray,  # shape (num_rows, num_cols)
    lon_idx: np.ndarray,  # shape (num_rows, num_cols)
) -> np.ndarray:
    """
    Expand a coarse ERA5 array to a non-separable model grid.

    Uses direct 2-D advanced indexing rather than the 1-D broadcasting
    trick, so it works for any irregular (row, col) → (lat, lon) mapping.

    Returns
    -------
    ndarray, shape (time, num_rows, num_cols)
    """
    # coarse_array[:, lat_idx, lon_idx] does the right thing:
    # for each (i,j), it picks coarse_array[:, lat_idx[i,j], lon_idx[i,j]]
    return coarse_array[:, lat_idx, lon_idx]


def apply_index_map_expand(
    coarse_array: np.ndarray,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
) -> np.ndarray:
    """
    Expand a coarse array to the fine grid using index maps.
    Dispatches to 1-D (broadcasting) or 2-D (direct indexing) based on
    lat_idx.ndim. Use this when reading from netCDF where index map
    dimensionality may be 1-D (separable) or 2-D (e.g. DEM).

    Returns
    -------
    ndarray, shape (time, num_rows, num_cols)
        For 2-D index maps this is direct. For 1-D, output is
        (time, len(lat_idx), len(lon_idx)); caller should ensure
        (lat_idx, lon_idx) align with (row, col) if needed.
    """
    if lat_idx.ndim == 1:
        return apply_index_map_1d(coarse_array, lat_idx, lon_idx)
    return apply_index_map(coarse_array, lat_idx, lon_idx)
