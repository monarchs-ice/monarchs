"""
Extract a single grid column from a progress.nc dump file and run it as a
standalone 1D MONARCHS test case.

This is primarily a debugging tool: if a full-domain run produces a NaN or
unexpected result at cell (row, col), this script lets you reproduce the
problem in isolation — with Numba and parallelism off by default — so you can
add print statements, step through the physics, and inspect every variable.

What it does
------------
1. Reads the vertical grid dimensions from the progress file.
2. Loads the full grid and extracts cell [row, col] into a 1×1 structured array,
   preserving every state variable exactly as it was at the dump point.
3. Slices the full-domain met_data.nc to a 1×1 (time, 1, 1) met file,
   starting from the met index stored in the dump so the forcing is identical.
4. Builds a minimal model_setup namespace (lateral movement OFF, Numba OFF
   by default) and calls monarchs.core.driver.main() directly.

Usage
-----
python scripts/extract_column_1d.py \\
    --progress /path/to/progress.nc \\
    --met      /path/to/met_data.nc \\
    --row 42   --col 17             \\
    [--days 10]                     \\
    [--outdir  output/debug_r42_c17]\\
    [--from-start]                  \\
    [--use-numba]

Arguments
---------
--progress      Path to the progress/dump .nc file from the full run.
--met           Path to the MONARCHS-format met_data.nc from the full run.
--row           Row index of the cell to extract (0-based).
--col           Column index of the cell to extract (0-based).
--days          How many model days to run the 1D case (default: 10).
--outdir        Output directory; defaults to output/debug_r<row>_c<col>/.
--from-start    Restart met forcing from t=0 instead of continuing from the
                met index stored in the dump file. Useful if you want to run
                the column from the very beginning of the met record.
--use-numba     Enable Numba JIT compilation (disabled by default so that
                debugging with plain Python is easier).
--t-steps-per-day
                Hourly timesteps per model day (default: 24).
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

# ── allow the script to be run from the repo root without installing ─────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from monarchs.core.model_grid import get_spec
from monarchs.core.dump_model_state import reload_from_dump
from monarchs.core import configuration
from monarchs.core.driver import main as monarchs_main

# ── met variable names as stored in a MONARCHS-format met file ───────────────
_MET_VARS = [
    "snowfall",
    "snow_dens",
    "temperature",
    "wind",
    "pressure",
    "dew_point_temperature",
    "LW_surf",
    "SW_surf",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--progress", required=True,
                   help="Path to the progress/dump .nc file from the full run.")
    p.add_argument("--met", required=True,
                   help="Path to the MONARCHS-format met_data.nc from the full run.")
    p.add_argument("--row", type=int, required=True,
                   help="Row index of the cell to extract (0-based).")
    p.add_argument("--col", type=int, required=True,
                   help="Column index of the cell to extract (0-based).")
    p.add_argument("--days", type=int, default=10,
                   help="Number of days to run the 1D test case (default: 10).")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: output/debug_r<row>_c<col>/).")
    p.add_argument("--from-start", action="store_true",
                   help="Restart met forcing from t=0 rather than continuing "
                        "from the met index stored in the dump file.")
    p.add_argument("--use-numba", action="store_true",
                   help="Enable Numba JIT (disabled by default for easier "
                        "debugging).")
    p.add_argument("--t-steps-per-day", type=int, default=24,
                   help="Hourly timesteps per model day (default: 24).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_grid_dimensions(progress_path: str) -> tuple[int, int, int]:
    """Return (vert_grid, vert_grid_lake, vert_grid_lid) from the progress file."""
    with Dataset(progress_path) as ds:
        vert_grid      = len(ds.dimensions["vert_grid"])
        vert_grid_lake = len(ds.dimensions["vert_grid_lake"])
        vert_grid_lid  = len(ds.dimensions["vert_grid_lid"])
    return vert_grid, vert_grid_lake, vert_grid_lid


def read_met_start_idx(progress_path: str) -> int:
    """Return the met_start_idx stored in the progress file."""
    with Dataset(progress_path) as ds:
        return int(ds.variables["met_start_idx"][:])


def extract_cell_to_1x1_grid(
    progress_path: str,
    row: int,
    col: int,
    vert_grid: int,
    vert_grid_lake: int,
    vert_grid_lid: int,
) -> np.ndarray:
    """
    Load the full-domain grid from the progress file, pull out cell [row, col],
    and return it as a 1×1 structured array with the same dtype.
    """
    dtype = get_spec(vert_grid, vert_grid_lake, vert_grid_lid)
    full_grid, _, _, _ = reload_from_dump(progress_path, dtype)

    grid_1x1 = np.zeros((1, 1), dtype=dtype)
    for field in dtype.names:
        grid_1x1[field][0, 0] = full_grid[field][row, col]

    # Fix row/col indices so the model knows it is at position (0, 0)
    grid_1x1["row"][0, 0]    = 0
    grid_1x1["column"][0, 0] = 0

    return grid_1x1


def write_1x1_met_file(
    src_met_path: str,
    dst_met_path: str,
    row: int,
    col: int,
    met_start_idx: int,
    n_steps: int,
) -> None:
    """
    Slice the full-domain met file to a single (row, col) cell and write a
    1×1 MONARCHS-format met file.

    The slice begins at ``met_start_idx`` so the 1D run receives the same
    forcing as the full run at the point the dump was taken.  If fewer than
    ``n_steps`` timesteps remain in the file, the met data wraps back to the
    beginning of the file to fill the remainder.
    """
    Path(dst_met_path).parent.mkdir(parents=True, exist_ok=True)

    with Dataset(src_met_path) as src:
        total_steps = src.variables["temperature"].shape[0]

        # Build a list of (start, end) slices, wrapping if needed
        slices: list[tuple[int, int]] = []
        steps_needed = n_steps
        offset = met_start_idx % total_steps
        while steps_needed > 0:
            end = min(offset + steps_needed, total_steps)
            slices.append((offset, end))
            steps_needed -= end - offset
            offset = 0  # wrap to beginning

        lat_val = float(src.variables["cell_latitude"][row, col])
        lon_val = float(src.variables["cell_longitude"][row, col])

        with Dataset(dst_met_path, "w") as dst:
            dst.createDimension("time",   None)
            dst.createDimension("row",    1)
            dst.createDimension("column", 1)

            v = dst.createVariable("cell_latitude",  "f8", ("row", "column"))
            v[:] = [[lat_val]]
            v = dst.createVariable("cell_longitude", "f8", ("row", "column"))
            v[:] = [[lon_val]]

            for vname in _MET_VARS:
                chunks = [
                    np.asarray(src.variables[vname][s:e, row, col],
                               dtype=np.float64)[:, np.newaxis, np.newaxis]
                    for s, e in slices
                ]
                data = np.concatenate(chunks, axis=0)  # (n_steps, 1, 1)
                v = dst.createVariable(vname, "f8", ("time", "row", "column"))
                v[:] = data

    actual_steps = sum(e - s for s, e in slices)
    wrapped = actual_steps > (total_steps - met_start_idx)
    print(f"  Wrote 1×1 met file  : {dst_met_path}")
    print(f"  Steps written       : {actual_steps}"
          + ("  (wrapped)" if wrapped else ""))
    print(f"  Forcing starts at   : original met index {met_start_idx}")


def build_model_setup(
    outdir: str,
    n_days: int,
    met_path: str,
    vert_grid: int,
    vert_grid_lake: int,
    vert_grid_lid: int,
    use_numba: bool,
    t_steps_per_day: int,
) -> types.SimpleNamespace:
    """
    Build a minimal model_setup-like namespace that monarchs_main() can
    consume.  Lateral movement is disabled (single column) and Numba is off
    by default to make debugging easier.
    """
    ms = types.SimpleNamespace()

    # Grid
    ms.row_amount              = 1
    ms.col_amount              = 1
    ms.lat_grid_size           = 1000

    # Vertical resolution — must match the dump exactly
    ms.vertical_points_firn    = vert_grid
    ms.vertical_points_lake    = vert_grid_lake
    ms.vertical_points_lid     = vert_grid_lid

    # Timestepping
    ms.num_days                = n_days
    ms.t_steps_per_day         = t_steps_per_day
    ms.lateral_timestep        = t_steps_per_day * 3600

    # Met data — point at the 1×1 file we just created.
    # met_input_filepath must exist so that create_defaults_for_missing_flags
    # recognises this as an ERA5-style run without raising a ValueError.
    # It is never actually opened because load_precalculated_met_data=True.
    ms.met_input_filepath          = met_path
    ms.met_output_filepath         = met_path
    ms.met_data_source             = "ERA5"
    ms.load_precalculated_met_data = True

    # Physics toggles
    ms.firn_column_toggle              = True
    ms.firn_heat_toggle                = True
    ms.lake_development_toggle         = True
    ms.lid_development_toggle          = True
    ms.lateral_movement_toggle         = False   # single column — no neighbours
    ms.lateral_movement_percolation_toggle = False
    ms.snowfall_toggle                 = True
    ms.percolation_toggle              = True
    ms.perc_time_toggle                = True
    ms.densification_toggle            = False
    ms.spinup                          = False

    # Parallelism / solver — all off for debugging
    ms.parallel                        = False
    ms.use_numba                       = use_numba
    ms.use_mpi                         = False
    ms.cores                           = 1
    ms.ignore_errors                   = False
    ms.single_column_toggle            = True

    # Output
    outdir_path = Path(outdir)
    ms.output_filepath         = str(outdir_path / "output_1d.nc")
    ms.dump_filepath           = str(outdir_path / "dump_1d.nc")
    ms.dump_data               = True
    ms.dump_format             = "NETCDF4"
    ms.dump_timestep           = 1
    ms.reload_from_dump        = False
    ms.save_output             = True
    ms.output_timestep         = 1
    ms.output_grid_size        = vert_grid
    ms.vars_to_save            = (
        "firn_temperature", "rho", "Sfrac", "Lfrac",
        "firn_depth", "lake_depth", "lid_depth",
        "lake", "lid", "v_lid", "ice_lens_depth", "water_level",
    )

    # Misc
    ms.verbose_logging             = True
    ms.rho_init                    = "default"
    ms.T_init                      = "default"
    ms.rho_sfc                     = 500
    ms.solver                      = "hybr"
    ms.flow_speed_scaling          = 1.0
    ms.outflow_proportion          = 1.0
    ms.catchment_outflow           = False
    ms.dump_checkpoint_frequency   = False
    ms.dump_data_pre_lateral_movement = False
    ms.met_timestep                = "hourly"
    ms.dask_scheduler              = "processes"
    ms.flow_into_land              = False
    ms.simulated_water_toggle      = False

    return ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    row, col = args.row, args.col
    outdir   = args.outdir or f"output/debug_r{row:03d}_c{col:03d}"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print("  MONARCHS 1D column extractor / diagnostic runner")
    print(f"{'='*62}")
    print(f"  Progress file : {args.progress}")
    print(f"  Met file      : {args.met}")
    print(f"  Cell          : row={row}, col={col}")
    print(f"  Days to run   : {args.days}")
    print(f"  Numba         : {args.use_numba}")
    print(f"  Output dir    : {outdir}")
    print(f"{'='*62}\n")

    # ── 1. Read grid dimensions ──────────────────────────────────────────────
    print("[1/4] Reading grid dimensions from progress file...")
    vert_grid, vert_grid_lake, vert_grid_lid = read_grid_dimensions(args.progress)
    print(f"  vert_grid={vert_grid}  vert_grid_lake={vert_grid_lake}"
          f"  vert_grid_lid={vert_grid_lid}")

    met_start_in_dump = read_met_start_idx(args.progress)
    met_start_idx     = 0 if args.from_start else met_start_in_dump
    print(f"  Met index in dump : {met_start_in_dump}")
    print(f"  Met index to use  : {met_start_idx}"
          f"  ({'from t=0' if args.from_start else 'continuing from dump'})")

    # ── 2. Extract cell state ────────────────────────────────────────────────
    print(f"\n[2/4] Extracting cell [{row}, {col}] from progress file...")
    grid_1x1 = extract_cell_to_1x1_grid(
        args.progress, row, col,
        vert_grid, vert_grid_lake, vert_grid_lid,
    )
    print(f"  firn_depth  = {grid_1x1['firn_depth'][0, 0]:.4f} m")
    print(f"  lake_depth  = {grid_1x1['lake_depth'][0, 0]:.4f} m"
          f"  (lake={bool(grid_1x1['lake'][0, 0])})")
    print(f"  lid_depth   = {grid_1x1['lid_depth'][0, 0]:.4f} m"
          f"  (lid={bool(grid_1x1['lid'][0, 0])})")
    print(f"  valid_cell  = {bool(grid_1x1['valid_cell'][0, 0])}")
    print(f"  rho[0]      = {grid_1x1['rho'][0, 0, 0]:.2f} kg m⁻³")

    if not grid_1x1["valid_cell"][0, 0]:
        print("\n  WARNING: this cell has valid_cell=False in the dump. "
              "The model will skip it. Re-run with the correct row/col, or "
              "manually set valid_cell=True if you want to force it through.")

    # ── 3. Write 1×1 met file ────────────────────────────────────────────────
    met_1x1_path = str(Path(outdir) / "met_1x1.nc")
    n_met_steps  = args.days * args.t_steps_per_day
    print(f"\n[3/4] Writing 1×1 met file ({n_met_steps} steps)...")
    write_1x1_met_file(
        src_met_path  = args.met,
        dst_met_path  = met_1x1_path,
        row           = row,
        col           = col,
        met_start_idx = met_start_idx,
        n_steps       = n_met_steps,
    )

    # ── 4. Configure and run ─────────────────────────────────────────────────
    print(f"\n[4/4] Running 1D test case for {args.days} day(s)...")
    model_setup = build_model_setup(
        outdir          = outdir,
        n_days          = args.days,
        met_path        = met_1x1_path,
        vert_grid       = vert_grid,
        vert_grid_lake  = vert_grid_lake,
        vert_grid_lid   = vert_grid_lid,
        use_numba       = args.use_numba,
        t_steps_per_day = args.t_steps_per_day,
    )

    # Apply MONARCHS defaults for any attributes not set above
    configuration.create_defaults_for_missing_flags(model_setup)
    configuration.handle_incompatible_flags(model_setup)
    configuration.handle_invalid_values(model_setup)

    if model_setup.use_numba:
        configuration.jit_modules()

    # Call main() directly — this bypasses initialise_model_data() since
    # we have already built the grid from the dump state.
    grid_out = monarchs_main(model_setup, grid_1x1)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  1D diagnostic run complete")
    print(f"{'='*62}")
    print(f"  firn_depth  (final) = {grid_out['firn_depth'][0, 0]:.4f} m")
    print(f"  lake_depth  (final) = {grid_out['lake_depth'][0, 0]:.4f} m")
    print(f"  lid_depth   (final) = {grid_out['lid_depth'][0, 0]:.4f} m")
    print(f"  Output written to   : {outdir}/")
    print(f"    output_1d.nc  — time-series of saved variables")
    print(f"    dump_1d.nc    — final model state")
    print(f"    met_1x1.nc    — forcing used for this run")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()

