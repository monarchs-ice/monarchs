# test_turbulent_mixing_no_heating.py
import numpy as np
import pytest
from monarchs.physics.lake import turbulent_mixing
from monarchs.core.model_grid import get_spec, initialise_iceshelf

@pytest.fixture
def make_cell():
    """
    Factory to build a single structured cell with a mixed lake.
    - Surface boundary: lake_temperature[0] (you can set it per test)
    - Bottom boundary: implemented in turbulent_mixing as 273.15 K
    """
    def _make(
        vert_grid=10,
        vert_grid_lake=8,
        vert_grid_lid=1,
        lake_depth_m=1.0,
        core_temp=280.0,
        surface_temp=273.15,
        firn_depth=5.0,
        firn_temp=263.0,
    ):
        dtype = get_spec(vert_grid, vert_grid_lake, vert_grid_lid)

        rho = np.full(vert_grid, 917.0, dtype=float)
        firn_T = np.full(vert_grid, float(firn_temp), dtype=float)

        lake_T = np.full(vert_grid_lake, float(core_temp), dtype=float)
        lake_T[0] = float(surface_temp)   # upper boundary in your scheme
        lake_T[-1] = 273.15               # explicit lower boundary

        grid = initialise_iceshelf(
            num_rows=1, num_cols=1,
            vert_grid=vert_grid,
            vert_grid_lake=vert_grid_lake,
            vert_grid_lid=vert_grid_lid,
            dtype=dtype,
            x=0, y=0,
            firn_depth=firn_depth,
            rho=rho,
            firn_temperature=firn_T,
            lake_depth=lake_depth_m,
            lake_temperature=lake_T,
            # Flags: open lake, no lid (albedo path is irrelevant as sw_in=0)
            melt=True, exposed_water=True, lake=True, lid=False, v_lid=False,
        )
        return grid[0, 0]
    return _make


def core_index(cell):
    return int(cell["vert_grid_lake"] / 2)


def run_mixing(cell, hours, sw_in=0.0):
    """
    Run mixing for `hours` hours with time step `dt` seconds and given `sw_in`.
    Prints inside turbulent_mixing are ignored by pytest unless -s is used.
    """
    for _ in range(hours):
        turbulent_mixing(cell, sw_in=sw_in, dt=3600)


@pytest.mark.parametrize(
    "initial_core, surface_boundary",
    [
        (290.0, 273.15),  # from above freezing to 0°C boundaries
    ],
)
def test_relax_toward_freezing_no_heating(make_cell, initial_core, surface_boundary):
    """
    With SW=0 (no internal heating) and both boundaries at (or effectively tending to) 0°C,
    the lake interior should monotonically approach 273.15 K.
    """
    cell = make_cell(core_temp=initial_core, surface_temp=surface_boundary)
    idx = core_index(cell)

    diffs = []
    # Track the deviation |T_core - 273.15| each hour over 10 hours
    for _ in range(10):
        run_mixing(cell, hours=10, sw_in=0.0)
        diffs.append(abs(float(cell["lake_temperature"][idx]) - 273.15))

    # Deviation should not increase over time (allow tiny numerical tolerance)
    for a, b in zip(diffs, diffs[1:]):
        assert b <= a + 1e-9

    # And we should be close to freezing after 10 hours
    assert diffs[-1] < 0.05  # within 0.05 K of 0°C


def test_no_change_if_already_at_boundary(make_cell):
    """
    If the interior starts at 0°C with 0°C boundaries and no solar,
    it should remain essentially unchanged.
    """
    cell = make_cell(core_temp=273.15, surface_temp=273.15)
    idx = core_index(cell)

    before = float(cell["lake_temperature"][idx])
    run_mixing(cell, hours=6, sw_in=0.0)  # 2 hours, no SW
    after = float(cell["lake_temperature"][idx])

    assert abs(after - before) < 1e-6


@pytest.mark.parametrize(
    "surface_boundary, core_start, expect_direction",
    [
        (276.0, 274.0, "up"),   # surface warmer than core → core should warm
        (270.0, 272.0, "down"), # surface colder than core → core should cool
    ],
)
def test_upper_boundary_influence_no_heating(make_cell, surface_boundary, core_start, expect_direction):
    """
    Optional: isolate the effect of the upper boundary when SW=0.
    The bottom boundary at 273.15 provides a restoring tendency; the net should
    still move the core toward the combined boundary influence.
    """
    cell = make_cell(core_temp=core_start, surface_temp=surface_boundary)
    idx = core_index(cell)

    before = float(cell["lake_temperature"][idx])
    run_mixing(cell, hours=6, sw_in=0.0)
    after = float(cell["lake_temperature"][idx])

    if expect_direction == "up":
        assert after >= before - 1e-9
    else:
        assert after <= before + 1e-9