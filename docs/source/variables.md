# MONARCHS grid variables

List of MONARCHS model grid variables. Generated from the variable catalogue (`monarchs.variables`). Don't edit this manually, run `python scripts/gen_variable_docs.py` to regenerate!
Units broadly follow the CF conventions - see https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch03.html for details.

## Fixed values

| Variable | Long name | Dim | Units | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `column` | Grid column index | scalar | 1 | from input |  |
| `row` | Grid row index | scalar | 1 | from input |  |
| `vert_grid` | Firn layer count | scalar | 1 | computed (`n_firn`) |  |
| `vert_grid_lake` | Lake layer count | scalar | 1 | computed (`n_lake`) |  |
| `vert_grid_lid` | Lid layer count | scalar | 1 | computed (`n_lid`) |  |
| `lat` | Latitude | scalar | degrees_north | from input |  |
| `lon` | Longitude | scalar | degrees_east | from input |  |
| `size_dx` | Cell size, east-west | scalar | m | `1000.0` |  |
| `size_dy` | Cell size, north-south | scalar | m | `1000.0` |  |
| `valid_cell` | Cell runs model physics (flag) | scalar | 1 | `True` |  |

## Firn

| Variable | Long name | Dim | Units | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `firn_depth` | Firn column total depth | scalar | m | from input |  |
| `vertical_profile` | Depth of each firn layer | firn | m | computed (`vertical_profile`) |  |
| `firn_temperature` | Firn column temperature | firn | K | from input |  |
| `rho` | Firn density | firn | kg m-3 | from input |  |
| `Sfrac` | Solid (ice) volume fraction | firn | 1 | computed (`sfrac_from_rho`) |  |
| `Lfrac` | Liquid (water) volume fraction | firn | 1 | `0.0` |  |
| `meltflag` | Meltwater present at layer (flag) | firn | 1 | `0.0` |  |
| `saturation` | Layer saturated (flag) | firn | 1 | `0.0` |  |
| `pore_closure` | Pore close-off density (unused; see constants) | scalar | kg m-3 | `0.0` |  |
| `ice_lens` | Ice lens present (flag) | scalar | 1 | `False` |  |
| `ice_lens_depth` | Layer index of highest ice lens | scalar | 1 | computed (`ice_lens_below_column`) |  |

## Surface

| Variable | Long name | Dim | Units | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `albedo` | Surface albedo | scalar | 1 | `0.0` |  |
| `melt` | Surface melt this step (flag) | scalar | 1 | `False` |  |
| `exposed_water` | Exposed surface water (flag) | scalar | 1 | `False` |  |
| `total_melt` | Cumulative melt depth | scalar | m | `0.0` |  |
| `snow_added` | Snow depth added | scalar | m | `0.0` |  |

## Lake

| Variable | Long name | Dim | Units | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `lake` | Lake present (flag) | scalar | 1 | `False` |  |
| `lake_depth` | Melt lake depth | scalar | m | `0.0` |  |
| `lake_temperature` | Lake temperature profile | lake | K | `273.15` |  |

## Lid

| Variable | Long name | Dim | Units | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `lid` | Frozen lid present (flag) | scalar | 1 | `False` |  |
| `lid_depth` | Frozen lid depth | scalar | m | `0.0` |  |
| `lid_temperature` | Frozen lid temperature profile | lid | K | `273.15` |  |
| `rho_lid` | Frozen lid density | lid | kg m-3 | `0.0` |  |
| `v_lid` | Virtual lid present (flag) | scalar | 1 | `False` |  |
| `v_lid_depth` | Virtual lid depth | scalar | m | `0.0` |  |
| `virtual_lid_temperature` | Virtual lid temperature | scalar | K | `273.15` |  |
| `has_had_lid` | Lid present this cycle (flag) | scalar | 1 | `False` |  |
| `lid_sfc_melt` | Tracked lid surface melt | scalar | m | `0.0` |  |
| `lid_snow_depth` | Snow depth on the lid | scalar | m | `0.0` |  |
| `snow_on_lid` | Snow-on-lid state (0/1/2) | scalar | 1 | `0` |  |

## Lateral

| Variable | Long name | Dim | Units | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `water` | Liquid water depth per layer (lateral flow) | firn | m | `0.0` |  |
| `water_level` | Water-table height for lateral flow | scalar | m | `0.0` |  |
| `water_direction` | Lateral outflow direction (0=NW..7=W) | directions | 1 | `0` |  |

## Diagnostic

| Variable | Long name | Dim | Units | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `firn_boundary_change` | Firn boundary change this day | scalar | m | `0.0` |  |
| `lake_boundary_change` | Lake boundary change this day | scalar | m | `0.0` |  |
| `lid_boundary_change` | Lid boundary change this day | scalar | m | `0.0` |  |

## Counter

| Variable | Long name | Dim | Units | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `melt_hours` | Cumulative surface-melt hours | scalar | h | `0` |  |
| `lid_melt_count` | Lid melt-step counter | scalar | 1 | `0` |  |
| `lake_refreeze_counter` | Lake refreeze counter | scalar | 1 | `0` |  |
| `exposed_water_refreeze_counter` | Exposed-water refreeze counter | scalar | 1 | `0` |  |
| `t_step` | Timestep within the current day | scalar | 1 | `0` |  |
| `day` | Model day | scalar | 1 | `0` |  |
| `visit_count` | Times this cell has been visited | scalar | 1 | `0` |  |

## Internal

| Variable | Long name | Dim | Units | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `reset_combine` | Lid/firn just combined (flag) | scalar | 1 | `False` |  |
| `error_flag` | Cell hit an error state (flag) | scalar | 1 | `False` |  |
| `numba` | Running under Numba (flag) | scalar | 1 | `False` |  |
