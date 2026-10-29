# Moderate traffic, varied road shapes

This is the scene set requested after the first high-density preview proved
too crowded. It contains two SUMO scenes each for straight roads, curved
roads, multi-arm intersections, and converging approaches. Each scene has
6–15 total vehicles. At 2 s, 5–12 other vehicles are within 80 m of ego.

The route files and `.sumocfg` files are in the scene folders. `passed_index.json`
lists all eight scenes. `road_shapes_t2.png` previews one scene per road form;
`road_shapes_10s.gif` shows their motion. `smoke_report.json` has exact
SUMO counts, positions, and screening results.

| Scene | Form | Inserted / planned | Other vehicles within 80 m at 2 s | Collisions |
|---|---|---:|---:|---:|
| straight_01 | Straight | 10/10 | 9 | 0 |
| straight_02 | Straight | 6/6 | 5 | 0 |
| curve_01 | Pronounced curve | 8/8 | 6 | 0 |
| curve_02 | Gentle curve | 10/10 | 9 | 0 |
| intersection_01 | Multi-arm intersection | 13/13 | 10 | 0 |
| intersection_02 | Multi-arm intersection | 15/15 | 12 | 0 |
| merge_01 | Converging approaches | 10/10 | 9 | 0 |
| merge_02 | Converging approaches | 10/10 | 9 | 0 |

The check uses plain SUMO for 10 s at 0.1 s time steps, with collision
checking at junctions. This set is a traffic preview without attacks; it
has not passed a UniAD/DreamForge closed-loop handover test. The ego's
initial same-lane exclusion is 22 m ahead and 18 m behind (center-to-center).
