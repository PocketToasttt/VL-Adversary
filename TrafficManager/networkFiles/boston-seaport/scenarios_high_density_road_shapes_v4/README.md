# High density road shape preview

Generated from the Boston Seaport SUMO network with
`TrafficManager/experiments/build_high_density_road_previews.py`.
These are **traffic previews without attacks**. They are not yet converted to
the three-attack closed-loop scenario schema, and do not validate UniAD or
DreamForge handover.

Each scene uses a straight road, a curved road, a multi-arm intersection, or
converging approaches. Traffic is placed on the ego road and connected nearby
roads. Vehicles depart at 0 s with 13.5 m target spacing. The ego's initial
same-lane exclusion is 22 m ahead and 18 m behind (center-to-center). Each
scene has a 10 s SUMO run at 0.1 s steps, FCD at 0.5 s, and junction collision
checks. Passing means every planned vehicle inserted by 10 s, ego present at
2 s, at least 10 other vehicles within 80 m at 2 s, and zero SUMO collisions.

| Scene | Road form | Inserted / planned | Other cars within 60 m at 2 s | Within 80 m | Collision | Result |
|---|---|---:|---:|---:|---:|---|
| straight_01 | Straight | 22/22 | 19 | 21 | 0 | PASS |
| straight_02 | Straight | 15/15 | 14 | 14 | 0 | PASS |
| curve_01 | Curved | 23/23 | 13 | 19 | 0 | PASS |
| curve_02 | Mild bend | 34/35 | 16 | 24 | 0 | FAIL: incomplete insertion |
| intersection_01 | Multi-arm intersection | 51/51 | 23 | 32 | 0 | PASS |
| intersection_02 | Multi-arm intersection | 50/50 | 23 | 34 | 0 | PASS |
| merge_01 | Converging approaches | 43/43 | 25 | 32 | 0 | PASS |
| merge_02 | Converging approaches | 43/43 | 23 | 33 | 0 | PASS |

Use `passed_index.json` to identify the seven passed previews. Each scene
folder contains its route file, SUMO config, scene metadata, FCD, collision
output, and log. The four-panel PNG shows one passed scene per road form at
2 s; the GIF shows those four 10 s SUMO trajectories. Vehicles join the road
over the first two seconds, so the animation's initial frame is sparser.
