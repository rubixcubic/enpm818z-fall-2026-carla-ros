# l5_bev_demo: three bird's-eye views of one CARLA scene

ENPM818Z Lecture 5, section "BEV and Occupancy". None of these views is
learned. Each one is the geometry of a method from the slides, so you can see
what the method does before a network is trained to do it.

| View | Topic | Built from | Slides |
|---|---|---|---|
| LiDAR height map | `/l5/bev/lidar_height` | the bridge's LiDAR | PointPillars, Steps 1 to 3 |
| LiDAR occupancy grid | `/l5/bev/occupancy` | the bridge's LiDAR | Occupancy; BEV and occupancy in industry (Autoware) |
| Camera IPM | `/l5/bev/ipm` | 4 roof RGB cameras | Before learning: IPM; IPM breaks for anything above the road |
| Semantic lift-splat | `/l5/bev/semantic` | 4 roof semantic + depth cameras | Lift-Splat-Shoot; Lift, by hand; Splat, by hand |

All four use one grid (`config/bev.yaml`): 250 x 250 cells of 0.2 m, from
-25 m to +25 m around the AV, in the `ego_vehicle` frame (x forward, y left,
z up, origin on the ground under the car). The images are drawn front up and
left on the left, so the four views line up cell for cell.

## Run it

The CARLA server must be running, and the L2 bridge must own the clock:

```bash
cd ~/enpm818z_ws
colcon build --symlink-install --packages-select l2_carla_demo l5_seg_demo l5_bev_demo
source install/setup.bash

ros2 launch l2_carla_demo demo.launch.py rviz:=false     # terminal 1
ros2 launch l5_bev_demo bev.launch.py                    # terminal 2
```

RViz opens with the occupancy grid and the point cloud under the AV, and the
three BEV images in their own panels. To save one moment as PNG files:

```bash
ros2 run l5_bev_demo snapshot --ros-args -p out:=bev_snapshot
```

It writes `lidar_height.png`, `occupancy.png`, `ipm.png`, `semantic.png` and
`all_four.png` (the four side by side, labeled).

## What each node does

**`surround_rig`** attaches four cameras to the bridge's vehicle (found by
`role_name: ego`): front, left, right and rear, 0.3 m above the roof, tilted
30 degrees down, 100 degree field of view, 480 x 360 pixels, 10 Hz. At each place
it mounts an RGB, a semantic segmentation and a depth camera with the same pose.
It publishes `/l5/<camera>/rgb` (bgra8), `/l5/<camera>/semantic` (mono8, the
class tag), `/l5/<camera>/depth` (32FC1, meters), `/l5/<camera>/camera_info`,
and the static transforms `ego_vehicle -> l5/<camera> -> l5/<camera>_optical`.
It also publishes `/l5/<camera>/ego_mask` once, latched (mono8, 255 where the
camera sees the AV itself): the pixels of the first depth image whose 3D point
falls inside the AV's bounding box. Measured: 26.8 percent of the front image
(the hood), 9.2 percent of each side image and 37.8 percent of the rear image
(the node's log, Tesla Model 3). `l5_seg_demo`'s `seg_eval` uses it to leave
the AV out of the grade. It never ticks the simulator.

**`lidar_bev`** turns each LiDAR sweep into two grids. Points inside the AV's
footprint are dropped first: the lowest beams hit the AV's own roof.
- Height map: the tallest point in each cell, colored 0 to 2.5 m; empty cells
  are black.
- Occupancy: per 1 degree ray, cells up to the farthest return are free; cells
  holding a point between 0.3 m and 2.5 m above the ground are occupied;
  everything behind the nearest obstacle stays unknown.

**`ipm_bev`** maps every cell's ground point (x, y, 0) into each camera with
the mounting transform and the pinhole model, once, then warps each frame with
`cv2.remap`. Where cameras overlap, the one that looks most directly at the
cell wins.

**`semantic_bev`** lifts every pixel to 3D at its true depth and splats its
class into the cell under it. CARLA's depth is measured along the optical axis
(a z-buffer), so the point is `((u - c_u) d / f, (v - c_v) d / f, d)`. Checked:
road pixels land at z = 0.01 m or less across the whole front image; taking
the depth along the ray instead puts the image edges about 0.46 m above the
road. When several classes fall in one cell, people and vehicles win over lane
markings, and lane markings over the road.

**`labels:=network`** (`ros2 launch l5_bev_demo bev.launch.py labels:=network`)
splats a trained network's classes instead of CARLA's true ones, still with
CARLA's true depth: it starts `l5_seg_demo`'s `seg_node` on the four roof
cameras (SegFormer-B0, Cityscapes), and `semantic_bev` reads
`/l5/<camera>/semantic_net`. That network has no lane-marking class, so the
lane lines disappear from the grid, and fewer roof images reach it, so the grid
is sparser. `l5_seg_demo`'s README has the numbers. The default,
`labels:=truth`, is unchanged.

## Tasks

1. **The grid.** Car B on the slides sits at (12.0, -2.0) m. Which cell is that
   on this package's grid (`half_extent` 25 m, `resolution` 0.2 m)? Compute
   `i` and `j` by hand with the slide's formula, then find the cell in
   `ipm.png` (row `n - 1 - i`, column `n - 1 - j`, n = 250).
2. **IPM.** Find a car or a building in `ipm.png`. Measure, in cells, how long
   its streak is, and explain the length with the slide "IPM breaks for anything
   above the road". Then find a lane line: is it straight and the right width?
3. **Lift-splat.** In `semantic.png` the far road breaks into separate rows.
   Explain the gaps with the slide "The camera squeezes the far road into a
   few rows". Set `stride: 2` in `config/bev.yaml`: what happens near the AV,
   and far away?
4. **Occupancy.** Find a gray (unknown) wedge in `occupancy.png`. What object
   casts it? Compare the same place in `semantic.png`: which view tells the
   planner more, and which one is honest about what it did not see?
5. **The three side by side.** Name one thing each view gets right that the
   other two get wrong.

## Measured on the course laptop (RTX 4060, CARLA 0.9.16, Town10HD)

With the bridge and all 12 rig cameras running, the server ran slower than
real time: the bridge's LiDAR arrived at 13 to 16 Hz instead of 20, the rig's
cameras at 3 to 5 Hz instead of 10. The BEV nodes kept up with their inputs:
LiDAR grids 12 to 15 Hz, IPM and semantic at their 10 Hz timers. Lower
`image_width` and `image_height` in `config/bev.yaml` if your machine is slower.
