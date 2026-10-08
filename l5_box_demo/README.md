# l5_box_demo: 3D boxes from the LiDAR, with no learning

ENPM818Z Lecture 5, section "3D Detection". A 3D detector reports, per object,
the 7 numbers of a 3D box: center (x, y, z), length, width, height and heading.
This package produces them from the L2 bridge's LiDAR with no network, so you
can see each step: cluster the points, then fit a rotated box to each cluster.
It is the step the Frustum Association slides mean by "real stacks fit a box
to the cluster".

## Run it

The CARLA server must be running, and the L2 bridge must own the clock:

```bash
cd ~/enpm818z_ws
colcon build --symlink-install \
  --packages-select l2_carla_demo l5_tracking_demo l5_box_demo
source install/setup.bash

ros2 launch l2_carla_demo demo.launch.py rviz:=false      # terminal 1
ros2 run l5_tracking_demo spawn_traffic                    # terminal 2: 40 vehicles
ros2 launch l5_box_demo boxes.launch.py evaluate:=true     # terminal 3
```

RViz shows the LiDAR and the boxes: each box, its heading axis drawn both
ways, and a label with its size and heading. With `evaluate:=true` the node
grades every sweep against CARLA's true boxes and logs a report every 10 s
(`out:=boxes.json` saves it).

To save one sweep as a picture (the AV's frame, x forward to the right):

```bash
ros2 run l5_box_demo snapshot --ros-args -p out:=boxes.png
```

It writes `boxes.png` and the sweep's data as `boxes_sweep.json`, so it does
not overwrite the evaluation's `boxes.json`.

To track the boxes instead of the cluster centroids:

```bash
ros2 launch l5_tracking_demo tracking.launch.py source:=boxes
```

## What the node does

**Steps 1 to 3** are `l5_tracking_demo`'s detector, shared code, so both
packages see the same clusters: points in the AV's frame, the AV's own points
dropped, the ground dropped (0.3 m to 2.5 m above the road kept), cells of a
0.4 m grid that hold a point, touching cells joined into clusters, clusters of
fewer than 5 points dropped.

**Step 4, L-shape fitting.** For each cluster, seen from above, try every
direction theta from 0 to 89 degrees in steps of 1 degree. In each direction,
project the points on the two axes e1 = (cos theta, sin theta) and
e2 = (-sin theta, cos theta); the rectangle with those axes that just holds the
points has four edges. Score the direction by how close the points are to the
edges: the sum over points of 1 / max(d, d0), d being a point's distance to
its nearest edge. Keep the best direction. This is the search-based fitting
with the closeness criterion of

> X. Zhang, W. Xu, C. Dong and J. M. Dolan, "Efficient L-Shape Fitting for
> Vehicle Detection Using Laser Scanners", IEEE Intelligent Vehicles Symposium
> (IV), 2017, pp. 54 to 59 (Algorithms 2 and 4).

The paper gives no values for the angle step or d0. The values here are
Autoware's, whose `autoware_shape_estimation` package runs the same method
(`lib/model/bounding_box.cpp`): a 1 degree step, d0 = 0.1 m, and points
farther than 0.4 m from every edge do not vote. Like Autoware, the distance to
the nearest edge is taken point by point.

**Step 5.** Length is the longer side, width the shorter; the heading runs
along the length. Height is the highest point of the cluster above the road,
and the box stands on the road. Boxes longer than 7 m (walls, buildings) are
dropped.

**Step 6.** The box goes into the fixed frame `map` with the AV's pose at the
sweep's time stamp, and is published on `/l5/boxes` (`vision_msgs/Detection3DArray`).

**The heading is known modulo 180 degrees.** One sweep shows a rectangle; it
cannot say which end is the front. Every published heading is in [-90, 90)
degrees, the RViz label says "mod 180", and the evaluation measures the
heading error modulo 180 degrees. A tracker, which knows which way the object
moved, can settle it.

## The evaluation

`box_detector` with `evaluate:=true` compares every sweep with CARLA's true
boxes at the same simulation time: the vehicles (actors), plus the parked cars
built into the map (`world.get_environment_objects`), within 30 m of the AV.
A true vehicle is "visible" when at least 5 kept LiDAR points fall in its box.
The definitions are nuScenes' (devkit, `python-sdk/nuscenes/eval`):

* a box matches a vehicle when their centers are within 0.5, 1, 2 or 4 m on the
  ground, one box per vehicle, closest pairs first (the devkit sorts by
  confidence; these boxes have none);
* for boxes matched within 2 m: center error, size error ASE = 1 minus the IoU
  of the two boxes once their centers and headings are aligned, and heading
  error, here modulo 180 degrees;
* the centroid error: the distance from the same cluster's centroid (what
  `l5_tracking_demo`'s detector reports) to the vehicle's center.

## Measured

Course laptop (RTX 4060 laptop GPU), CARLA 0.9.16, Town10HD, the L2 bridge's
LiDAR (32 channels, 50 m, about 20 sweeps per second), 40 vehicles from
`spawn_traffic`, 120 s per run. Each run ran the box detector (with
`evaluate:=true`), `l5_tracking_demo`'s centroid detector, and two trackers
with their evaluators, all on the same sweeps.

**The boxes**, two runs (about 2,400 sweeps each; 3.1 visible vehicles per
sweep within 30 m):

| | run A | run B |
|---|---|---|
| boxes per sweep | 13.7 | 15.6 |
| recall within 2 m, visible vehicles | 0.57 | 0.60 |
| recall within 2 m, all vehicles in 30 m | 0.41 | 0.46 |
| precision within 2 m | 0.13 | 0.12 |
| center error, mean (median), m | 0.91 (0.75) | 0.94 (0.72) |
| **centroid error, same objects**, mean (median), m | 1.29 (1.19) | 1.44 (1.46) |
| size error ASE | 0.70 | 0.65 |
| length error, mean, m | -1.94 | -1.76 |
| width error, mean, m | -1.02 | -0.94 |
| heading error mod 180 deg, median (mean), deg | 3.2 (34.3) | 2.6 (27.1) |

Read it this way:

* Fitting the box moves the center about 0.4 to 0.5 m closer to the truth
  than the centroid (mean), because the box covers the side the LiDAR
  cannot see.
* When the fit works, the heading is right: the median heading error is 2.6
  to 3.2 degrees. The mean is far higher because a car seen from one face only
  gives a thin box whose long side may be the car's width: a 90 degree error.
* The boxes are too small (length 1.8 to 1.9 m short, width about 1 m short, on
  average): a box only
  covers the points it sees. Autoware corrects this with the class's typical
  size; this package does not.
* Most boxes are not vehicles: walls, fences, poles and trees also form
  clusters, which is why the precision is low. This detector has no classes.

**Tracking the boxes instead of the centroids**, same runs,
`l5_tracking_demo`'s evaluator (CARLA's vehicles within 30 m, a track matches
within 3 m; three runs):

| | centroids | boxes |
|---|---|---|
| track position error, m (runs 1, 2, 3) | 1.42, 1.31, 1.64 | 1.09, 1.02, 1.26 |
| recall | 0.67, 0.53, 0.63 | 0.72, 0.55, 0.64 |
| ID switches | 72, 83, 78 | 75, 80, 65 |

The tracks sit 0.3 to 0.4 m closer to the vehicles with boxes; the ID switches
do not change in a consistent way.

(The first run's box evaluation is not reported: a bug in the truth reader,
fixed since, moved the true boxes. Its tracking numbers use
`l5_tracking_demo`'s own truth and are valid.)

## Limits you can see in the snapshot

* **Split clusters.** A car hit by few points, or a truck seen at its corner,
  can fall into two or three clusters, each with its own thin box. Try
  `cell_size: 0.6` in `config/boxes.yaml`.
* **One face only.** A car seen straight from behind or from the side gives a
  line of points; its box is a thin strip along that face, and its center sits
  on the face, not in the car.
* **Hidden cars** have no points and no box: a vehicle behind another is
  missed.

## Tasks

1. **Car B by hand.** `test/test_lshape.py` fits car B of the slides: center
   (12.0, -2.0) m, 4.5 x 1.9 m, heading 30 deg, with 15 points on its rear and
   26 on its left side. The centroid is (10.99, -1.89); the fit gives (11.93,
   -2.00), 4.41 x 1.91 m, heading 29 deg. Run the test, then explain why the
   length comes out 0.09 m short.
2. **The criterion.** Plot the closeness score against theta for car B's
   points (`l5_box_demo.lshape.closeness`). Where is the peak, and how sharp is it?
3. **The heading.** Take a box with heading 30 deg mod 180. Which two
   directions could the car be driving? What would tell them apart?
4. **Split clusters.** Run with `cell_size: 0.6`, then 0.8. What happens to
   the recall and to the boxes per sweep, and why?
5. **Track the boxes.** Run `tracking.launch.py source:=boxes evaluate:=true`
   and `source:=lidar evaluate:=true` and compare the position errors with the
   table above.
