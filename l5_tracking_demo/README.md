# l5_tracking_demo: a multi-object tracker in CARLA

ENPM818Z Lecture 5, section "Tracking". One Kalman filter per object, the
gate, NN or GNN association, and the track lifecycle, run live on the L2
bridge, and graded against CARLA's ground truth.

| Slide | Where it is in the code |
|---|---|
| One Kalman filter per object | `tracker_core.py`, `KalmanCV`: state `[px, py, vx, vy]` in the fixed frame `map`, constant velocity |
| Every frame, the same loop | `MultiTracker.step`: predict every track, associate, update, manage the rest |
| Distance in units of the track's uncertainty | `KalmanCV.epsilon`: epsilon = nu^T S^-1 nu, gate 9.21 |
| Four ways to associate | `associate_nn` (each track in turn, by ID) and `associate_gnn` (Hungarian, lowest total) |
| The track lifecycle | tentative, confirmed after M of N (3 of 5), coasting, deleted after 1.0 s with no detection |
| Tempe, in tracking terms | `reset_on_class_change: true`: a new class starts a new track |
| Exercise: two tracks, three detections | `test/test_tracker_core.py` runs it: NN gives 7.0 or 2.5 by order, GNN 2.5 |

`tracker_core.py` has no ROS in it. Run the checks with
`python3 -m pytest test/test_tracker_core.py` from this folder.

## Run it

The CARLA server must be running, and the L2 bridge must own the clock:

```bash
cd ~/enpm818z_ws
colcon build --symlink-install --packages-select l2_carla_demo l5_tracking_demo
source install/setup.bash

ros2 launch l2_carla_demo demo.launch.py rviz:=false      # terminal 1
ros2 run l5_tracking_demo spawn_traffic                    # terminal 2: 40 vehicles
ros2 launch l5_tracking_demo tracking.launch.py            # terminal 3
```

RViz shows the LiDAR, the detections (blue boxes) and the tracks: gray
tentative, green confirmed, orange coasting, each with its 1-sigma position
ellipse, its velocity as an arrow of 1 s of travel, and its ID.

Launch arguments:

| Argument | Default | What it changes |
|---|---|---|
| `source` | `lidar` | `lidar`: cluster the bridge's LiDAR (centroids). `boxes`: the centers of `l5_box_demo`'s fitted boxes, same clusters (build `l5_box_demo` too). `truth`: CARLA's objects with noise, misses and clutter added |
| `association` | `gnn` | `gnn` or `nn` |
| `nn_order` | `ascending` | NN only: the oldest track (lowest ID) chooses first, or `descending` |
| `cost` | `epsilon` | how pairs inside the gate are ranked: `epsilon`, or `likelihood` = epsilon + ln\|S\| |
| `confirmed_first` | `true` | confirmed and coasting tracks are matched before tentative ones (below) |
| `reset_on_class_change` | `false` | `true`: Tempe |
| `flip_prob` | `0.0` | `truth` source only: the chance that a detection carries the wrong class |
| `evaluate` | `false` | grade detections and tracks against CARLA; `out:=file.json` saves the report |
| `rviz` | `true` | |

Every other number is in `config/tracker.yaml`, each with a comment.

To save one frame as a picture (the AV's frame, x forward to the right):

```bash
ros2 run l5_tracking_demo snapshot --ros-args -p out:=tracking.png -p min_ego_speed:=3.0
```

## The nodes

**`detector`** turns each LiDAR sweep into detections, with no learning:
drop the points that hit the AV itself, drop the ground (keep 0.3 m to 2.5 m
above the road; flat roads only), mark the cells of a 0.4 m grid that hold a
point, join touching cells (diagonals included) into clusters, and give one
detection per cluster of at least 5 points and at most 7 m across: its
centroid, moved into `map` with the AV's pose at the sweep's time stamp.
Every detection reports R = 0.5^2 I m^2.

The centroid falls short of the object's center: the LiDAR only sees the faces
turned toward the AV (L5, Frustum Association, Step 3). Measured below: 1.3 m
on average.

**`truth_detector`** publishes CARLA's own vehicles within 40 m of the AV,
seen or hidden, with what you choose added: Gaussian noise (`noise_sigma`,
0.3 m), misses (`miss_prob`, 0.1), clutter (`clutter_rate`, 0.5 false
detections per frame on average) and wrong class labels (`flip_prob`). It
publishes every second tick, 10 Hz. Use it to change one effect at a time.

**`tracker`** runs `MultiTracker` on each detection message and publishes
`/l5/tracks` (vision_msgs/Detection3DArray, the confirmed and coasting
tracks, `id` = track ID), `/l5/tracks/json` (every track with its status,
velocity and covariance) and `/l5/tracks/markers` (RViz). It logs the frame's
counts every 50 frames.

**`evaluate`** compares, at the same simulation time, the detections and the
tracks the planner is told about (confirmed and coasting) with CARLA's
vehicles within 30 m of the AV. A detection or track within 3 m of a vehicle
matches it (one to one, Hungarian). It reports the detection rate, the recall
(the share of vehicles with a track), false tracks per frame, ID switches (a
vehicle's matched track ID changes), and the mean position and speed errors.

**`spawn_traffic`** spawns vehicles on CARLA's autopilot (Traffic Manager)
and removes them on Ctrl+C. It never ticks.

## Confirmed tracks first

The slides rank pairs by epsilon alone. A new track has a large S, so epsilon
makes it look close to everything, and GNN, which minimizes the total, will
hand it the detection of an older track whenever that lowers the sum. The
older track then coasts, the new one is confirmed, and the object has changed
ID. The fix used by real trackers (for example DeepSORT's "matching cascade")
is to match the confirmed and coasting tracks first, and the tentative tracks
only on the detections left over. That is `confirmed_first: true`, the
default. Measured below, with it off and on.

## Measured on the course laptop

Setup for every run: CARLA 0.9.16, Town10HD, RTX 4060 laptop GPU, the L2
bridge at 20 Hz with its default 32-channel LiDAR, `spawn_traffic` with 60
vehicles requested (60, 60 and 59 spawned), each run 150 s of wall time,
graded by `evaluate` (30 m, 3 m). To compare settings fairly, the trackers
being compared ran side by side on the **same** detection stream: a second and
third `tracker` node, remapped to their own topics, each with its own
`evaluate`.

**LiDAR detector**, 3034 sweeps (20 Hz), 14.4 detections per sweep on
average, 7.3 CARLA vehicles within 30 m per sweep:

| | GNN, confirmed first (default) | NN, oldest first | GNN, all tracks at once |
|---|---|---|---|
| detection rate | 0.56 | 0.56 | 0.56 |
| recall | 0.60 | 0.60 | 0.60 |
| false tracks per frame | 16.7 | 15.4 | 18.4 |
| ID switches | 119 | 161 | 202 |
| mean position error | 1.33 m | 1.33 m | 1.31 m |
| mean speed error | 0.61 m/s | 0.59 m/s | 0.68 m/s |

- The detection rate counts hidden vehicles too: 0.56 means 44 percent of the
  vehicles within 30 m gave no cluster within 3 m, mostly because something
  stood between them and the LiDAR.
- Most "false tracks" are real things that are not CARLA actors: parked cars
  that are part of the map, poles, trees, walls. A size filter cannot tell a
  parked car from a moving one; a learned detector with classes can.
- The position error is mostly the centroid falling short of the center.

**Truth detector**, default noise (0.3 m, 10 percent misses, 0.5 clutter per
frame), 1489 frames (10 Hz), 5.0 vehicles within 30 m per frame:

| | GNN, confirmed first (default) | NN, oldest first | GNN, all tracks at once |
|---|---|---|---|
| detection rate | 0.89 | 0.89 | 0.89 |
| recall | 0.998 | 0.998 | 0.998 |
| false tracks per frame | 0.49 | 0.50 | 1.19 |
| ID switches | 31 | 32 | 147 |
| mean speed error | 0.66 m/s | 0.65 m/s | 0.69 m/s |

GNN and NN gave about the same result here (31 and 32 switches), because in
normal traffic two vehicles rarely compete for one detection. Without
`confirmed_first`, GNN was much worse in both runs (147 and 202).

On a third recorded run, replayed offline through every setting (same
detections, same truth), GNN with all tracks at once made 283 ID switches
with `cost: epsilon` and 77 with `cost: likelihood`; with `confirmed_first`
it made 32 with either cost, and NN made 35. NN with the newest track
choosing first (`nn_order: descending`, all tracks at once) made 223.

**Tempe**: the truth detector with `flip_prob: 0.3` (3 detections in 10 carry
a wrong class), 1488 frames, both trackers on the same detections:

| | `reset_on_class_change: false` | `true` |
|---|---|---|
| recall | 0.998 | 0.357 |
| ID switches | 28 | 849 |
| mean speed error | 0.61 m/s | 1.39 m/s |

With the reset, a track has to pass 3 of 5 frames with no class change to be
confirmed, and each restart throws its velocity away: two out of three
vehicles within 30 m have no confirmed track at a given moment, and the ones that
do have a speed error more than twice as large. That is the chain on the
slide: no history, no velocity, no predicted path.

## Tasks

1. **The exercise.** Read `test_nn_order_changes_the_answer` and
   `test_gnn_picks_the_lowest_total` in `test/test_tracker_core.py`, then
   change the epsilon table and predict NN's and GNN's answers before you run
   them.
2. **A wrong R.** In `config/tracker.yaml`, give the truth detector
   `noise_sigma: 1.0` and `reported_sigma: 0.5`: the detections now claim to
   be twice as precise as they are. What happens to the ID switches, and why?
   Then set `reported_sigma: 1.0` and compare.
3. **NN against GNN.** Find a scene where they differ: two vehicles crossing
   close to each other, with the truth source. In normal traffic we measured
   no real difference (31 and 32 switches above). Script the two vehicles with
   CARLA's autopilot or with `set_target_velocity`, and use `nn_order` to
   reproduce the exercise's order effect.
4. **Confirmed first.** Turn `confirmed_first` off with GNN and watch RViz:
   which tracks take the detections, and what does it do to the IDs?
5. **Tempe.** Run `flip_prob:=0.3` with `reset_on_class_change` off and on.
   Watch the velocity arrows. Which design keeps the planner informed?
6. **M of N.** Try 2 of 3 and 5 of 8. What changes in recall, false tracks,
   and how fast a new vehicle is reported?
