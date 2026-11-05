# l9_planning_demo

The Lecture 9 sampling planner on Town03 in CARLA, with every candidate path
drawn in RViz. Slide: "Demo: the same planner in CARLA".

The AV starts on the slides' stretch of Town03 road 67, at `s = 60 m` on the
right lane's center, with a car parked at `s = 118 m`. Every 0.5 s the node
samples the 31 candidates of the slides (10 end offsets times 3 maneuver lengths,
plus stop behind), checks them with three circles, scores them with the four cost
terms and picks the cheapest. Between plans the AV moves along the first 0.5 s of
that path at 10 m/s. At the end of the stretch it starts again.

The node **places** the AV on the path with `set_transform`; it does not steer
or brake. There is no controller here: L10 adds one, and GP4 asks you to write
it.

## Run it

```bash
carla_start                                   # or your own CARLA 0.9.16 server
cd ~/enpm818z_ws && colcon build --symlink-install --packages-select l9_planning_demo
source install/setup.bash
ros2 launch l9_planning_demo demo.launch.py             # one parked car, as the slides
ros2 launch l9_planning_demo demo_obstacles.launch.py   # or three parked cars
```

Both launch files take `host:=`, `port:=` (default 2000) and `rviz:=` (default
`true`). Start CARLA first: the node stops with "Could not reach the CARLA server"
if no server answers.

The node loads Town03 if another town is loaded (a few seconds). It ticks the
server in synchronous mode at 20 Hz: do not run `l2_carla_demo`'s bridge, or any
other client that ticks, at the same time. On exit it removes every actor it
spawned (the AV, the parked cars, the camera) and gives the server back its
asynchronous settings.

## How the node runs

`demo_node.py` is one node, `planning_demo`. It is a CARLA client and a ROS 2
publisher.

**At start:**

1. Connect to CARLA, load Town03 if needed, and put the server in synchronous mode
   with a step of 0.05 s (20 Hz). From then on the node advances the simulation.
2. Read the road from `data/l9_scene.npz`: the right lane's center line on road 67,
   a point every 0.5 m. This is the reference path of the Frenet frame.
3. Spawn the AV (`vehicle.tesla.model3`, role `hero`), one parked car
   (`vehicle.audi.a2`) per obstacle, and a camera on the AV. Physics is off for
   the cars: the node places them, nothing drives them.
4. Publish the road once on `/l9/road`, then make the first plan.

**Every step (0.05 s):** move the AV 0.5 m along the current plan (10 m/s times
0.05 s), place it there with `set_transform`, move CARLA's spectator above it,
and call `world.tick()`. The camera sends one frame per step to `/l9/camera`.

**Every 0.5 s:** plan again from where the AV is now, with its `d`, `d'` and
`d''` at that point, so the new candidates start smoothly. Publish the plan on
`/l9/plan`.

**At the end of the stretch** (`s` past 237.9 − 70 − 5 = 162.9 m, so the 70 m
plan still fits on the road): start again at `s = 60 m`.

## Where the obstacles come from: they are given, not detected

Nothing in this package detects anything. The parked cars are numbers:

- **One car** (`demo.launch.py`): `s = 118 m`, `d = −0.3 m`, from the slides
  (parameter `car_s`).
- **Several cars** (`demo_obstacles.launch.py`): the parameter `obstacles`, a flat
  list `[s1, d1, s2, d2, ...]` in meters.

For each `(s, d)`, `planner.parked_car()` turns the position into a box on the map:
its center, its heading (along the lane), and the centers of its three circles.
The node gives the planner these boxes, and places a CARLA car at the same pose so
you can see it. The planner never reads CARLA: it does not look at the cars, the
camera or the map.

In a full stack the boxes come from perception (the detector and the tracker),
and moving cars get a predicted path. This demo skips both on purpose: the cars
do not move, and their boxes are exact.

## The planner

`planner.py` is a sampling planner in the Frenet frame, the one on the slides
(Stanford's Junior in the 2007 Urban Challenge rolled out its paths this way;
the Frenet sampling follows Werling et al., 2010).

1. **Frame.** The reference is the right lane's center. `s` is the distance along
   it, `d` the distance to its left. The AV and every car are turned into
   `(s, d)` and back with `to_frenet` and `from_frenet`.
2. **Generate.** 31 candidates, every 0.5 m along 70 m of road:
   - 10 end offsets, `d_end` from −0.5 to 4.0 m every 0.5 m (both lane centers,
     0 and 3.5 m, are among them);
   - 3 maneuver lengths, 20, 30 and 40 m: the distance along `s` over which `d`
     goes from where the AV is to `d_end`, as a quintic polynomial (no bend at
     either end); after it the candidate holds `d_end`;
   - 1 stop behind: keep the right lane and stop with the front bumper 2.0 m
     short of the nearest car ahead in that lane.
3. **Check.** Each candidate is turned back into map points `(x, y, heading)`.
   Three circles of radius 1.21 m cover the AV's 4.5 m by 1.9 m body (centers
   0.15 m behind, 1.35 m and 2.85 m ahead of the base link); each car gets three
   too. A candidate is dropped as a **collision** if any circle-to-circle gap
   along it is under 0.3 m, and as **too sharp** if its peak sideways
   acceleration, speed² times curvature, passes 3 m/s².
4. **Score.** Each kept candidate costs the sum of four terms:

   | term | number | weight |
   |---|---|---|
   | comfort | peak sideways acceleration, m/s² | `w_comfort` 2 |
   | clearance | how much the smallest gap is short of 1.5 m (0 if it is not) | `w_clearance` 4 |
   | lane | where it ends: 0 lanes from the route's lane (the right lane) or 1, and m from the nearest lane center | `w_lane` 2, `w_center` 2 |
   | progress | m of the 70 m it does not cover | `w_progress` 0.5 |

5. **Pick** the cheapest. With one car at `s = 118 m` and the AV at `s = 60 m`, it
   ends at `d = 3.5 m` over 40 m, cost 5.07, as on the slides.

All numbers in `planner.py` (the AV, the scene, the candidates, the weights, the
margins) are chosen for the lecture; the road is measured from `Town03.xodr`.
`planner.py` is a copy of `lecture9/l9common.py` in the course's Python repo,
without the garage: change both together.

## What this demo does not do

- **No perception, no prediction.** The cars are given (see above), and they
  never move.
- **No speed choice.** The AV always plans at 10 m/s. It can stop behind a car,
  but it cannot slow down to wait for a gap.
- **One lane change per candidate.** A candidate moves to its `d_end` once and
  holds it. There is no "pass, come back, pass again" path in one plan.
- **No way out.** If every candidate collides, the cheapest of them still wins
  and the AV drives it. Not every layout of `obstacles` is safe for this reason;
  the one in `demo_obstacles.launch.py` was checked: every plan along its drive
  has a collision-free cheapest path.
- **No controller.** The AV is placed on the path, not driven.

## Change the planner while it runs

Each weight is a parameter. A change takes effect at the next plan, 0.5 s later:

```bash
ros2 param set /planning_demo w_clearance 10.0   # the AV passes farther left (4.0 m, cost 5.86)
ros2 param set /planning_demo w_progress 0.0     # the AV stops behind the parked car
ros2 param set /planning_demo car_s 90.0         # move the parked car (one-car run only)
```

| parameter | default | what it is |
|---|---|---|
| `w_comfort` | 2.0 | per m/s² of peak sideways acceleration |
| `w_clearance` | 4.0 | per m that the smallest gap to a car falls under 1.5 m |
| `w_lane` | 2.0 | per lane the path ends away from the route's lane |
| `w_center` | 2.0 | per m the path ends away from a lane center |
| `w_progress` | 0.5 | per m of the 70 m horizon not covered |
| `car_s` | 118.0 | where the parked car is, in m along the right lane (one car only) |
| `obstacles` | unset | several parked cars, `[s1, d1, s2, d2, ...]` in m; unset, the one car at `car_s` |
| `speed` | 10.0 | the AV's speed, m/s |
| `period` | 0.5 | s between plans |
| `delta` | 0.05 | s per simulation step |
| `host`, `port`, `timeout` | localhost, 2000, 20.0 | the CARLA server, and how long to wait for it (s) |
| `town` | Town03 | the map the node loads |
| `vehicle`, `parked` | `vehicle.tesla.model3`, `vehicle.audi.a2` | the CARLA models of the AV and of the parked cars |
| `follow_camera` | true | CARLA's spectator follows the AV (seen only in a server with a window) |
| `camera` | true | a camera behind the AV, published on `/l9/camera` |
| `camera_width`, `camera_height`, `camera_fov` | 800, 450, 90 | its image size in pixels and its field of view in degrees |

### Three parked cars

`demo_obstacles.launch.py` places cars at `s = 78 m` in the left lane, `118 m` in
the right lane (the slides' car) and `165 m` in the right lane. The AV keeps its
lane until it has passed the first, moves left to pass the second, and stays left
because the third blocks the right lane.

## What RViz shows

RViz opens a top view of the stretch on a white background, as on the slides, and
a Camera panel.

| topic | type | drawn |
|---|---|---|
| `/l9/road` | `MarkerArray`, latched | the right lane's edges (gray), the line between the two lanes, and the road's center line at `d = 5.25 m` (yellow) |
| `/l9/plan` | `MarkerArray` | every candidate, in the slides' colors: green the cheapest, gray kept, rust collides, purple too sharp, black stop behind; the AV (dark box) at the start of the plan with its three circles; each parked car (blue box); the three cheapest costs and the weights as text |
| `/l9/camera` | `sensor_msgs/Image`, bgr8 | what a camera 7 m behind and 3.5 m above the AV sees, in RViz's Camera panel |

**Frames.** Everything is in the `map` frame: `x` as CARLA's `x`, `y` as minus
CARLA's `y`, so the frame is right-handed, as on the slides. The node publishes no
TF, so RViz's Global Status says "No tf data" for `map`; the markers are in `map`
themselves and still show.

## The files

| file | what it is |
|---|---|
| `l9_planning_demo/demo_node.py` | the node: CARLA, the plan every 0.5 s, the markers, the camera |
| `l9_planning_demo/planner.py` | the planner: a copy of `lecture9/l9common.py` from the course's Python repo, without the garage |
| `data/l9_scene.npz` | the stretch of road 67, exported from `Town03.xodr` (the same file as in the Python repo) |
| `launch/demo.launch.py` | the node with one parked car, and RViz |
| `launch/demo_obstacles.launch.py` | the node with three parked cars, and RViz |
| `rviz/l9_planning.rviz` | the top view and the Camera panel |
| `test/test_planner.py` | `python3 -m pytest -q test`: the slides' costs, no CARLA needed |

## If something goes wrong

- *Could not reach the CARLA server*: start it, and wait until it accepts
  connections (`carla_start` waits for you).
- *The server crashes while it loads Town03*: start it at a higher quality level.
  A server started with `-quality-level=Low` crashed with signal 11 on every
  `load_world("Town03")` in our tests; Epic loaded it in 3.6 s.
- *RViz stays white*: the background is white by design, so a white view means no
  markers yet. The node publishes only after Town03 has loaded; wait a few
  seconds. Check with `ros2 topic hz /l9/plan` (2 Hz when it runs).
- *RViz shows nothing at all*: set the fixed frame to `map`, and check that both
  MarkerArray displays are on.
- *The Camera panel stays black*: check `ros2 topic hz /l9/camera` (about 20 Hz),
  and look for an error from `spawn_actor` in the node's output.
