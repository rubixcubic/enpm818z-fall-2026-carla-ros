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
ros2 launch l9_planning_demo demo.launch.py   # port:=2000 rviz:=true by default
```

The node loads Town03 if another town is loaded (this takes a few seconds). It
ticks the server in synchronous mode at 20 Hz: do not run `l2_carla_demo`'s
bridge, or any other client that ticks, at the same time. On exit it removes its
two cars and gives the server back its asynchronous settings.

The CARLA window follows the AV from above. RViz shows the same scene in the
`map` frame (`x` as CARLA's `x`, `y` as minus CARLA's `y`, as on the slides).

## Change the planner while it runs

Each weight is a parameter. A change takes effect at the next plan, 0.5 s later:

```bash
ros2 param set /planning_demo w_clearance 10.0   # the AV passes farther left (4.0 m, cost 5.86)
ros2 param set /planning_demo w_progress 0.0     # the AV stops behind the parked car
ros2 param set /planning_demo car_s 90.0         # move the parked car
```

| parameter | default | what it is |
|---|---|---|
| `w_comfort` | 2.0 | per m/s² of peak sideways acceleration |
| `w_clearance` | 4.0 | per m that the smallest gap to the car falls under 1.5 m |
| `w_lane` | 2.0 | per lane the path ends away from the route's lane |
| `w_center` | 2.0 | per m the path ends away from a lane center |
| `w_progress` | 0.5 | per m of the 70 m horizon not covered |
| `car_s` | 118.0 | where the parked car is, in m along the right lane |
| `host`, `port` | localhost, 2000 | the CARLA server |
| `follow_camera` | true | the CARLA window follows the AV |

All weights are chosen for the lecture, as the slides say.

## What RViz shows

| topic | drawn |
|---|---|
| `/l9/road` | the right lane's edges (gray), the line between the two lanes, and the road's center line at `d = 5.25 m` (yellow) |
| `/l9/plan` | every candidate, in the slides' colors: green the cheapest, gray kept, rust collides, purple too sharp, black stop behind; the AV (dark box) at the start of the plan with its three circles; the parked car (blue box); the three cheapest costs and the weights as text |

## The files

| file | what it is |
|---|---|
| `l9_planning_demo/demo_node.py` | the node: CARLA, the plan every 0.5 s, the markers |
| `l9_planning_demo/planner.py` | a copy of `lecture9/l9common.py` from the course's Python repo, without the garage: change both together |
| `data/l9_scene.npz` | the stretch of road 67, exported from `Town03.xodr` (the same file as in the Python repo) |
| `launch/demo.launch.py` | the node and RViz |
| `rviz/l9_planning.rviz` | a top view of the stretch |
| `test/test_planner.py` | `python3 -m pytest -q test`: the slides' costs, no CARLA needed |

## If something goes wrong

- *Could not reach the CARLA server*: start it, and wait until it accepts
  connections (`carla_start` waits for you).
- *The server crashes while it loads Town03*: start it at a higher quality level.
  A server started with `-quality-level=Low` crashed with signal 11 on every
  `load_world("Town03")` in our tests; Epic loaded it in 3.6 s.
- *RViz shows nothing*: set the fixed frame to `map`, and check that both
  MarkerArray displays are on.
