# l8_foxglove_demo: L8's predictions, live, in Foxglove

ENPM818Z Lecture 8 reading, "Foxglove" (Pre-Read Material on the course site). One
launch file starts the L2 bridge, `foxglove_bridge` and a node, `l8_scene`, that
spawns traffic near the AV and publishes every car's true box and its
constant-velocity (CV) prediction, 5 s ahead. A Foxglove layout shows them next to
the front camera, the LiDAR, the speeds and the log.

| L8 | Where it is here |
|---|---|
| CV: the road user keeps the velocity it has now | `cv_predict.py`, `cv_path`: x(h) = x0 + vx h, y(h) = y0 + vy h |
| Every 0.1 s for 5 s | `config/l8_scene.yaml`: `step: 0.1`, `horizon: 5.0`; the node publishes at 10 Hz |
| A dot every second | `dot_every: 1.0`: the dots on each blue path |
| A miss: more than 2 m off | `warn_distance: 2.0`: the log warns when a car's CV path comes within 2 m of the AV's at the same time |
| Two paths that cross at different times never meet | `cv_predict.py`, `closest_approach`: the distance is taken time by time |

`cv_predict.py` has no ROS and no CARLA in it, and `markers.py` has no CARLA. The
tests run without a simulator.

**The predictions start from CARLA's true velocity**, not from a tracker. This demo
is about the viewer; L8 itself fits the velocity from a noisy track.

## Run it

The CARLA server must be running. Give it 30 to 60 s to load the level first.

```bash
cd ~/enpm818z_ws
colcon build --symlink-install --packages-select l2_carla_demo l8_foxglove_demo
source install/setup.bash

ros2 launch l8_foxglove_demo l8_foxglove.launch.py
```

Then, in the Foxglove app:

1. **Open connection**, **Foxglove WebSocket**, address `ws://localhost:8765`.
2. **Layout**, **Import from file**, and pick
   `install/l8_foxglove_demo/share/l8_foxglove_demo/foxglove/l8_layout.json`
   (or `src/enpm818z-fall-2026-carla-ros/l8_foxglove_demo/foxglove/l8_layout.json`).

Ctrl+C stops everything: `l8_scene` removes the cars it spawned, and the bridge
removes the AV and hands the clock back to the server.

Launch arguments:

| Argument | Default | What it changes |
|---|---|---|
| `host`, `port` | `localhost`, `2000` | where the CARLA server is |
| `town` | (empty) | a map to load, for example `Town01`; empty keeps the loaded one |
| `vehicles` | `8` | traffic cars, at the spawn points nearest the AV |
| `foxglove_port` | `8765` | the WebSocket port Foxglove connects to |

Every other number is in `config/l8_scene.yaml`, each with a comment.

## What the layout shows

| Panel | Topics |
|---|---|
| 3D, following the AV | `/carla/ego_vehicle/lidar` (colored by intensity), `/l8/vehicles` (dark: the AV; orange: the other cars, each with its speed), `/l8/predictions` (blue: each car's CV path; gray: the AV's own) |
| Image | `/carla/ego_vehicle/rgb_front/image` |
| Plot | `/l8/ego/speed` and `/l8/nearest/speed`, in m/s |
| Log | `/rosout`: the bridge's messages and the 2 m warnings |
| Raw Messages | `/l8/nearest/distance`: the nearest car's distance to the AV, in m |

There is no Map panel: Town01's GNSS is set at latitude 0, longitude 0
(`+lat_0=0 +lon_0=0` in `Town01.xodr`), which is open sea on a real map.

## Topics

| Topic | Type | What |
|---|---|---|
| `/l8/vehicles` | `visualization_msgs/MarkerArray` | each car's box (its true pose and size) and a speed label |
| `/l8/predictions` | `visualization_msgs/MarkerArray` | each car's CV path (line) and its dots, one per second |
| `/l8/ego/speed` | `std_msgs/Float32` | the AV's speed, m/s |
| `/l8/nearest/speed` | `std_msgs/Float32` | the speed of the car nearest the AV, m/s |
| `/l8/nearest/distance` | `std_msgs/Float32` | its distance to the AV, m |

All markers are in the `map` frame and stamped with CARLA's simulation time, like the
bridge's transforms.

## Troubleshooting

- **`time-out ... waiting for the simulator`**: the server is not up yet, or is still
  loading a map. Wait and launch again. With `town:=...`, the bridge loads the map
  first; if that times out, launch again: the map is loaded by then. `l8_scene` waits
  and retries on its own (`CARLA is busy, trying again` in the log).
- **Foxglove connects but shows nothing**: check the address and port, then import the
  layout again. The 3D panel needs `/tf` to follow the AV.
- **`address already in use` from `foxglove_bridge`**: another bridge holds 8765. Stop
  it, or launch with `foxglove_port:=8766` and connect to `ws://localhost:8766`.
- **The cars do not move**: the Traffic Manager must be synchronous. The bridge sets it;
  start this launch file on its own, not next to another bridge.

## Tests

```bash
colcon test --packages-select l8_foxglove_demo && colcon test-result --verbose
```

`test_cv_predict.py` checks L8's own CV numbers for car A, `test_markers.py` the
markers, and `test_layout.py` that every topic in the layout is one the bridge or
`l8_scene` publishes.
