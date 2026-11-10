# l10_topics_demo: a controller's five input topics, live from CARLA

ENPM818Z Lecture 10. One launch file starts the L2 bridge, which drives the AV on
CARLA's autopilot in Town03, and a node, `l10_topics`, that publishes the topics a
path-following controller reads. Nothing here steers the AV: watch the inputs with
`ros2 topic echo`.

| Topic | Message type | What it holds here |
|---|---|---|
| `/planning/path` | `nav_msgs/Path` | the lane center, 50 m ahead, one point per meter (51 points), every 0.5 s |
| `/localization/odom` | `nav_msgs/Odometry` | the AV's true pose in `map` and its velocity in its own frame |
| `/carla/imu/data` | `sensor_msgs/Imu` | the bridge's IMU, renamed by the launch file; `angular_velocity.z` is the turn rate |
| `/carla/speedometer` | `geometry_msgs/TwistStamped` | the AV's speed, m/s, in `twist.linear.x` |
| `/behavior/target_speed` | `geometry_msgs/TwistStamped` | the road's speed limit, m/s, in `twist.linear.x` (30 km/h = 8.33 m/s) |

**Stand-ins, said plainly.** The pose is CARLA's truth, not an estimate. The target
speed is the road's speed limit, not a behavior layer's choice. The path is the lane
center ahead, not a local planner's path; at a junction it takes the branch that
turns least, which may not be the one the autopilot takes.

## Run it

The CARLA server must be running. Give it 30 to 60 s to load the level first.

```bash
cd ~/enpm818z_ws
colcon build --symlink-install --packages-select l2_carla_demo l10_topics_demo
source install/setup.bash

ros2 launch l10_topics_demo l10_topics.launch.py
```

In a second terminal (`source install/setup.bash` first):

```bash
ros2 topic list
ros2 topic echo /behavior/target_speed --once
ros2 topic echo /localization/odom --once --no-arr
ros2 topic hz /planning/path
```

Arguments: `town:=Town03` (the default), `port:=2000`, `host:=localhost`.
