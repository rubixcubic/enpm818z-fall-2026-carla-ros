# enpm818z-fall-2026-carla-ros

ROS 2 packages for ENPM818Z.

| Package | What it is |
|---|---|
| `l2_carla_demo` | The lecture 2 demo bridge. Publishes a CARLA sensor suite onto ROS 2 topics. Read it; you will import from it. |
| `l3_ekf_demo` | The lecture 3 hands-on: fuses the bridge's GNSS and IMU with an EKF and logs the NIS. Subscribes to `l2_carla_demo`'s topics. Start the bridge with `gnss_noise_m:=1.5`, the GNSS noise the filter's R assumes. |
| `l5_bev_demo` | The lecture 5 hands-on: three bird's-eye views of one scene (LiDAR height and occupancy grid, camera IPM, semantic lift-splat). Adds four roof cameras to the bridge's vehicle. See its README. |
| `l5_tracking_demo` | The lecture 5 tracking hands-on: a multi-object tracker (one Kalman filter per object, gate, GNN or NN, track lifecycle) on detections from the bridge's LiDAR or from CARLA's truth with noise added, graded against CARLA's truth. See its README. |
| `l5_box_demo` | The lecture 5 3D Detection hands-on: 3D boxes (center, size, heading mod 180 deg) from the bridge's LiDAR with no learning, by clustering and L-shape fitting, graded against CARLA's true boxes; can feed `l5_tracking_demo` (`source:=boxes`). See its README. |
| `l5_seg_demo` | The lecture 5 Segmentation hands-on: semantic segmentation of the bridge's front camera by a trained network (SegFormer-B0, Cityscapes), graded pixel by pixel against CARLA's semantic camera mounted at the same place (IoU per class, mIoU); optional YOLOv8s-seg instance masks; can give `l5_bev_demo` the network's classes (`labels:=network`). Needs `pip install transformers`. See its README. |
| `l8_foxglove_demo` | The lecture 8 reading, Foxglove: one launch file starts the L2 bridge, `foxglove_bridge` and a node that spawns traffic near the AV and publishes every car's true box and its constant-velocity prediction, 5 s ahead, with the speeds and a warning when a prediction comes within 2 m of the AV's. `foxglove/l8_layout.json` is the layout to import. See its README. |
| `gp1_starter/ads_pipeline` | **The GP1 starter.** The cumulative ADS pipeline that GP1 through GP4 build out. Copy it into your team repository; this copy carries a `COLCON_IGNORE` and is never built here. |
| `gp2_starter/ads_pipeline` | **The GP2 starter.** The files GP2 adds to your GP1 package: ground truth, the detector node, detector grading and an image tracker, with tests and tools. Copy them into your team's `ads_pipeline`; this copy carries a `COLCON_IGNORE` and is never built here. |

## Build

```bash
cd ~/enpm818z_ws
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` means edits to Python files take effect without a
rebuild. You still need to rebuild after touching `setup.py`, a launch file,
or a YAML.

## For GP1

Copy `gp1_starter/ads_pipeline/` into your private team repository and work
there; the GP1 page shows the exact commands. The task descriptions, the grading rubric and
**your team's assigned sensor rig** are on the course site.

What is given, and what is not:

| Given | Yours |
|---|---|
| `l2_carla_demo/conversions.py` — every CARLA measurement to its ROS 2 message, with the axis flip and the timestamps handled | `sensor_manager.py` — spawning, YAML parameters, TF, clean shutdown (9 TODOs) |
| `lidar_projection.py` — intrinsics, projection maths, depth colouring, overlay, and a driver that produces all three required images | `build_extrinsic()` — about six lines, and the whole point of Task 5 |
| `launch/sensors_launch.py` — complete, and the pattern to copy | `launch/record_launch.py` — Task 3 |
| `package.xml`, `setup.py` — dependencies declared | entry points, team details |
| `rviz/ads_pipeline.rviz` — displays laid out | connect the topics, Task 4 |

**The conversions are not duplicated.** `ads_pipeline` imports them from
`l2_carla_demo`, which is the same code the L2 lecture demo runs. Writing
your own is allowed but pointless, and it is not what GP1 grades.

That import is a build-time dependency: `l2_carla_demo` must be built in
the same workspace as `ads_pipeline`, which `colcon build` at the top of
this repo does. Your submission stays `ads_pipeline/` only; graders build
it next to this repo's copy of `l2_carla_demo`.

Nothing here is a solution. Every function that raises
`NotImplementedError` is deliberate.

## For GP2

GP2 adds files to the package your team built in GP1. Copy the contents of
`gp2_starter/ads_pipeline/` into your team's `ads_pipeline/`; the GP2 page
shows the exact commands, the two lines to add to `setup.py` and
`package.xml`, and the pip installs. Your team's two recorded drives are
given (download link on Canvas): GP2 needs no CARLA.

| Given | Yours |
|---|---|
| `tools/record_gp2.py`: how your drives were recorded (RGB, depth, semantic, and the actors' 3D boxes); you do not need to run it | `gt_boxes.py`: `project_points`, `box_from_corners`, `actor_pixels` (Task 1) |
| `gt_boxes.py`: frame loader, walker margin, near-plane clipping, the frame-level assembly | `detector_core.py`: `load_model`, `predict`; `detector_node.py`: `to_detection_array` (Task 2) |
| `detector_node.py`: the node; `launch/perception_launch.py`: every parameter a launch argument | `detection_eval.py`: `match_frame`, `pr_curve`, `average_precision`, `mean_ap` (Task 3) |
| `tools/run_detector.py`, `grade.py`, `check_coco.py`, `show_frame.py`, `replay_drive.py` (a drive played as the camera topic, for the live node) | `box_tracker.py`: `BoxKalman`, `iou_cost`, `associate`, `BoxTracker.step` (Task 4) |
| `test/test_gp2_*.py`: the checks for every function you write | the report (Task 5) |

## Before you start

Put your team's rig into your copy of `ads_pipeline/config/carla_config.yaml`. The values
in there now are placeholders. Every team has a different vehicle, spawn
point, sensor geometry and camera pitch, so every team's Task 5 extrinsic is
a different matrix, and graders recompute the expected one from your
assigned row.

Three things in the skeleton are worth reading before you write anything,
because they are what most often costs a week:

1. Synchronous mode goes in before any callback, not after.
2. Timestamps come off the measurement, never from the wall clock.
3. RViz2 shows nothing until you publish a TF tree.

## Checking your work

```bash
ros2 topic list                  # seven sensor topics
ros2 topic hz /carla/lidar/points
ros2 run tf2_tools view_frames   # TF tree exists and is connected
ros2 bag info <your-bag>         # all seven topics, >= 2 minutes
```
