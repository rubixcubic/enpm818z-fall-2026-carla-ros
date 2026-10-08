# l5_seg_demo: semantic segmentation of the front camera, graded against CARLA

ENPM818Z Lecture 5, section "Segmentation". Semantic segmentation gives a
class to every pixel of the image: road, sidewalk, car, sky. On an AV a
trained network computes these classes from the camera image, every frame. In
CARLA a semantic segmentation camera writes the true class of every pixel. This
package does both on the L2 bridge's front camera and compares them pixel by
pixel, so you can see where the network is right and where it is wrong.

## Run it

The CARLA server must be running, and the L2 bridge must own the clock. The
network needs PyTorch (installed for L4) and the `transformers` package:

```bash
python3 -m pip install --user --break-system-packages transformers   # once
```

```bash
cd ~/enpm818z_ws
colcon build --symlink-install \
  --packages-select l2_carla_demo l5_tracking_demo l5_seg_demo
source install/setup.bash

ros2 launch l2_carla_demo demo.launch.py rviz:=false      # terminal 1
ros2 run l5_tracking_demo spawn_traffic                    # terminal 2: 40 vehicles
ros2 launch l5_seg_demo seg.launch.py evaluate:=true       # terminal 3
```

The first run downloads the network's weights (15 MB) into
`~/.cache/enpm818z-weights/huggingface`. RViz shows the camera with the
network's classes on top and a legend, the network's classes alone, and
CARLA's true classes. With `evaluate:=true` the log shows, every 10 s, the IoU
of each class and the mIoU so far (`out:=seg_report.json` saves them).

To save one moment as the picture above (and its numbers as JSON):

```bash
ros2 run l5_seg_demo snapshot --ros-args -p out:=seg.png -p skip:=20
```

`instances:=true` also runs YOLOv8s-seg, the instance model of the reading
slides, on the same image: one mask per object on `/l5/seg/instances` (weights
in `~/.cache/enpm818z-weights/yolov8s-seg.pt`, as in L4).

## The nodes

| Node | In | Out |
|---|---|---|
| `seg_node` | `/carla/ego_vehicle/rgb_front/image` | `/l5/seg/labels` (mono8, one class per pixel), `/l5/seg/classes` (colors), `/l5/seg/overlay` (camera plus classes and legend), `/l5/seg/legend` (class names and colors, JSON), `/l5/seg/instances` (with `instances:=true`) |
| `seg_truth` | CARLA | `/l5/seg/truth_tags` (mono8, CARLA's true class per pixel), `/l5/seg/truth` (colors) |
| `seg_eval` | `/l5/seg/labels`, `/l5/seg/truth_tags` | the log, and a JSON report with `out:=` |
| `snapshot` | the camera, `/l5/seg/labels`, `/l5/seg/truth_tags` | one PNG, its JSON and an `.npz` with the three images |

**`seg_node`, the network.** SegFormer-B0 trained on Cityscapes (Hugging Face
model `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`; E. Xie et al.,
"SegFormer: Simple and Efficient Design for Semantic Segmentation with
Transformers", NeurIPS 2021). A Transformer encoder, attention between image
patches as in L4's ViT, at four scales, then a small decoder. It gives 19
scores per pixel, one per class, at a quarter of the image's size; the node
resizes them to the full 1280 x 720 image and keeps the class with the highest
score. The image goes in at its own size. The weights were set by training, on
Cityscapes: photos of German streets, not CARLA. 3.72 million parameters
(counted from the loaded model).

**`seg_truth`, the answer key.** The bridge has no semantic camera, so this
node attaches one to the AV at the same place as the bridge's front camera,
with the same size and field of view. It finds the bridge's camera among the
AV's sensors and copies its pose relative to the AV. It logs the check: in
every run here, the two cameras were 0.00 mm apart, with 0.000 degrees between
their yaws and pitches. Both cameras fire on the same tick, so their images
carry the same time stamp. This node never ticks the simulator.

**`seg_eval`, the grade.** It pairs the network's answer and CARLA's truth of
the same tick (same time stamp) and counts, for each class c, over every
graded pixel of every paired frame:

* TP, true positive: truth c, network c
* FP, false positive: truth not c, network c
* FN, false negative: truth c, network not c

and IoU(c) = TP / (TP + FP + FN): the pixels both call c, divided by the
pixels either calls c. mIoU is the mean of IoU(c) over the classes CARLA showed
during the run. The counts are summed over all frames before dividing, as the
Cityscapes benchmark does. The log also says what the network called the
pixels of each class.

## Two class lists

The network knows the 19 Cityscapes classes. CARLA 0.9.16 has 29 tags
(`Docs/ref_sensors.md`, "Semantic segmentation camera"). CARLA's tags 1 to 19
are the 19 Cityscapes classes in the same order and with the same colors, so
the network's class t is CARLA's tag t + 1 (`classes.py`; the tests check the
names and colors against `cityscapesScripts`' `labels.py`).

| CARLA tag | How it is graded | Why |
|---|---|---|
| 1 to 19: road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person (CARLA: Pedestrian), rider, car, truck, bus, train, motorcycle, bicycle | as the same class | the same class in both lists |
| 24 RoadLine (lane marking) | as road | Cityscapes has no lane-marking class. Its road is "Part of ground on which cars usually drive, i.e. all lanes, all directions, all streets. Including the markings on the road." (cityscapes-dataset.com, class definitions) |
| 0 Unlabeled, 20 Static, 21 Dynamic, 22 Other, 23 Water, 25 Ground, 26 Bridge, 27 RailTrack, 28 GuardRail | not graded | no Cityscapes class; Cityscapes marks its own static, dynamic, ground, bridge, rail track and guard rail "ignoreInEval" |

**So this network cannot answer the slide's question about lane markings.**
The slide asks which pixels are road and which are lane marking. The network
was trained on labels where the markings are road: in every run below it
called 99.6 to 99.9 percent of CARLA's lane-marking pixels road. An AV that
needs lane lines uses a model trained with a lane class, or a separate lane
detector.

## Measured

Course laptop (RTX 4060 laptop GPU, 8 GB, plugged in), CARLA 0.9.16 in its
Apptainer container on the same GPU, Town10HD, the L2 bridge's front camera
(1280 x 720, 90 degree field of view, 20 Hz), 39 or 40 vehicles from
`spawn_traffic`, no pedestrians. Three runs of about 120 s each.

**Speed.** One 1280 x 720 image through the network, from the image in memory
to the classes on the CPU, median of 30 runs after 5 warm-up runs, nothing else
on the GPU: 34.2 ms in float32, 19.6 ms in float16 (`half: true`, the
default). The classes barely change: on the slides' saved CARLA frame the
mIoU differs by 0.0001. In the node, with CARLA rendering on the same GPU,
float16 takes 29 to 34 ms per image (the node's log, after the first 10 s).
YOLOv8s-seg (`instances:=true`) adds 11 ms per image.

**How many frames are graded.** The node segments 3 to 11 images per second
(the node's log, per 10 s). The limit is not the network: the bridge publishes
its 3.7 MB images with "best effort" delivery, and most do not arrive. A test
subscriber received 43 to 46 of the 160 images the bridge sent in 8 s. The truth images are sent with
reliable delivery (with best effort, 73 of 160 arrived, and few pairs formed).

**The grade.** IoU per class, summed over each run (`seg_eval`'s JSON). Runs 1
and 3 drove the same route (same spawn point and traffic seed), so their large
classes agree; run 2 drove elsewhere.

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| frames graded | 795 | 875 | 504 |
| **mIoU** | **0.408** | **0.339** | **0.390** |
| pixel accuracy | 0.936 | 0.904 | 0.937 |
| road (with lane markings) | 0.988 | 0.962 | 0.989 |
| building | 0.885 | 0.876 | 0.886 |
| sky | 0.847 | 0.858 | 0.847 |
| vegetation | 0.778 | 0.696 | 0.778 |
| sidewalk | 0.547 | 0.600 | 0.563 |
| car | 0.506 | 0.395 | 0.396 |
| pole | 0.130 | 0.259 | 0.128 |
| truck | 0.159 | 0.003 | 0.040 |
| traffic light | 0.048 | 0.122 | 0.049 |
| traffic sign | 0.004 | 0.225 | 0.002 |

(Run 3 also ran the rig mode below on the same GPU, so it graded fewer frames.
Classes with less than 0.1 percent of the pixels are left out of the table;
they are in the JSON.)

Read it this way, with run 3's "the network called it" line:

* **Big, flat classes are easy:** road, building, sky, vegetation all above 0.7.
* **Cars are found, but drawn too big.** The network calls 92 percent of the
  true car pixels car, yet the car IoU is only 0.40: the false positives are
  0.9 to 1.5 times the true positives (runs 1 to 3). The network's outlines are coarse (its scores
  are computed at a quarter of the image's size), and part of what it calls
  car is CARLA's truck (next point).
* **CARLA's "truck" is mostly "car" to the network** (61 percent of the truck
  pixels in run 3). Two class lists with the same names do not always draw the
  line in the same place.
* **Thin objects fail:** poles, traffic lights and signs are mostly called
  building (62, 80 and 92 percent in run 3). They are a few pixels wide, and
  the network's quarter-size scores blur them into what is behind them.
* **This is a domain gap.** The network was trained on real German streets and
  tested on CARLA's rendered town. Its published Cityscapes score is not what
  you get here, and this table is why AV teams test on their own data.

## With l5_bev_demo: a network's classes, from above

`l5_bev_demo`'s `semantic_bev` splats CARLA's true classes onto the grid seen
from above. `labels:=network` makes it splat the network's classes instead
(CARLA's true depth is still used):

```bash
ros2 launch l5_bev_demo bev.launch.py labels:=network
```

The roof cameras are not the bridge's camera (480 x 360, 100 degree field of
view, tilted 30 degrees down), so this starts a second `seg_node` with
`rig:=true` on the four roof cameras; it publishes `/l5/<camera>/semantic_net`.
Measured in run 3: 8.9 to 9.9 ms per 480 x 360 image (after the first 10 s).
In that run the simulation slowed to about 10 ticks per second (seg_truth sent
about 96 images every 10 s instead of 200), so the four roof cameras, at 10
images per simulated second each, sent about 190 images every 10 s; 80 to 110
of them reached the node, again because of best-effort delivery. So the network's grid is sparser than the truth's: a
camera whose classes and depth from the same tick did not both arrive is left
out of that grid. The lane markings are missing from it, as above.

To grade the network on the front roof camera (its truth publishes best
effort):

```bash
ros2 run l5_seg_demo seg_eval --ros-args -r __node:=seg_eval_rig \
  -p truth_topic:=/l5/front/semantic -p labels_topic:=/l5/front/semantic_net \
  -p reliable:=false
```

This grading was not measured: the run that tried it subscribed with reliable
delivery, which cannot receive best-effort images, and `reliable:=false` was
added after it (tested with a fake best-effort publisher, not in CARLA).

## License of the network

The weights are NVIDIA's, under the NVIDIA Source Code License for SegFormer
(the model card links it). Section 3.3: "The Work and any derivative works
thereof only may be used or intended for use non-commercially. ... As used
herein, "non-commercially" means for research or evaluation purposes only."
Course use is evaluation. The weights are downloaded on first run and are not
in this repository.

## Tasks

1. **The class map.** Open `classes.py`. Which CARLA tags does the grade
   ignore, and what does it do with lane markings? Why can this network not
   find lane lines, whatever its accuracy?
2. **IoU by hand.** `test/test_seg_eval.py` grades a 2 x 4 image by hand. Work
   out road's and car's IoU on paper, then run the test.
3. **Summed, not averaged.** One frame has 1 car pixel, missed; the next has 99,
   all found. What is the car IoU summed over both frames, and what would the
   mean of the two frames' IoUs be? Which one does Cityscapes report?
4. **Thin things.** Run with `evaluate:=true` and read the "the network called
   it" column for pole and traffic light. Look at the snapshot: where do those
   pixels go?
5. **float16.** Set `half: false` in `config/seg.yaml`. How much slower is each
   image (the log), and does the mIoU change?
6. **From above.** Run `bev.launch.py labels:=network` and compare the
   semantic grid with the default. What is missing, and why?
