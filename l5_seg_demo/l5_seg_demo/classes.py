"""The two class lists, the network's and CARLA's, and how one maps onto the other.

The network: SegFormer-B0 trained on Cityscapes predicts 19 classes, numbered
0 to 18 (Cityscapes calls the numbers "trainIds"; the names are the model's
id2label, nvidia/segformer-b0-finetuned-cityscapes-1024-1024, config.json).

CARLA 0.9.16: the semantic segmentation camera writes one tag per pixel, 0 to
28 (Docs/ref_sensors.md, "Semantic segmentation camera"). Tags 1 to 19 are the
19 Cityscapes classes, in the same order and with the same colors:

    Cityscapes trainId t   <->   CARLA tag t + 1

The other CARLA tags have no Cityscapes class:

    24 RoadLine   Cityscapes has no lane-marking class. Its road is "Part of
                  ground on which cars usually drive, i.e. all lanes, all
                  directions, all streets. Including the markings on the road."
                  (cityscapes-dataset.com, class definitions). So for grading,
                  CARLA's lane markings count as
                  road, and the network cannot tell you where a lane line is.
    0 Unlabeled, 20 Static, 21 Dynamic, 22 Other, 23 Water, 25 Ground,
    26 Bridge, 27 RailTrack, 28 GuardRail
                  Cityscapes has static, dynamic, ground, bridge, rail track
                  and guard rail too, but marks them "ignoreInEval" (trainId
                  255, cityscapesScripts helpers/labels.py). These pixels are
                  not graded.

Everything on the wire uses CARLA's tags, so the network's answer and CARLA's
truth can be drawn with one palette and compared pixel by pixel.
"""

from __future__ import annotations

import numpy as np

IGNORE = 255

# Cityscapes trainIds 0 to 18: the network's output.
CITYSCAPES = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)

# CARLA 0.9.16 tags: name and color (RGB), from the sensor reference table.
CARLA_TAGS = {
    0: ("unlabeled", (0, 0, 0)),
    1: ("road", (128, 64, 128)),
    2: ("sidewalk", (244, 35, 232)),
    3: ("building", (70, 70, 70)),
    4: ("wall", (102, 102, 156)),
    5: ("fence", (190, 153, 153)),
    6: ("pole", (153, 153, 153)),
    7: ("traffic light", (250, 170, 30)),
    8: ("traffic sign", (220, 220, 0)),
    9: ("vegetation", (107, 142, 35)),
    10: ("terrain", (152, 251, 152)),
    11: ("sky", (70, 130, 180)),
    12: ("person", (220, 20, 60)),        # CARLA: Pedestrian
    13: ("rider", (255, 0, 0)),
    14: ("car", (0, 0, 142)),
    15: ("truck", (0, 0, 70)),
    16: ("bus", (0, 60, 100)),
    17: ("train", (0, 80, 100)),
    18: ("motorcycle", (0, 0, 230)),
    19: ("bicycle", (119, 11, 32)),
    20: ("static", (110, 190, 160)),
    21: ("dynamic", (170, 120, 50)),
    22: ("other", (55, 90, 80)),
    23: ("water", (45, 60, 150)),
    24: ("lane marking", (157, 234, 50)),  # CARLA: RoadLine
    25: ("ground", (81, 0, 81)),
    26: ("bridge", (150, 100, 100)),
    27: ("rail track", (230, 150, 140)),
    28: ("guard rail", (180, 165, 180)),
}
ROAD, ROADLINE = 1, 24

# The network's trainId t -> CARLA tag t + 1.
TRAIN_TO_CARLA = np.arange(1, len(CITYSCAPES) + 1, dtype=np.uint8)

# CARLA tag -> the trainId it is graded as, or IGNORE.
CARLA_TO_TRAIN = np.full(256, IGNORE, dtype=np.uint8)
CARLA_TO_TRAIN[1:20] = np.arange(len(CITYSCAPES), dtype=np.uint8)
CARLA_TO_TRAIN[ROADLINE] = 0                       # lane markings count as road

# Lookup table: CARLA tag -> RGB color. Unknown values draw black.
COLORS_RGB = np.zeros((256, 3), dtype=np.uint8)
for _tag, (_name, _rgb) in CARLA_TAGS.items():
    COLORS_RGB[_tag] = _rgb
COLORS_BGR = np.ascontiguousarray(COLORS_RGB[:, ::-1])


def tag_name(tag: int) -> str:
    return CARLA_TAGS.get(int(tag), (f"tag {tag}", None))[0]


def network_to_tags(train_ids: np.ndarray) -> np.ndarray:
    """The network's trainIds (0 to 18) as CARLA tags (1 to 19)."""
    return TRAIN_TO_CARLA[train_ids]


def tags_to_graded(tags: np.ndarray) -> np.ndarray:
    """CARLA tags as the trainIds they are graded as; IGNORE where not graded."""
    return CARLA_TO_TRAIN[tags]
