"""The class mapping between the network (Cityscapes) and CARLA."""

import numpy as np

from l5_seg_demo.classes import (CARLA_TAGS, CITYSCAPES, COLORS_RGB, IGNORE, ROADLINE,
                                 network_to_tags, tags_to_graded)

# cityscapesScripts, helpers/labels.py: (name, trainId, color) of the 19 graded classes.
CITYSCAPES_LABELS = [
    ("road", 0, (128, 64, 128)), ("sidewalk", 1, (244, 35, 232)),
    ("building", 2, (70, 70, 70)), ("wall", 3, (102, 102, 156)),
    ("fence", 4, (190, 153, 153)), ("pole", 5, (153, 153, 153)),
    ("traffic light", 6, (250, 170, 30)), ("traffic sign", 7, (220, 220, 0)),
    ("vegetation", 8, (107, 142, 35)), ("terrain", 9, (152, 251, 152)),
    ("sky", 10, (70, 130, 180)), ("person", 11, (220, 20, 60)),
    ("rider", 12, (255, 0, 0)), ("car", 13, (0, 0, 142)), ("truck", 14, (0, 0, 70)),
    ("bus", 15, (0, 60, 100)), ("train", 16, (0, 80, 100)),
    ("motorcycle", 17, (0, 0, 230)), ("bicycle", 18, (119, 11, 32)),
]


def test_network_names_are_cityscapes_train_ids():
    assert [n for n, _, _ in CITYSCAPES_LABELS] == list(CITYSCAPES)


def test_train_id_t_is_carla_tag_t_plus_1_with_the_same_color():
    for name, t, color in CITYSCAPES_LABELS:
        tag = int(network_to_tags(np.array([t], dtype=np.uint8))[0])
        assert tag == t + 1
        assert CARLA_TAGS[tag][0] == name
        assert tuple(COLORS_RGB[tag]) == color


def test_round_trip_network_to_tags_to_graded():
    t = np.arange(19, dtype=np.uint8)
    assert np.array_equal(tags_to_graded(network_to_tags(t)), t)


def test_lane_marking_is_graded_as_road():
    assert tags_to_graded(np.array([ROADLINE], dtype=np.uint8))[0] == 0


def test_carla_only_tags_are_not_graded():
    for tag in (0, 20, 21, 22, 23, 25, 26, 27, 28):
        assert tags_to_graded(np.array([tag], dtype=np.uint8))[0] == IGNORE


def test_every_carla_tag_has_a_color():
    assert len(CARLA_TAGS) == 29
    for tag, (_, rgb) in CARLA_TAGS.items():
        assert tuple(COLORS_RGB[tag]) == rgb
