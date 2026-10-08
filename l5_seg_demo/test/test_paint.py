"""paint(): LiDAR points take the class of the pixel they project to."""

import numpy as np

from l5_seg_demo.seg_lidar_node import paint, quat_to_matrix

# A 100 x 80 image, focal length 50 px, principal point at its center (chosen).
K = np.array([[50.0, 0.0, 50.0], [0.0, 50.0, 40.0], [0.0, 0.0, 1.0]])
LABELS = np.zeros((80, 100), dtype=np.uint8)
LABELS[:, 50:] = 14          # right half: car
LABELS[:, :50] = 1           # left half: road

# The bridge's optical frame: quaternion (-0.5, 0.5, -0.5, 0.5) from a body frame
# (x forward, y left, z up) to x right, y down, z forward.
ROT = quat_to_matrix(-0.5, 0.5, -0.5, 0.5).T
NO_SHIFT = np.zeros(3)


def test_optical_rotation():
    # 10 m ahead in the body frame is 10 m along the optical axis.
    assert np.allclose(ROT @ [10.0, 0.0, 0.0], [0.0, 0.0, 10.0])
    # 1 m to the left is 1 m in -x (left in the image).
    assert np.allclose(ROT @ [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0])


def test_left_and_right_take_their_pixels_class():
    pts = np.array([[10.0, 2.0, 0.0],     # ahead and left: u = 50 - 50 * 2 / 10 = 40
                    [10.0, -2.0, 0.0]])   # ahead and right: u = 60
    tags, seen = paint(pts, ROT, NO_SHIFT, K, LABELS)
    assert seen.all()
    assert list(tags) == [1, 14]


def test_behind_and_outside_are_not_seen():
    pts = np.array([[-5.0, 0.0, 0.0],     # behind the camera
                    [10.0, 30.0, 0.0],    # far left: u = 50 - 150 < 0
                    [0.2, 0.0, 0.0]])     # 0.2 m ahead, closer than min_depth
    tags, seen = paint(pts, ROT, NO_SHIFT, K, LABELS)
    assert not seen.any()
    assert (tags == 255).all()
