"""The bird's-eye-view grid shared by all three nodes.

Slide: 'The BEV grid: from meters to a cell'.

A BEV grid is a square of cells laid on the ground around the AV, in the
ego_vehicle frame (x forward, y left, z up, origin on the ground under the
vehicle). Every node in this package uses the same grid, so the three images
line up cell for cell and can be compared directly.

    i = floor((x + half_extent) / resolution)     cell index along x, forward
    j = floor((y + half_extent) / resolution)     cell index along y, left

For display the grid is drawn front up and left on the left, the way you would
look down on the AV from above:

    image row    = n - 1 - i      (far ahead is at the top)
    image column = n - 1 - j      (the AV's left is on the left)
"""

import math

import numpy as np
from sensor_msgs.msg import Image


class BevGrid:
    def __init__(self, half_extent: float, resolution: float) -> None:
        self.half_extent = float(half_extent)
        self.resolution = float(resolution)
        self.n = int(round(2.0 * self.half_extent / self.resolution))

        # Ground position (x, y) of every cell center, in image layout. Used by
        # the IPM node, which asks "what does the camera see at this cell?".
        centers = (np.arange(self.n) + 0.5) * self.resolution - self.half_extent
        x_of_row = centers[::-1]          # row 0 is the far front
        y_of_col = centers[::-1]          # column 0 is the far left
        self.cell_x, self.cell_y = np.meshgrid(x_of_row, y_of_col, indexing="ij")

    def index(self, x: np.ndarray, y: np.ndarray):
        """Cell indices (i, j) of ground points, and which points fall inside."""
        i = np.floor((x + self.half_extent) / self.resolution).astype(np.int64)
        j = np.floor((y + self.half_extent) / self.resolution).astype(np.int64)
        inside = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        return i, j, inside

    def to_image_rc(self, i: np.ndarray, j: np.ndarray):
        """Cell indices to image row and column (front up, left on the left)."""
        return self.n - 1 - i, self.n - 1 - j

    def draw_ego(self, image: np.ndarray, length: float = 4.8,
                 width: float = 2.2) -> None:
        """Outline the AV's footprint, centered on the grid, in white."""
        half_l = int(round(length / 2.0 / self.resolution))
        half_w = int(round(width / 2.0 / self.resolution))
        c = self.n // 2
        image[c - half_l:c + half_l, c - half_w] = 255
        image[c - half_l:c + half_l, c + half_w] = 255
        image[c - half_l, c - half_w:c + half_w] = 255
        image[c + half_l, c - half_w:c + half_w] = 255


def matrix_from_transform(tf) -> np.ndarray:
    """geometry_msgs/Transform to a 4 x 4 homogeneous matrix."""
    q = tf.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = [tf.translation.x, tf.translation.y, tf.translation.z]
    return m


def image_msg(bgr: np.ndarray, header) -> Image:
    """A bgr8 numpy image (rows x cols x 3) as sensor_msgs/Image."""
    msg = Image()
    msg.header = header
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = 3 * msg.width
    msg.data = np.ascontiguousarray(bgr, dtype=np.uint8).tobytes()
    return msg


def focal_length(width: int, fov_deg: float) -> float:
    """Pinhole focal length in pixels (L2): f = (W / 2) / tan(FOV / 2)."""
    return width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
