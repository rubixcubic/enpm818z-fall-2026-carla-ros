#!/usr/bin/env python3
"""
lidar_projection.py  --  GP1 Task 5.

Project LiDAR points onto the camera image using the known extrinsic between
the two sensors.

Everything here is given to you EXCEPT build_extrinsic. That one function is
the whole point of the task, and it is about eight lines.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Given: the pinhole intrinsics.
# ---------------------------------------------------------------------------
def build_intrinsic_matrix(image_w: int, image_h: int,
                           fov_deg: float) -> np.ndarray:
    """The 3x3 intrinsic matrix K from image size and horizontal FOV."""
    focal = image_w / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array([
        [focal, 0.0,   image_w / 2.0],
        [0.0,   focal, image_h / 2.0],
        [0.0,   0.0,   1.0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# YOURS.
# ---------------------------------------------------------------------------
def build_extrinsic(lidar_loc, camera_loc, camera_pitch_deg=0.0) -> np.ndarray:
    """Return T_cam_lidar: LiDAR frame -> camera OPTICAL frame.

    Both locations are (x, y, z) in CARLA vehicle coordinates, taken from
    your team's assigned rig. camera_pitch_deg is that rig's camera pitch.

    Three things decide this matrix, and L2 covers all of them.

    1. AXIS CONVENTION. K assumes the optical convention: x right, y down,
       z forward along the optical axis. CARLA uses x forward, y right,
       z up. Same three physical directions, different names, so the axes
       have to be relabelled before K ever sees them. Derive the relabelling
       from your own convention rather than copying one; the signs follow
       from which way each axis points.

    2. MOUNT ROTATION. Most teams have a camera that is not level with the
       vehicle. The relabelling above is then not sufficient on its own:
       you must compose it with the mount rotation, and the order matters.
       Getting the order backwards gives an overlay that looks almost right
       and is wrong by a few degrees, which is L2's one-degree error made
       visible.

    3. TRANSLATION. Work out where the LiDAR origin sits AS SEEN FROM the
       camera, then express that in the optical frame.

    Two checks you can run before hunting for bugs:

      * If the camera sits 1.5 m ahead of the LiDAR, a point 10 m ahead of
        the LiDAR must come out at a depth of 8.5 m, not 11.5 m.
      * Project a point and project it back. The round trip should return
        the number you started with.

    Returns:
        (4, 4) float64 array.
    """
    raise NotImplementedError('GP1 Task 5: implement the extrinsic.')


# ---------------------------------------------------------------------------
# Given: the projection itself, the colouring and the overlay.
# ---------------------------------------------------------------------------
def project_lidar_to_image(points_xyz: np.ndarray, K: np.ndarray,
                           T_cam_lidar: np.ndarray,
                           image_w: int, image_h: int):
    """Project (N, 3) LiDAR points to pixels.

    Returns (pixels, depths): an (M, 2) int array of (u, v) and an (M,)
    array of depths, for the M points that land inside the image.
    """
    n = points_xyz.shape[0]
    homogeneous = np.hstack([points_xyz, np.ones((n, 1))]).T      # (4, N)

    in_camera = T_cam_lidar @ homogeneous                          # (4, N)

    # Keep only what is in front of the camera. Note this tests the THIRD
    # row, which is the optical z axis. If your extrinsic is wrong, this is
    # where the points vanish.
    in_front = in_camera[2, :] > 0.1
    in_camera = in_camera[:, in_front]
    if in_camera.shape[1] == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,))

    projected = K @ in_camera[:3, :]
    projected /= projected[2:3, :]

    u = projected[0, :].astype(int)
    v = projected[1, :].astype(int)
    depth = in_camera[2, :]

    on_image = (u >= 0) & (u < image_w) & (v >= 0) & (v < image_h)
    return np.stack([u[on_image], v[on_image]], axis=1), depth[on_image]


def colorize_depth(depths: np.ndarray, max_depth: float = 50.0) -> np.ndarray:
    """Map depths to BGR colours with a jet colormap."""
    normalized = np.clip(depths / max_depth, 0.0, 1.0)
    normalized = (normalized * 255).astype(np.uint8)
    # reshape, not squeeze: a single point must still be a (1, 3) row.
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET).reshape(-1, 3)


def overlay_projection(image_bgr: np.ndarray, pixels: np.ndarray,
                       colors: np.ndarray, dot_size: int = 3) -> np.ndarray:
    """Draw the projected points on a copy of the image."""
    out = image_bgr.copy()
    for (u, v), color in zip(pixels, colors):
        cv2.circle(out, (int(u), int(v)), dot_size,
                   [int(c) for c in color], -1)
    return out


# ---------------------------------------------------------------------------
# Deliverable driver. Produces the three overlays Task 5 asks for.
# ---------------------------------------------------------------------------
def run_three_cases(points_xyz, image_bgr, K, T_correct, out_dir='results'):
    """Save the correct overlay and the two deliberately broken ones.

    Task 5 wants all three, plus the point count for each, plus a paragraph
    on which failure would be more dangerous in a real vehicle.

    You supply T_correct. The two broken variants are derived from it here
    so that the comparison is honest: same scene, same frame, one thing
    changed at a time.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    h, w = image_bgr.shape[:2]

    T_no_rotation = T_correct.copy()
    T_no_rotation[:3, :3] = np.eye(3)

    T_wrong_sign = T_correct.copy()
    T_wrong_sign[1, :] *= -1.0

    counts = {}
    for name, T in (('correct', T_correct),
                    ('no_rotation', T_no_rotation),
                    ('wrong_sign', T_wrong_sign)):
        pixels, depths = project_lidar_to_image(points_xyz, K, T, w, h)
        counts[name] = len(pixels)
        overlay = image_bgr.copy()
        if len(pixels):
            overlay = overlay_projection(overlay, pixels,
                                         colorize_depth(depths))
        cv2.imwrite(os.path.join(out_dir, f'projection_{name}.png'), overlay)

    print('points landing on the image:')
    for name, n in counts.items():
        print(f'  {name:14} {n}')
    return counts
