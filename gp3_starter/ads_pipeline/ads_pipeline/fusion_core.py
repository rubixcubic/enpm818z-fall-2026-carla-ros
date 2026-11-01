"""GP3 Task 3: one 2D box plus the LiDAR gives one 3D point on the map.

Frustum association: keep the points whose pixel falls in the box, group
them with DBSCAN, keep the cluster nearest the camera.

Axes. /carla/lidar/points is in ROS axes (x forward, y LEFT, z up): GP1's
lidar_to_msg flips CARLA's y. GP1's build_extrinsic works in CARLA's axes
(y RIGHT). project_keep_points flips y for the projection only, so every
point keeps its ROS coordinates and gets its pixel.
"""
import math

import numpy as np
from scipy.spatial import cKDTree

ROS_TO_CARLA = np.array([1.0, -1.0, 1.0])


def project_keep_points(points_ros, K, T_cam_lidar, image_w, image_h):
    """Given. GP1's project_lidar_to_image, but it also returns the points.

    Returns (kept, uv, depth): the (M, 3) points in ROS axes that land in
    the image, their (M, 2) pixels and their (M,) depths along the camera's
    axis.
    """
    pts_carla = points_ros * ROS_TO_CARLA
    hom = np.hstack([pts_carla, np.ones((len(pts_carla), 1))]).T
    cam = T_cam_lidar @ hom
    front = cam[2, :] > 0.1
    proj = K @ cam[:3, front]
    proj /= proj[2:3, :]
    u, v = proj[0, :].astype(int), proj[1, :].astype(int)
    on = (u >= 0) & (u < image_w) & (v >= 0) & (v < image_h)
    kept = points_ros[front][on]
    return kept, np.stack([u[on], v[on]], axis=1), cam[2, front][on]


def ground_point(box, K, T_cam_lidar, lidar_height, max_range):
    """Given. Where a person's feet touch the road, in the LiDAR frame (ROS
    axes), from the bottom center of their box; None when that pixel looks
    above the horizon or the point is beyond max_range.

    CARLA's ray-cast LiDAR returns almost no points on people (it hits a
    simplified collision shape), so a pedestrian box is placed with the
    camera alone: the ray through the pixel meets the flat road, which sits
    lidar_height below the LiDAR.
    """
    u, v = (box[0] + box[2]) / 2.0, box[3]
    T_inv = np.linalg.inv(T_cam_lidar)
    origin = T_inv[:3, 3]                                   # camera center, CARLA axes
    ray = T_inv[:3, :3] @ np.linalg.solve(K, np.array([u, v, 1.0]))
    if ray[2] >= -1e-6:                                     # level or upward: no road
        return None
    p = (origin + (-lidar_height - origin[2]) / ray[2] * ray) * ROS_TO_CARLA
    return None if np.hypot(p[0], p[1]) > max_range else p


def remove_ground(points, lidar_height, clearance=0.3):
    """Given. Drop the points within clearance (m) of the road, which sits
    lidar_height below the LiDAR. Without it, the road in front of a
    person's feet, inside the person's box, can be the nearest cluster."""
    return points[points[:, 2] > -lidar_height + clearance]


def dbscan(points, eps, min_pts):
    """Given. DBSCAN: labels per point, -1 for noise."""
    tree = cKDTree(points)
    nbrs = tree.query_ball_point(points, eps)
    labels = np.full(len(points), -1)
    cluster = 0
    for i in range(len(points)):
        if labels[i] != -1 or len(nbrs[i]) < min_pts:
            continue
        labels[i] = cluster
        queue = list(nbrs[i])
        while queue:
            j = queue.pop()
            if labels[j] == -1:
                labels[j] = cluster
                if len(nbrs[j]) >= min_pts:
                    queue.extend(nbrs[j])
        cluster += 1
    return labels


def associate_lidar_to_detection(box, pts_lidar, uv, depth, min_range,
                                 max_range, eps, min_pts):
    """YOURS. Frustum association. Keep the points whose pixel (uv) is inside box,
    with depth > 0 and a range from the LiDAR between min_range and max_range.
    Fewer than min_pts points: return None. Group them with dbscan(eps,
    min_pts), and return the centroid of the cluster NEAREST the LiDAR (by
    the median range of its points). No cluster: return None.
    """
    raise NotImplementedError('GP3 Task 3: associate_lidar_to_detection')


def to_map(p_lidar, T_ego_lidar, pose):
    """YOURS. Move p_lidar into the vehicle frame with T_ego_lidar (4 x 4), then into
    the map frame with the car's pose (x, y, theta). Returns a (3,) array;
    z stays the height in the vehicle frame.
    """
    raise NotImplementedError('GP3 Task 3: to_map')
