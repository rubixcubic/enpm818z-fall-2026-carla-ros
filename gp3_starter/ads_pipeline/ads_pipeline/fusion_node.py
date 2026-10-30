"""GP3 Task 3 node: GP2's boxes plus the LiDAR give objects on the map. Given.

Reads your GP1 YAML (rig_config) for the camera and the mounts, so K and
the extrinsic come from GP1's build_intrinsic_matrix and build_extrinsic.
Each box is paired with the LiDAR sweep of the SAME stamp (the detector
keeps the image's stamp). The car's pose comes from pose_topic:
/localization/pose (your EKF) or /ground_truth/ego (the truth, for the
error budget in the report). The array is published even when empty.

Yours (fusion_core.py): associate_lidar_to_detection and to_map.
"""
import bisect
import math
from collections import OrderedDict

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import (Detection2DArray, Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)

from ads_pipeline import fusion_core
from ads_pipeline.lidar_projection import build_extrinsic, build_intrinsic_matrix


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def load_rig(path):
    """K, T_cam_lidar (CARLA axes), T_ego_lidar (ROS axes), image size."""
    p = yaml.safe_load(open(path))['sensor_manager']['ros__parameters']
    cam, m = p['camera'], p['mounts']
    w, h = int(cam['width']), int(cam['height'])
    K = build_intrinsic_matrix(w, h, float(cam['fov']))
    lid, c = m['lidar_top'], m['camera_rgb_front']
    T_cam_lidar = build_extrinsic((lid['x'], lid['y'], lid['z']),
                                  (c['x'], c['y'], c['z']), c['pitch'])
    T_ego_lidar = np.eye(4)
    T_ego_lidar[:3, 3] = (lid['x'], -lid['y'], lid['z'])   # level LiDAR
    return K, T_cam_lidar, T_ego_lidar, w, h


def box_and_hypothesis(d):
    """[u1, v1, u2, v2] and the class hypothesis of one Detection2D, or None
    when the detector gave it no hypothesis."""
    if not d.results:
        return None
    cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
    return ([cx - d.bbox.size_x / 2, cy - d.bbox.size_y / 2,
             cx + d.bbox.size_x / 2, cy + d.bbox.size_y / 2], d.results[0].hypothesis)


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        def p(name, default):
            return self.declare_parameter(name, default).value

        self.K, self.T_cam_lidar, self.T_ego_lidar, self.w, self.h = load_rig(
            p('rig_config', 'config/carla_config.yaml'))
        self.min_range = p('min_range', 1.0)
        self.max_range = p('max_range', 60.0)
        self.eps = p('dbscan_eps', 0.7)
        self.min_pts = p('dbscan_min_pts', 5)
        self.ground_clearance = p('ground_clearance', 0.3)
        pose_topic = p('pose_topic', '/localization/pose')

        self.clouds = OrderedDict()          # stamp -> (N, 3) points, ROS axes
        self.pose_t, self.poses = [], []     # sorted stamps, (x, y, theta)
        self.pub = self.create_publisher(Detection3DArray,
                                         '/perception/fused_objects', 10)
        self.create_subscription(PointCloud2, '/carla/lidar/points',
                                 self.on_cloud, 5)
        if pose_topic.startswith('/ground_truth'):
            self.create_subscription(Odometry, pose_topic,
                                     lambda m: self.on_pose(m.header, m.pose.pose), 50)
        else:
            self.create_subscription(PoseStamped, pose_topic,
                                     lambda m: self.on_pose(m.header, m.pose), 50)
        self.create_subscription(Detection2DArray, '/perception/detections',
                                 self.on_detections, 10)

    def on_cloud(self, msg):
        pts = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'),
                                    skip_nans=True).astype(float)
        self.clouds[round(stamp_s(msg.header), 3)] = pts
        while len(self.clouds) > 40:
            self.clouds.popitem(last=False)

    def on_pose(self, header, pose):
        t = stamp_s(header)
        if self.pose_t and t <= self.pose_t[-1]:
            return
        self.pose_t.append(t)
        self.poses.append((pose.position.x, pose.position.y,
                           yaw_of(pose.orientation)))
        if len(self.pose_t) > 2000:
            del self.pose_t[:1000], self.poses[:1000]

    def pose_at(self, t):
        i = bisect.bisect_right(self.pose_t, t + 1e-3) - 1
        return None if i < 0 else self.poses[i]

    def cloud_at(self, t):
        key = round(t, 3)
        if key in self.clouds:
            return self.clouds[key]
        if not self.clouds:
            return None
        near = min(self.clouds, key=lambda k: abs(k - t))
        if abs(near - t) > 0.05:
            return None
        self.get_logger().warn(f'no sweep at {t:.3f}; using {near:.3f}',
                               throttle_duration_sec=5.0)
        return self.clouds[near]

    def on_detections(self, msg):
        out = Detection3DArray()
        out.header.stamp, out.header.frame_id = msg.header.stamp, 'map'
        t = stamp_s(msg.header)
        cloud, pose = self.cloud_at(t), self.pose_at(t)
        if cloud is not None and pose is not None and msg.detections:
            above = fusion_core.remove_ground(cloud, self.T_ego_lidar[2, 3],
                                              self.ground_clearance)
            kept, uv, depth = fusion_core.project_keep_points(
                above, self.K, self.T_cam_lidar, self.w, self.h)
            for d in msg.detections:
                bh = box_and_hypothesis(d)
                if bh is None:
                    continue
                box, hypothesis = bh
                c = fusion_core.associate_lidar_to_detection(
                    box, kept, uv, depth, self.min_range, self.max_range,
                    self.eps, self.min_pts)
                if c is None:
                    continue
                p = fusion_core.to_map(c, self.T_ego_lidar, pose)
                o = Detection3D()
                o.header = out.header
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis = hypothesis
                o.results.append(hyp)
                (o.bbox.center.position.x, o.bbox.center.position.y,
                 o.bbox.center.position.z) = (float(v) for v in p)
                out.detections.append(o)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(FusionNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
