"""Small helpers shared by the nodes: time stamps, detection messages, CARLA truth.

Detections travel as vision_msgs/Detection3DArray, a standard message, so the
package needs no custom .msg file:
    bbox.center.position    the detection's position in the fixed frame (map)
    bbox.size               its extent along x, y, z (display only)
    results[0].hypothesis   class label and score
    results[0].pose.covariance[0], [7]   the variances of x and y (R)
"""

from __future__ import annotations

import math

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from l5_tracking_demo.tracker_core import Detection


def spin(node) -> None:
    """rclpy.spin, quiet on Ctrl+C.

    Jazzy raises ExternalShutdownException from spin() on SIGINT, and when the
    signal lands while spin() is building its wait set, an RCLError about an
    invalid context. Both mean "shutting down"; anything else is a real error.
    """
    import rclpy
    from rclpy.executors import ExternalShutdownException
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if node.context.ok():
            raise


def stamp_to_sec(stamp: TimeMsg) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def sec_to_stamp(t: float) -> TimeMsg:
    msg = TimeMsg()
    msg.sec = int(t)
    msg.nanosec = int(round((t - int(t)) * 1e9)) % 1_000_000_000
    return msg


def time_key(t: float) -> int:
    """Simulation time in whole milliseconds: a dictionary key that survives float noise."""
    return int(round(t * 1000.0))


def make_detection(x: float, y: float, z: float, size: tuple, label: str,
                   sigma: float, yaw: float = 0.0) -> Detection3D:
    det = Detection3D()
    det.bbox.center.position.x = float(x)
    det.bbox.center.position.y = float(y)
    det.bbox.center.position.z = float(z)
    det.bbox.center.orientation.z = math.sin(yaw / 2.0)
    det.bbox.center.orientation.w = math.cos(yaw / 2.0)
    det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = (float(s) for s in size)
    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = label
    hyp.hypothesis.score = 1.0
    hyp.pose.pose.position = det.bbox.center.position
    cov = [0.0] * 36
    cov[0] = cov[7] = sigma * sigma
    hyp.pose.covariance = cov
    det.results.append(hyp)
    return det


def detections_from_msg(msg: Detection3DArray, default_sigma: float) -> list[Detection]:
    out = []
    for det in msg.detections:
        p = det.bbox.center.position
        R = default_sigma ** 2 * np.eye(2)
        label = "object"
        if det.results:
            hyp = det.results[0]
            label = hyp.hypothesis.class_id or label
            c = hyp.pose.covariance
            if c[0] > 0.0 and c[7] > 0.0:
                R = np.array([[c[0], c[1]], [c[6], c[7]]])
        out.append(Detection(np.array([p.x, p.y]), R, label,
                             (det.bbox.size.x, det.bbox.size.y, det.bbox.size.z)))
    return out


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# --------------------------------------------------------------------------
# CARLA truth, in the ROS map frame
# --------------------------------------------------------------------------
def actor_kind(type_id: str) -> str | None:
    if type_id.startswith("vehicle."):
        return "vehicle"
    if type_id.startswith("walker.pedestrian"):
        return "pedestrian"
    return None


def truth_from_snapshot(snapshot, kinds: dict[int, str]) -> list[tuple]:
    """(id, x, y, vx, vy, yaw, kind) per actor, in the ROS map frame.

    CARLA is left-handed with y to the right; ROS has y to the left. So y, the
    y velocity and the yaw all change sign (the L2 axis trap). The actor's
    location is the middle of its bounding box on the ground.
    """
    out = []
    for a in snapshot:
        kind = kinds.get(a.id)
        if kind is None:
            continue
        tf = a.get_transform()
        v = a.get_velocity()
        out.append((a.id, tf.location.x, -tf.location.y, v.x, -v.y,
                    -math.radians(tf.rotation.yaw), kind))
    return out
