"""GP3 Task 4 node: fused objects in, tracks on the map out. Given.

Reads /perception/fused_objects, steps your MapTracker with the time
between stamps, and publishes /perception/tracks (ads_msgs), the objects
GP4's planner and behavior node read, with their speeds.
"""
import rclpy
from ads_msgs.msg import TrackedObject, TrackedObjectArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray

from ads_pipeline.map_tracker import MapTracker


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def to_tracked_array(tracks, header):
    msg = TrackedObjectArray()
    msg.header = header
    for t in tracks:
        o = TrackedObject()
        o.id, o.class_id = str(t['track_id']), t['cls']
        o.position.x, o.position.y = float(t['xy'][0]), float(t['xy'][1])
        o.velocity.x, o.velocity.y = float(t['v'][0]), float(t['v'][1])
        o.covariance = [float(v) for v in t['P'].flatten()]
        msg.objects.append(o)
    return msg


class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')
        self.tracker = MapTracker(gate=self.declare_parameter('gate', 9.21).value)
        self.t_last = None
        self.pub = self.create_publisher(TrackedObjectArray, '/perception/tracks', 10)
        self.create_subscription(Detection3DArray, '/perception/fused_objects',
                                 self.on_objects, 10)

    def on_objects(self, msg):
        t = stamp_s(msg.header)
        dt = 0.0 if self.t_last is None else max(0.0, t - self.t_last)
        self.t_last = t if self.t_last is None else max(self.t_last, t)
        dets = [{'xy': (d.bbox.center.position.x, d.bbox.center.position.y),
                 'cls': d.results[0].hypothesis.class_id}
                for d in msg.detections if d.results]
        self.pub.publish(to_tracked_array(self.tracker.step(dets, dt), msg.header))


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(TrackerNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
