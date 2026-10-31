"""Ground truth for GP3: where the car and every object really were.

Given. Call GroundTruth.publish(stamp) in sensor_manager right after
world.tick(), so its stamps match the sensors' stamps. No filter or fusion
node may subscribe to these topics: they exist to grade the estimates.
"""
import math

from nav_msgs.msg import Odometry
from std_msgs.msg import String
from vision_msgs.msg import (Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)

from l2_carla_demo.conversions import header, quaternion_from_euler


# CARLA blueprint pattern -> GP2's class names (detector_node.COLORS).
CLASS_OF = {'vehicle.*': 'vehicle', 'walker.pedestrian.*': 'pedestrian'}


def light_message(state, at_light, dist):
    """'<state> <1 or 0> <signed distance to the stop line, m>'."""
    return f'{state} {int(bool(at_light))} {dist:.2f}'


def parse_light(text):
    state, at, dist = text.split()
    return state, at == '1', float(dist)


def ahead_distance(ego_xy, ego_yaw, point_xy):
    """How far ahead of the car (along its heading) a point is, in meters;
    negative once the car has passed it."""
    return round((point_xy[0] - ego_xy[0]) * math.cos(ego_yaw)
                 + (point_xy[1] - ego_xy[1]) * math.sin(ego_yaw), 6)


def carla_to_map_xy(x, y):
    """CARLA's y points right; the map frame's y points left."""
    return x, -y


def carla_to_map_yaw(yaw_deg):
    """CARLA yaw is in degrees and turns the other way."""
    return -math.radians(yaw_deg)


def ego_odometry(transform, velocity, stamp):
    """The car's true pose and velocity in the map frame."""
    msg = Odometry()
    msg.header = header('map', stamp)
    msg.child_frame_id = 'ego_vehicle'
    x, y = carla_to_map_xy(transform.location.x, transform.location.y)
    msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
    msg.pose.pose.position.z = transform.location.z
    msg.pose.pose.orientation = quaternion_from_euler(
        0.0, 0.0, carla_to_map_yaw(transform.rotation.yaw))
    msg.twist.twist.linear.x, msg.twist.twist.linear.y = carla_to_map_xy(
        velocity.x, velocity.y)
    return msg


def box_detection(actor_id, cls, center_map, size, yaw_map):
    """One true box: CARLA ID in .id, class in results[0]."""
    d = Detection3D()
    d.id = str(actor_id)
    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = cls
    hyp.hypothesis.score = 1.0
    d.results.append(hyp)
    (d.bbox.center.position.x, d.bbox.center.position.y,
     d.bbox.center.position.z) = center_map
    d.bbox.center.orientation = quaternion_from_euler(0.0, 0.0, yaw_map)
    d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = size
    return d


class GroundTruth:
    """Publishes /ground_truth/ego and /ground_truth/objects."""

    def __init__(self, node, world, ego, radius=80.0, gp4=False):
        self.world, self.ego, self.radius = world, ego, radius
        self.node, self.actors = node, []
        if gp4:
            # GP4: collisions and the traffic light that affects the car.
            self.pub_col = node.create_publisher(String, '/ground_truth/collision', 10)
            self.pub_light = node.create_publisher(String, '/ground_truth/traffic_light', 10)
            bp = world.get_blueprint_library().find('sensor.other.collision')
            col = world.spawn_actor(bp, __import__('carla').Transform(), attach_to=ego)
            col.listen(self._on_collision)
            self.actors.append(col)
        self.gp4 = gp4
        self.pub_ego = node.create_publisher(Odometry, '/ground_truth/ego', 10)
        self.pub_obj = node.create_publisher(
            Detection3DArray, '/ground_truth/objects', 10)

    def publish(self, stamp):
        tf = self.ego.get_transform()
        self.pub_ego.publish(ego_odometry(tf, self.ego.get_velocity(), stamp))
        arr = Detection3DArray()
        arr.header = header('map', stamp)
        for pattern, cls in CLASS_OF.items():
            for a in self.world.get_actors().filter(pattern):
                if a.id == self.ego.id:
                    continue
                atf = a.get_transform()
                if atf.location.distance(tf.location) > self.radius:
                    continue
                c = atf.transform(a.bounding_box.location)
                x, y = carla_to_map_xy(c.x, c.y)
                e = a.bounding_box.extent
                arr.detections.append(box_detection(
                    a.id, cls, (x, y, c.z), (2 * e.x, 2 * e.y, 2 * e.z),
                    carla_to_map_yaw(atf.rotation.yaw)))
        self.pub_obj.publish(arr)
        if self.gp4:
            self._publish_light(tf, stamp)

    def _on_collision(self, event):
        msg = String()
        msg.data = f'{event.timestamp:.3f} {event.other_actor.type_id}'
        self.pub_col.publish(msg)

    def _publish_light(self, tf, stamp):
        state, at, dist = str(self.ego.get_traffic_light_state()), False, float('nan')
        light = self.ego.get_traffic_light()
        if light is not None:
            at = self.ego.is_at_traffic_light()
            ego_xy = carla_to_map_xy(tf.location.x, tf.location.y)
            yaw = carla_to_map_yaw(tf.rotation.yaw)
            stops = [carla_to_map_xy(w.transform.location.x, w.transform.location.y)
                     for w in light.get_stop_waypoints()]
            if stops:
                dist = min((ahead_distance(ego_xy, yaw, p) for p in stops), key=abs)
        msg = String()
        msg.data = light_message(state, at, dist)
        self.pub_light.publish(msg)
