"""Ground truth for GP3: where the car and every object really were.

Given. Call GroundTruth.publish(stamp) in sensor_manager right after
world.tick(), so its stamps match the sensors' stamps. No filter or fusion
node may subscribe to these topics: they exist to grade the estimates.
GP4 adds /carla/speedometer, a sensor (a car's wheel-speed reading), which
the controller may use, and the traffic light ahead of the car.
"""
import math

from geometry_msgs.msg import TwistStamped
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


def light_ahead(ego_xy, ego_yaw, stops, max_ahead=50.0, behind=5.0, lane_half=2.0,
                max_yaw=0.5):
    """(key, signed distance) of the stop line the car drives toward, or None.
    stops: [(key, (x, y), yaw)], map frame. A stop line counts when it faces
    the car's way (within max_yaw rad), lies in the car's lane (within
    lane_half m to the side) and is at most max_ahead m ahead, or at most
    behind m behind, so the crossing itself is seen."""
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    best = None
    for key, (x, y), yaw in stops:
        if abs((yaw - ego_yaw + math.pi) % (2 * math.pi) - math.pi) > max_yaw:
            continue
        dx, dy = x - ego_xy[0], y - ego_xy[1]
        ahead, side = c * dx + s * dy, -s * dx + c * dy
        if abs(side) <= lane_half and -behind <= ahead <= max_ahead:
            if best is None or ahead < best[1]:
                best = (key, round(ahead, 6))
    return best


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
            # A speedometer, as a car's wheel-speed sensors give it: m/s in twist.linear.x.
            self.pub_speed = node.create_publisher(TwistStamped, '/carla/speedometer', 10)
            # Every stop line in the town, once: CARLA reports a light to the car
            # only inside its trigger zone, 2 to 3 m before the line, too late to stop.
            self.lights = {l.id: l for l in world.get_actors().filter('traffic.traffic_light')}
            self.stops = [(lid, carla_to_map_xy(w.transform.location.x, w.transform.location.y),
                           carla_to_map_yaw(w.transform.rotation.yaw))
                          for lid, l in self.lights.items() for w in l.get_stop_waypoints()]
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
            v = self.ego.get_velocity()
            sp = TwistStamped()
            sp.header = header('ego_vehicle', stamp)
            sp.twist.linear.x = math.hypot(v.x, v.y)
            self.pub_speed.publish(sp)

    def _on_collision(self, event):
        msg = String()
        msg.data = f'{event.timestamp:.3f} {event.other_actor.type_id}'
        self.pub_col.publish(msg)

    def _publish_light(self, tf, stamp):
        state, at, dist = 'Green', False, float('nan')
        found = light_ahead(carla_to_map_xy(tf.location.x, tf.location.y),
                            carla_to_map_yaw(tf.rotation.yaw), self.stops)
        if found is not None:
            state, at, dist = str(self.lights[found[0]].get_state()), True, found[1]
        msg = String()
        msg.data = light_message(state, at, dist)
        self.pub_light.publish(msg)
