#!/usr/bin/env python3
"""The L9 sampling planner on Town03 in CARLA, drawn in RViz.

Slide: 'Demo: the same planner in CARLA'.

    ros2 launch l9_planning_demo demo.launch.py
    ros2 param set /planning_demo w_clearance 10.0

The AV starts on the slides' stretch of Town03 road 67, at s = 60 m on the right
lane's center, with a car parked at s = 118 m. Every 0.5 s the node samples the
31 candidates of the slides, checks them with circles, scores them and picks the
cheapest; between plans the AV moves along the first 0.5 s of that path at
10 m/s. The node places the AV on the path with set_transform: there is no
controller here (L10 adds one). At the end of the stretch the AV starts again.

The node owns the simulation clock: it puts the server in synchronous mode and
calls world.tick() from a timer. Do not run a second client that also ticks
(l2_carla_demo's carla_bridge does).

Topics (frame "map": x as CARLA's x, y as minus CARLA's y, as in the slides):
    /l9/road        the lanes of the stretch (latched)
    /l9/plan        the candidates, the cars, the AV's circles and the costs
    /l9/camera      what a camera behind and above the AV sees (sensor_msgs/Image, bgr8)
"""
import math
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import SetParametersResult
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray

import carla

from l9_planning_demo import planner as pl

FRAME = "map"
# The colors of the slides and of l9_handson.py, as (r, g, b) from 0 to 1.
COLORS = {"best": "#1E9E74", "kept": "#7A828C", "collision": "#D9694A",
          "too sharp": "#7B5EA7", "stop": "#1E2939", "av": "#1E2939", "car": "#2D6CA2",
          "line": "#B0B0B0", "center": "#E0B000"}
WEIGHT_NAMES = ("comfort", "clearance", "lane", "center", "progress")


def rgb(name, alpha=1.0):
    h = COLORS[name].lstrip("#")
    return tuple(int(h[k:k + 2], 16) / 255.0 for k in (0, 2, 4)) + (alpha,)


class PlanningDemo(Node):

    def __init__(self) -> None:
        super().__init__("planning_demo")
        self.declare_parameter("host", "localhost")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("town", "Town03")
        self.declare_parameter("delta", 0.05)          # 20 Hz simulation step
        self.declare_parameter("period", 0.5)          # s between plans, as the slides
        self.declare_parameter("speed", pl.SCENE["speed"])
        self.declare_parameter("car_s", pl.SCENE["car_s"])
        # several parked cars: a flat list [s1, d1, s2, d2, ...] in meters (launch file
        # demo_obstacles.launch.py); unset, the one car of the slides at car_s
        self.declare_parameter("obstacles", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("vehicle", "vehicle.tesla.model3")
        self.declare_parameter("parked", "vehicle.audi.a2")
        self.declare_parameter("follow_camera", True)
        # a chase camera on the AV, shown in RViz's Camera panel (CARLA renders it even
        # with -RenderOffScreen)
        self.declare_parameter("camera", True)
        self.declare_parameter("camera_width", 800)
        self.declare_parameter("camera_height", 450)
        self.declare_parameter("camera_fov", 90.0)
        for n in WEIGHT_NAMES:
            self.declare_parameter("w_" + n, pl.WEIGHTS[n])

        self.delta = self.get_parameter("delta").value
        self.period = self.get_parameter("period").value
        self.speed = self.get_parameter("speed").value
        self.weights = {n: self.get_parameter("w_" + n).value for n in WEIGHT_NAMES}
        self.scene = dict(pl.SCENE, car_s=self.get_parameter("car_s").value)
        try:
            flat = self.get_parameter("obstacles").value
        except ParameterUninitializedException:     # declared with a type only: unset
            flat = None
        if flat and len(flat) % 2 == 0:
            self.obstacles = [(float(flat[k]), float(flat[k + 1])) for k in range(0, len(flat), 2)]
            self.several = True
        else:
            self.obstacles = [(self.scene["car_s"], self.scene["car_d"])]
            self.several = False

        share = get_package_share_directory("l9_planning_demo")
        self.ref = pl.load_scene(os.path.join(share, "data", "l9_scene.npz"))
        self.s_end = float(self.ref["s"][-1]) - self.scene["horizon"] - 5.0

        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_road = self.create_publisher(MarkerArray, "/l9/road", latched)
        self.pub_plan = self.create_publisher(MarkerArray, "/l9/plan", 10)
        self.pub_camera = self.create_publisher(Image, "/l9/camera", 2)

        self.actors = []
        self.original_settings = None
        self._connect()
        self._spawn()
        self._publish_road()
        self._restart()
        self.add_on_set_parameters_callback(self._on_parameters)

        self.timer = self.create_timer(self.delta, self._tick)
        self.get_logger().info(
            f"planning every {self.period} s on {self.world.get_map().name}; "
            f"weights {self.weights}")

    # ---------------------------------------------------------------- CARLA
    def _connect(self) -> None:
        host = self.get_parameter("host").value
        port = self.get_parameter("port").value
        client = carla.Client(host, port)
        client.set_timeout(self.get_parameter("timeout").value)
        try:
            client.get_server_version()
        except RuntimeError as exc:
            raise SystemExit(f"Could not reach the CARLA server at {host}:{port} ({exc}). "
                             "Is it running, and did you give it time to load the level?")
        town = self.get_parameter("town").value
        world = client.get_world()
        if town and not world.get_map().name.endswith(town):
            client.set_timeout(max(60.0, self.get_parameter("timeout").value))  # a map load is slow
            world = client.load_world(town)
            client.set_timeout(self.get_parameter("timeout").value)
        self.client, self.world = client, world
        self.map = world.get_map()
        self.original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta
        world.apply_settings(settings)

    def _transform(self, x, y, heading, lift=0.05):
        """A CARLA transform for a pose in the map frame. The base link is the middle of
        the rear axle; CARLA places a vehicle by the center of its bounding box, so the
        caller passes the body center. z comes from the road under the point."""
        loc = carla.Location(x=float(x), y=float(-y), z=0.0)
        wp = self.map.get_waypoint(loc, project_to_road=True)
        loc.z = wp.transform.location.z + lift
        return carla.Transform(loc, carla.Rotation(yaw=-math.degrees(heading)))

    def _body_center(self, x, y, heading, rear_to_ref):
        """The body center, from a reference point rear_to_ref meters in from the rear."""
        ahead = pl.AV["length"] / 2 - rear_to_ref
        return x + ahead * math.cos(heading), y + ahead * math.sin(heading)

    def _spawn(self) -> None:
        bp_lib = self.world.get_blueprint_library()
        av_bp = bp_lib.find(self.get_parameter("vehicle").value)
        av_bp.set_attribute("role_name", "hero")
        car_bp = bp_lib.find(self.get_parameter("parked").value)
        far = carla.Transform(carla.Location(x=0.0, y=0.0, z=500.0))
        self.av = self.world.spawn_actor(av_bp, far)
        self.cars = [self.world.spawn_actor(car_bp, carla.Transform(carla.Location(x=10.0 * (k + 1), y=0.0, z=500.0)))
                     for k in range(len(self.obstacles))]
        for a in [self.av] + self.cars:
            a.set_simulate_physics(False)   # placed, not driven
            self.actors.append(a)
        self._place_car()
        if self.get_parameter("camera").value:
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(self.get_parameter("camera_width").value))
            cam_bp.set_attribute("image_size_y", str(self.get_parameter("camera_height").value))
            cam_bp.set_attribute("fov", str(self.get_parameter("camera_fov").value))
            # 7 m behind the body center, 3.5 m up, looking a little down: the AV, the
            # road ahead and the parked car when it comes up
            mount = carla.Transform(carla.Location(x=-7.0, z=3.5), carla.Rotation(pitch=-12.0))
            self.camera = self.world.spawn_actor(cam_bp, mount, attach_to=self.av)
            self.actors.append(self.camera)
            self.camera.listen(self._on_image)

    def _on_image(self, image) -> None:
        """A CARLA frame (BGRA) as a ROS image (bgr8: the alpha channel dropped)."""
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height, msg.width = image.height, image.width
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = image.width * 3
        msg.data = np.ascontiguousarray(bgra[:, :, :3]).tobytes()
        self.pub_camera.publish(msg)

    def _place_car(self) -> None:
        """Every parked car where its (s, d) says. The planner is given these same boxes:
        nothing detects them (no perception in this demo)."""
        self.car_poses = [pl.parked_car(self.ref, dict(self.scene, car_s=s, car_d=d))
                          for s, d in self.obstacles]
        for car, c in zip(self.cars, self.car_poses):
            car.set_transform(self._transform(c["x"], c["y"], c["heading"]))

    # ---------------------------------------------------------------- planning
    def _restart(self) -> None:
        self.s0, self.state = self.scene["av_s"], (self.scene["av_d"], 0.0, 0.0)
        self._plan()

    def _plan(self) -> None:
        """Plan from (s0, state): every candidate scored, the cheapest first."""
        # the stop candidate keeps the right lane: it stops behind the nearest car ahead in
        # that lane (none: no stop candidate)
        lane = self.ref["lane_width"] / 2
        ahead = [s for s, d in self.obstacles if abs(d) < lane and s > self.s0]
        scene = dict(self.scene, car_s=min(ahead) if ahead else self.s0 + 1e3)
        self.cands = pl.plan(self.ref, self.s0, self.state, self.car_poses, speed=self.speed,
                             weights=self.weights, scene=scene)
        self.best = self.cands[0][0]
        self.t_plan = 0.0
        self._publish_plan()

    def _pose_on_best(self, t):
        """The AV's (s, d, d', d'') t seconds into the current plan."""
        adv = self.speed * t
        if self.best.get("stop"):
            adv = min(adv, float(self.best["s"][-1] - self.s0))
        L = self.best["maneuver"]
        d, d1, d2 = pl.lateral(self.best["coef"], L, adv)
        if adv >= L:
            d1, d2 = 0.0, 0.0
        return self.s0 + adv, float(d), float(d1), float(d2)

    def _tick(self) -> None:
        self.t_plan += self.delta
        s, d, d1, d2 = self._pose_on_best(self.t_plan)
        if self.t_plan >= self.period - 1e-9:
            self.s0, self.state = s, (d, d1, d2)
            if self.s0 > self.s_end:
                self._restart()
            else:
                self._plan()
            s, d, d1 = self.s0, self.state[0], self.state[1]
        x, y = pl.from_frenet(self.ref, s, d)
        _, _, hr, kr = pl.ref_at(self.ref, s)
        heading = float(hr + math.atan2(d1, 1 - kr * d))
        cx, cy = self._body_center(float(x), float(y), heading, pl.AV["rear_to_base"])
        self.av.set_transform(self._transform(cx, cy, heading))
        self.av_pose = (float(x), float(y), heading)
        if self.get_parameter("follow_camera").value:
            self._follow(cx, cy)
        self.world.tick()

    def _follow(self, x, y) -> None:
        """The spectator looks down on the AV from 45 m, ahead of it, so the window
        shows the AV, the parked car and the lanes."""
        t = carla.Transform(carla.Location(x=float(x) + 15.0, y=float(-y), z=45.0),
                            carla.Rotation(pitch=-90.0, yaw=0.0))
        self.world.get_spectator().set_transform(t)

    def _on_parameters(self, params) -> SetParametersResult:
        for p in params:
            if p.name.startswith("w_") and p.name[2:] in self.weights:
                if p.value < 0:
                    return SetParametersResult(successful=False, reason="weights are 0 or more")
                self.weights[p.name[2:]] = float(p.value)
            elif p.name == "car_s":
                if self.several:
                    return SetParametersResult(successful=False,
                                               reason="car_s moves the one car; this run has several")
                self.scene["car_s"] = float(p.value)
                self.obstacles = [(self.scene["car_s"], self.scene["car_d"])]
                self._place_car()
        self.get_logger().info(f"weights {self.weights}, parked cars at s = {[s for s, _ in self.obstacles]} m; "
                               "they take effect at the next plan")
        return SetParametersResult(successful=True)

    # ---------------------------------------------------------------- RViz
    def _marker(self, ns, mid, kind, color, width=0.1):
        m = Marker()
        m.header.frame_id = FRAME
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, kind, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = width
        m.color.r, m.color.g, m.color.b, m.color.a = color
        return m

    def _strip(self, ns, mid, xs, ys, color, width, z=0.0):
        from geometry_msgs.msg import Point
        m = self._marker(ns, mid, Marker.LINE_STRIP, color, width)
        m.points = [Point(x=float(a), y=float(b), z=z) for a, b in zip(xs, ys)]
        return m

    def _publish_road(self) -> None:
        """The stretch's lane lines: the right lane's two edges, the line between the
        two lanes, and the road's center line at d = 5.25 m."""
        s = self.ref["s"]
        out = MarkerArray()
        w = self.ref["lane_width"]
        for k, (d, name) in enumerate(((-w / 2, "line"), (w / 2, "line"), (1.5 * w, "center"))):
            x, y = pl.from_frenet(self.ref, s, np.full_like(s, d))
            out.markers.append(self._strip("road", k, x, y, rgb(name), 0.15, z=-0.05))
        self.pub_road.publish(out)

    def _box(self, ns, mid, x, y, heading, color, rear_to_ref):
        cx, cy = self._body_center(x, y, heading, rear_to_ref)
        m = self._marker(ns, mid, Marker.CUBE, color)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = cx, cy, 0.75
        m.pose.orientation.z, m.pose.orientation.w = math.sin(heading / 2), math.cos(heading / 2)
        m.scale.x, m.scale.y, m.scale.z = pl.AV["length"], pl.AV["width"], 1.5
        return m

    def _publish_plan(self) -> None:
        out = MarkerArray()
        clear = Marker()
        clear.header.frame_id = FRAME
        clear.action = Marker.DELETEALL
        out.markers.append(clear)
        for k, (p, sc) in enumerate(self.cands):
            if p is self.best:
                kind, width, z = "best", 0.35, 0.2
            elif p.get("stop"):
                kind, width, z = "stop", 0.2, 0.15
            elif sc["dropped"]:
                kind, width, z = sc["dropped"], 0.06, 0.05
            else:
                kind, width, z = "kept", 0.12, 0.1
            out.markers.append(self._strip("candidates", k, p["x"], p["y"], rgb(kind), width, z))
        # the AV at the start of this plan, its three circles, the parked car
        x, y = pl.from_frenet(self.ref, self.s0, self.state[0])
        _, _, hr, kr = pl.ref_at(self.ref, self.s0)
        h = float(hr + math.atan2(self.state[1], 1 - kr * self.state[0]))
        out.markers.append(self._box("av", 0, float(x), float(y), h, rgb("av", 0.8), pl.AV["rear_to_base"]))
        r = pl.circle_radius()
        ccx, ccy = pl.circles(float(x), float(y), h, pl.circle_offsets())
        for k, (a, b) in enumerate(zip(ccx[0], ccy[0])):
            m = self._marker("av_circles", k, Marker.CYLINDER, rgb("av", 0.25))
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(a), float(b), 0.02
            m.scale.x = m.scale.y = 2 * r
            m.scale.z = 0.04
            out.markers.append(m)
        for k, c in enumerate(self.car_poses):
            out.markers.append(self._box("car", k, c["x"], c["y"], c["heading"], rgb("car", 0.9),
                                         pl.AV["length"] / 2))
        # the costs of the three cheapest, beside the AV
        lines = []
        for p, sc in self.cands[:3]:
            name = "stop behind" if p.get("stop") else f"d_end {p['d_end']:.1f} m, {p['maneuver']:.0f} m"
            lines.append(f"{name}: {sc['total']:.2f}")
        lines.append("w: " + ", ".join(f"{n} {self.weights[n]:g}" for n in WEIGHT_NAMES))
        t = self._marker("costs", 0, Marker.TEXT_VIEW_FACING, (0.1, 0.1, 0.1, 1.0))
        t.pose.position.x, t.pose.position.y, t.pose.position.z = float(x), float(y) + 9.0, 2.0
        t.scale.z = 1.6
        t.text = "\n".join(lines)
        out.markers.append(t)
        self.pub_plan.publish(out)

    # ---------------------------------------------------------------- shutdown
    def destroy_node(self) -> None:
        for a in self.actors:
            if a is not None and a.is_alive:
                if isinstance(a, carla.Sensor):
                    a.stop()                      # no callback after the node is gone
                a.destroy()
        if self.original_settings is not None:
            self.world.apply_settings(self.original_settings)
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlanningDemo()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
