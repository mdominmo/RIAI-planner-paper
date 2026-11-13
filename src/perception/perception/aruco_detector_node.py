"""Detección Distribuida: on-board ArUco detection and georeferencing.

One instance runs per UAV. For every image captured while perception is
active it detects the ArUco markers, solves a PnP problem for each one using
the known marker side (^C T_M), and expresses the marker centre in the world
frame with the camera extrinsics (^R T_C) and the UAV pose estimated by its
autopilot at the image time (^W T_R):

    ^W T_M = ^W T_R · ^R T_C · ^C T_M

Only the resulting position estimates p_n^(i) are sent to the global fusion
module (swarm_interfaces/MarkerEstimate on /perception/marker_estimates).

IDs not in ``marker_ids`` are discarded: with small dictionaries the
detector occasionally decodes a wrong ID (e.g. 24 when only 0-5 exist).
"""
import math

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from px4_msgs.msg import VehicleOdometry
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

from swarm_interfaces.msg import MarkerEstimate
from perception import geometry as g

PERCEPTION_ACTIVE_TOPIC = '/perception/active'
ESTIMATES_TOPIC = '/perception/marker_estimates'


def latched_qos():
    return QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


def make_detector_parameters():
    # OpenCV < 4.7 (Ubuntu 24.04 ships 4.6): DetectorParameters() builds an
    # uninitialised object that segfaults inside detectMarkers().
    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params


class ArucoDetectorNode(Node):

    def __init__(self):
        super().__init__('aruco_detector')
        self.declare_parameter('uav_id', 1)
        self.declare_parameter('world_name', 'riai_planner_paper_world')
        self.declare_parameter('image_topic', '')
        self.declare_parameter('odometry_topic', '')
        # Valid task IDs: anything else is a false decoding.
        self.declare_parameter('marker_ids', [0, 1, 2, 3, 4, 5])
        self.declare_parameter('aruco_dictionary', 'DICT_4X4_250')
        self.declare_parameter('marker_length', 2.0)            # side [m]
        # Calibrated camera (mono_cam in x500_vision): intrinsics ...
        self.declare_parameter('image_width', 1240)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('horizontal_fov', 1.047)
        self.declare_parameter('distortion', [0.0, 0.0, 0.0, 0.0, 0.0])
        # ... and extrinsics ^R T_C (camera link pose in the UAV body frame).
        self.declare_parameter('camera_xyz', [0.12, 0.03, 0.242])
        self.declare_parameter('camera_rpy', [0.0, 0.8, 0.0])
        # Quality gates.
        self.declare_parameter('max_reprojection_error', 2.0)   # [px]
        self.declare_parameter('max_distance', 60.0)            # [m]
        self.declare_parameter('max_pose_dt', 0.1)              # image/pose [s]
        self.declare_parameter('image_latency', 0.0)            # capture->reception [s]
        self.declare_parameter('always_active', False)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.uav_id = int(gp('uav_id'))
        world = gp('world_name')
        image_topic = gp('image_topic') or (
            f'/world/{world}/model/x500_vision_{self.uav_id}'
            '/link/mono_cam/base_link/sensor/camera/image/compressed')
        odom_topic = gp('odometry_topic') or f'/px4_{self.uav_id}/fmu/out/vehicle_odometry'

        self.valid_ids = {int(i) for i in gp('marker_ids')}
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, gp('aruco_dictionary')))
        self.params = make_detector_parameters()
        self.object_points = g.marker_object_points(float(gp('marker_length')))
        self.k = g.camera_matrix(int(gp('image_width')), int(gp('image_height')),
                                 float(gp('horizontal_fov')))
        self.dist = np.asarray(gp('distortion'), dtype=float)
        self.t_rc, self.r_rc = g.camera_extrinsics(gp('camera_xyz'), gp('camera_rpy'))
        self.max_reproj = float(gp('max_reprojection_error'))
        self.max_distance = float(gp('max_distance'))
        self.max_pose_dt = float(gp('max_pose_dt'))
        self.image_latency = float(gp('image_latency'))

        self.active = bool(gp('always_active'))
        self.poses = g.PoseBuffer(horizon=2.0)
        self.stats = {'images': 0, 'sent': 0, 'bad_id': 0, 'no_pose': 0, 'rejected': 0}
        self.seen = set()

        px4_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             durability=QoSDurabilityPolicy.VOLATILE,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(VehicleOdometry, odom_topic, self.odometry_callback, px4_qos)
        self.create_subscription(CompressedImage, image_topic, self.image_callback, 2)
        self.create_subscription(Bool, PERCEPTION_ACTIVE_TOPIC, self.active_callback, latched_qos())
        self.pub = self.create_publisher(MarkerEstimate, ESTIMATES_TOPIC, 100)
        self.create_timer(10.0, self.report)
        self.get_logger().info(
            f'UAV {self.uav_id}: ArUco detector on {image_topic} (pose: {odom_topic}), '
            f'valid ids {sorted(self.valid_ids)}')

    def active_callback(self, msg):
        if msg.data != self.active:
            self.get_logger().info(f'UAV {self.uav_id}: perception {"on" if msg.data else "off"}')
        self.active = msg.data or bool(self.get_parameter('always_active').value)

    def odometry_callback(self, msg):
        if math.isnan(msg.position[0]) or math.isnan(msg.q[0]):
            return
        # timestamp_sample [us] is PX4 time translated by the uXRCE-DDS time
        # sync to the agent clock (system time), not the image clock.
        p, r = g.px4_to_world(msg.position, msg.q)
        self.poses.add(msg.timestamp_sample * 1e-6, p, r)

    def image_callback(self, msg):
        if not self.active:
            return
        t_recv = self.get_clock().now().nanoseconds * 1e-9
        self.stats['images'] += 1
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.dictionary, parameters=self.params)
        if ids is None:
            return

        pose = self.poses.lookup(self.image_time(msg, t_recv), self.max_pose_dt)
        if pose is None:
            self.stats['no_pose'] += 1
            return
        p_wr, r_wr = pose

        for c, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in self.valid_ids:
                self.stats['bad_id'] += 1
                continue
            sol = g.solve_marker_pnp(c, self.object_points, self.k, self.dist)
            if sol is None:
                self.stats['rejected'] += 1
                continue
            t_cm, _, err = sol
            distance = float(np.linalg.norm(t_cm))
            if err > self.max_reproj or distance > self.max_distance:
                self.stats['rejected'] += 1
                continue
            p_wm = g.marker_in_world(t_cm, p_wr, r_wr, self.t_rc, self.r_rc)

            obs = MarkerEstimate()
            obs.header.stamp = msg.header.stamp
            obs.header.frame_id = 'world'
            obs.uav_id = self.uav_id
            obs.marker_id = marker_id
            obs.position.x, obs.position.y, obs.position.z = (float(v) for v in p_wm)
            obs.distance = distance
            obs.reprojection_error = err
            self.pub.publish(obs)
            self.stats['sent'] += 1
            if marker_id not in self.seen:
                self.seen.add(marker_id)
                self.get_logger().info(
                    f'UAV {self.uav_id}: marker {marker_id} at '
                    f'({p_wm[0]:.2f}, {p_wm[1]:.2f}, {p_wm[2]:.2f}), d={distance:.1f} m')

    def image_time(self, msg, t_recv):
        """Image time in the clock of the pose buffer.

        If the image stamp and the autopilot share the time base (real UAV,
        synchronised clocks) the stamp is used. In simulation the image is
        stamped with simulation time while PX4 reports system time: then the
        reception time minus the expected transport latency is used.
        """
        t_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        latest = self.poses.latest_time()
        if latest is not None and abs(t_stamp - latest) < 1.0:
            return t_stamp
        return t_recv - self.image_latency

    def report(self):
        if self.active or self.stats['images']:
            s = self.stats
            self.get_logger().info(
                f'UAV {self.uav_id}: {s["images"]} images, {s["sent"]} estimates sent, '
                f'{s["bad_id"]} invalid ids, {s["rejected"]} rejected, {s["no_pose"]} without pose')


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
