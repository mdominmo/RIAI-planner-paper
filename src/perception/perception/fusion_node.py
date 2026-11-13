"""Módulo de Fusión Global: fuses the marker estimates sent by every UAV.

For each marker n the fused position is the mean of all its measurements,
from all the UAVs that observed it (paper, Sec. Percepción):

    p_n = 1 / (sum_{i in R_n} |M_n^(i)|) · sum_{i in R_n} sum_{p in M_n^(i)} p

A marker enters the map as soon as one UAV observes it (``min_measurements``
can raise that). The map is published latched on /perception/task_map whenever
it changes and returned on demand by the /perception/get_task_map service.
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Point

from swarm_interfaces.msg import MarkerEstimate, TaskMap
from swarm_interfaces.srv import GetTaskMap
from perception.geometry import MarkerFusion
from perception.aruco_detector_node import ESTIMATES_TOPIC

TASK_MAP_TOPIC = '/perception/task_map'
TASK_MAP_SERVICE = '/perception/get_task_map'


class FusionNode(Node):

    def __init__(self):
        super().__init__('fusion_node')
        self.declare_parameter('min_measurements', 1)
        self.declare_parameter('publish_period', 0.5)   # [s] while the map changes
        self.min_measurements = int(self.get_parameter('min_measurements').value)

        self.fusion = MarkerFusion()
        self.known = set()
        self.dirty = False
        self.create_subscription(MarkerEstimate, ESTIMATES_TOPIC, self.estimate_callback, 200)
        self.map_pub = self.create_publisher(
            TaskMap, TASK_MAP_TOPIC,
            QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.create_service(GetTaskMap, TASK_MAP_SERVICE, self.get_map_callback)
        self.create_timer(float(self.get_parameter('publish_period').value), self.publish_if_dirty)
        self.get_logger().info('Global fusion module ready.')

    def estimate_callback(self, obs):
        p = (obs.position.x, obs.position.y, obs.position.z)
        self.fusion.add(obs.uav_id, obs.marker_id, p)
        self.dirty = True
        n = obs.marker_id
        if n not in self.known and self.fusion.count[n] >= self.min_measurements:
            self.known.add(n)
            est = self.fusion.estimate(n)
            self.get_logger().info(
                f'New task detected: {n} (UAV {obs.uav_id}) at '
                f'({est[0]:.2f}, {est[1]:.2f}, {est[2]:.2f})')
            self.publish_map()          # new tasks are announced immediately

    def build_map(self):
        msg = TaskMap()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        for n, p, count, n_uavs in self.fusion.targets(self.min_measurements):
            msg.ids.append(int(n))
            msg.positions.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
            msg.num_measurements.append(int(count))
            msg.num_uavs.append(int(n_uavs))
        return msg

    def publish_map(self):
        self.map_pub.publish(self.build_map())
        self.dirty = False

    def publish_if_dirty(self):
        if self.dirty:
            self.publish_map()

    def get_map_callback(self, request, response):
        response.map = self.build_map()
        for n, p, count, n_uavs in zip(response.map.ids, response.map.positions,
                                       response.map.num_measurements, response.map.num_uavs):
            self.get_logger().info(
                f'Task {n}: ({p.x:.2f}, {p.y:.2f}, {p.z:.2f}) from {count} measurements, {n_uavs} UAVs')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
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
