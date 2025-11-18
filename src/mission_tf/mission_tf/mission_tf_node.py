import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import StaticTransformBroadcaster
from gz.transport13 import Node as GzNode
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.navsat_pb2 import NavSat
from functools import partial
from px4_msgs.msg import VehicleLocalPosition # type: ignore
from geographiclib.geodesic import Geodesic
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy


class MissionTfNode(Node):
    def __init__(self):
        super().__init__('mission_tf_node')

        self.mission_frame_geopose = ()
        self.mission_frame_pose = None
        self.mission_tf_published = False
        
        self.declare_parameter('mission_frame_pose_topic', "/model/cone/pose")
        self.declare_parameter('mission_frame_geopose_topic', "/mission_frame_geopose")
        
        self.mission_frame_topic = self.get_parameter('mission_frame_pose_topic').get_parameter_value().string_value
        self.mission_frame_geopose_topic = self.get_parameter('mission_frame_geopose_topic').get_parameter_value().string_value

        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
    
        self.gz_node = GzNode()
        self.gz_frame_pose_sub = self.gz_node.subscribe(
            Pose, self.mission_frame_topic, lambda msg: self.mission_frame_pose_callback(msg)
        )

        self.gz_frame_geo_pose_sub = self.gz_node.subscribe(
            NavSat, self.mission_frame_geopose_topic, lambda msg: self.mission_frame_geopose_callback(msg)
        )

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.ros2_subs = []
        pattern = f"/fmu/out/vehicle_local_position"
        n_uavs = len([topic for topic, _ in self.get_topic_names_and_types() if pattern in topic])
        self.uav_tfs_published = [False for _ in range(n_uavs)]
        for uav_id in range(1,n_uavs+1):
            self.ros2_subs.append(
                self.create_subscription(
                    VehicleLocalPosition,
                    f"/px4_{uav_id}/fmu/out/vehicle_local_position",
                    partial(self.uav_local_frame_callback, uav_id=uav_id),
                    qos_profile
                )
            )
        
        self.create_timer(1.0, self.publish_mission_frame_transform)


    def offset_between_gps_coordinates(self, lat1, lon1, alt1, lat2, lon2, alt2):
        
        geod = Geodesic.WGS84
        g = geod.Inverse(lat1, lon1, lat2, lon2)

        distance = g['s12']
        azimuth_rad = math.radians(g['azi1'])
        dx_north = distance * math.cos(azimuth_rad)
        dy_east = distance * math.sin(azimuth_rad)
        dz_up = alt2 - alt1
        x_east = dy_east
        y_north = dx_north
        z_up = dz_up

        return x_east, y_north, z_up
    

    def mission_frame_geopose_callback(self, msg: NavSat):

        self.mission_frame_geopose = (
            msg.latitude_deg,
            msg.longitude_deg, 
            msg.altitude
        )


    def mission_frame_pose_callback(self, msg: Pose):
        self.mission_frame_pose = msg


    def publish_mission_frame_transform(self):
        
        if self.mission_frame_pose.position.x != float('nan'):
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "world"
            t.child_frame_id = "mission"

            t.transform.translation.x = self.mission_frame_pose.position.x
            t.transform.translation.y = self.mission_frame_pose.position.y
            t.transform.translation.z = self.mission_frame_pose.position.z
            t.transform.rotation = Quaternion(
                x=self.mission_frame_pose.orientation.x,
                y=self.mission_frame_pose.orientation.y,
                z=self.mission_frame_pose.orientation.z,
                w=self.mission_frame_pose.orientation.w
            )
            self.static_tf_broadcaster.sendTransform(t)
            self.get_logger().debug(f"Mission frame tf published.")
 

    def uav_local_frame_callback(self, msg, uav_id):
    
        if self.mission_frame_geopose:
            dx, dy, dz = self.offset_between_gps_coordinates(
                self.mission_frame_geopose[0], 
                self.mission_frame_geopose[1], 
                self.mission_frame_geopose[2],
                msg.ref_lat, msg.ref_lon, msg.ref_alt
            )

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "mission"
            t.child_frame_id = f"uav_{uav_id}/local_frame"

            t.transform.translation.x = dx
            t.transform.translation.y = dy
            t.transform.translation.z = dz

            self.static_tf_broadcaster.sendTransform(t)
            self.get_logger().debug(f"UAV {uav_id} tf published.")


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = MissionTfNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()