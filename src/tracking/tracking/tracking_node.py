import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import subprocess
from gz.transport13 import Node as GzNode
from gz.msgs10.pose_pb2 import Pose as GzPose
from swarm_interfaces.srv import Tracking

class TrackingNode(Node):
  
    def __init__(self):
        super().__init__('tracking_node')
        self.gz_node = GzNode()
        
        self.subs = []
        self.tracks = []
        self.obstacle_poses = {}
        self.target_poses = {}

        cmd = "gz topic -l | grep obstacle"
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
        obstacle_pose_topics = [line.strip() for line in out.splitlines() if line.strip()]
        for topic in obstacle_pose_topics:
            self.subs.append(
                self.gz_node.subscribe(
                    GzPose,
                    topic,
                    lambda msg: self.obstacle_pose_callback(msg)
                )
            )   
        cmd = "gz topic -l | grep target"
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
        target_pose_topics = [line.strip() for line in out.splitlines() if line.strip()]
        for topic in target_pose_topics:
            self.subs.append(
                self.gz_node.subscribe(
                    GzPose,
                    topic,
                    lambda msg: self.target_pose_callback(msg)
                )
            )  
        while len(self.obstacle_poses) != len(obstacle_pose_topics) or len(self.target_poses) != len(target_pose_topics):
            rclpy.spin_once(self, timeout_sec=2.0)
            self.get_logger().info("Waiting for tracking info.")
        self.srv = self.create_service(Tracking, "tracking_service", self.tracking_callback)
        self.get_logger().info("Starting tracking node.")

    def obstacle_pose_callback(self, msg):
        self.obstacle_poses[msg.name] = Pose()
        self.obstacle_poses[msg.name].position.x = msg.position.x
        self.obstacle_poses[msg.name].position.y = msg.position.y
        self.obstacle_poses[msg.name].position.z = msg.position.z


    def target_pose_callback(self, msg):
        self.target_poses[msg.name] = Pose()
        self.target_poses[msg.name].position.x = msg.position.x
        self.target_poses[msg.name].position.y = msg.position.y
        self.target_poses[msg.name].position.z = msg.position.z


    def tracking_callback(self, req, res):
        res.target_poses = list(self.target_poses.values())
        res.obstacle_poses = list(self.obstacle_poses.values())
        return res


def main(args=None):
  
    rclpy.init(args=args)
    node = TrackingNode()    
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

