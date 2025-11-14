import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geographic_msgs.msg import GeoPose
from geometry_msgs.msg import Pose
from offboard_control.control.multi_offboard_controller_assembler import MultiOffboardControllerAssembler
from offboard_control.service.sim_uavs_configuration_service import SimUAVSConfigurationService
from offboard_control.domain.constant.states import States
import time
import cv2
from planning.planning.planner import Planner
from planning.planning.utils import enu_ned
from riai_msgs.srv import Tracking

METHODS = {
    0: "Random",
    1: "Hungarian",
    2: "RTT*"
}

class MissionNode(Node):
  
    def __init__(self):
        super().__init__('mission_node')

        self.declare_parameter('mode', 'execution')
        self.declare_parameter('perception', 'global')
        self.declare_parameter('plan_type', 2)
        self.declare_parameter('targets', 4)
        self.declare_parameter("vehicle_ids", [1,2])
        self.declare_parameter('n_points', 400)
        
        self.declare_parameter('mission_frame', [.0, .0, .0])
        self.declare_parameter('mission_radius', 5.0)
        self.declare_parameter('mission_height', 4.0)
        self.declare_parameter('step_size', 1,0)
        self.declare_parameter('n_steps', 2000)
        self.declare_parameter('space_coef', .5)
        self.declare_parameter('time_coef', .5)

        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        self.perception = self.get_parameter('perception').get_parameter_value().string_value
        self.plan_type = self.get_parameter('plan_type').get_parameter_value().integer_value
        self.n_targets = self.get_parameter('targets').get_parameter_value().integer_value
        self.vehicle_ids = self.get_parameter('vehicle_ids').get_parameter_value().integer_array_value
        self.n_points = self.get_parameter('n_points').get_parameter_value().integer_value

        self.mission_frame = self.get_parameter('mission_frame').get_parameter_value().double_array_value
        self.mission_radius = self.get_parameter('mission_radius').get_parameter_value().double_value
        self.mission_height = self.get_parameter('mission_height').get_parameter_value().double_value
        self.step_size = self.get_parameter('step_size').get_parameter_value().double_value
        self.n_steps = self.get_parameter('n_steps').get_parameter_value().integer_value
        self.space_coef = self.get_parameter('space_coef').get_parameter_value().double_value
        self.time_coef = self.get_parameter('time_coef').get_parameter_value().double_value

        self.bridge = CvBridge()
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = cv2.aruco.DetectorParameters()
        self.multi_offboard_controller = None
        self.target_poses = {}
        self.obstacle_poses = {}
        self.rgb_subs = []
        self.detected_ids = set()
        self.planner = None

        self.configure()


    def run(self):
        self.start_formation()
        self.run_perception()
        self.execute_mission(self.plan_type)


    def start_formation(self):

        arm_future = self.multi_offboard_controller.arm_all()
        while not arm_future.done():
            rclpy.spin_once(self)

        self.multi_offboard_controller.offboard_mode_all()
        self.multi_offboard_controller.check_offboard()
        self.get_logger().info(f"Vehicles waiting for offboard mode.")

        self.get_logger().info(f"Vehicles taking off.")
        takeoff_future = self.multi_offboard_controller.take_off_all(10.0)
        while not takeoff_future.done():
            rclpy.spin_once(self)

        self.get_logger().info(f"Vehicles going to initial formation.")
        trajectories, asigned_vehicles = self.planner.get_initial_trajectory(self.get_vehicle_poses())
        future = self.multi_offboard_controller.trajectory_following(
            asigned_vehicles,
            trajectories
        )
        while not future.done():
            rclpy.spin_once(self)


    def run_perception(self):

        self.get_logger().info(f"Starting Perception")        
        start_time = time.perf_counter()

        trajectories, asigned_vehicles = self.planner.get_perception_trajectory()
        future = self.multi_offboard_controller.trajectory_following(
            asigned_vehicles,
            trajectories    
        )
        while not future.done() or not self.check_perception() :
            rclpy.spin_once(self)

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        self.get_logger().info(f"Perception finish.")
        self.get_logger().info(f"Time elapsed: {elapsed}")


    def execute_mission(self, plan_type: int): 
           
        trajectories, asigned_vehicles = self.planner.get_execution_planning(
            self.get_vehicle_poses, self.target_poses
        )
        self.get_logger().info(f"Executing mission.")
        
        start_time = time.perf_counter()
        future = self.multi_offboard_controller.trajectory_following(
            asigned_vehicles,
            trajectories
        )        
        while not future.done():
            rclpy.spin_once(self)
        
        future = self.multi_offboard_controller.land_all()
                
        while not future.done():
            rclpy.spin_once(self)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        self.get_logger().info(f"Method: {METHODS[plan_type]}, Time elapsed: {elapsed}")

        self.multi_offboard_controller.hold_all()
        self.multi_offboard_controller.disarm_all()

        
    def get_vehicle_poses(self):
        return [enu_ned(controller.state_manager.state_repositories[States.LOCAL_POSITION].get().get_pose()) for controller in self.multi_offboard_controller.controllers]
    

    def get_tracked_poses(self):
        
        cli = self.create_client(Tracking, "tracking_service")
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for tracking service...")

        req = Tracking.Request()
        future = cli.call_async(req)

        while not future.done():
            rclpy.spin_once(self, timeout_sec=2.0)
        
        response = future.result()
        self.target_poses = response.target_poses
        self.obstacle_poses = response.obstacle_poses
        

    def check_perception(self):
        if len(self.detected_ids) == len(self.target_poses):
            self.get_logger().info("¡Todos los ArUco IDs han sido detectados!")
            return True
        return False
    

    def image_callback(self, msg):
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        _, ids, _ = cv2.aruco.detectMarkers(cv_image, self.dictionary, parameters=self.parameters)

        if ids is not None:
            for aruco_id in ids.flatten():
                if aruco_id not in self.detected_ids and aruco_id < len(self.target_poses):
                    self.detected_ids.add(aruco_id)
                    self.get_logger().info(f"Nuevo ArUco detectado: {aruco_id}")


    def configure(self):

        self.gps_origin = GeoPose()
        self.gps_origin.position.latitude = 47.397971057728974
        self.gps_origin.position.longitude = 8.5461637398001464
        self.gps_origin.position.altitude = .0

        self.configuration_service = SimUAVSConfigurationService(
            self,
            self.vehicle_ids    
        )
        multi_offboard_controller_assembler = MultiOffboardControllerAssembler()
        self.multi_offboard_controller = multi_offboard_controller_assembler.assemble(
            self,
            self.configuration_service.get_offboard_configurations()
        ) 
        self.multi_offboard_controller.set_home(self.gps_origin)
    
        for i in self.vehicle_ids:
            self.rgb_subs.append(self.create_subscription(
                CompressedImage,
                f"/world/riai_planner_paper_world/model/x500_mono_cam_{i}/link/mono_cam/base_link/sensor/camera/image/compressed",
                self.image_callback,
                10
            ))
        
        self.planner = Planner(
            self.mission_frame,
            self.mission_radius,
            self.mission_height,
            len(self.vehicle_ids),
            self.step_size,
            self.n_steps,
            self.space_coef,
            self.time_coef
        )
        

def main(args=None):
  
    rclpy.init(args=args)
    node = MissionNode()    
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

