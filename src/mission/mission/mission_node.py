import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from offboard_control.control.multi_offboard_controller_assembler import MultiOffboardControllerAssembler
from offboard_control.service.sim_uavs_configuration_service import SimUAVSConfigurationService
from offboard_control.domain.constant.states import States
import time
import cv2
from planning.planner import Planner
from planning.utils import enu_ned, enu_ned_trajectories, bspline_trajectory, plot_pose_list
from riai_msgs.srv import Tracking


ASIGNATION_METHODS = {
    0: "Random",
    1: "RRT* + Hungarian",
    2: "Only RTT*",
    3: "By euclidean distance"
}


class MissionNode(Node):
    def __init__(self):
        super().__init__('mission_node')

        self.declare_parameter('mode', 'execution')
        self.declare_parameter('perception', 'global')
        self.declare_parameter('plan_type', 0)
        self.declare_parameter('targets', 4)
        self.declare_parameter("vehicle_ids", [1,2])
        self.declare_parameter('n_points', 200)
        self.declare_parameter('mission_frame', [108.28299713134766, -94.181564331054688, 1.9263170957565308])
        self.declare_parameter('mission_radius', 5.0)
        self.declare_parameter('mission_height', 8.0)
        self.declare_parameter('step_size', 1.0)
        self.declare_parameter('n_steps', 2000)
        self.declare_parameter('space_coef', .8)
        self.declare_parameter('time_coef', .2)
        self.declare_parameter('avg_speed', 1.0)
        self.declare_parameter('spatial_tol', .5)
        self.declare_parameter('time_tol', 100.0)
        self.declare_parameter('cylinder_height', 1.0)
        self.declare_parameter('cylinder_radius', 1.0)
        
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
        self.avg_speed = self.get_parameter('avg_speed').get_parameter_value().double_value
        self.spatial_tol = self.get_parameter('spatial_tol').get_parameter_value().double_value
        self.time_tol = self.get_parameter('time_tol').get_parameter_value().double_value
        self.cylinder_height = self.get_parameter('cylinder_height').get_parameter_value().double_value
        self.cylinder_radius = self.get_parameter('cylinder_radius').get_parameter_value().double_value

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

        self.multi_offboard_controller.offboard_mode_all()
        arm_future = self.multi_offboard_controller.arm_all()
        while not arm_future.done():
            rclpy.spin_once(self, timeout_sec=2.0)
        self.multi_offboard_controller.check_offboard()
        self.get_logger().info(f"Vehicles waiting for offboard mode.")

        self.get_logger().info(f"Vehicles taking off.")
        takeoff_future = self.multi_offboard_controller.take_off_all(10.0)
        while not takeoff_future.done():
            rclpy.spin_once(self, timeout_sec=2.0)

        trajectories = []
        while len(trajectories) != len(self.vehicle_ids):
            self.get_logger().info(f"Computing initial trajectory.")
            trajectories = self.planner.get_initial_trajectory(self.get_vehicle_poses(), self.obstacle_poses)
        
        self.get_logger().info(f"Vehicles going to initial formation.")
        future = self.multi_offboard_controller.trajectory_following(
            range(len(self.vehicle_ids)),
            self.transform_to_local_frame_trajectory(trajectories)
        )
        while not future.done():
            rclpy.spin_once(self, timeout_sec=2.0)


    def run_perception(self):

        self.get_logger().info(f"Starting Perception")        
        start_time = time.perf_counter()

        future = self.multi_offboard_controller.trajectory_following(
            range(len(self.vehicle_ids)),
            self.transform_to_local_frame_trajectory(self.planner.get_perception_trajectory())  
        )
        while not future.done():
            rclpy.spin_once(self, timeout_sec=2.0)

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        self.get_logger().info(f"Perception finish.")
        self.get_logger().info(f"Time elapsed: {elapsed}")


    def execute_mission(self, plan_type: int): 
        
        targets_poses = [self.target_poses[int(id)] for id in self.detected_ids]
        
        trajectories = []
        while len(trajectories) != len(self.vehicle_ids):
            self.get_logger().info(f"Computing execution trajectory.")
            trajectories = self.planner.get_tasks_planning(
                self.get_vehicle_poses(), 
                targets_poses,
                self.obstacle_poses,
                self.plan_type
            )

        self.get_logger().info(f"Executing mission.")
        start_time = time.perf_counter()
        future = self.multi_offboard_controller.trajectory_following(
            range(len(self.vehicle_ids)),
            self.transform_to_local_frame_trajectory(trajectories)
        )        
        while not future.done():
            rclpy.spin_once(self, timeout_sec=2.0)
        
        future = self.multi_offboard_controller.land_all()
        while not future.done():
            rclpy.spin_once(self, timeout_sec=2.0)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        self.get_logger().info(f"Method: {ASIGNATION_METHODS[plan_type]}, Time elapsed: {elapsed}")

        self.multi_offboard_controller.hold_all()
        self.multi_offboard_controller.disarm_all()

        
    def get_vehicle_poses(self):
        
        in_map_poses = []
        for uav_id, controller in enumerate(self.multi_offboard_controller.controllers, start=1):
            local_pose = enu_ned(controller.state_manager.state_repositories[States.LOCAL_POSITION].get().get_pose())
            in_map_poses.append(local_pose)
           
        return in_map_poses


    def get_tracked_poses(self):
        
        cli = self.create_client(Tracking, "tracking_service")
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for tracking service...")

        future = cli.call_async(Tracking.Request())
        while not future.done():
            rclpy.spin_once(self, timeout_sec=2.0)
        
        response = future.result()
        self.target_poses = response.target_poses
        for p in self.target_poses:
            p.position.z = 6.0
        
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

    
    def transform_to_local_frame_trajectory(self, trajectories):
        trajectories = self.smooth_trajectories(enu_ned_trajectories(trajectories))
        return trajectories 


    def smooth_trajectories(self, trajectories):
        for trajectory in trajectories:
            smooth_poses = bspline_trajectory(
                trajectory[0],
                self.n_points,
                smoothing=.0,
                degree=3,
                periodic=False,
                speed=self.avg_speed)
            
            trajectory[0] = [s["pose"] for s in smooth_poses]
            trajectory[1] = [s["vel"] for s in smooth_poses]
            trajectory[2] = [s["yaw"] for s in smooth_poses]
            trajectory[3] = [s["dt"] for s in smooth_poses]

        return trajectories


    def configure(self):

        self.configuration_service = SimUAVSConfigurationService(
            self,
            self.vehicle_ids    
        )
        multi_offboard_controller_assembler = MultiOffboardControllerAssembler()
        self.multi_offboard_controller = multi_offboard_controller_assembler.assemble(
            self,
            self.configuration_service.get_offboard_configurations()
        ) 
        for i in self.vehicle_ids:
            self.rgb_subs.append(self.create_subscription(
                CompressedImage,
                f"/world/riai_planner_paper_world/model/x500_vision_{i}/link/mono_cam/base_link/sensor/camera/image/compressed",
                self.image_callback,
                10
            ))
        
        self.mission_frame_pose = Pose()
        self.mission_frame_pose.position.x = self.mission_frame[0]
        self.mission_frame_pose.position.y = self.mission_frame[1] 
        self.mission_frame_pose.position.z = self.mission_frame[2] 

        self.planner = Planner(
            self.mission_frame_pose,
            self.mission_radius,
            self.mission_height,
            len(self.vehicle_ids),
            self.step_size,
            self.n_steps,
            self.space_coef,
            self.time_coef,
            self.avg_speed,
            self.spatial_tol,
            self.time_tol,
            self.cylinder_height,
            self.cylinder_radius
        )
        self.get_tracked_poses()
        

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

