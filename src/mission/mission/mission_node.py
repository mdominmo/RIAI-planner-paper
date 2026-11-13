import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from offboard_control.control.multi_offboard_controller_assembler import MultiOffboardControllerAssembler
from offboard_control.service.sim_uavs_configuration_service import SimUAVSConfigurationService
from offboard_control.domain.constant.states import States
import time
from planning.planner import Planner
from planning.utils import enu_ned, enu_ned_trajectories, bspline_trajectory
from swarm_interfaces.msg import TaskMap
from swarm_interfaces.srv import Tracking, GetTaskMap
from datetime import datetime
from mission import batch_processing


class MissionNode(Node):
    def __init__(self):
        super().__init__('mission_node')
        self.declare_parameter('plan_type', 1)
        self.declare_parameter("num_vehicles", 1)
        self.declare_parameter('n_points', 500)
        self.declare_parameter('mission_frame', [108.28299713134766, -94.181564331054688, 15.0])
        self.declare_parameter('mission_radius', 20.0)
        self.declare_parameter('mission_height', 17.0)
        self.declare_parameter('step_size', 1.0)
        self.declare_parameter('n_steps', 2000)
        self.declare_parameter('space_coef', .8)
        self.declare_parameter('time_coef', .2)
        self.declare_parameter('avg_speed', 1.0)
        self.declare_parameter('spatial_tol', .45)
        self.declare_parameter('time_tol', 100.0)
        self.declare_parameter('cylinder_height', 2.0)
        self.declare_parameter('cylinder_radius', 1.5)
        
        self.plan_type = self.get_parameter('plan_type').get_parameter_value().integer_value
        self.num_vehicles = self.get_parameter('num_vehicles').get_parameter_value().integer_value
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

        self.multi_offboard_controller = None
        self.obstacle_poses = {}
        self.detected_ids = set()
        self.planner = None

        # Cooperative perception: the ArUco detectors on board each UAV only
        # process images while this is true; the global fusion module
        # (perception/fusion_node) publishes the fused task map.
        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.perception_active_pub = self.create_publisher(Bool, "/perception/active", latched)
        self.set_perception_active(False)
        self.create_subscription(TaskMap, "/perception/task_map", self.task_map_callback, latched)

        self._perception_start_time = None
        self._assignation_start_time = None
        self._execution_start_time = None
        self._execution_end_time = None

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
        takeoff_future = self.multi_offboard_controller.take_off_all(5.0)
        while not takeoff_future.done():
            rclpy.spin_once(self, timeout_sec=2.0)

        trajectories = []
        while not self._check_trajectories(trajectories):
            self.get_logger().info(f"Computing initial trajectory.")
            future = self.planner.get_initial_trajectory(self.get_vehicle_poses(), self.obstacle_poses)
            while not future.done():
                rclpy.spin_once(self, timeout_sec=2.0)
            assigned_uavs, trajectories = future.result()

        self.get_logger().info(f"Vehicles going to initial formation.")
        future = self.multi_offboard_controller.trajectory_following(
            assigned_uavs,
            self.transform_to_local_frame_trajectory(trajectories)
        )
        while not future.done():
            rclpy.spin_once(self, timeout_sec=2.0)


    def run_perception(self):
        
        assigned_uavs, trajectories = self.planner.get_perception_trajectory()

        self.get_logger().info(f"Starting Perception")   
        self.set_perception_active(True)
        self._perception_start_time = time.perf_counter()
        future = self.multi_offboard_controller.trajectory_following(
            assigned_uavs,
            self.transform_to_local_frame_trajectory(trajectories)  
        )
        while not future.done():
            rclpy.spin_once(self, timeout_sec=2.0)

        self.set_perception_active(False)
        end_time = time.perf_counter()
        elapsed = end_time - self._perception_start_time
        self.get_logger().info(f"Perception finish.")
        self.get_logger().info(f"Time elapsed: {elapsed}")
        

    def execute_mission(self, plan_type: int): 
        
        targets_poses = []
        tasks_ids = []
        trajectories = []
        assigned_uavs = []

        for task_id, position in self.get_task_map():
            pose = Pose()
            pose.position.x = position.x
            pose.position.y = position.y
            pose.position.z = self.mission_height
            targets_poses.append(pose)
            tasks_ids.append(task_id)

        if not targets_poses:
            self.get_logger().warn("No tasks detected during perception: landing.")
            future = self.multi_offboard_controller.land_all()
            while not future.done():
                rclpy.spin_once(self, timeout_sec=2.0)
            self.multi_offboard_controller.hold_all()
            self.multi_offboard_controller.disarm_all()
            return

        assignation_initial_vehicle_poses = self.get_vehicle_poses()
        self._assignation_start_time = time.perf_counter()
        while not self._check_trajectories(trajectories, len(assigned_uavs)):
            self.get_logger().info(f"Computing execution trajectory.")
            future = self.planner.get_tasks_planning(
                self.get_vehicle_poses(), 
                targets_poses,
                self.obstacle_poses,
                plan_type
            )
            while not future.done():
                rclpy.spin_once(self, timeout_sec=2.0)
            assigned_uavs, goal_ids, trajectories = future.result()

        trajectories = self.transform_to_local_frame_trajectory(trajectories)
        batch_processing.save_trajectories(self.mission_id, trajectories, self.avg_speed)
        self._execution_start_time = time.perf_counter()
        
        self.get_logger().info(f"Executing mission.")
        future = self.multi_offboard_controller.trajectory_following(
            assigned_uavs,
            trajectories
        )        
        while not future.done():
            rclpy.spin_once(self, timeout_sec=2.0)
        
        future = self.multi_offboard_controller.land_all()
        while not future.done():
            rclpy.spin_once(self, timeout_sec=2.0)
        self.multi_offboard_controller.hold_all()
        self.multi_offboard_controller.disarm_all()

        self._execution_end_time = time.perf_counter()
        
        for uav, goal in zip(assigned_uavs, goal_ids):
            batch_processing.save_assignation_row(
                self.mission_id,
                uav,
                assignation_initial_vehicle_poses[uav],
                tasks_ids[goal]   
            )

        batch_processing.save_mission_row(
            self.mission_id,
            self.num_vehicles,
            self.plan_type,
            self._perception_start_time,
            self._assignation_start_time,
            self._execution_start_time,
            self._execution_end_time,
            self.spatial_tol
        )

        
    def get_vehicle_poses(self):
        
        in_map_poses = []
        for uav_id, controller in enumerate(self.multi_offboard_controller.controllers, start=1):
            local_pose = enu_ned(controller.state_manager.state_repositories[States.LOCAL_POSITION].get().get_pose())
            in_map_poses.append(local_pose)
           
        return in_map_poses


    def get_tracked_poses(self):
        """Obstacle poses (static obstacles are known a priori)."""
        cli = self.create_client(Tracking, "tracking_service")
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for tracking service...")

        future = cli.call_async(Tracking.Request())
        while not future.done():
            rclpy.spin_once(self, timeout_sec=2.0)
        
        response = future.result()
        self.obstacle_poses = response.obstacle_poses


    def set_perception_active(self, active: bool):
        self.perception_active_pub.publish(Bool(data=active))


    def task_map_callback(self, msg):
        """Logs each task the first time it appears in the fused map."""
        for task_id, position in zip(msg.ids, msg.positions):
            if task_id in self.detected_ids:
                continue
            self.detected_ids.add(task_id)
            self.get_logger().info(f"New task detected: {task_id}")
            pose = Pose()
            pose.position = position
            batch_processing.save_perception_row(
                self.mission_id,
                task_id,
                pose,
                time.perf_counter()
            )


    def get_task_map(self):
        """Fused task map [(ID_n, p_n)] from the global fusion module."""
        cli = self.create_client(GetTaskMap, "/perception/get_task_map")
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for the global fusion module...")

        future = cli.call_async(GetTaskMap.Request())
        while not future.done():
            rclpy.spin_once(self, timeout_sec=2.0)

        task_map = future.result().map
        for task_id, p, n in zip(task_map.ids, task_map.positions, task_map.num_measurements):
            self.get_logger().info(f"Task {task_id}: ({p.x:.2f}, {p.y:.2f}) from {n} measurements")
        return list(zip(task_map.ids, task_map.positions))


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
        
        self.mission_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

        self.configuration_service = SimUAVSConfigurationService(
            self,
            list(range(1, self.num_vehicles+1))    
        )
        multi_offboard_controller_assembler = MultiOffboardControllerAssembler()
        self.multi_offboard_controller = multi_offboard_controller_assembler.assemble(
            self,
            self.configuration_service.get_offboard_configurations()
        ) 
        self.mission_frame_pose = Pose()
        self.mission_frame_pose.position.x = self.mission_frame[0]
        self.mission_frame_pose.position.y = self.mission_frame[1] 
        self.mission_frame_pose.position.z = self.mission_frame[2] 

        self.planner = Planner(
            self.mission_frame_pose,
            self.mission_radius,
            self.mission_height,
            self.num_vehicles,
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


    def _check_trajectories(self, trajectories, num_vehicles=None):
        
        if num_vehicles is None or num_vehicles == 0:
            num_vehicles = self.num_vehicles

        if len(trajectories) == 0:
            return False
        
        valid_trajectories = [traj for traj in trajectories if traj and all(isinstance(traj[i], list) and len(traj[i]) > 0 for i in range(4))]

        if len(valid_trajectories) != num_vehicles:
            self.get_logger().warn(
                f"Trajectories missing for some vehicles!"
                f"{len(valid_trajectories)}/{num_vehicles} valid"
            )
            return False
        return True

        
def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()    
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

