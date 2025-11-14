from geometry_msgs.msg import Pose
from planning.utils import generate_loiter_formation
from planning.rtt_star_planner import RttStarPlanner

class Planner():
    
    
    def __init__(
            self,
            mission_frame: Pose,
            mission_radius: float,
            mission_height: float,
            n_vehicles: int,
            step_size: float,
            n_steps: int,
            space_coef: float,
            time_coef: float
        ):
        self._mission_frame = mission_frame
        self._mission_frame.position.z = mission_height

        self.limit = True
        self.bias_prob = .6
        
        #Bounding box
        self.rtt_start_planner = RttStarPlanner(
            lower_limit=[mission_frame.position.x - mission_radius, mission_frame.position.y - mission_radius],
            upper_limit=[mission_frame.position.x + mission_radius, mission_frame.position.y + mission_radius],
            step_size=step_size,
            n_steps=n_steps,
            space_coef=space_coef,
            time_coef=time_coef
        )
        
        self.perception_trajectories = generate_loiter_formation(
            mission_frame,
            mission_radius,
            n_vehicles
        )
        self.assigned_vehicles = [] 


    def get_initial_trajectory(self, vehicle_poses):
        trajectories, self.asigned_vehicles = self.rtt_star_planner.plan(
            vehicle_poses,
            [self.perception_trajectories[n][0][0] for n in range(len(self.perception_trajectories))]
        )
        return trajectories, self.asigned_vehicles


    def get_perception_trajectory(self):
        return self.perception_trajectories, self.asigned_vehicles

    #TODO For multiple vehicles
    def get_execution_planning(self, vehicle_poses, target_poses, avg_speed, obstacles):
        trajectories, self.asigned_vehicles = self.rtt_star_planner.plan(
            vehicle_poses,
            target_poses,
            avg_speed,
            obstacles,
            bias=self.rtt_bias,
            limit=self.limit,
        )
        return trajectories, self.asigned_vehicles