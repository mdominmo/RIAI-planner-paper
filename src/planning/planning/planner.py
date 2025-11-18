from geometry_msgs.msg import Pose
from .utils import generate_loiter_formation, square_bounds_from_circle
from .multi_rrt_star_planner import MultiRRTStarPlanner
from .hungarian_tasks_planner import HungarianTasksPlanner
from .assignation_methods import AssignationMethods
import numpy as np


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
            time_coef: float,
            avg_speed: float,
            spatial_tol: float,
            time_tol: float,
            cylinder_height: float,
            obstacle_radius: float
        ):
        self._t_final = 70.0
        self._mission_frame = mission_frame
        self._step_size = step_size
        self._n_steps = n_steps
        self._space_coef = space_coef
        self._time_coef = time_coef
        self._limit = True
        self._bias_prob = .8
        self._avg_speed = avg_speed
        self._spatial_tol = spatial_tol
        self._time_tol = time_tol
        self._cylinder_height = cylinder_height
        self._obstacle_radius = obstacle_radius
        self.rtt_planner = MultiRRTStarPlanner()
        self.hungarian_planner = HungarianTasksPlanner()

        self._mission_frame.position.z = mission_height
        self.perception_trajectories = generate_loiter_formation(
            center=mission_frame,
            radius=mission_radius,
            n_drones=n_vehicles,
            n_points=200,
            speed=self._avg_speed
        )
        self.assigned_vehicles = [] 
        self._lower_limit, self._upper_limit = square_bounds_from_circle(
            mission_frame,
            mission_radius
        )
        self._lower_limit.append(.0)
        self._upper_limit.append(mission_height)


    def get_initial_trajectory(self, vehicle_poses, obstacles_poses):
    
        start_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, .0])) for n, p in enumerate(vehicle_poses)]
        goal_poses = [self.perception_trajectories[n][0][0] for n in range(len(vehicle_poses))]
        goal_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, self._t_final])) for n, p in enumerate(goal_poses)]
        
        return self.multi_rrt_plan(start_poses, goal_poses, self._model_static_obstacles(obstacles_poses))
  

    def get_perception_trajectory(self):
        return self.perception_trajectories


    def get_tasks_planning(
        self, 
        vehicle_poses, 
        goal_poses, 
        obstacles_poses,
        plan_type
    ):  
        match plan_type:
            
            case AssignationMethods.ONLY_RRT_STAR:
                start_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, .0])) for n, p in enumerate(vehicle_poses)]
                goal_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, self._t_final])) for n, p in enumerate(goal_poses)]

                return self.multi_rrt_plan(start_poses, goal_poses, self._model_static_obstacles(obstacles_poses))
            
            #TODO
            case AssignationMethods.RRT_STAR_HUNGARIAN:
                start_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, .0])) for n, p in enumerate(vehicle_poses)]
                goal_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, self._t_final])) for n, p in enumerate(goal_poses)]

                return self.multi_rrt_plan(start_poses, goal_poses, self._model_static_obstacles(obstacles_poses))
            
            case _:
                start_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, .0])) for n, p in enumerate(vehicle_poses)]
                goal_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, self._t_final])) for n, p in enumerate(goal_poses)]
                
                return self.multi_rrt_plan(start_poses, goal_poses, self._model_static_obstacles(obstacles_poses))
            

    def _model_static_obstacles(self, obstacles_poses):
        
        obstacles = []
        for obstacle in obstacles_poses:
            delta_t = .5
            t_obs = np.arange(0, self._t_final + delta_t, delta_t)
            n_points = len(t_obs)
            x_obs = np.full(n_points, obstacle.position.x)
            y_obs = np.full(n_points, obstacle.position.y)
            z_obs = np.full(n_points, self._cylinder_height)
            r_obs = np.full(n_points, self._obstacle_radius)

            obstacle = np.column_stack([x_obs, y_obs, z_obs, t_obs, r_obs])
            obstacles.append(obstacle)
        
        return obstacles


    def multi_rrt_plan(self, start_poses, goal_poses, obstacles):
        return self.rtt_planner.plan(
            start_poses,        
            goal_poses,
            self._lower_limit,
            self._upper_limit,
            self._step_size,
            self._n_steps,
            self._space_coef,
            self._time_coef,
            self._avg_speed,
            obstacles,
            self._bias_prob,
            self._limit,
            self._spatial_tol,
            self._time_tol,
            self._cylinder_height,
            self._obstacle_radius
        )