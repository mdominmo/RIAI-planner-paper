from geometry_msgs.msg import Pose
from .utils import generate_loiter_formation, square_bounds_from_circle, model_static_obstacles
from .multi_rrt_star_planner import MultiRRTStarPlanner
from .hungarian_tasks_planner import HungarianTasksPlanner
from .assignation_methods import AssignationMethods
import numpy as np

from rclpy.task import Future
from concurrent.futures import ThreadPoolExecutor


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
        self._executor = ThreadPoolExecutor(max_workers=1)
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
        self._theta_gamma = 1.1
        self._mission_frame.position.z = mission_height + 2.0
        
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
        
        self.hungarian_planner = HungarianTasksPlanner()
        self.rtt_planner = MultiRRTStarPlanner(
            self._lower_limit,
            self._upper_limit,
            self._step_size,
            self._n_steps,
            self._space_coef,
            self._time_coef,
            self._theta_gamma
        )


    def get_initial_trajectory(self, vehicle_poses, obstacles_poses):
        
        future = Future()

        def task():
            start_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, .0])) for n, p in enumerate(vehicle_poses)]
            goal_poses = [self.perception_trajectories[n][0][0] for n in range(len(vehicle_poses))]
            goal_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, self._t_final])) for n, p in enumerate(goal_poses)]
            
            future.set_result(self.multi_rrt_plan(start_poses, goal_poses, model_static_obstacles(
                obstacles_poses,
                self._t_final,
                self._cylinder_height,
                self._obstacle_radius
            )))
        
        self._executor.submit(task)
        return future
  

    def get_perception_trajectory(self):
        return self.perception_trajectories


    def get_tasks_planning(
        self, 
        vehicle_poses, 
        goal_poses, 
        obstacles_poses,
        plan_type
    ):  
        
        future = Future()

        def task():

            match plan_type:
                
                case AssignationMethods.ONLY_RRT_STAR:
                    start_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, .0])) for n, p in enumerate(vehicle_poses)]
                    goal_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, self._t_final])) for n, p in enumerate(goal_poses)]

                    future.set_result(self.multi_rrt_plan(
                        start_poses, 
                        goal_poses,     
                        model_static_obstacles(
                            obstacles_poses,
                            self._t_final,
                            self._cylinder_height,
                            self._obstacle_radius
                        )))
                
                case AssignationMethods.RRT_STAR_HUNGARIAN:
                    start_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, .0])) for n, p in enumerate(vehicle_poses)]
                    goal_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, self._t_final])) for n, p in enumerate(goal_poses)]

                    future.set_result(self.multi_rrt_hungarian_plan(
                        start_poses, 
                        goal_poses,     
                        model_static_obstacles(
                            obstacles_poses,
                            self._t_final,
                            self._cylinder_height,
                            self._obstacle_radius
                        )))
                
                case _:
                    start_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, .0])) for n, p in enumerate(vehicle_poses)]
                    goal_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, self._t_final])) for n, p in enumerate(goal_poses)]
                    
                    future.set_result(self.multi_rrt_plan(
                        start_poses, 
                        goal_poses,     
                        model_static_obstacles(
                            obstacles_poses,
                            self._t_final,
                            self._cylinder_height,
                            self._obstacle_radius
                        )))
                
        self._executor.submit(task)
        return future
            

    def multi_rrt_hungarian_plan(self, start_poses, goal_poses, obstacles):
        
        results = self.rtt_planner.run_all_combinations(
            start_poses,        
            goal_poses,
            self._avg_speed,
            obstacles,
            self._bias_prob,
            self._limit,
            self._spatial_tol,
            self._time_tol
        )

        dt = .5
        costs = [[.0 for _ in range(len(goal_poses))] for _ in range(len(start_poses))]
        for id in results.keys(): 

            agent_idx = int(id.split('->')[0].strip())
            goal_idx = int(id.split('->')[1].strip())
            costs[agent_idx][goal_idx] = results[id]["final_cost"]
        
        agent_idx, goal_idx = self.hungarian_planner.plan(costs) 
        goal_poses =  [goal_poses[i] for i in goal_idx]

        return self.rtt_planner.plan_paths(
            start_poses, goal_poses
        )


    def multi_rrt_plan(self, start_poses, goal_poses, obstacles):
        return self.rtt_planner.plan(
            start_poses,        
            goal_poses,
            self._avg_speed,
            obstacles,
            self._bias_prob,
            self._limit,
            self._spatial_tol,
            self._time_tol,
            self._cylinder_height,
            self._obstacle_radius
        )