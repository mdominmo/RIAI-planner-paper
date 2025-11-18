from geometry_msgs.msg import Pose
from scipy.optimize import linear_sum_assignment
from math import sqrt

class HungarianTasksPlanner():
    
    
    def _euclidean_distance(self, pose1: Pose, pose2: Pose) -> float:
        dx = pose2.position.x - pose1.position.x
        dy = pose2.position.y - pose1.position.y
        dz = pose2.position.z - pose1.position.z
        return sqrt(dx**2 + dy**2 + dz**2)


    def build_cost_matrix(self, drones, targets):
        cost_matrix = []
        for drone in drones:
            row = [self._euclidean_distance(drone, target) for target in targets]
            cost_matrix.append(row)
        return cost_matrix
    

    def plan(self, cost_matrix):
        return linear_sum_assignment(cost_matrix)
    