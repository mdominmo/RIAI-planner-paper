import numpy as np


def compute_cost(child_position, parent, space_coef, time_coef):
    spatial_cost = np.linalg.norm(child_position[:3] - parent._position[:3])
    temporal_cost = abs(child_position[3] - parent._position[3])
    return parent._cost + space_coef * spatial_cost + time_coef * temporal_cost


class Node():
    def __init__(
            self,
            position
        ):

        self._position = position
        self._parent = None
        self._cost = .0
    

    def set_parent(
        self, 
        parent, 
        space_coef, 
        time_coef
    ):    
        self._parent = parent
        self._cost = compute_cost(
            self._position, 
            self._parent,
            space_coef,
            time_coef
        )


class RttStarPlanner():

    def __init__(
            self,
            lower_limit,
            upper_limit,
            step_size,
            n_steps,
            space_coef,
            time_coef,
            theta_gamma = 1.1
        ):
        
        self._tree = []
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit
        self._step_size = step_size
        self._theta_gamma = theta_gamma
        self._n_steps = n_steps
        self._space_coef = space_coef
        self._time_coef = time_coef

        self._delta_t = .5


    def _q_rand(self, goal=None, prob=.0) -> np.ndarray:
        if goal is not None and np.random.rand() < prob:
            return goal[:3]  # Only spatial samples.
        return np.random.uniform(low=self._lower_limit[:3], high=self._upper_limit[:3])  


    def _q_nearest(self, q_rand):
        positions = np.array([node._position for node in self._tree])
        distances = np.linalg.norm(positions[:,:3] - q_rand, axis=1)
        return self._tree[np.argmin(distances)]._position

    
    def _q_new(self, q_rand, q_nearest, speed=5.0):
        new_space = q_nearest[:3] + self._step_size * (q_rand[:3] - q_nearest[:3]) / np.linalg.norm(q_rand[:3] - q_nearest[:3])
        travel_dist = np.linalg.norm(new_space - q_nearest[:3])
        new_time = q_nearest[3] + travel_dist / speed
        
        return np.concatenate([new_space, [new_time]])


    def _neighbor_radius(self):
        return self._step_size * 2.5


    def _find_neighbors(self, q_new, radius):
        
        neighbors = [node for node in self._tree if np.linalg.norm(node._position - q_new) < radius]
        
        if not neighbors:
            return None, None    
        
        best_parent = np.argmin([compute_cost(q_new, n, self._space_coef, self._time_coef) for n in neighbors])
        return neighbors.pop(best_parent), neighbors
    

    def _rewire(self, neighbors, new_node, obstacles):
        
        for neighbor in neighbors:

            through_q_new_cost = compute_cost(neighbor._position, new_node, self._space_coef, self._time_coef)
            if through_q_new_cost < neighbor._cost:
                if self._check_restrictions(neighbor, new_node, obstacles):
                    neighbor._parent = new_node
                    neighbor._cost = through_q_new_cost

    
    def _check_restrictions(self, node, parent, obstacles):
        
        # height restriction
        if node._position[2] <= 0.0 or node._position[2] >= self._upper_limit[2]:
            return False
        
        # Obstacle interception
        t_travel = node._position[3] - parent._position[3]
        if t_travel <= 1e-9:
            return False
        
        n_samples = max(2,int(t_travel / self._delta_t))
        t_samples = np.linspace(parent._position[3], node._position[3], n_samples)
        
        robot_interp = np.array([
                parent._position[:3] + 
                (node._position[:3] - parent._position[:3]) * 
                ((t - parent._position[3]) / t_travel) 
                for t in t_samples
        ])
        
        for obstacle in obstacles:
            
            if obstacle[-1,3] < t_samples[0] or obstacle[0,3] > t_samples[-1]:
                continue
            
            obstacle_interp = np.array([
                np.interp(t_samples, obstacle[:,3], obstacle[:,dim]) for dim in range(3)]).T
            
            for i in range(n_samples):
                
                dist_xy = np.linalg.norm(robot_interp[i][:2] - obstacle_interp[i][:2])
                dz = abs(robot_interp[i][2] - obstacle_interp[i][2])

                if dist_xy <= obstacle[0,4]:
                    if 0.0 <= dz <= obstacle[0,2]:
                        return False
        return True


    def plan(self, start, goal, speed, obstacles, bias_prob=.0, limit=False, tol_space=2.0, tol_time=5.0):
        
        goal_node = None
        self._tree.append(Node(start))
        
        for n_iterations in range(self._n_steps):
            
            q_rand = self._q_rand(goal, bias_prob)
            q_nearest = self._q_nearest(q_rand)
            q_new = self._q_new(q_rand, q_nearest, speed)
            
            new_node = Node(q_new)
            best_parent, neighbors = self._find_neighbors(q_new, self._neighbor_radius())
            
            if best_parent is not None:
                new_node.set_parent(best_parent, self._space_coef, self._time_coef)

                if self._check_restrictions(new_node, best_parent, obstacles):
                    self._tree.append(new_node)
                    if neighbors:
                        self._rewire(neighbors, new_node, obstacles)
                    spatial_dist = np.linalg.norm(new_node._position[:3] - goal[:3])
                    temporal_dist = abs(new_node._position[3] - goal[3])
                    
                    if spatial_dist < tol_space:
                        goal_node = new_node
                        if limit:
                            break
                
        return goal_node, self._tree, n_iterations+1
