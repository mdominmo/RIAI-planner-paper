import numpy as np


def compute_cost(child, parent):
    return np.linalg.norm(child._position - parent._position) + parent._cost


class Node():
    def __init__(
            self,
            position
        ):

        self._position = position
        self._parent = None
        self._cost = .0
    

    def set_parent(self, parent):
        self._parent = parent
        self._cost = compute_cost(self, self._parent)


class RttStarPlanner():

    def __init__(
            self,
            lower_limit,
            upper_limit,
            step_size,
            n_steps,
            theta_gamma = 1.1
        ):
        
        self._tree = []
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit
        self._step_size = step_size
        self._theta_gamma = theta_gamma
        self._n_steps = n_steps


    def _q_rand(self) -> np.ndarray:
        return np.random.uniform(low=self._lower_limit, high=self._upper_limit)


    def _q_nearest(self, q_rand):
        positions = np.array([node._position for node in self._tree])
        distances = np.linalg.norm(positions - q_rand, axis=1)
        return self._tree[np.argmin(distances)]._position

    
    def _q_new(self, q_rand, q_nearest):
        return q_nearest + self._step_size * (q_rand - q_nearest) / np.linalg.norm(q_rand - q_nearest)
    

    def _neighbor_radius(self):

        #Karaman & Frazzoli 2011
        d = len(self._upper_limit)
        n = len(self._tree)

        free_region = np.prod([self._upper_limit[i] - self._lower_limit[i] for i in range(len(self._lower_limit))])
        gamma = 2 * (1 + 1/d)**(1/d) * free_region**(1/d) * self._theta_gamma

        # return max(self._step_size * 2.5, gamma * (np.log(n)/n)**(1/d))
        return self._step_size * 2.5

    def _find_neighbors(self, q_new, radius):
        
        neighbors = [node for node in self._tree if np.linalg.norm(node._position - q_new) < radius]
        if not neighbors:
            return None, None    
        best_parent = np.argmin([n._cost + np.linalg.norm(q_new - n._position) for n in neighbors])
    
        return neighbors.pop(best_parent), neighbors
    

    def _rewire(self, neighbors, new_node):
        
        for neighbor in neighbors:
            through_q_new_cost = compute_cost(neighbor, new_node)
            if through_q_new_cost < neighbor._cost:
                neighbor.set_parent(new_node)

    
    def _check_restrictions(self, q_new):
        # TODO
        return True


    def solve(self, start, goal, tol=2.0):
        
        goal_node = None
        self._tree.append(Node(start))
        
        for _ in range(self._n_steps):

            q_rand = self._q_rand()
            q_nearest = self._q_nearest(q_rand)
            q_new = self._q_new(q_rand, q_nearest)
            
            if self._check_restrictions(q_new):
                
                new_node = Node(q_new)
                best_parent, neighbors = self._find_neighbors(q_new, self._neighbor_radius())
                
                if best_parent:
                    new_node.set_parent(best_parent)
                
                self._tree.append(new_node)
                
                if neighbors:
                    self._rewire(neighbors, new_node)

                if np.linalg.norm(new_node._position - goal) < tol:
                    goal_node = new_node
                    break
                
        return goal_node, self._tree
