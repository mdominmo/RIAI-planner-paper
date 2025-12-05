from .rtt_star_planner import RttStarPlanner 
import numpy as np
from geometry_msgs.msg import Pose, Twist
from .utils import plot_trajectories

class MultiRRTStarPlanner():
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
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit
        self._step_size = step_size
        self._theta_gamma = theta_gamma
        self._n_steps = n_steps
        self._space_coef = space_coef
        self._time_coef = time_coef

        self._planner = RttStarPlanner(
            self._lower_limit,
            self._upper_limit,
            self._step_size,
            self._n_steps,
            self._space_coef,
            self._time_coef
        )


    def plan_paths(
            self,
            start_poses, 
            goal_poses,
            speed, 
            obstacles,
            bias_prob, 
            limit,
            spatial_tol, 
            time_tol,
            obstacle_radius
        ):

        dt = .5
        trajectories = [[] for _ in range(len(start_poses))]
        for agent_idx, start_pose, goal_pose in enumerate(zip(start_poses, goal_poses)):
            
            poses = []
            dts = [dt*n for n in range(len(poses))]
            velocities = [Twist() for _ in range(len(poses))]
            yaws = [float('nan') for _ in range(len(poses))]

            goal_node, _, _ = self._planner.plan(
                start_pose, goal_pose, 
                speed, obstacles, 
                bias_prob, limit, 
                spatial_tol, time_tol
            )
            
            while(goal_node._parent is not None):
                p = Pose()
                p.position.x = node._position[0]
                p.position.y = node._position[1]
                p.position.z = node._position[2]
                poses.append(p)
                node = node._parent

            trajectories[agent_idx] = [
                poses[::-1],
                velocities,
                yaws,
                dts
            ]
            obstacles.append(
                self.path_to_obstacle(
                    goal_node, obstacle_radius)
            )

        return trajectories
            

    def run_all_combinations(
            self, 
            starts, goals,      
            speed, obstacles,
            bias_prob, limit,
            spatial_tol, time_tol
        ):
        
        results = {}

        for s_name, s in starts:
            for g_name, g in goals:

                planner = RttStarPlanner(
                    self._lower_limit,
                    self._upper_limit,
                    self._step_size,
                    self._n_steps,
                    self._space_coef,
                    self._time_coef
                )
                goal_node, tree, n_iterations = planner.plan(
                    s, g, 
                    speed, obstacles, 
                    bias_prob, limit, 
                    spatial_tol, time_tol
                )
                key = f"{s_name} -> {g_name}"

                if goal_node is not None:
                    final_cost = goal_node._cost
                    final_time = goal_node._position[3]
                else:
                    final_cost = None
                    final_time = None
                results[key] = {
                    "start": s,
                    "goal": g,
                    "goal_node": goal_node,
                    "tree": tree,
                    "n_iterations": n_iterations,
                    "final_cost": final_cost,
                    "final_time": final_time 
                }
                
        return results
    

    def path_to_obstacle(self, goal_node, radius):
        """
        Convierte el camino desde el start hasta goal_node
        en un "obstáculo dinámico" de radio `radius`.

        Devuelve un array de shape (N, 5):
        [x, y, z, t, r]
        """
        # Recorrer el path desde goal hasta el inicio
        path = []
        n = goal_node
        while n is not None:
            path.append(n._position.copy())
            n = n._parent

        # Invertimos para tenerlo en orden temporal creciente
        path.reverse()
        path = np.array(path)  # shape (N, 4) -> [x, y, z, t]

        xs = path[:, 0]
        ys = path[:, 1]
        zs = path[:, 2]
        ts = path[:, 3]
        rs = np.full_like(ts, radius)

        obstacle = np.column_stack([xs, ys, zs, ts, rs])
        return obstacle


    def choose_and_plan(
            self, 
            starts, 
            goals,
            speed, 
            obstacles,
            bias_prob, 
            limit,
            spatial_tol, 
            time_tol,
            obstacle_height,
            obstacle_radius
        ):
        """
        Devuelve best result que es un diccionario con esto por cada "Start X -> Goal Y" 

        {
                    "start": np.ndarray        # posición inicial 4D [x, y, z, t]
                    "goal": np.ndarray         # posición objetivo 4D [x, y, z, t]
                    "goal_node": Node          # nodo final alcanzado en el árbol
                    "tree": List[Node]         # todos los nodos generados para esa ruta
                    "n_iterations": int        # iteraciones realizadas
                    "final_cost": float | None # coste total del camino (si existe)
                    "final_time": float | None # tiempo final t del nodo objetivo
        }
        """
        remaining_starts = list(starts)
        remaining_goals = list(goals)

        best_results = {} 

        step = 1
        while remaining_starts and remaining_goals:
            current_results = self.run_all_combinations(
                remaining_starts, remaining_goals,
                speed, obstacles,
                bias_prob, limit,
                spatial_tol, time_tol
            )
            best_key = None
            best_cost = np.inf

            for key, data in current_results.items():
                cost = data["final_cost"]
                if cost is not None and cost < best_cost:
                    best_cost = cost
                    best_key = key

            if best_key is None:
                break

            chosen = current_results[best_key]
            goal_node = chosen["goal_node"]
            pruned_goal = self.prune_path(goal_node, obstacles)

            chosen["goal_node"] = pruned_goal
            chosen["final_cost"] = pruned_goal._cost
            chosen["final_time"] = pruned_goal._position[3]

            best_results[best_key] = chosen
            if pruned_goal is not None:
                path_obstacle = self.path_to_obstacle(pruned_goal, obstacle_radius)
                obstacles.append(path_obstacle)

            start_name, goal_name = best_key.split(" -> ")

            remaining_starts = [s for s in remaining_starts if s[0] != start_name]
            remaining_goals  = [g for g in remaining_goals  if g[0] != goal_name]
            step += 1

        return best_results
    

    def prune_path(
            self, 
            goal_node, 
            obstacles
        ):
        """
        Recibe goal_node y elimina nodos intermedios siempre que el segmento directo
        entre puntos consecutivos no tenga colisión.
        Reescribe los parents y recalcula costes.

        Devuelve el nodo final (goal_node) con el nuevo parent-chain.
        """

        # 1) obtener lista de nodos
        path = []
        n = goal_node
        while n is not None:
            path.append(n)
            n = n._parent
        path.reverse()  # orden correcto

        # 2) ejecutamos el podado greedy
        pruned = [path[0]]
        i = 0
        while True:
            j = i + 2
            last_valid = i + 1

            while j < len(path):
                if self._planner._check_restrictions(path[j], pruned[-1], obstacles):
                    last_valid = j
                    j += 1
                else:
                    break

            pruned.append(path[last_valid])

            if last_valid == len(path)-1:
                break

            i = last_valid

        # 3) reconstruir parent-chain y recalcular costes
        pruned[0]._parent = None
        pruned[0]._cost = 0.0

        for k in range(1, len(pruned)):
            pruned[k].set_parent(pruned[k-1], self._planner._space_coef, self._planner._time_coef)

        return pruned[-1]  # nuevo goal_node podado


    def plan(
        self, 
        starts, 
        goals,
        speed, 
        obstacles,
        bias_prob, 
        limit,
        spatial_tol, 
        time_tol,
        obstacle_height,
        obstacle_radius
    ):
        results = self.choose_and_plan(
            starts, goals,
            speed, obstacles,
            bias_prob, limit,
            spatial_tol, time_tol,obstacle_height, obstacle_radius
        )
    
        dt = .5
        trajectories = [[] for _ in range(len(starts))]
        for id in results.keys():  
            positions = []
            node = results[id]["goal_node"]
            while(node._parent is not None): 
                p = Pose()
                p.position.x = node._position[0]
                p.position.y = node._position[1]
                p.position.z = node._position[2]
                positions.append(p)
                node = node._parent
            
            positions = positions[::-1]
            dts = [dt*n for n in range(len(positions))]
            velocities = [Twist() for _ in range(len(positions))]
            yaws = [float('nan') for _ in range(len(positions))]
            
            id = int(id.split('->')[0].strip())
            print(f"{id}")
            trajectories[id] = [
                positions,
                velocities,
                yaws,
                dts
            ]
            
        return trajectories