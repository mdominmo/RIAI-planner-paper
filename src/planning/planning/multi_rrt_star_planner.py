from .rtt_star_planner import RttStarPlanner 
import numpy as np
from geometry_msgs.msg import Pose, Twist


class MultiRRTStarPlanner():
    def __init__(self):
        pass


    def run_all_combinations(self, starts, goals,
                            lower_limit, upper_limit,
                            STEP_SIZE, N_STEPS,
                            SPACE_COEF, TIME_COEF,
                            SPEED, obstacles,
                            BIAS_PROB, LIMIT,
                            SPATIAL_TOL, TIME_TOL):
        
        results = {}

        for s_name, s in starts:
            for g_name, g in goals:

                planner = RttStarPlanner(
                    lower_limit,
                    upper_limit,
                    STEP_SIZE,
                    N_STEPS,
                    SPACE_COEF,
                    TIME_COEF
                )
                goal_node, tree, n_iterations = planner.plan(
                    s, g, 
                    SPEED, obstacles, 
                    BIAS_PROB, LIMIT, 
                    SPATIAL_TOL, TIME_TOL
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


    def choose_and_plan(self, starts, goals,
                        lower_limit, upper_limit,
                        STEP_SIZE, N_STEPS,
                        SPACE_COEF, TIME_COEF,
                        SPEED, obstacles,
                        BIAS_PROB, LIMIT,
                        SPATIAL_TOL, TIME_TOL,CYLINDER_HEIGHT,OBSTACLE_RADIUS):
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
        # assignments = [] 

        step = 1
        while remaining_starts and remaining_goals:
            current_results = self.run_all_combinations(
                remaining_starts, remaining_goals,
                lower_limit, upper_limit,
                STEP_SIZE, N_STEPS,
                SPACE_COEF, TIME_COEF,
                SPEED, obstacles,
                BIAS_PROB, LIMIT,
                SPATIAL_TOL, TIME_TOL
            )
            # print(f"{current_results}")
            best_key = None
            best_cost = np.inf

            for key, data in current_results.items():
                cost = data["final_cost"]
                if cost is not None and cost < best_cost:
                    best_cost = cost
                    best_key = key

            if best_key is None:
                break

            # best_results[best_key] = current_results[best_key]
            chosen = current_results[best_key]

            goal_node = chosen["goal_node"]
            planner = RttStarPlanner(
                lower_limit, upper_limit,
                STEP_SIZE, N_STEPS,
                SPACE_COEF, TIME_COEF
            )
            planner._tree = chosen["tree"]

            pruned_goal = self.prune_path(goal_node, planner, obstacles)

            chosen["goal_node"] = pruned_goal
            chosen["final_cost"] = pruned_goal._cost
            chosen["final_time"] = pruned_goal._position[3]

            best_results[best_key] = chosen
            # assignments.append((best_key, best_cost))


            

            # best_goal_node = current_results[best_key]["goal_node"]
            if pruned_goal is not None:
                path_obstacle = self.path_to_obstacle(pruned_goal, OBSTACLE_RADIUS)
                obstacles.append(path_obstacle)

            start_name, goal_name = best_key.split(" -> ")

            remaining_starts = [s for s in remaining_starts if s[0] != start_name]
            remaining_goals  = [g for g in remaining_goals  if g[0] != goal_name]
            step += 1

        return best_results
    
    def prune_path(goal_node, planner, obstacles):
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
                if planner._check_restrictions(path[j], pruned[-1], obstacles):
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
            pruned[k].set_parent(pruned[k-1], planner._space_coef, planner._time_coef)

        return pruned[-1]  # nuevo goal_node podado


    def plan(
        self, 
        starts, 
        goals,
        lower_limit, 
        upper_limit,
        STEP_SIZE, 
        N_STEPS,
        SPACE_COEF, 
        TIME_COEF,
        SPEED, 
        obstacles,
        BIAS_PROB, 
        LIMIT,
        SPATIAL_TOL, 
        TIME_TOL,
        CYLINDER_HEIGHT,
        OBSTACLE_RADIUS
    ):
        results = self.choose_and_plan(
            starts, goals,
            lower_limit, upper_limit,
            STEP_SIZE, N_STEPS,
            SPACE_COEF, TIME_COEF,
            SPEED, obstacles,
            BIAS_PROB, LIMIT,
            SPATIAL_TOL, TIME_TOL,CYLINDER_HEIGHT,OBSTACLE_RADIUS
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