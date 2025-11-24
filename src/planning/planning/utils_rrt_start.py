from rtt_star_planner import RttStarPlanner, Node
import numpy as np
from typing import List



def run_all_combinations(starts, goals,
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

            goal_node, tree, n_iterations = planner.plan(s, g, SPEED, obstacles, BIAS_PROB, LIMIT, SPATIAL_TOL, TIME_TOL)

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
                "final_time": final_time #TODO
            }

            print(f"✔ Terminada: {key}")
            
    print("\n=== RESUMEN DE RESULTADOS ===")
    for key, data in results.items():
        print(
            f"{key}: "
            f"cost={data['final_cost']}, "
            f"time={data['final_time']}"
        )

    return results

def path_to_obstacle(goal_node, radius):
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


def choose_and_plan(starts, goals,
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
    assignments = [] 

    step = 1
    while remaining_starts and remaining_goals:
        print(f"\n=== ITERACIÓN GREEDY {step} ===")
        print("Starts disponibles:", [s[0] for s in remaining_starts])
        print("Goals disponibles:", [g[0] for g in remaining_goals])

        current_results = run_all_combinations(
            remaining_starts, remaining_goals,
            lower_limit, upper_limit,
            STEP_SIZE, N_STEPS,
            SPACE_COEF, TIME_COEF,
            SPEED, obstacles,
            BIAS_PROB, LIMIT,
            SPATIAL_TOL, TIME_TOL
        )

        best_key = None
        best_cost = np.inf

        for key, data in current_results.items():
            cost = data["final_cost"]
            if cost is not None and cost < best_cost:
                best_cost = cost
                best_key = key

        if best_key is None:
            print("⚠ No se ha encontrado ninguna ruta válida en esta iteración. Parando.")
            break

        print(f"✅ Mejor asignación en esta iteración: {best_key} con coste {best_cost:.4f}")

        best_results[best_key] = current_results[best_key]
        assignments.append((best_key, best_cost))

        best_goal_node = current_results[best_key]["goal_node"]

        if best_goal_node is not None:
            best_goal_node = prune_path_nodes(
                best_goal_node,
                obstacles,
                lower_limit,
                upper_limit,
                SPACE_COEF,
                TIME_COEF,
                STEP_SIZE
            )

            best_results[best_key]["goal_node"] = best_goal_node

            path_obstacle = path_to_obstacle(best_goal_node, OBSTACLE_RADIUS)
            obstacles.append(path_obstacle)


        start_name, goal_name = best_key.split(" -> ")

        remaining_starts = [s for s in remaining_starts if s[0] != start_name]
        remaining_goals  = [g for g in remaining_goals  if g[0] != goal_name]

        step += 1

    print("\n=== ASIGNACIONES FINALES (GREEDY) ===")
    for key, cost in assignments:
        print(f"{key}: coste = {cost:.4f}")

    return best_results


def extract_path_nodes(goal_node: Node) -> List[Node]:
    """Devuelve la lista de nodos desde start hasta goal_node."""
    path = []
    n = goal_node
    while n is not None:
        path.append(n)
        n = n._parent
    path.reverse()
    return path


def prune_path_nodes(
    goal_node: Node,
    obstacles: List[np.ndarray],
    lower_limit: np.ndarray,
    upper_limit: np.ndarray,
    space_coef: float,
    time_coef: float,
    step_size: float = 1.0
) -> Node:
    """
    Poda el path eliminando nodos intermedios si el atajo entre vecinos
    no colisiona. Devuelve el NUEVO goal_node cuyo _parent-chain es el
    path podado.
    """

    if goal_node is None:
        return None

    checker = RttStarPlanner(
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        step_size=step_size,
        n_steps=1,
        space_coef=space_coef,
        time_coef=time_coef
    )

    path = extract_path_nodes(goal_node)
    if len(path) <= 2:
        return goal_node

    i = 1
    while i < len(path) - 1:
        prev_node = path[i-1]
        next_node = path[i+1]

        if checker._check_restrictions(next_node, prev_node, obstacles):
            path.pop(i) 
        else:
            i += 1

    path[0]._parent = None
    path[0]._cost = 0.0

    for k in range(1, len(path)):
        path[k].set_parent(path[k-1], space_coef, time_coef)

    new_goal_node = path[-1]
    return new_goal_node
