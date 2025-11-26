import numpy as np
from rtt_star_planner import RttStarPlanner


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

            goal_node, tree, n_iterations = planner.plan(
                s, g, SPEED, obstacles, BIAS_PROB,
                LIMIT, SPATIAL_TOL, TIME_TOL
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

            print(f"✔ Terminada: {key}")
            
    print("\n=== RESUMEN DE RESULTADOS ===")
    for key, data in results.items():
        print(f"{key}: cost={data['final_cost']}, time={data['final_time']}")

    return results



def path_to_obstacle(goal_node, radius):
    path = []
    n = goal_node
    while n is not None:
        path.append(n._position.copy())
        n = n._parent

    path.reverse()
    path = np.array(path)

    xs = path[:, 0]
    ys = path[:, 1]
    zs = path[:, 2]
    ts = path[:, 3]
    rs = np.full_like(ts, radius)

    return np.column_stack([xs, ys, zs, ts, rs])


# ---------------------------------------------------------------------
# PODADO DE LA RUTA
# ---------------------------------------------------------------------

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




# ---------------------------------------------------------------------
# GREEDY MULTI-PLANNING + PODADO
# ---------------------------------------------------------------------
def choose_and_plan(starts, goals,
                    lower_limit, upper_limit,
                    STEP_SIZE, N_STEPS,
                    SPACE_COEF, TIME_COEF,
                    SPEED, obstacles,
                    BIAS_PROB, LIMIT,
                    SPATIAL_TOL, TIME_TOL,
                    CYLINDER_HEIGHT, OBSTACLE_RADIUS):

    remaining_starts = list(starts)
    remaining_goals = list(goals)

    best_results = {}
    # assignments = []

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
        best_planner = None

        for key, data in current_results.items():
            if data["final_cost"] is not None and data["final_cost"] < best_cost:
                best_cost = data["final_cost"]
                best_key = key

        if best_key is None:
            print("⚠ No rutas válidas.")
            break

        print(f"✅ Mejor asignación: {best_key} con coste {best_cost:.4f}")

        chosen = current_results[best_key]

        # ------------------------------------------------------------------
        # 🔧 APLICAMOS EL PODADO A LA RUTA GANADORA
        # ------------------------------------------------------------------
        goal_node = chosen["goal_node"]
        planner = RttStarPlanner(
            lower_limit, upper_limit,
            STEP_SIZE, N_STEPS,
            SPACE_COEF, TIME_COEF
        )
        planner._tree = chosen["tree"]

        pruned_goal = prune_path(goal_node, planner, obstacles)

        chosen["goal_node"] = pruned_goal
        chosen["final_cost"] = pruned_goal._cost
        chosen["final_time"] = pruned_goal._position[3]

        best_results[best_key] = chosen
        # assignments.append((best_key, best_cost))

        # añadir trayectoria como obstáculo
        obstacles.append(path_to_obstacle(pruned_goal, OBSTACLE_RADIUS))

        start_name, goal_name = best_key.split(" -> ")
        remaining_starts = [s for s in remaining_starts if s[0] != start_name]
        remaining_goals = [g for g in remaining_goals if g[0] != goal_name]

        step += 1

    print("\n=== ASIGNACIONES FINALES ===")
    # for key, cost in assignments:
    #     print(f"{key}: coste = {cost:.4f}")

    return best_results
