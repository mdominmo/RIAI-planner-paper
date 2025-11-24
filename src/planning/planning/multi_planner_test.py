import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
from utils_rrt_start import choose_and_plan

CYLINDER_HEIGHT = 1.3 
OBSTACLE_RADIUS = 1.5
Z_BASE = 0.0 
BIAS_PROB = .8
SPACE_COEF = .2
TIME_COEF = .8
SPATIAL_TOL = 0.9
TIME_TOL = 50.0
SPEED = 5.0
LIMIT=True
STEP_SIZE = 1.0
N_STEPS = 4000

upper_limit = np.array([20.0, 20.0, 1.5])
lower_limit = np.array([0.0, 0.0, 0.0])

starts = [
    ("Start 1", np.array([0.0, 0.0, 1.5, .0])),
    ("Start 2", np.array([2.0, 18.0, 1.5, 0.0])),
    ("Start 3", np.array([2.0, 2.0, 1.5, 0.0])),
    ("Start 4", np.array([18.0, 18.0, 1.5, 0.0]))
]

goals = [
    ("Goal 1", np.array([20.0, 20.0, 1.5, 70.0])),
    ("Goal 2", np.array([18.0, 2.0,  1.5, 70.0])),
    ("Goal 3", np.array([10.0, 18.0, 1.5, 70.0])),
    ("Goal 4", np.array([18.0, 10.0, 1.5, 70.0]))
]

results = {}

obstacles_0 = np.array([[5.0, 5.0, CYLINDER_HEIGHT, OBSTACLE_RADIUS],
                      [12.0, 6.0, CYLINDER_HEIGHT, OBSTACLE_RADIUS],
                      [4.0, 3.0, CYLINDER_HEIGHT, OBSTACLE_RADIUS],
                      [3.0, 10.0, CYLINDER_HEIGHT, OBSTACLE_RADIUS], 
                      [15.0, 15.0, CYLINDER_HEIGHT, OBSTACLE_RADIUS]])
obstacles = []
for obstacle in obstacles_0:
    delta_t = 0.5
    t_obs = np.arange(0, 70 + delta_t, delta_t)
    n_points = len(t_obs)
    x_obs = np.full(n_points, obstacle[0])
    y_obs = np.full(n_points, obstacle[1])
    z_obs = np.full(n_points, CYLINDER_HEIGHT)
    r_obs = np.full(n_points, OBSTACLE_RADIUS)

    obstacle = np.column_stack([x_obs, y_obs, z_obs, t_obs, r_obs])
    obstacles.append(obstacle)
    


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

for obstacle in obstacles_0:

    Xc, Yc, R_dummy, R = obstacle[0], obstacle[1], obstacle[2], obstacle[3]
    u = np.linspace(0, 2 * np.pi, 50) 
    v = np.linspace(Z_BASE, Z_BASE + CYLINDER_HEIGHT, 50) 
    U, V = np.meshgrid(u, v)
    
    X = Xc + R * np.cos(U)
    Y = Yc + R * np.sin(U)
    Z = V
    
    ax.plot_surface(X, Y, Z, color='gray', alpha=0.5)
    
    X_base = Xc + R * np.cos(u)
    Y_base = Yc + R * np.sin(u)
    
    Z_base_const = Z_BASE * np.ones_like(X_base)
    Z_top_const = (Z_BASE + CYLINDER_HEIGHT) * np.ones_like(X_base)
    
    ax.plot_trisurf(X_base, Y_base, Z_base_const, color='gray', alpha=0.6) 
    ax.plot_trisurf(X_base, Y_base, Z_top_const, color='gray', alpha=0.6) 


results = choose_and_plan(
    starts, goals,
    lower_limit, upper_limit,
    STEP_SIZE, N_STEPS,
    SPACE_COEF, TIME_COEF,
    SPEED, obstacles,
    BIAS_PROB, LIMIT,
    SPATIAL_TOL, TIME_TOL,CYLINDER_HEIGHT,OBSTACLE_RADIUS
)



def debug_path_cost(goal_node, space_coef, time_coef):
    """Trace back path and calculate expected vs actual cost."""
    if goal_node is None:
        print("No goal node found")
        return
    
    path = []
    node = goal_node
    while node is not None:
        path.append(node)
        node = node._parent
    
    path.reverse()
    
    print(f"\n=== PATH ANALYSIS ===")
    print(f"Number of nodes in path: {len(path)}")
    print(f"Space coef: {space_coef}, Time coef: {time_coef}")
    
    accumulated_cost = 0.0
    accumulated_dist = 0.0
    accumulated_time = 0.0
    
    for i in range(len(path)):
        node = path[i]
        print(f"\nNode {i}:")
        print(f"  Position: {node._position}")
        print(f"  Stored cost: {node._cost:.4f}")
        
        if i > 0:
            parent = path[i-1]
            spatial_dist = np.linalg.norm(node._position[:3] - parent._position[:3])
            temporal_dist = abs(node._position[3] - parent._position[3])
            edge_cost = space_coef * spatial_dist + time_coef * temporal_dist
            
            accumulated_dist += spatial_dist
            accumulated_time += temporal_dist
            accumulated_cost += edge_cost
            
            print(f"  Edge from parent:")
            print(f"    Spatial dist: {spatial_dist:.4f} m")
            print(f"    Temporal dist: {temporal_dist:.4f} s")
            print(f"    Edge cost: {edge_cost:.4f}")
            print(f"    Expected accumulated: {accumulated_cost:.4f}")
            print(f"    Difference: {node._cost - accumulated_cost:.4f}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total spatial distance: {accumulated_dist:.4f} m")
    print(f"Total temporal distance: {accumulated_time:.4f} s")
    print(f"Expected final cost: {accumulated_cost:.4f}")
    print(f"Actual final cost: {goal_node._cost:.4f}")
    print(f"Final time: {goal_node._position[3]:.4f} s")

#debug_path_cost(goal_node, SPACE_COEF, TIME_COEF)


colors = ['green', 'blue', 'magenta', 'orange', 'cyan', 'purple', 'brown', 'black']

for i, (key, data) in enumerate(results.items()):
    goal_node = data["goal_node"]
    tree = data["tree"]

    color = colors[i % len(colors)]
    label = key  # para la leyenda

    # 1) (Opcional) Dibujar el árbol completo de esa planificación en finito
    for node in tree:
        if node._parent:
            ax.plot(
                [node._position[0], node._parent._position[0]],
                [node._position[1], node._parent._position[1]],
                [node._position[2], node._parent._position[2]],
                color=color,
                linewidth=0.3,
                alpha=0.2
            )

    # 2) Dibujar SOLO el camino óptimo de esa planificación más grueso
    if goal_node is not None:
        n = goal_node
        while n._parent:
            ax.plot(
                [n._position[0], n._parent._position[0]],
                [n._position[1], n._parent._position[1]],
                [n._position[2], n._parent._position[2]],
                color=color,
                linewidth=3.0,
                alpha=1.0,
                label=label
            )
            # Para que en la leyenda solo salga una vez cada clave
            label = None
            n = n._parent



start  = starts[0][1]
start_2 = starts[1][1]
start_3 = starts[2][1]
start_4 = starts[3][1]

goal  = goals[0][1]
goal_2 = goals[1][1]
goal_3 = goals[2][1]
goal_4 = goals[3][1]



ax.scatter(start[0], start[1], start[2], color='green', s=100, marker='o', label='Start')
ax.scatter(goal[0], goal[1], goal[2], color='red', s=100, marker='*', label='Goal')


ax.scatter(start_2[0], start_2[1], start_2[2], color='cyan', s=100, marker='^', label='Start 2')
ax.scatter(goal_2[0], goal_2[1], goal_2[2], color='magenta', s=100, marker='X', label='Goal 2')


ax.scatter(start_3[0], start_3[1], start_3[2], color='yellow', s=100, marker='o', label='Start 3')
ax.scatter(start_4[0], start_4[1], start_4[2], color='orange', s=100, marker='o', label='Start 4')

ax.scatter(goal_3[0], goal_3[1], goal_3[2], color='purple', s=100, marker='*', label='Goal 3')
ax.scatter(goal_4[0], goal_4[1], goal_4[2], color='brown', s=100, marker='*', label='Goal 4')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(lower_limit[0], upper_limit[0])
ax.set_ylim(lower_limit[1], upper_limit[1])
ax.set_zlim(lower_limit[2], upper_limit[2])

ax.set_title(f'RTT* Tree PLanner 3D.', fontsize=12)


parameter_summary = (
    f"Multi RRT* greedy\n"
    f"Num paths: {len(results)}\n"
    f"Tol: {SPATIAL_TOL} m\n"
    f"Spatial coeficient (α): {SPACE_COEF}\n"
    f"Temporal coeficient(β): {TIME_COEF}\n"
    f"Step size: {STEP_SIZE} m\n"
    f"Bias prob: {BIAS_PROB}"
)


fig.text(
    0.02, 
    0.02, 
    parameter_summary, 
    fontsize=9, 
    bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=0.5')
)

handles, labels = ax.get_legend_handles_labels()
obstacle_proxy = Rectangle((0, 0), 1, 1, fc='gray', alpha=0.5) 

unique_handles = {}
for h, l in zip(handles, labels):
    if l not in unique_handles:
        unique_handles[l] = h

if 'Obstacles' not in unique_handles:
    final_handles = [obstacle_proxy] + list(unique_handles.values())
    final_labels = ['Obstacles'] + list(unique_handles.keys())
else:
    final_handles = list(unique_handles.values())
    final_labels = list(unique_handles.keys())

ax.legend(final_handles, final_labels)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

ax.grid(True)
plt.tight_layout()
plt.show()