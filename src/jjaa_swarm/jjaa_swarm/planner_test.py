import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
from rtt_star_planner import RttStarPlanner 

CYLINDER_HEIGHT = 1.3 
OBSTACLE_RADIUS = 2.5
Z_BASE = 0.0 
BIAS_PROB = .5

upper_limit = np.array([20.0, 20.0, 1.5])
lower_limit = np.array([0.0, 0.0, 0.0])

start = np.array([0.0, 0.0, 1.5])
goal = np.array([20.0, 20.0, 1.5])

obstacles = np.array([[5.0, 5.0, CYLINDER_HEIGHT, OBSTACLE_RADIUS],[12.0, 6.0, CYLINDER_HEIGHT, OBSTACLE_RADIUS],[4.0, 3.0, CYLINDER_HEIGHT, OBSTACLE_RADIUS],[3.0, 10.0, CYLINDER_HEIGHT, OBSTACLE_RADIUS], [15.0, 15.0, CYLINDER_HEIGHT, OBSTACLE_RADIUS]])

step_size = 1.0
n_steps = 2000
tol = 1.0

planner = RttStarPlanner(
    lower_limit,
    upper_limit,
    step_size,
    n_steps
)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

for obstacle in obstacles:

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

goal_node, tree, n_iterations = planner.solve(start, goal, obstacles, BIAS_PROB, True, tol)

for node in tree:
    if node._parent:
        ax.plot([node._position[0], node._parent._position[0]],
                [node._position[1], node._parent._position[1]],
                [node._position[2], node._parent._position[2]],
                color='blue', linewidth=0.5, alpha=0.5)

if goal_node:
    n = goal_node
    while n._parent:
        ax.plot([n._position[0], n._parent._position[0]],
                [n._position[1], n._parent._position[1]],
                [n._position[2], n._parent._position[2]],
                color='green', linewidth=4.5, alpha=1.0, label='Path')
        n = n._parent

x = [node._position[0] for node in tree]
y = [node._position[1] for node in tree]
z = [node._position[2] for node in tree]

ax.scatter(x, y, z, color='red', marker='.', s=10, label='Nodes')
ax.scatter(start[0], start[1], start[2], color='green', s=100, marker='o', label='Start')
ax.scatter(goal[0], goal[1], goal[2], color='red', s=100, marker='*', label='Goal')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(lower_limit[0], upper_limit[0])
ax.set_ylim(lower_limit[1], upper_limit[1])
ax.set_zlim(lower_limit[2], upper_limit[2])
ax.set_title(f'RTT Tree*, Iterations: {n_iterations}, tol: {tol} m, step_size: {step_size} m, bias: {BIAS_PROB}.')

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
ax.grid(True)

plt.show()