import numpy as np
import matplotlib.pyplot as plt
from rtt_star_planner import RttStarPlanner

upper_limit = np.array([20.0, 20.0])
lower_limit = np.array([0.0, 0.0])

start = np.array([2.0, 0.0])
goal = np.array([19.0, 10.0])

step_size = 1.0
n_steps = 1000
tol = .5

planner = RttStarPlanner(
    lower_limit,
    upper_limit,
    step_size,
    n_steps
)

goal_node, tree = planner.solve(start, goal, tol)

for node in tree:
    if node._parent:
        plt.plot([node._position[0], node._parent._position[0]],
                [node._position[1], node._parent._position[1]],
                color='blue', linewidth=0.5, alpha=0.5)

n = goal_node
while n._parent:
    plt.plot([n._position[0], n._parent._position[0]],
            [n._position[1], n._parent._position[1]],
            color='green', linewidth=1.5, alpha=1.0)
    n = n._parent

x = [node._position[0] for node in tree]
y = [node._position[1] for node in tree]

plt.scatter(x, y, color='red', marker='o', label='Nodes')
plt.scatter(*start, color='green', s=100, marker='o', label='start')
plt.scatter(*goal, color='red', s=100, marker='*', label='goal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'RTT Tree*, Iterations: {len(tree)}, tol: {tol} m, step_size: {step_size} m')
plt.legend()
plt.grid(True)

plt.show()