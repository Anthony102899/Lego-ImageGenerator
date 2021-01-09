import numpy as np

from solvers.rigidity_solver.gradient import gradient_analysis
import testcases

model = testcases.equilateral_triangle()
points = model.point_matrix()
edges = model.edge_matrix()

hinges = model.joints
hinge_axes = np.array([h.axis for h in hinges])
hinge_pivots = np.array([h.pivot_point for h in hinges])
hinge_point_indices = model.joint_point_indices()

print(hinge_axes)
print(hinge_pivots)
print(hinge_point_indices)

gradient_analysis(points, edges, hinge_axes, hinge_pivots, hinge_point_indices)