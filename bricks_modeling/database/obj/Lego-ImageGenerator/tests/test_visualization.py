from visualization.model_visualizer import visualize_2D
from util import geometry_util as geo
from scipy.linalg import null_space
import numpy as np

# points = np.array([
#     [-80, 223],
#     [-204, -250],
#     [85, 228],
#     [100, -260]
# ]) / 100

points = np.array([
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
])

edges = ((0, 1), (2, 3))

rot90 = np.array([
    [0, 1],
    [-1, 0]
])

rigid_motion_whole = geo.trivial_basis(points)
rigid_motion_parts = np.block([
    [points[0] - points[1], np.zeros(2), np.zeros(4)],
    [np.zeros(4), points[2] - points[3], np.zeros(2)]
])
joint_motion = np.concatenate([rot90 @ p for p in points[:2]] + [np.zeros(4)])
allowed = np.vstack((rigid_motion_whole, rigid_motion_parts, joint_motion))
prohibitive = null_space(allowed).T

allowed = np.array([
    [1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, -1, 0, 0, -1, 1, 0],

    [-1, 1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, -1, -1, 1],

    [0, 1, -1, 0, 0, 0, 0, 0],

    [0, -1, -1, -2, 0, 1, 1, 0],
    # [-3, -1, -1, 1, 3, 1, 1, -1]
])
prohibitive = null_space(allowed).T / 0.577350269

print(null_space(geo.trivial_basis(points[:2])).T)

print(prohibitive)

import matplotlib.pyplot as plt
for motion in prohibitive:
    vectors = motion.reshape(-1, 2)
    print(vectors)
    plt.scatter([0], [0], s=10)
    visualize_2D(points, edges, arrows=vectors)