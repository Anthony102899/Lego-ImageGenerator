from tqdm import tqdm
import numpy as np
from itertools import product

from solvers.rigidity_solver.algo_core import spring_energy_matrix
import util.geometry_util as geo_util

x_range = np.linspace(-3, 3, num=500 // 4)
y_range = np.linspace(-0.2, 1.3, num=125 // 4)

xy_range = product(x_range, y_range)

points = np.array([
    [-0.5, 0, 0],
    [0.5, 0, 0],
    [0, 1, 0],
])

edges = np.array([
    [0, 1],
    [1, 2],
    [2, 0],
])

objectives = []

for x, y in tqdm(xy_range):
    points[2, 0] = x
    points[2, 1] = y

    Q = spring_energy_matrix(points, edges)
    pairs = geo_util.eigen(Q, symmetric=True)

    objectives.append(pairs[6][0])


from matplotlib import pyplot as plt

Z = np.array(objectives).reshape(len(x_range), len(y_range)).transpose()
fig = plt.figure(figsize=(16, 4), facecolor=(0.95, 0.95, 0.95), dpi=100)
fig.patch.set_visible(False)
ax = plt.gca()
ax.set_aspect('auto')
plt.contour(x_range, y_range, Z, levels=50)
plt.axis("off")
# plt.axis("tight")
plt.savefig("eigenvalues.png", bbox_inches="tight")
y_ind, x_ind = np.unravel_index(np.argmax(Z), Z.shape)
# plt.scatter([x_range[x_ind]], [y_range[y_ind]])
plt.show()
