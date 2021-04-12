#%%
import sys
sys.path.append("../..")

from tqdm import tqdm
import numpy as np
from itertools import product
import scipy

from solvers.rigidity_solver.algo_core import spring_energy_matrix
import util.geometry_util as geo_util

#%%

x_range = np.linspace(-3, 3, num=500 // 4)
y_range = np.linspace(-0.2, 1.3, num=125 // 4)

xy_range = product(x_range, y_range)

dim = 3
points = np.array([
    [-0.5, 0, 0],
    [0.5, 0, 0],
    [0, 1, 0],
])

init_points = np.array([
    [-0.5, 0, 0],
    [0.5, 0, 0],
    [0, 1, 0],
])

edges = np.array([
    [0, 1],
    [1, 2],
    [2, 0],
])

init_K = spring_energy_matrix(init_points, edges, dim=dim)

objectives = []
energies = []
pair_trace = []

fix_2_points = True
print("Fix 2 points?", fix_2_points)

for x, y in tqdm(xy_range):
    points[2, 0] = x
    points[2, 1] = y

    K = spring_energy_matrix(points, edges, dim=dim)
    delta_x2 = points[2] - init_points[2]
    energy = delta_x2.T @ init_K @ delta_x2

    ## fix points
    if fix_2_points:
        constr = np.hstack((np.eye(6), np.zeros((6, 3))))
        B = scipy.linalg.null_space(constr)
        T = np.transpose(B) @ B
        S = B.T @ K @ B
        L = np.linalg.cholesky(T)
        L_inv = np.linalg.inv(L)
        Q = L_inv.T @ S @ L_inv
    else:
        Q = K

    pairs = geo_util.eigen(Q, symmetric=True)

    if fix_2_points:
        obj = pairs[1][0] # the first eigenvalue is always 0, vector pointing z axis
    else:
        obj = pairs[6][0]

    objectives.append(energy)
    pair_trace.append(pairs)

#%%
# if fix_2_points:
#     objectives = [pairs[1][0] for pairs in pair_trace]
# else:
#     objectives = [pairs[6][0] for pairs in pair_trace]

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D

Z = np.array(objectives).reshape(len(x_range), len(y_range)).transpose()

plot_mode = "contour"
# plot_mode = "3d"

print(f"plot_mode: {plot_mode}")

if plot_mode == "3d":
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y = np.meshgrid(x_range, y_range)
    print(X)
    print(Y)
    print(X.shape, Y.shape)
    ax = plt.gca()
    scale_x, scale_y, scale_z = 1, 1, 0.5
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=True)
    ax.set_aspect('auto')
elif plot_mode == "contour":
    Z = np.array(objectives).reshape(len(x_range), len(y_range)).transpose()
    fig = plt.figure(figsize=(16, 4), facecolor=(0.95, 0.95, 0.95), dpi=100)
    ax = plt.gca()
    ax.set_aspect('auto')
    plt.contour(x_range, y_range, Z, levels=50)

plt.savefig(f"eigenvalues-{plot_mode}-{np.max(x_range)}-{np.max(y_range)}.png", bbox_inches="tight")
y_ind, x_ind = np.unravel_index(np.argmax(Z), Z.shape)
# plt.scatter([x_range[x_ind]], [y_range[y_ind]])
plt.show()
