#%%
import sys
sys.path.append("../..")

from tqdm import tqdm
import numpy as np
from itertools import product
import scipy

from solvers.rigidity_solver.algo_core import spring_energy_matrix, generalized_courant_fischer
import util.geometry_util as geo_util

#%%

x_range = np.linspace(-0.8, 0.8, num=300 // 4)
y_range = np.linspace(-0.8, 0.8, num=300 // 4)

xy_range = product(x_range, y_range)

dim = 3

case = "beam"
points = np.array([
    [0, -1, 0],
    [0, 0, 0],
])
edges = np.array([
    [0, 1],
])

init_points = np.copy(points)

constr = np.vstack((
    np.hstack((np.eye(3), np.zeros((3, 3)))),
    np.array([0, 0, 0, 0, 0, 1])
))

init_K = spring_energy_matrix(init_points, edges, dim=dim)
init_Q, init_B = generalized_courant_fischer(init_K, constr)
init_eigenpairs = geo_util.eigen(init_Q)
print(init_eigenpairs)

objectives = []
energies = []
pair_trace = []

fix_2_points = True
print("Fix 2 points?", fix_2_points)

for x, y in tqdm(xy_range):
    points[1, 0] = x
    points[1, 1] = y

    K = spring_energy_matrix(points, edges, dim=dim)
    delta_x2 = np.array([0, 0, 0, x, y, 0])
    energy = delta_x2.T @ init_K @ delta_x2

    ## fix points
    if fix_2_points:
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
        obj = pairs[0][0]  # the first eigenvalue is always 0, vector pointing z axis
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

# plot_mode = "contour"
plot_mode = "3d"

print(f"plot_mode: {plot_mode}")

if plot_mode == "3d":
    # fig = plt.figure(figsize=(50, 50))
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y = np.meshgrid(x_range, y_range)
    ax.set_zlim(0, 1.3)
    ax.view_init(azim=15)

    arrow_x = np.zeros((8,))
    arrow_y = np.zeros((8,))
    arrow_z = np.zeros((8,))
    # vectors = [-(init_B @ v)[6:] for e, v in init_eigenpairs]
    vectors = [geo_util.normalize(np.array((i, j, 0))) for i in (-1, 0, 1) for j in (-1, 0, 1) if i != 0 or j != 0]
    print(vectors)
    arrow_u, arrow_v, arrow_w = zip(*vectors)
    ax.quiver(arrow_x, arrow_y, arrow_z, arrow_u, arrow_v, arrow_w, length=0.2, colors=(1, 0, 0))

    scale_x, scale_y, scale_z = 1, 1, 1
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=True, zsort='min')
    ax.set_aspect('auto')
elif plot_mode == "contour":
    Z = np.array(objectives).reshape(len(x_range), len(y_range)).transpose()
    fig = plt.figure(facecolor=(0.95, 0.95, 0.95), dpi=100)
    ax = plt.gca()
    ax.set_aspect('auto')
    plt.contour(x_range, y_range, Z, levels=50)

plt.savefig(f"beam-eigenvalue-{plot_mode}-{np.max(x_range)}-{np.max(y_range)}.png", bbox_inches="tight",
            dpi=800)
y_ind, x_ind = np.unravel_index(np.argmax(Z), Z.shape)
# plt.scatter([x_range[x_ind]], [y_range[y_ind]])
plt.show()
