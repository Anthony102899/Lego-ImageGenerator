#%%
import sys
sys.path.append("../..")

from collections import defaultdict

import solvers.rigidity_solver
from solvers.rigidity_solver.algo_core import *
from solvers.rigidity_solver.internal_structure import *
from visualization.model_visualizer import visualize_2D
from bricks_modeling.file_IO.model_reader import *
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.geometry_util import *
import numpy as np

from tqdm import tqdm

filename = "../../data/full_models/hole_axle_test.ldr"
bricks = read_bricks_from_file(filename)
graph = ConnectivityGraph(bricks)

#%%
dim = 2
points = np.array([
    -0.5, 0,
    1 / 2, np.sqrt(3) / 2,
    1, 0
]).reshape(-1, dim)

edges = np.array([
    0, 1,
    1, 2,
    2, 0
]).reshape(-1, 2)

K, P, A = spring_energy_matrix(points, edges, dim=dim, matrices=True)
M = spring_energy_matrix(points, edges, dim=2, matrices=False)
eigenpairs = eigen(M, symmetric=True)

def diff_unit_vector(vector, edge, diff_index, dim):
    p_ind, q_ind = edge
    norm = np.linalg.norm(vector)
    col_vec = vector.reshape(dim, 1)
    point_index, offset = divmod(diff_index, dim)
    if p_ind != point_index and q_ind != point_index:
        return np.zeros(dim)

    I = np.identity(dim)
    sign = 1 if point_index == p_ind else -1
    jacobian = sign * (I / norm - col_vec @ col_vec.T / np.power(norm, 3))
    return jacobian[:, offset]

def diff_norm_reciprocal(vector, edge, diff_index, dim):
    p_ind, q_ind = edge
    norm = np.linalg.norm(vector)
    point_index, offset = divmod(diff_index, dim)
    if p_ind != point_index and q_ind != point_index:
        return 0
    sign = 1 if point_index == p_ind else -1
    diff_vec = -sign * vector / np.power(norm, 3)
    return diff_vec[offset]
    

def diff_projection_matrix(diff_index, P, edges, dim):
    diff_P = np.zeros_like(P)
    for i, edge in enumerate(edges):
        original_vector = P[i, i * dim: (i + 1) * dim]
        diff_P[i, i * dim: (i + 1) * dim] = diff_unit_vector(original_vector, edge, diff_index, dim)

    return diff_P

def diff_stiffness_matrix(diff_index, points, edges, dim):
    edge_vectors = [points[p] - points[q] for p, q in edges]
    diag_entries = [
        diff_norm_reciprocal(vector, edge, diff_index, dim) 
        for vector, edge in zip(edge_vectors, edges)]
    return np.diag(diag_entries)

def diff_spring_energy_matrix(diff_index, points, edges, dim):
    K, P, A = spring_energy_matrix(points, edges, dim=dim, matrices=True)
    diff_K = diff_stiffness_matrix(diff_index, points, edges, dim)
    diff_P = diff_projection_matrix(diff_index, P, edges, dim)

    diff_PKP = np.linalg.multi_dot([diff_P.T, K, P]) + \
               np.linalg.multi_dot([P.T, diff_K, P]) + \
               np.linalg.multi_dot([P.T, K, diff_P])
    diff_M = np.linalg.multi_dot([A.T, diff_PKP, A])

    return diff_M


def diff_eigenvalue(eigenpair, diff_index, L_inv, B, points, edges, dim):
    eigenvalue, eigenvector = eigenpair
    diff_M = diff_spring_energy_matrix(diff_index, points, edges, dim)
    
    return eigenvector.T @ L_inv.T @ B.T @ diff_M @ B @ L_inv @ eigenvector

def diff_min_nonzero_eigenvalue_wrt_index(eigenpairs, diff_index, L_inv, B, points, edges, dim):
    nonzero_pairs = [
        (val, vec) 
        for val, vec in eigenpairs 
        if val > 1e-8]
    nonzero_pairs.sort(key=lambda p: p[0])
    min_nonzero_pair = nonzero_pairs[0]

    nonzero_eigenvalues, _ = zip(*nonzero_pairs)

    # if len(nonzero_eigenvalues) < 2:
    #     print("eigen pairs", eigenpairs)

    if len(nonzero_eigenvalues) == 1 or abs(nonzero_eigenvalues[1] - nonzero_eigenvalues[0]) > 1e-8:
        eigenvalue, eigenvector = min_nonzero_pair
        diff_M = diff_spring_energy_matrix(diff_index, points, edges, dim)
        return np.linalg.multi_dot([eigenvector.T, L_inv.T, B.T, diff_M, B, L_inv, eigenvector])
    else:
        return 0

def diff_min_nonzero_eigenvalue(points, edges, fixed_indices, joints, dim):
    M = spring_energy_matrix(points, edges, dim=dim)
    A = constraint_matrix(points, edges, joints, fixed_indices, dim)
    B = null_space(A)
    T = np.transpose(B) @ B
    L = cholesky(T)
    L_inv = np.linalg.inv(L)
    S = B.T @ M @ B

    eigenpairs = geo_util.eigen(L_inv.T @ S @ L_inv, symmetric=True)

    return np.asarray([
        diff_min_nonzero_eigenvalue_wrt_index(eigenpairs, i, L_inv, B, points, edges, dim)
        for i in range(len(points) * dim)
    ])

def min_nonzero_spring_eigenvalue(points, edges, dim):
    # K, P, A = spring_energy_matrix(points, edges, [], dim=dim, matrices=True)
    Q = spring_energy_matrix(points, edges, dim=dim)
    eigenpairs = eigen(Q, symmetric=True)
    min_pair = min([pair for pair in eigenpairs if pair[0] > 1e-8], key=lambda p: p[0])
    return min_pair[0]

def min_nonzero_eigenvalue(points, edges, fixed_points_idx, joints, dim, returns_pair=False):
    # K, P, A = spring_energy_matrix(points, edges, [], dim=dim, matrices=True)
    M = spring_energy_matrix(points, edges, dim=dim)
    A = constraint_matrix(points, edges, joints, fixed_points_idx, dim)

    # mathmatical computation
    B = null_space(A)
    T = np.transpose(B) @ B
    S = B.T @ M @ B
    L = cholesky(T)
    L_inv = inv(L)
    Q = np.linalg.multi_dot([L_inv.T, S, L_inv])

    eigenpairs = eigen(Q, symmetric=True)
    nonzero_pairs = [(val, B @ vec) for val, vec in eigenpairs if abs(val) > 1e-6]
    if returns_pair:
        return nonzero_pairs[0]
    else:
        return nonzero_pairs[0][0]

# %%
import solvers.rigidity_solver.test_cases.cases_2D as cases
case_name = 'case_6_1'
w = 24
case = vars(cases)[case_name](width=w, layer_num=3, unit=8 / w)
# case = cases.case_6()
points, fixed_points_idx, edges, joints = case
print(f"Case 6_1 with w = {w}")
moving_index = 8

test_points = \
    np.vstack([
        points[0: moving_index],
        [0, 1],
        points[moving_index + 1: ],
    ])
# min_nonzero_eigenvalue(test_points, edges, fixed_indices, joints, dim)
# diff_min_nonzero_eigenvalue(test_points, edges, fixed_indices, joints, dim)
# %%
from itertools import product
# x_range = np.linspace(-1, 1, num=40)
# y_range = np.linspace(-1, 1, num=40)

cx, cy = points[moving_index]
dt = 1.5
x_num, y_num = 40, 40
x_range = np.linspace(cx - dt, cx + dt, num=40)
y_range = np.linspace(cy - dt, cy + dt, num=40)

print("Generating points")
point_set_with_point_1_varying = [
    np.vstack([
        points[0: moving_index],
        [x, y],
        points[moving_index + 1: ],
    ])
    for x, y in product(x_range, y_range)

]

print("Computing eigenvalues")
coords_eigenvalues = np.asarray(
    [min_nonzero_eigenvalue(pts, edges, fixed_points_idx, joints, dim)
        for pts in point_set_with_point_1_varying]
).reshape(40, 40).transpose()
Z = coords_eigenvalues

print("Computing gradient")
coords_gradients = np.asarray(
    [diff_min_nonzero_eigenvalue(pts, edges, fixed_points_idx, joints, dim=dim)[moving_index * dim: (moving_index + 1) * dim]
        for pts in point_set_with_point_1_varying]
).reshape(40, 40, -1).transpose(1, 0, 2)
U, V = zip(*coords_gradients.reshape(-1, 2))


# np.array(
#     [aggregate_nonzero_eigval(np.array([x, y, 0]), min) 
#     for x, y in product(x_range, y_range)]).reshape((200, 100)).transpose()
from matplotlib import pyplot as plt
X, Y = np.meshgrid(x_range, y_range)
C = np.hypot(U, V)
fig = plt.figure()
mp = plt.quiver(X, Y, U, V, C)
plt.colorbar(mp)
plt.title(f"Grad - {case_name}")
plt.xlim(x_range[0], x_range[-1])
plt.ylim(y_range[0], y_range[-1])
plt.gca().set_aspect('equal')
plt.savefig(f"eigenvalue-gradients-{case_name}", dpi=330)
plt.show()

plt.clf()
ctmp = plt.contour(X, Y, Z, levels=25)
plt.colorbar(mp)
plt.title(f"Contour - {case_name}")
for i, j in edges:
    plt.plot(
        [points[i, 0], points[j, 0]],
        [points[i, 1], points[j, 1]],
        'r--' if moving_index in [i, j] else 'b'
    )
plt.scatter([points[moving_index, 0], 0], [points[moving_index, 1], 1], s=100, c='r', zorder=100)
plt.xlim(x_range[0], x_range[-1])
plt.ylim(y_range[0], y_range[-1])
plt.gca().set_aspect('equal')
plt.savefig(f"eigenvalue-contour-{case_name}", dpi=330)
plt.show()
#%%

num_iter = 10000
step_size = 1e-3 * len(points)
print("Step size =", step_size)

plt.clf()
plt.title('Analytical Gradient Optim')

from tqdm import tqdm
for i in tqdm(range(num_iter)):
    frac = (i + 1) / num_iter


    grad = diff_min_nonzero_eigenvalue(points, edges, fixed_points_idx, joints, dim)
    delta = grad.reshape(-1, dim) * step_size
    delta[fixed_points_idx] = np.zeros_like(points[0])
    points += delta

    if i % 1000 == 0:
        pair = min_nonzero_eigenvalue(points, edges, fixed_points_idx, joints, dim, returns_pair=True)

        visualize_2D(points, edges, pair[1].reshape(points.shape))

        clr = [frac, 0, 0]
        eigval = min_nonzero_eigenvalue(points, edges, fixed_points_idx, joints, dim)
        for pind, qind in edges:
            p, q = points[pind], points[qind]
            plt.plot([p[0], q[0]], [p[1], q[1]], c=clr)

        x, y = zip(*points[fixed_points_idx])
        plt.scatter(x, y)

        plt.title(f"Iter: {i + 1} Eigenvalue: {eigval:.6f}")
        plt.gca().set_aspect('equal')
        plt.show()

#%%
points_3 = np.array([
    [0, 1],
    [0, 0],
    [1, 0],
], dtype=np.float64)
step_size = 1e-3

point_trace = []
cur_points = np.copy(points_3)
for step in range(1000):
    gradients = diff_min_nonzero_eigenvalue(cur_points, edges, dim)
    one_step = (gradients * step_size).reshape(-1, dim)
    cur_points[0:2] += one_step[0:2]
    point_trace.append(np.copy(cur_points).reshape(-1))

point_trace = np.asarray(point_trace)

#%%
plt.clf()
fig, ax = plt.subplots()
ax.set_aspect("equal")
for p, q in itertools.combinations(points_3, 2):
    x, y = zip(p, q)
    plt.plot(x, y, c='black')

for p, q in itertools.combinations(point_trace[-1].reshape(-1, 2), 2):
    x, y = zip(p, q)
    plt.plot(x, y, c='blue')

for i in range(points_3.shape[0]):
    x, y = point_trace[:, i * dim: i * dim + dim].transpose()
    plt.plot(x, y, c='blue')
    print(x[0], y[0])
    print(x[-1], y[-1])
plt.title('Gradient Descent Example')
plt.show()
# %%
import solvers.rigidity_solver.test_cases.cases_2D as cases2d

points, edges,  = cases2d.case_6()

