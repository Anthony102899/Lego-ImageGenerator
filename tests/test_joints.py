from solvers.rigidity_solver.models import *

from solvers.rigidity_solver.algo_core import solve_rigidity, spring_energy_matrix
from numpy import linalg as LA
from scipy.linalg import null_space
from numpy.linalg import cholesky, inv, matrix_rank

import util.geometry_util as geo_util
from visualization.model_visualizer import visualize_3D, visualize_hinges

from testcases import simple, tetra, joint

# model = tetra.square_pyramid_axes()

points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],

    [0, 1, 0],
    [1, 1, 0],
    [2, 0, 0],

    [2, 0, 1],
    [3, 0, 1],
    [2, 1, 1],
    [3, 1, 1],
    [4, 1, 1],
    [5, 2, 1],
])

model = Model()
beams = [
    Beam(points[:3]),
    Beam(points[3:6]),
    Beam(points[6:]),
]
joints = [
    Joint(beams[0], beams[1], pivot=np.array([1, 0, 0]), rotation_axes=np.array([0, 1, 0])),
    Joint(beams[1], beams[2], pivot=np.array([1, 1, 0])),
]
model.add_beams(beams)
model.add_joints(joints)

dim = 3
points = model.point_matrix()
edges = model.edge_matrix()
A = model.constraint_matrix()

A = A if A.size != 0 else np.zeros((1, len(points) * dim))

trivial_motions = geo_util.trivial_basis(points, dim=3)
count = 1
fixed_coordinates = np.zeros((len(model.beams[0].points) * 3, points.shape[0] * 3))
for r, c in enumerate(range(len(model.beams[0].points) * 3)):
    fixed_coordinates[r, c] = 1
# A = np.vstack((A, fixed_coordinates))
A = np.vstack((A, np.take(trivial_motions, [0, 1, 2, 3, 4, 5], axis=0)))

# pivots = np.array([j.pivot_point for j in model.joints])
# axes = np.array([j.axis for j in model.joints])
# visualize_hinges(points, edges=edges, pivots=pivots, axes=axes)

M = spring_energy_matrix(points, edges, dim=dim)
print("M rank:", matrix_rank(M))

# mathmatical computation
B = null_space(A)
T = np.transpose(B) @ B
S = B.T @ M @ B

L = cholesky(T)
L_inv = inv(L)

Q = LA.multi_dot([L_inv.T, S, L_inv])
print("Q shape", Q.shape)
# compute eigenvalues / vectors
eigen_pairs = geo_util.eigen(Q, symmetric=True)
eigen_pairs = [(e_val, B @ e_vec) for e_val, e_vec in eigen_pairs]

# determine rigidity by the number of zero eigenvalues
zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) < 1e-6]
print("DoF:", len(zero_eigenspace))

trivial_motions = geo_util.trivial_basis(points, dim=3)

non_zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) >= 1e-8]

if len(zero_eigenspace) > 0:
    print("Non-rigid")
    for e, v in zero_eigenspace:
        arrows = v.reshape(-1, 3)
        print(e)
        visualize_3D(points, edges=edges, arrows=arrows)
        # visualize_3D(points, edges=edges)
else:
    print("rigid")
    # for e, v in non_zero_eigenspace:
    #     arrows = v.reshape(-1, 3)
    #     visualize_3D(points, edges=edges, arrows=arrows)
    #
    e, v = non_zero_eigenspace[0]
    print("smallest eigenvalue:", e)
    arrows = v.reshape(-1, 3)
    # visualize_3D(points, edges=edges)
    visualize_3D(points, edges=edges, arrows=np.where(np.isclose(arrows, 0), 0, arrows))
