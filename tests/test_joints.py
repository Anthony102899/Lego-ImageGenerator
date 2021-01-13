from solvers.rigidity_solver.joints import *

model = Model()

from solvers.rigidity_solver.algo_core import solve_rigidity, spring_energy_matrix
from numpy import linalg as LA
from scipy.linalg import null_space
from numpy.linalg import cholesky, inv, matrix_rank

import util.geometry_util as geo_util
from visualization.model_visualizer import visualize_3D

from testcases import simple

model = simple.hinge_with_yaw_axis()

dim = 3
points = model.point_matrix()
edges = model.edge_matrix()
A = model.constraint_matrix()

A = A if A.size != 0 else np.zeros((1, len(points) * dim))

trivial_motions = geo_util.trivial_basis(points, dim=3)
fixed_coordinates = np.zeros((len(model.beams[0].points) * 3, points.shape[0] * 3))
for r, c in enumerate(range(len(model.beams[0].points) * 3)):
    fixed_coordinates[r, c] = 1
# A = np.vstack((A, fixed_coordinates))
# A = np.vstack((A, np.take(trivial_motions, [0, 1, 2, 3, 4, 5], axis=0)))


M = spring_energy_matrix(points, edges, dim=dim)
print("M rank:", matrix_rank(M))

# mathmatical computation
B = null_space(A)
T = np.transpose(B) @ B
S = B.T @ M @ B

print("A shape:", A.shape)
print("T rank:", matrix_rank(T))
print("B rank:", matrix_rank(B))
print("S rank:", matrix_rank(S))
print("S shape", S.shape)

L = cholesky(T)
L_inv = inv(L)

Q = LA.multi_dot([L_inv.T, S, L_inv])
print("Q shape", Q.shape)
# compute eigenvalues / vectors
eigen_pairs = geo_util.eigen(Q, symmetric=True)
eigen_pairs = [(e_val, B @ e_vec) for e_val, e_vec in eigen_pairs]

# determine rigidity by the number of non-zero eigenvalues
zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) < 1e-6]
print("DoF:", len(zero_eigenspace))

trivial_motions = geo_util.trivial_basis(points, dim=3)
non_rigid_motion = np.array([
    [0, 0, 1],
    [-1, 1, 1],
    [0, 0, 2],
    [0, 0, 1],
    [1, -1, 1],
    [0, 0, 2],
]).reshape(-1)

print("motion rank", matrix_rank(np.vstack([non_rigid_motion, trivial_motions])))
print(A @ non_rigid_motion)
for motion in trivial_motions:
    print(A @ motion)


non_zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) >= 1e-6]

if len(zero_eigenspace) > 0:
    print("Non-rigid")
    for e, v in zero_eigenspace:
        arrows = v.reshape(-1, 3)
        for i, v in enumerate(arrows):
            print(i, points[i], v)
        # visualize_3D(points, edges=edges, arrows=arrows)
        # visualize_3D(points, edges=edges)
else:
    print("rigid")
    print([e for e, v in non_zero_eigenspace])
    # for e, v in non_zero_eigenspace:
    #     arrows = v.reshape(-1, 3)
    #     visualize_3D(points, edges=edges, arrows=arrows)
    #
    e, v = non_zero_eigenspace[0]
    arrows = v.reshape(-1, 3)
    # visualize_3D(points, edges=edges)
    # visualize_3D(points, edges=edges, arrows=np.where(np.isclose(arrows, 0), 0, arrows))