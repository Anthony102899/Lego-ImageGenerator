#%%
import sys
sys.path.append("..")

import scipy
import numpy as np
from numpy.linalg import matrix_rank, matrix_power, cholesky, inv

import util.geometry_util as geo_util

from solvers.rigidity_solver.gradient import gradient_analysis
from solvers.rigidity_solver.internal_structure import tetrahedronize
from solvers.rigidity_solver.algo_core import solve_rigidity, spring_energy_matrix
from solvers.rigidity_solver.joints import Beam, Model, Hinge
from solvers.rigidity_solver.gradient import gradient_analysis

from visualization.model_visualizer import visualize_3D

import testcases
#%%

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(npoints, ndim)
    vec = geo_util.rowwise_normalize(vec)
    return vec

axes_list = [
    sample_spherical(4) for i in range(10000)
]

objectives = []

from tqdm import tqdm
for axes in (axes_list):
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ]) * 150

    beams = [
        Beam.tetra(points[i], points[(i + 1) % 4], thickness=20) for i in range(4)
    ]
    hinges = [
        Hinge(beams[i], beams[(i + 1) % 4], axis=ax, pivot_point=points[(i + 1) % 4])
        for i, ax in zip(range(4), axes)
    ]

    model = Model()
    model.add_beams(beams[:2])
    model.add_joints(hinges[:1])

    points = model.point_matrix()
    edges = model.edge_matrix()
    A = model.constraint_matrix()

    hinges = model.joints
    hinge_axes = np.array([h.axis for h in hinges])
    hinge_pivots = np.array([h.pivot_point for h in hinges])
    hinge_point_indices = model.joint_point_indices()
    print(hinge_point_indices)

    extra_constraints = np.zeros((len(model.beams[0].points) * 3, points.shape[0] * 3))
    for r, c in enumerate(range(len(model.beams[0].points) * 3)):
        extra_constraints[r, c] = 1
    trivial_motions = geo_util.trivial_basis(points, dim=3)
    extra_constraints = trivial_motions
    A = np.vstack([A, trivial_motions])

    M = spring_energy_matrix(points, edges, dim=3)

    # mathematical computation
    B = scipy.linalg.null_space(A)
    T = np.transpose(B) @ B
    S = B.T @ M @ B

    L = cholesky(T)
    L_inv = inv(L)

    Q = np.linalg.multi_dot([L_inv.T, S, L_inv])

    pairs = geo_util.eigen(Q, symmetric=True)
    # print([v for v, e in pairs])
    obj, eigenvector = pairs[0]
    arrows = B @ eigenvector

    # torch_obj = gradient_analysis(
    #     points,
    #     edges,
    #     hinge_axes,
    #     hinge_pivots,
    #     hinge_point_indices,
    #     extra_constraints=extra_constraints,
    #     iters=1
    # ).detach().numpy()

    print(obj)

    # visualize_3D(points, edges=edges, arrows=arrows.reshape(-1, 3))

    objectives.append((obj, axes))


print(objectives)

#%%

"""
fixed_coordinates = np.zeros((len(model.beams[0].points) * 3, points.shape[0] * 3))
for r, c in enumerate(range(len(model.beams[0].points) * 3)):
    fixed_coordinates[r, c] = 1
A = np.vstack((A, fixed_coordinates))

M = spring_energy_matrix(points, edges, dim=3)
print("M rank:", matrix_rank(M))

# mathmatical computation
B = scipy.linalg.null_space(A)
T = np.transpose(B) @ B
S = B.T @ M @ B

print("A shape:", A.shape)
print("T rank:", matrix_rank(T))
print("B rank:", matrix_rank(B))
print("S rank:", matrix_rank(S))
print("S shape", S.shape)

L = cholesky(T)
L_inv = inv(L)

Q = np.linalg.multi_dot([L_inv.T, S, L_inv])
print("Q shape", Q.shape)
# compute eigenvalues / vectors
eigen_pairs = geo_util.eigen(Q, symmetric=True)
eigen_pairs = [(e_val, B @ e_vec) for e_val, e_vec in eigen_pairs]

# determine rigidity by the number of non-zero eigenvalues
zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) < 1e-6]
print("DoF:", len(zero_eigenspace))

trivial_motions = geo_util.trivial_basis(points, dim=3)

non_zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) >= 1e-6]

if len(zero_eigenspace) > 0:
    print("Non-rigid")
    for e, v in zero_eigenspace:
        arrows = v.reshape(-1, 3)
        for i, v in enumerate(arrows):
            print(i, points[i], v)
        visualize_3D(points, edges=edges, arrows=arrows)
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
    visualize_3D(points, edges=edges, arrows=np.where(np.isclose(arrows, 0), 0, arrows))
"""