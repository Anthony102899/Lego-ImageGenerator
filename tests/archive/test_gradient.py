#%%
import sys
sys.path.append("../..")

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

import tests.testsamples
#%%

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(npoints, ndim)
    vec = geo_util.rowwise_normalize(vec)
    return vec


axes_list = [
    sample_spherical(4) for i in range(10000)
]


objectives = []

from testsamples import tetra
from tqdm import tqdm
# for axes in tqdm(axes_list):
if True:
    model = tetra.square_centering_axes()

    points = model.point_matrix()
    edges = model.edge_matrix()
    A = model.constraint_matrix()

    hinges = model.joints
    hinge_axes = np.array([h.axis for h in hinges])
    hinge_pivots = np.array([h.pivot_point for h in hinges])
    hinge_point_indices = model.joint_point_indices()

    extra_constraints = np.zeros((len(model.beams[0].points) * 3, points.shape[0] * 3))
    for r, c in enumerate(range(len(model.beams[0].points) * 3)):
        extra_constraints[r, c] = 1
    trivial_motions = geo_util.trivial_basis(points, dim=3)
    extra_constraints = trivial_motions
    A = np.vstack([A, extra_constraints])

    M = spring_energy_matrix(points, edges, dim=3)

    # mathematical computation
    B = scipy.linalg.null_space(A)
    T = np.transpose(B) @ B
    S = B.T @ M @ B

    L = cholesky(T)
    L_inv = inv(L)

    Q = np.linalg.multi_dot([L_inv.T, S, L_inv])

    pairs = geo_util.eigen(Q, symmetric=True)
    eigenvalues = np.array([v for v, e in pairs])
    print("DoF:", np.sum(np.isclose(eigenvalues, 0)))
    obj, eigenvector = pairs[0]
    arrows = B @ eigenvector

    torch_obj = gradient_analysis(
        points,
        edges,
        hinge_axes,
        hinge_pivots,
        hinge_point_indices,
        extra_constraints=extra_constraints,
        iters=1
    ).detach().numpy()

    print(torch_obj, obj)

    # visualize_3D(points, edges=edges, arrows=arrows.reshape(-1, 3))
    objectives.append((obj, axes))


print(objectives)

#%%
print(max(objectives, key=lambda p: p[0]))
print(min(objectives, key=lambda p: p[0]))
