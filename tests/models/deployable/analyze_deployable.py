import open3d as o3d
import scipy
import numpy as np
from numpy import linalg as LA
from scipy.linalg import null_space
from numpy.linalg import cholesky, inv, matrix_rank
from solvers.rigidity_solver.eigen_analysis import eigen_analysis
import solvers.rigidity_solver.algo_core as core
from visualization.model_visualizer import visualize_3D
import util.geometry_util as geo_util
import time

from model_deployable import define

from util.logger import logger

log = logger()

stage = 1
definition = define(stage)
model = definition["model"]

log.debug(f"model definition: {definition}")
log.debug(f"model: {model.report()}")

points = model.point_matrix()
edges = model.edge_matrix()
dim = 3

trivial_motions = geo_util.trivial_basis(points, dim=3)

start = time.time()

log.debug("computing A")
A = model.constraint_matrix()

log.debug("assembled joint constraint matrix, time - {}".format(time.time() - start))
A = A if A.size != 0 else np.zeros((1, len(points) * dim))

trivial_motions = geo_util.trivial_basis(points, dim=3)
count = len(model.beams[0].points)
# count = 3
fixed_coordinates = np.zeros((count * 3, points.shape[0] * 3))
for r, c in enumerate(range(count * 3)):
    fixed_coordinates[r, c] = 1
extra_constraints = np.take(trivial_motions, [0, 1, 2, 3, 4, 5], axis=0)
# extra_constraints = fixed_coordinates
A = np.vstack((A, extra_constraints))

log.debug("constraint A matrix, shape {}, time - {}".format(A.shape, time.time() - start))

M = core.spring_energy_matrix(points, edges, dim)

log.debug("stiffness matrix, time - {}".format(time.time() - start))
log.debug("using sparse matrix type {}".format(type(M)))

log.debug("computing B")
B = null_space(A)

log.debug("null space of constraints, B shape {} , time - {}".format(B.shape, time.time() - start))

rank_A = np.linalg.matrix_rank(A)
log.debug(f"rank of A {rank_A}")
log.debug(f"nullity of A: {A.shape[1] - rank_A}")

T = np.transpose(B) @ B
log.debug("T = B.T @ B computed, time - {}".format(time.time() - start))

S = B.T @ M @ B
log.debug("S computed. time - {}".format(time.time() - start))

L = cholesky(T)
log.debug("cholesky on T shape {} time - {}".format(L.shape, time.time() - start))

L_inv = np.linalg.inv(L)
log.debug("inverse of L, shape {} time - {}".format(L_inv.shape, time.time() - start))

Q = L_inv.T @ S @ L_inv
log.debug("merged stiffness and constraint matrix into Q {}, time - {}".format(Q.shape, time.time() - start))

eigen_pairs = geo_util.eigen(Q, symmetric=True)
log.debug("eigen decomposition on Q, time - {}".format(time.time() - start))

zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) < 1e-6]
log.debug(f"DoF: {len(zero_eigenspace)}")

trivial_motions = geo_util.trivial_basis(points, dim=3)

non_zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) >= 1e-8]
log.debug("non zero eigenspace, time - {}".format(time.time() - start))

log.debug(f"smallest 6 eigenvalue: {[e for e, _ in eigen_pairs[:6]]}")
if len(zero_eigenspace) > 0:
    log.debug("Non-rigid")
    for i, (e, v) in enumerate(zero_eigenspace):
        eigenvector = B @ v
        force = M @ eigenvector
        # force /= np.linalg.norm(force)
        arrows = force.reshape(-1, 3)
        log.debug(e)
        np.savez(f"data/rigid_deployable{stage}_non_rigid_{i}.npz",
                 eigenvalue=np.array(e),
                 points=points,
                 edges=edges,
                 eigenvector=eigenvector,
                 force=force,
                 stiffness=M)
        visualize_3D(points, edges=edges, arrows=arrows, show_point=False)
else:
    log.debug("rigid")
    e, v = non_zero_eigenspace[0]
    eigenvector = B @ v
    force = M @ eigenvector
    force /= np.linalg.norm(force)
    arrows = force.reshape(-1, 3)

    # part1_f /= np.linalg.norm(part1_f)

    np.savez(f"data/rigid_deployable_stage{stage}.npz",
             eigenvalue=np.array(e),
             points=points,
             edges=edges,
             eigenvector=eigenvector,
             force=force,
             stiffness=M)
    visualize_3D(points, edges=edges, arrows=force.reshape(-1, 3), show_point=False)
    visualize_3D(points, edges=edges, arrows=eigenvector.reshape(-1, 3), show_point=False)
