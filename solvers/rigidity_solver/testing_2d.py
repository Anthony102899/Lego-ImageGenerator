from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from solvers.rigidity_solver.algo_core import solve_rigidity
from solvers.rigidity_solver.internal_structure import structure_sampling
import visualization.model_visualizer as vis
from solvers.rigidity_solver.eigen_analysis import get_motions, get_weakest_displacement
import solvers.rigidity_solver.test_cases.cases_2D as cases2d
from solvers.rigidity_solver.algo_core import spring_energy_matrix
import numpy as np
import util.geometry_util as geo_util
from numpy import linalg as LA
from scipy.linalg import null_space
from numpy.linalg import cholesky
from numpy.linalg import inv

def get_constraits_for_allowed_motions(allowed_motions, target_points, dim):
    allowed_motion_vecs = np.zeros((len(allowed_motions), len(target_points) * dim))
    for idx, constraint in enumerate(allowed_motions):
        if constraint[0] == "T": # translation
            for j in range(len(target_points)):
                allowed_motion_vecs[idx, j * dim : j * dim + 2] = constraint[1]
        elif constraint[0] == "R":
            for j in range(len(target_points)):
                rot_arm = target_points[j] - constraint[1]
                allowed_motion_vecs[idx, j * dim]     = -rot_arm[1]
                allowed_motion_vecs[idx, j * dim + 1] = rot_arm[0]

    return null_space(allowed_motion_vecs).T

def constraints_for_joints2(jo1_pts, jo1_pts_idx, jo2_pts, jo2_pts_idx, points, allowed_motions, fixed_points_index, dim = 2):
    connection_mat  = np.zeros((len(jo2_pts) * dim, len(points)*dim))

    forbidden_motion_vecs = get_constraits_for_allowed_motions(allowed_motions, jo2_pts, dim = 2)
    constraints_mat = np.zeros((len(forbidden_motion_vecs) + len(fixed_points_index) * dim, len(jo2_pts) * dim))
    constraints_mat[:forbidden_motion_vecs.shape[0], :forbidden_motion_vecs.shape[1]] = forbidden_motion_vecs

    for i in range(len(jo2_pts)):
        jo2_idx = jo2_pts_idx[i]
        jo1_idx = jo1_pts_idx[i]
        connection_mat[i * dim: i * dim + dim, jo2_idx * dim: jo2_idx * dim + dim] = np.identity(2)
        connection_mat[i * dim: i * dim + dim, jo1_idx * dim: jo1_idx * dim + dim] = np.identity(2)

    A = constraints_mat @ connection_mat

    for row, point_idx in enumerate(fixed_points_index):
        A[len(forbidden_motion_vecs) + row * dim: len(forbidden_motion_vecs) + row * dim + dim,
        point_idx * dim: point_idx * dim + dim] = np.identity(dim)

    B = null_space(A)
    T = np.transpose(B) @ B
    L = cholesky(T)

    return B, L

def solve_rigidity_new(dim = 2):
    global eigen_pairs
    eigen_pairs = geo_util.eigen(inv(L).T @ S @ inv(L), symmetric=True)
    eigen_pairs = [(e_val, B @ e_vec) for e_val, e_vec in eigen_pairs]
    # collect all eigen vectors with zero eigen value
    zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) < 1e-6]
    non_zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) >= 1e-6]
    if len(zero_eigenspace) == 0 or (len(zero_eigenspace) == (3 if dim == 2 else 6)):
        return True, non_zero_eigenspace
    else:
        return False, zero_eigenspace


if __name__ == "__main__":
    debugger = MyDebugger("test")

    points, fixed_points_index, edges, abstract_edges = cases2d.case_seperate_parts()

    fixed_points_index.sort()
    M = spring_energy_matrix(points, edges, fixed_points_index, dim=2)
    B, L = constraints_for_joints2(points[0:3], list(range(0, 3)),
                                   points[3:6], list(range(3,6)),
                                   points,
                                   allowed_motions = [("R", np.array([0,1])), ("T", np.array([0,1])), ("T", np.array([1,0]))],
                                   fixed_points_index = [0,1,2]
                                   )
    S = B.T @ M @ B

    is_rigid, eigen_pairs = solve_rigidity_new()

    if is_rigid:
        vec, value = get_weakest_displacement(eigen_pairs, dim=2)
        print(f"worse case value: {value}")
        # print(vec)
        vis.visualize_2D(points, edges, vec)
    else:
        motion_vecs = get_motions(eigen_pairs, points, dim=2)
        vis.visualize_2D(points, edges, motion_vecs[0])

    print("The structure is", "rigid" if is_rigid else "not rigid.")