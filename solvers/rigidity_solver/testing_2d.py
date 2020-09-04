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
from numpy.linalg import matrix_rank

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

def constraints_for_joints(source_pts, source_pts_idx, target_pts, target_pts_idx, points, allowed_motions, fixed_points_index, dim = 2):
    points_num = len(points)

    forbidden_motion_vecs = get_constraits_for_allowed_motions(allowed_motions, target_pts, dim=dim)

    constraint_mat = np.zeros((len(forbidden_motion_vecs) + len(fixed_points_index) * dim, (len(target_pts) * dim)))
    project_mat = np.zeros((len(target_pts) * dim, points_num * dim))

    basis1 = (source_pts[1] - source_pts[0]) / LA.norm(source_pts[1] - source_pts[0])
    basis2 = np.array([basis1[1], -basis1[0]])
    idx_0, idx_1 = source_pts_idx[0], source_pts_idx[1]

    # assemble projection matrix
    for i in range(len(target_pts)):
        jo2_idx = target_pts_idx[i]
        # assembly the coordinate m
        project_mat[i * dim, jo2_idx * dim : jo2_idx * dim + dim ]  = np.transpose(basis1)
        project_mat[i * dim, idx_0 * dim   : idx_0 * dim + dim] = np.transpose(points[idx_0] - points[jo2_idx] - basis1)
        project_mat[i * dim, idx_1   * dim : idx_1   * dim + dim ] = np.transpose(points[jo2_idx] - points[idx_0])

        # assembly the coordinate n
        project_mat[i * dim + 1, jo2_idx * dim : jo2_idx * dim + dim ] = np.transpose(basis2)
        vec = points[jo2_idx] - points[idx_0]
        project_mat[i * dim, idx_0 * dim]     = -vec[1] + basis2[0]
        project_mat[i * dim, idx_0 * dim + 1] =  vec[0] + basis2[1]
        project_mat[i * dim, idx_1 * dim]     = vec[1]
        project_mat[i * dim, idx_1 * dim + 1] = -vec[0]

    # assemble the constraint matrix via forbidden motion
    for idx, motion_vec in enumerate(forbidden_motion_vecs):
        for j in range(len(target_pts)):
            projected_value_1 = np.transpose(motion_vec[j*dim : j*dim + dim]) @ basis1
            projected_value_2 = np.transpose(motion_vec[j*dim : j*dim + dim]) @ basis2
            constraint_mat[idx, j * dim]     = projected_value_1
            constraint_mat[idx, j * dim + 1] = projected_value_2

    A = constraint_mat @ project_mat

    current_row = len(forbidden_motion_vecs)
    # for fixed joints
    for row, point_idx in enumerate(fixed_points_index):
        A[current_row + row*dim : current_row + row*dim + dim, point_idx* dim : point_idx * dim + dim] = np.identity(dim)

    print(project_mat)
    print("")
    print(constraint_mat)
    print("")
    print(A)

    B = null_space(A)
    T = np.transpose(B) @ B
    L = cholesky(T)

    return B, T, L


def solve_rigidity_new(dim = 2):
    global eigen_pairs
    eigen_pairs = geo_util.eigen(inv(L).T @ S @ inv(L), symmetric=True)
    eigen_pairs = [(e_val, B @ e_vec) for e_val, e_vec in eigen_pairs]
    # collect all eigen vectors with zero eigen value
    zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) < 1e-6]
    print("DoF:", len(zero_eigenspace))
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
    B, T, L = constraints_for_joints(points[0:2], list(range(0, 2)),
                                  points[3:5], list(range(3, 5)),
                                  points,
                                  allowed_motions = np.array([("T", np.array([1,0]))]),
                                  fixed_points_index = [3]
                                  )
    S = B.T @ M @ B

    print("T rank:", matrix_rank(T))
    print("S rank:", matrix_rank(S))

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