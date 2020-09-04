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
    assert len(target_points) == 2
    allowed_motion_vecs = np.zeros((len(allowed_motions) + 1, len(target_points) * dim))

    # allow changes of the distance between two points
    dir_vec = (target_points[0] - target_points[1]) / LA.norm(target_points[0] - target_points[1])
    allowed_motion_vecs[0, 0:2] = dir_vec.T
    allowed_motion_vecs[0, 2:4] = -dir_vec.T

    for idx, constraint in enumerate(allowed_motions):
        if constraint[0] == "T": # translation
            for j in range(len(target_points)):
                allowed_motion_vecs[idx+1, j * dim : j * dim + 2] = constraint[1]
        elif constraint[0] == "R":
            for j in range(len(target_points)):
                rot_arm = target_points[j] - constraint[1]
                allowed_motion_vecs[idx+1, j * dim]     = -rot_arm[1]
                allowed_motion_vecs[idx+1, j * dim + 1] = rot_arm[0]

    return null_space(allowed_motion_vecs).T

def constraints_for_joints(source_pts, source_pts_idx, target_pts, target_pts_idx, points, allowed_motions, fixed_points_index, dim = 2):
    points_num = len(points)

    forbidden_motion_vecs = get_constraits_for_allowed_motions(allowed_motions, target_pts, dim=dim)

    constraint_mat = np.zeros((len(forbidden_motion_vecs) + len(fixed_points_index) * dim, (len(target_pts) * dim)))
    project_mat = np.zeros((len(target_pts) * dim, points_num * dim))

    b1_length = LA.norm(source_pts[1] - source_pts[0])
    basis1 = (source_pts[1] - source_pts[0]) / b1_length
    basis2 = np.array([-basis1[1], basis1[0]])
    idx_0, idx_1 = source_pts_idx[0], source_pts_idx[1]

    # assemble projection matrix
    for i in range(len(target_pts)):
        jo2_idx = target_pts_idx[i]
        # assembly the coordinate m
        project_mat[i * dim, jo2_idx * dim : jo2_idx * dim + dim ]  = np.transpose(basis1)
        project_mat[i * dim, idx_0 * dim   : idx_0 * dim + dim] = np.transpose((points[idx_0] - points[jo2_idx])/b1_length - basis1)
        project_mat[i * dim, idx_1   * dim : idx_1   * dim + dim ] = np.transpose((points[jo2_idx] - points[idx_0])/b1_length)

        # assembly the coordinate n
        vec = (points[jo2_idx] - points[idx_0]) / b1_length
        project_mat[i * dim + 1, jo2_idx * dim : jo2_idx * dim + dim ] = np.transpose(basis2)
        project_mat[i * dim + 1, idx_0 * dim]     = -vec[1] - basis2[0]
        project_mat[i * dim + 1, idx_0 * dim + 1] =  vec[0] - basis2[1]
        project_mat[i * dim + 1, idx_1 * dim]     =  vec[1]
        project_mat[i * dim + 1, idx_1 * dim + 1] = -vec[0]

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

    print(f"constraint matrix: rank{matrix_rank(constraint_mat)}")
    print(constraint_mat)
    print(f"projection matrix: rank{matrix_rank(project_mat)}")
    print(project_mat)
    print(f"A: rank{matrix_rank(A)}")
    print(A)

    return A


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

    points, fixed_points_index, edges, joints = cases2d.case_8_new()

    M = spring_energy_matrix(points, edges, dim=2, fixed_points_idx = [])

    j = 0
    eidx_1, eidx_2 = joints[j][0], joints[j][1]
    e_1, e_2 = edges[eidx_1], edges[eidx_2]

    A = constraints_for_joints(points[np.array(list(e_1))], list(e_1),
                               points[np.array(list(e_2))], list(e_2),
                               points,
                               allowed_motions = joints[j][2],
                               fixed_points_index = fixed_points_index
                               )
    B = null_space(A)
    T = np.transpose(B) @ B
    L = cholesky(T)
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