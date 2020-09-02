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

def constraints_for_joints(jo1_pts, jo1_pts_idx, jo2_pts, jo2_pts_idx, points, forbidden_motions, fixed_points_index, dim = 2):
    points_num = len(points)

    constraint_mat = np.zeros((len(forbidden_motions) + len(fixed_points_index) * dim, (len(jo2_pts) * dim)))
    project_mat = np.zeros((len(jo2_pts) * dim, points_num * dim))

    basis1 = (jo1_pts[1] - jo1_pts[0]) / LA.norm(jo1_pts[1] - jo1_pts[0])
    basis2 = (jo1_pts[2] - jo1_pts[0]) / LA.norm(jo1_pts[2] - jo1_pts[0])
    idx_i, idx_j, idx_k = jo1_pts_idx[0], jo1_pts_idx[1], jo1_pts_idx[2]

    # assemble projection matrix
    for i in range(len(jo2_pts)):
        jo2_idx = jo2_pts_idx[i]
        project_mat[i * dim, jo2_idx * dim : jo2_idx * dim + dim ] = np.transpose(basis1)
        project_mat[i * dim, idx_j   * dim : idx_j   * dim + dim ] = np.transpose(points[jo2_idx] - points[idx_i])
        project_mat[i * dim, idx_i   * dim : idx_i   * dim + dim ] = np.transpose(points[idx_i] - points[jo2_idx] - basis1)

        project_mat[i * dim + 1, jo2_idx * dim : jo2_idx * dim + dim ] = np.transpose(basis2)
        project_mat[i * dim + 1, idx_k * dim   : idx_k   * dim + dim] = np.transpose(points[jo2_idx] - points[idx_i])
        project_mat[i * dim + 1, idx_i * dim   : idx_i   * dim + dim] = np.transpose(points[idx_i] - points[jo2_idx] - basis2)

    # assemble the constraint matrix
    for idx, cons in enumerate(forbidden_motions):
        if cons[0] == "T": # translation
            projected_value_1 = np.transpose(cons[1]) @ basis1
            projected_value_2 = np.transpose(cons[1]) @ basis2
            for j in range(len(jo2_pts)):
                constraint_mat[idx, j * dim]     = projected_value_1
                constraint_mat[idx, j * dim + 1] = projected_value_2
        elif cons[0] == "R":
            for j in range(len(jo2_pts)):
                rot_arm = jo2_pts[j] - cons[1]
                projected_p_1 = np.transpose(rot_arm) @ basis1
                projected_p_2 = np.transpose(rot_arm) @ basis2
                constraint_mat[idx, j * dim]     = -projected_p_2
                constraint_mat[idx, j * dim + 1] = projected_p_1

    A = constraint_mat @ project_mat

    for row, point_idx in enumerate(fixed_points_index):
        A[len(forbidden_motions) + row*dim : len(forbidden_motions) + row*dim + dim, point_idx* dim : point_idx * dim + dim] = np.identity(dim)

    print(project_mat)
    print("")
    print(constraint_mat)
    print("")
    print(A)



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
    B, L = constraints_for_joints(points[0:3], list(range(0, 3)),
                         points[3:6], list(range(3,6)),
                                  points,
                                  forbidden_motions = [("T", np.array([0,1])),("T", np.array([1,0])),("R", np.array([0,0]))],
                                  fixed_points_index = [0]
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