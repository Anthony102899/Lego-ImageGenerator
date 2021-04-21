from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from solvers.rigidity_solver import constraints_3d
from util.debugger import MyDebugger
from bricks_modeling.connections.conn_type import ConnType
import numpy as np
import scipy
import util.geometry_util as geo_util
import open3d as o3d
import copy
from typing import List
import itertools
from numpy import linalg as LA
from numpy.linalg import inv, matrix_rank
from visualization.model_visualizer import visualize_3D, visualize_2D
from scipy.linalg import null_space, cholesky

def rigidity_matrix(
    points: np.ndarray,
    edges: np.ndarray,
    dim: int
    ) -> np.ndarray:
    """
    points: (n, d) array, n points in a d-dimensional space
    edges : (m, 2) array, m edges, store indices of the points they join
    dim   : int, dimension order
    """
    assert len(points.shape) == 2 and points.shape[1] == dim
    n, m = len(points), len(edges)

    # constructing the rigidity matrix R
    R = np.zeros((m, dim * n))
    for i, (p_ind, q_ind) in enumerate(edges):
        q_minus_p = points[q_ind, :] - points[p_ind, :]
        R[i, q_ind * dim : (q_ind + 1) * dim] = q_minus_p
        R[i, p_ind * dim : (p_ind + 1) * dim] = -q_minus_p

    return R

def _remove_fixed_edges(points: np.ndarray, edges: np.ndarray, fixed_points_idx):
    """
    subroutine used by spring_energy_matrix. remove the fixed points and edges by deleting them from inputs
    """
    if len(fixed_points_idx) == 0:
        return points, edges

    fixed_edges_idx = [
        index
        for index, edge in enumerate(edges)
        if edge[0] in fixed_points_idx and edge[1] in fixed_points_idx
    ]

    if len(fixed_edges_idx) > 0:
        edges = np.delete(edges, fixed_edges_idx, axis=0)

    return points, edges


def spring_energy_matrix(
        points: np.ndarray,
        edges: np.ndarray,
        dim: int = 3,
        matrices=False,
    ):
    """
    matrices: return K, P, A if true
    fix_stiffness: use constant for value K if true, use 1/norm(vec) if false
    """

    n, m = len(points), len(edges)

    K = np.zeros((m, m))
    P = np.zeros((m, m * dim))
    A = np.zeros((m * dim, n * dim))

    normalized = lambda v: v / LA.norm(v)

    # forming P and K
    for idx, e in enumerate(edges):
        if len(e) == 2:
            edge_vec = points[e[0]] - points[e[1]]
        else: # virtual edge
            assert len(e) == 2 + dim
            assert LA.norm(points[e[0]] - points[e[1]]) < 1e-6
            edge_vec = np.array(e[2:])
            edge_vec = normalized(edge_vec) / 1e-4 # making the spring strong by shorter the edge

        P[idx, idx * dim : idx * dim + dim] = normalized(edge_vec).T
        K[idx, idx] = 1 / LA.norm(edge_vec)


        for d in range(dim):
            A[dim * idx + d, dim * e[0] + d] = 1
            A[dim * idx + d, dim * e[1] + d] = -1

    if matrices:
        return K, P, A
    else:
        return np.linalg.multi_dot([A.T, P.T, K, P, A])


def transform_matrix_fitting(points_start, points_end, dim=3):
    assert len(points_start) == len(points_end)
    A = np.zeros((len(points_start) * dim, dim * dim + dim))
    b = points_end.reshape(-1)
    for row, p in enumerate(points_start):
        for i in range(dim):
            A[row * dim + i][dim * i : dim * i + dim] = p.T
            A[row * dim + i][i - dim] = 1

    Q = inv(A.T @ A) @ A.T @ b
    T = Q[-dim:]
    M = Q[: dim * dim].reshape(dim, dim)

    return M, T

def constraint_matrix(points, edges, joints, fixed_points_idx, dim):
    A = np.zeros((1, len(points) * dim))
    for joint in joints:
        edge_idx, motion_points, allowed_motions = joint[0], joint[1], joint[2]

        A_joint = constraints_for_one_joint(
            points[np.array(list(edges[edge_idx]))], list(edges[edge_idx]),
            points[np.array(motion_points)], motion_points,
            points,
            allowed_motions=allowed_motions, dim=dim)
        A = np.append(A, A_joint, 0)

    # adding the fixed points constraints
    C = get_matrix_for_fixed_points(fixed_points_idx, len(points), dim)
    A = np.append(A, C, 0)
    return A


def generalized_courant_fischer(original_stiffness, constraints):
    K = original_stiffness
    B = scipy.linalg.null_space(constraints)
    T = np.transpose(B) @ B
    S = B.T @ K @ B
    L = np.linalg.cholesky(T)
    L_inv = np.linalg.inv(L)
    Q = L_inv.T @ S @ L_inv

    return Q, B

def quadratic_matrix(points, edges, joints, fixed_points_idx, dim, verbose=False):
    """
    compute the quad matrix (the one used to extract eigenvalues)

    se_matrix:   spring_energy_matrix
    cstr_matrix: constraint_matrix
    """
    M = spring_energy_matrix(points, edges, dim=dim)
    A = constraint_matrix(points, edges, joints, fixed_points_idx, dim)

    # mathmatical computation
    B = null_space(A)
    T = np.transpose(B) @ B
    S = B.T @ M @ B

    if verbose:
        print("T rank:", matrix_rank(T))
        print("S rank:", matrix_rank(S))

    L = cholesky(T)
    L_inv = inv(L)

    Q = LA.multi_dot([L_inv.T, S, L_inv])
    return Q


def solve_rigidity(points: np.ndarray, edges: np.ndarray, joints, fixed_points_idx=None, dim: int = 3) -> (bool, List[np.ndarray]):
    if fixed_points_idx is None:
        fixed_points_idx = []

    M = spring_energy_matrix(points, edges, dim=dim)
    A = constraint_matrix(points, edges, joints, fixed_points_idx, dim)

    # mathmatical computation
    B = null_space(A)
    T = np.transpose(B) @ B
    S = B.T @ M @ B

    print("T rank:", matrix_rank(T))
    print("S rank:", matrix_rank(S))

    L = cholesky(T)
    L_inv = inv(L)

    Q = LA.multi_dot([L_inv.T, S, L_inv])
    # compute eigenvalues / vectors
    eigen_pairs = geo_util.eigen(Q, symmetric=True)
    eigen_pairs = [(e_val, B @ e_vec) for e_val, e_vec in eigen_pairs]

    # determine rigidity by the number of non-zero eigenvalues
    print("DoF:", len([(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) < 1e-6]))

    trivial_motions     = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if
                           abs(e_val) < 1e-6 and geo_util.is_trivial_motions([e_vec], points, dim)]

    non_trivial_motions = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if
                           abs(e_val) < 1e-6 and (not geo_util.is_trivial_motions([e_vec], points, dim))]

    non_zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) >= 1e-6]

    return trivial_motions, non_trivial_motions, non_zero_eigenspace

def constraints_for_one_joint(source_pts, source_pts_idx, target_pts, target_pts_idx, points, allowed_motions, dim = 2):
    forbidden_motion_vecs = get_constraits_for_points_allowed_motions(allowed_motions, target_pts, dim=dim)

    constraint_mat = np.zeros((len(forbidden_motion_vecs), (len(target_pts) * dim)))

    if dim == 2:
        project_mat, basis1, basis2 = project_matrix_2D(source_pts, source_pts_idx,target_pts, target_pts_idx, points)
    else:
        project_mat, basis1, basis2, basis3 = project_matrix_3D(source_pts, source_pts_idx,target_pts, target_pts_idx, points)

    # assemble the constraint matrix via forbidden motion
    for idx, motion_vec in enumerate(forbidden_motion_vecs):
        for j in range(len(target_pts)):
            projected_value_1 = np.transpose(motion_vec[j*dim : j*dim + dim]) @ basis1
            projected_value_2 = np.transpose(motion_vec[j*dim : j*dim + dim]) @ basis2
            constraint_mat[idx, j * dim]     = projected_value_1
            constraint_mat[idx, j * dim + 1] = projected_value_2
            if dim == 3:
                projected_value_3 = np.transpose(motion_vec[j * dim: j * dim + dim]) @ basis3
                constraint_mat[idx, j * dim + 2] = projected_value_3

    A = constraint_mat @ project_mat

    # print(f"constraint matrix: rank{matrix_rank(constraint_mat)}")
    # print(constraint_mat)
    # print(f"projection matrix: rank{matrix_rank(project_mat)}")
    # print(project_mat)
    # print(f"A: rank{matrix_rank(A)}")
    # print(A)

    return A

def project_matrix_2D(source_pts, source_pts_idx,target_pts, target_pts_idx, points, dim = 2):
    assert len(source_pts) == 2

    points_num = len(points)
    project_mat = np.zeros((len(target_pts) * dim, points_num * dim))

    b1_length = LA.norm(source_pts[1] - source_pts[0])
    basis1 = (source_pts[1] - source_pts[0]) / b1_length
    basis2 = np.array([-basis1[1], basis1[0]])
    idx_0, idx_1 = source_pts_idx[0], source_pts_idx[1]

    # assemble projection matrix
    for i in range(len(target_pts)):
        jo2_idx = target_pts_idx[i]
        # assembly the coordinate m
        project_mat[i * dim, jo2_idx * dim: jo2_idx * dim + dim] = np.transpose(basis1)
        project_mat[i * dim, idx_0 * dim: idx_0 * dim + dim] = np.transpose(
            (points[idx_0] - points[jo2_idx]) / b1_length - basis1)
        project_mat[i * dim, idx_1 * dim: idx_1 * dim + dim] = np.transpose(
            (points[jo2_idx] - points[idx_0]) / b1_length)

        # assembly the coordinate n
        vec = (points[jo2_idx] - points[idx_0]) / b1_length
        project_mat[i * dim + 1, jo2_idx * dim: jo2_idx * dim + dim] = np.transpose(basis2)
        project_mat[i * dim + 1, idx_0 * dim] = -vec[1] - basis2[0]
        project_mat[i * dim + 1, idx_0 * dim + 1] = vec[0] - basis2[1]
        project_mat[i * dim + 1, idx_1 * dim] = vec[1]
        project_mat[i * dim + 1, idx_1 * dim + 1] = -vec[0]

    return project_mat, basis1, basis2

# TODO: not tested yet
def project_matrix_3D(source_pts, source_pts_idx,target_pts, target_pts_idx, points):
    assert len(source_pts) == 3
    dim = 3

    points_num = len(points)
    project_mat = np.zeros((len(target_pts) * dim, points_num * dim))

    l_b1 = LA.norm(source_pts[1] - source_pts[0])
    basis1 = (source_pts[1] - source_pts[0]) / l_b1

    l_b2 = LA.norm(np.cross(basis1, source_pts[2] - source_pts[0]))
    basis2 = np.cross(basis1, source_pts[2] - source_pts[0]) / l_b2
    basis3 = np.cross(basis1, basis2)

    if abs(l_b1) < 1e-6:
        print("Warning l_b1", l_b1)
    if abs(l_b2) < 1e-6:
        print("Warning l_b2", l_b2)

    Da1, Da2, Da3 = source_pts_idx[0] * dim, source_pts_idx[0] * dim + 1, source_pts_idx[0] * dim + 2
    Db1, Db2, Db3 = source_pts_idx[1] * dim, source_pts_idx[1] * dim + 1, source_pts_idx[1] * dim + 2
    Dc1, Dc2, Dc3 = source_pts_idx[2] * dim, source_pts_idx[2] * dim + 1, source_pts_idx[2] * dim + 2

    a, b, c = source_pts[0], source_pts[1], source_pts[2]

    # assemble projection matrix
    for i in range(len(target_pts)):
        Dv1, Dv2, Dv3 = target_pts_idx[i], target_pts_idx[i] + 1, target_pts_idx[i] + 2
        v = target_pts[i]

        # assembly the coordinate m
        project_mat[i * dim, Da1: Da1+3] = (a-b) / l_b1 + (a-v) / l_b1
        project_mat[i * dim, Db1: Db1+3] = (v-a) / l_b1
        project_mat[i * dim, Dv1: Dv1+3] = (b-a) / l_b1

        # assembly the coordinate n
        project_mat[i * dim, Da1] = ((b[2] - c[2])*(a[1] - v[1]))/l_b2 - ((b[1] - c[1])*(a[2] - v[2]))/l_b2 - ((a[1] - b[1])*(a[2] - c[2]) - (a[2] - b[2])*(a[1] - c[1]))/l_b2
        project_mat[i * dim, Da2] = ((a[0] - b[0])*(a[2] - c[2]) - (a[2] - b[2])*(a[0] - c[0]))/l_b2 + ((b[0] - c[0])*(a[2] - v[2]))/l_b2 - ((b[2] - c[2])*(a[0] - v[0]))/l_b2
        project_mat[i * dim, Da3] = ((b[1] - c[1])*(a[0] - v[0]))/l_b2 - ((b[0] - c[0])*(a[1] - v[1]))/l_b2 - ((a[0] - b[0])*(a[1] - c[1]) - (a[1] - b[1])*(a[0] - c[0]))/l_b2
        project_mat[i * dim, Db1] = ((a[1] - c[1])*(a[2] - v[2]))/l_b2 - ((a[2] - c[2])*(a[1] - v[1]))/l_b2
        project_mat[i * dim, Db2] = ((a[2] - c[2])*(a[0] - v[0]))/l_b2 - ((a[0] - c[0])*(a[2] - v[2]))/l_b2
        project_mat[i * dim, Db3] = ((a[0] - c[0])*(a[1] - v[1]))/l_b2 - ((a[1] - c[1])*(a[0] - v[0]))/l_b2
        project_mat[i * dim, Dc1] = ((a[2] - b[2])*(a[1] - v[1]))/l_b2 - ((a[1] - b[1])*(a[2] - v[2]))/l_b2
        project_mat[i * dim, Dc2] = ((a[0] - b[0])*(a[2] - v[2]))/l_b2 - ((a[2] - b[2])*(a[0] - v[0]))/l_b2
        project_mat[i * dim, Dc3] = ((a[1] - b[1])*(a[0] - v[0]))/l_b2 - ((a[0] - b[0])*(a[1] - v[1]))/l_b2
        project_mat[i * dim, Dv1] = ((a[1] - b[1])*(a[2] - c[2]) - (a[2] - b[2])*(a[1] - c[1]))/l_b2
        project_mat[i * dim, Dv2] = -((a[0] - b[0])*(a[2] - c[2]) - (a[2] - b[2])*(a[0] - c[0]))/l_b2
        project_mat[i * dim, Dv3] = ((a[0] - b[0])*(a[1] - c[1]) - (a[1] - b[1])*(a[0] - c[0]))/l_b2

        # assembly the coordinate o
        project_mat[i * dim, Da1] =(((a[1] - b[1])*(b[1] - c[1]))/(l_b1*l_b2) + ((a[2] - b[2])*(b[2] - c[2]))/(l_b1*l_b2))*(a[0] - v[0]) - (((a[0] - b[0])*(a[1] - c[1]) - (a[1] - b[1])*(a[0] - c[0]))/(l_b1*l_b2) + ((a[0] - b[0])*(b[1] - c[1]))/(l_b1*l_b2))*(a[1] - v[1]) - (((a[0] - b[0])*(a[2] - c[2]) - (a[2] - b[2])*(a[0] - c[0]))/(l_b1*l_b2) + ((a[0] - b[0])*(b[2] - c[2]))/(l_b1*l_b2))*(a[2] - v[2]) + (((a[0] - b[0])*(a[1] - c[1]) - (a[1] - b[1])*(a[0] - c[0]))*(a[1] - b[1]))/(l_b1*l_b2) + (((a[0] - b[0])*(a[2] - c[2]) - (a[2] - b[2])*(a[0] - c[0]))*(a[2] - b[2]))/(l_b1*l_b2)
        project_mat[i * dim, Da2] =(((a[0] - b[0])*(b[0] - c[0]))/(l_b1*l_b2) + ((a[2] - b[2])*(b[2] - c[2]))/(l_b1*l_b2))*(a[1] - v[1]) + (((a[0] - b[0])*(a[1] - c[1]) - (a[1] - b[1])*(a[0] - c[0]))/(l_b1*l_b2) - ((a[1] - b[1])*(b[0] - c[0]))/(l_b1*l_b2))*(a[0] - v[0]) - (((a[1] - b[1])*(a[2] - c[2]) - (a[2] - b[2])*(a[1] - c[1]))/(l_b1*l_b2) + ((a[1] - b[1])*(b[2] - c[2]))/(l_b1*l_b2))*(a[2] - v[2]) - (((a[0] - b[0])*(a[1] - c[1]) - (a[1] - b[1])*(a[0] - c[0]))*(a[0] - b[0]))/(l_b1*l_b2) + (((a[1] - b[1])*(a[2] - c[2]) - (a[2] - b[2])*(a[1] - c[1]))*(a[2] - b[2]))/(l_b1*l_b2)
        project_mat[i * dim, Da3] =(((a[0] - b[0])*(b[0] - c[0]))/(l_b1*l_b2) + ((a[1] - b[1])*(b[1] - c[1]))/(l_b1*l_b2))*(a[2] - v[2]) + (((a[0] - b[0])*(a[2] - c[2]) - (a[2] - b[2])*(a[0] - c[0]))/(l_b1*l_b2) - ((a[2] - b[2])*(b[0] - c[0]))/(l_b1*l_b2))*(a[0] - v[0]) + (((a[1] - b[1])*(a[2] - c[2]) - (a[2] - b[2])*(a[1] - c[1]))/(l_b1*l_b2) - ((a[2] - b[2])*(b[1] - c[1]))/(l_b1*l_b2))*(a[1] - v[1]) - (((a[0] - b[0])*(a[2] - c[2]) - (a[2] - b[2])*(a[0] - c[0]))*(a[0] - b[0]))/(l_b1*l_b2) - (((a[1] - b[1])*(a[2] - c[2]) - (a[2] - b[2])*(a[1] - c[1]))*(a[1] - b[1]))/(l_b1*l_b2)
        project_mat[i * dim, Db1] =(((a[0] - b[0])*(a[1] - c[1]) - (a[1] - b[1])*(a[0] - c[0]))/(l_b1*l_b2) + ((a[0] - b[0])*(a[1] - c[1]))/(l_b1*l_b2))*(a[1] - v[1]) - (((a[1] - b[1])*(a[1] - c[1]))/(l_b1*l_b2) + ((a[2] - b[2])*(a[2] - c[2]))/(l_b1*l_b2))*(a[0] - v[0]) + (((a[0] - b[0])*(a[2] - c[2]) - (a[2] - b[2])*(a[0] - c[0]))/(l_b1*l_b2) + ((a[0] - b[0])*(a[2] - c[2]))/(l_b1*l_b2))*(a[2] - v[2])
        project_mat[i * dim, Db2] =(((a[1] - b[1])*(a[2] - c[2]) - (a[2] - b[2])*(a[1] - c[1]))/(l_b1*l_b2) + ((a[1] - b[1])*(a[2] - c[2]))/(l_b1*l_b2))*(a[2] - v[2]) - (((a[0] - b[0])*(a[1] - c[1]) - (a[1] - b[1])*(a[0] - c[0]))/(l_b1*l_b2) - ((a[1] - b[1])*(a[0] - c[0]))/(l_b1*l_b2))*(a[0] - v[0]) - (((a[0] - b[0])*(a[0] - c[0]))/(l_b1*l_b2) + ((a[2] - b[2])*(a[2] - c[2]))/(l_b1*l_b2))*(a[1] - v[1])
        project_mat[i * dim, Db3] =- (((a[0] - b[0])*(a[0] - c[0]))/(l_b1*l_b2) + ((a[1] - b[1])*(a[1] - c[1]))/(l_b1*l_b2))*(a[2] - v[2]) - (((a[0] - b[0])*(a[2] - c[2]) - (a[2] - b[2])*(a[0] - c[0]))/(l_b1*l_b2) - ((a[2] - b[2])*(a[0] - c[0]))/(l_b1*l_b2))*(a[0] - v[0]) - (((a[1] - b[1])*(a[2] - c[2]) - (a[2] - b[2])*(a[1] - c[1]))/(l_b1*l_b2) - ((a[2] - b[2])*(a[1] - c[1]))/(l_b1*l_b2))*(a[1] - v[1])
        project_mat[i * dim, Dc1] =((a[1] - b[1])**2/(l_b1*l_b2) + (a[2] - b[2])**2/(l_b1*l_b2))*(a[0] - v[0]) - ((a[0] - b[0])*(a[1] - b[1])*(a[1] - v[1]))/(l_b1*l_b2) - ((a[0] - b[0])*(a[2] - b[2])*(a[2] - v[2]))/(l_b1*l_b2)
        project_mat[i * dim, Dc2] =((a[0] - b[0])**2/(l_b1*l_b2) + (a[2] - b[2])**2/(l_b1*l_b2))*(a[1] - v[1]) - ((a[0] - b[0])*(a[1] - b[1])*(a[0] - v[0]))/(l_b1*l_b2) - ((a[1] - b[1])*(a[2] - b[2])*(a[2] - v[2]))/(l_b1*l_b2)
        project_mat[i * dim, Dc3] =((a[0] - b[0])**2/(l_b1*l_b2) + (a[1] - b[1])**2/(l_b1*l_b2))*(a[2] - v[2]) - ((a[0] - b[0])*(a[2] - b[2])*(a[0] - v[0]))/(l_b1*l_b2) - ((a[1] - b[1])*(a[2] - b[2])*(a[1] - v[1]))/(l_b1*l_b2)
        project_mat[i * dim, Dv1] =- (((a[0] - b[0])*(a[1] - c[1]) - (a[1] - b[1])*(a[0] - c[0]))*(a[1] - b[1]))/(l_b1*l_b2) - (((a[0] - b[0])*(a[2] - c[2]) - (a[2] - b[2])*(a[0] - c[0]))*(a[2] - b[2]))/(l_b1*l_b2)
        project_mat[i * dim, Dv2] =(((a[0] - b[0])*(a[1] - c[1]) - (a[1] - b[1])*(a[0] - c[0]))*(a[0] - b[0]))/(l_b1*l_b2) - (((a[1] - b[1])*(a[2] - c[2]) - (a[2] - b[2])*(a[1] - c[1]))*(a[2] - b[2]))/(l_b1*l_b2)
        project_mat[i * dim, Dv3] =(((a[0] - b[0])*(a[2] - c[2]) - (a[2] - b[2])*(a[0] - c[0]))*(a[0] - b[0]))/(l_b1*l_b2) + (((a[1] - b[1])*(a[2] - c[2]) - (a[2] - b[2])*(a[1] - c[1]))*(a[1] - b[1]))/(l_b1*l_b2)

    return project_mat, basis1, basis2, basis3

def get_matrix_for_fixed_points(fixed_points_index, points_num, dim):
    C = np.zeros((len(fixed_points_index) * dim, points_num * dim))
    # for fixed joints
    for row, point_idx in enumerate(fixed_points_index):
        C[row * dim: row * dim + dim, point_idx * dim: point_idx * dim + dim] = np.identity(dim)

    return C

# get constraint for a single point
def get_constraits_for_points_allowed_motions(allowed_motions, target_points, dim):
    allowed_motion_vecs = np.zeros((len(allowed_motions) * len(target_points), len(target_points) * dim))

    for idx, constraint in enumerate(allowed_motions):
        for j in range(len(target_points)):
            row_num = idx * len(target_points) + j
            if constraint[0] == "T":  # translation
                allowed_motion_vecs[row_num, j * dim: j * dim + 2] = constraint[1]
            elif constraint[0] == "R":
                rot_arm = target_points[j] - constraint[1]
                allowed_motion_vecs[row_num, j * dim] = -rot_arm[1]
                allowed_motion_vecs[row_num, j * dim + 1] = rot_arm[0]

    return null_space(allowed_motion_vecs).T
