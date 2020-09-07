from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from bricks_modeling.connections.conn_type import ConnType
import numpy as np
import util.geometry_util as geo_util
import open3d as o3d
import copy
from typing import List
import itertools
from numpy import linalg as LA
from numpy.linalg import inv, matrix_rank
from visualization.model_visualizer import visualize_3D, visualize_2D
from scipy.linalg import null_space
from numpy.linalg import cholesky
from numpy.linalg import inv
from numpy.linalg import matrix_rank

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
    fixed_points_idx=None,
    dim: int = 3,
    matrices=False,
    ) -> np.ndarray:
    """
    fixed_points_idx: list of indices of points to be fixed
    matrices: return K, P, A if true
    """

    if fixed_points_idx is not None:
        points, edges = _remove_fixed_edges(points, edges, fixed_points_idx)

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
            edge_vec = normalized(edge_vec)/1e-4 # making the spring strong by shorter the edge

        P[idx, idx * dim : idx * dim + dim] = normalized(edge_vec).T
        K[idx, idx] = 1 / LA.norm(edge_vec)
        # K[idx, idx] = 1 # set as the same material for debugging

        for d in range(dim):
            A[dim * idx + d, dim * e[0] + d] = 1
            A[dim * idx + d, dim * e[1] + d] = -1

    deleting_indices = []
    for idx in fixed_points_idx:
        deleting_indices.extend([idx * dim + i for i in range(dim)])

    A = np.delete(A, deleting_indices, axis=1)

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

# to predict the rigidity of the structure
def solve_rigidity(points: np.ndarray, edges: np.ndarray, fixed_points = [], dim: int = 3) -> (bool, List[np.ndarray]):
    fixed_points.sort()
    M = spring_energy_matrix(points, edges, fixed_points, dim)
    # the eigenvectors here don't have entries for fixed_points
    e_pairs = geo_util.eigen(M, symmetric=True)

    inserting_indices = []
    for idx in fixed_points:
        inserting_indices.extend([dim*idx + i for i in range(dim)])

    def fill_zeros_at_fixed_points(vector):
        for index in inserting_indices:
            vector = np.insert(vector, index, values=0)
        return vector

    # fill zeros into the eigenvectors at indices of fixed_points
    e_pairs = [(e_val, fill_zeros_at_fixed_points(e_vec)) for e_val, e_vec in e_pairs]

    # collect all eigen vectors with zero eigen value
    zero_eigenspace     = [(e_val, e_vec) for e_val, e_vec in e_pairs if abs(e_val) < 1e-6]
    non_zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in e_pairs if abs(e_val) >= 1e-6]

    if len(zero_eigenspace) == 0 or (len(fixed_points) == 0 and len(zero_eigenspace) == (3 if dim == 2 else 6)):
        return True, non_zero_eigenspace
    else:
        return False, zero_eigenspace

def solve_rigidity_new(points: np.ndarray, edges: np.ndarray, joints, fixed_points_idx = [], dim: int = 3) -> (bool, List[np.ndarray]):
    M = spring_energy_matrix(points, edges, dim=dim, fixed_points_idx=[])

    A = np.zeros((1, len(points) * dim))
    for joint in joints:
        edge_idx, motion_points, allowed_motions = joint[0], joint[1], joint[2]

        A_joint = constraints_for_one_joint(points[np.array(list(edges[edge_idx]))], list(edges[edge_idx]),
                                      points[np.array(motion_points)], motion_points,
                                      points,
                                      allowed_motions=allowed_motions)
        A = np.append(A, A_joint, 0)

    # adding the fixed points constraints
    C = get_matrix_for_fixed_points(fixed_points_idx, len(points), dim)
    A = np.append(A, C, 0)

    # mathmatical computation
    B = null_space(A)
    T = np.transpose(B) @ B
    L = cholesky(T)
    S = B.T @ M @ B
    print("T rank:", matrix_rank(T))
    print("S rank:", matrix_rank(S))

    # compute eigen value / vectors
    eigen_pairs = geo_util.eigen(inv(L).T @ S @ inv(L), symmetric=True)
    eigen_pairs = [(e_val, B @ e_vec) for e_val, e_vec in eigen_pairs]

    # judge rigid or not
    print("DoF:", len([(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) < 1e-6]))

    trivial_motions     = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if
                           abs(e_val) < 1e-6 and geo_util.is_trivial_motions([e_vec], points, dim)]

    non_trivial_motions = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if
                           abs(e_val) < 1e-6 and (not geo_util.is_trivial_motions([e_vec], points, dim))]

    non_zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) >= 1e-6]

    return trivial_motions, non_trivial_motions, non_zero_eigenspace

def constraints_for_one_joint(source_pts, source_pts_idx, target_pts, target_pts_idx, points, allowed_motions, dim = 2):
    points_num = len(points)

    forbidden_motion_vecs = get_constraits_for_points_allowed_motions(allowed_motions, target_pts, dim=dim)

    constraint_mat = np.zeros((len(forbidden_motion_vecs), (len(target_pts) * dim)))
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
        project_mat[i * dim, idx_0 * dim   : idx_0 * dim + dim]     = np.transpose((points[idx_0] - points[jo2_idx])/b1_length - basis1)
        project_mat[i * dim, idx_1   * dim : idx_1   * dim + dim ]  = np.transpose((points[jo2_idx] - points[idx_0])/b1_length)

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

    print(f"constraint matrix: rank{matrix_rank(constraint_mat)}")
    print(constraint_mat)
    print(f"projection matrix: rank{matrix_rank(project_mat)}")
    print(project_mat)
    print(f"A: rank{matrix_rank(A)}")
    print(A)

    return A

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
