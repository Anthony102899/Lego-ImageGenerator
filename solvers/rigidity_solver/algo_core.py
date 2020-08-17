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

def _remove_fixed_points_edges(points: np.ndarray, edges: np.ndarray, fixed_points_idx):
    """
    subroutine used by spring_energy_matrix. remove the fixed points and edges by deleting them from inputs
    """
    if len(fixed_points_idx) == 0:
        return points, edges

    fixed_edges_idx = [
        index
        for index, (pt_a, pt_b) in enumerate(edges) 
        if pt_a in fixed_points_idx and pt_b in fixed_points_idx
    ]

    if len(fixed_points_idx) > 0:
        points = np.delete(points, fixed_points_idx, axis=0)
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

    if fixed_points_idx is None:
        fixed_points_idx = []
    # remove the fixed items by deleting them from inputs
    points, edges = _remove_fixed_points_edges(points, edges, fixed_points_idx)


    K = np.zeros((len(edges), len(edges)))
    P = np.zeros((len(edges), len(edges) * dim))
    A = np.zeros((len(edges) * dim, len(points) * dim))

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

        P[idx][idx * dim : idx * dim + dim] = normalized(edge_vec).T
        K[idx][idx] = 1 / LA.norm(edge_vec)  # set as the same material for now
        for d in range(dim):
            A[dim * idx + d][dim * e[0] + d] = 1
            A[dim * idx + d][dim * e[1] + d] = -1

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
    M = spring_energy_matrix(points, edges, fixed_points, dim)
    e_pairs = geo_util.eigen(M, symmetric=True)

    # collect all eigen vectors with zero eigen value
    zero_eigenspace     = [(e_val, e_vec) for e_val, e_vec in e_pairs if abs(e_val) < 1e-6]
    non_zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in e_pairs if abs(e_val) >= 1e-6]

    if len(zero_eigenspace) == (3 if dim == 2 else 6):
        return True, non_zero_eigenspace
    else:
        return False, zero_eigenspace



if __name__ == "__main__":
    debugger = MyDebugger("test")

    #### Test data #1
    # dimension = 2
    # points = np.array([[0, 0], [1, 0], [0, 2], [0, 2]])
    # edges = [(0, 1), (1, 2), (0, 3)]
    # abstract_edges = [(2, 3, 1.0, 0.0)]
    # points_on_parts = {0: [0, 1], 1: [1, 2], 2: [0, 3]}

    #### Test data #2
    dimension = 2
    points = np.array([[0, 0], [1, 0], [0, 1]])
    fixed_points_index = [0,1]
    edges = [(0, 1), (1, 2)]
    abstract_edges = []
    points_on_parts = {0: [0, 1], 1: [1, 2], 2: [0, 2]}

    is_rigid, eigen_pairs = solve_rigidity(points, edges + abstract_edges, fixed_points=fixed_points_index, dim=dimension)
    if is_rigid:
        vec, value = get_weakest_displacement(eigen_pairs, dim=dimension)
        visualize_2D(points, edges, vec)
    else:
        motion_vecs = get_motions(eigen_pairs, dim=dimension)
        visualize_2D(points, edges, motion_vecs[0])


    print(is_rigid)