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
from numpy.linalg import inv
from numpy.linalg import matrix_rank
from scipy.linalg import polar


def rigidity_matrix(points: np.ndarray, edges: np.ndarray, dim: int) -> np.ndarray:
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


def spring_energy_matrix(
    points: np.ndarray, edges: np.ndarray, dim: int = 3
) -> np.ndarray:
    K = np.zeros((len(edges), len(edges)))
    P = np.zeros((len(edges), len(edges) * dim))
    A = np.zeros((len(edges) * dim, len(points) * dim))

    normalized = lambda v: v / LA.norm(v)

    # forming P and K
    for idx, e in enumerate(edges):
        p1 = points[e[0]]
        p2 = points[e[1]]
        edge_vec = p1 - p2
        assert LA.norm(edge_vec) > 1e-6
        normalized_edge_vec = normalized(edge_vec)
        P[idx][idx * dim : idx * dim + dim] = normalized_edge_vec.T
        K[idx][idx] = 1/LA.norm(edge_vec)  # set as the same material for now
        # K[idx][idx] = 1  # set as the same material for now

    # forming A
    for idx, e in enumerate(edges):
        for d in range(dim):
            A[dim * idx + d][dim * e[0] + d] = 1
            A[dim * idx + d][dim * e[1] + d] = -1

    return A.T @ P.T @ K @ P @ A


def tranform_matrix_fitting(points_start, points_end, dim=3):
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

if __name__ == "__main__":
    points = np.array([[0, 0], [0, 1], [1,1], [1, 0]]) * 20 + 5
    edges = np.array([[0, 1], [1, 2],[2,3], [3, 1]])

    # points = np.array([[0, 0], [0, 1], [1, 0]]) * 20 + 5
    # edges = np.array([[0, 1], [1, 2], [2, 0]])

    M = spring_energy_matrix(points, edges, dim=2)
    # M = zhenyuan_method()
    from util.geometry_util import eigen

    pairs = eigen(M, symmetric=True)
    for p in pairs:
        print("======")
        print(p[0])
        points_before = points
        points_after = points_before + 1 * p[1].reshape(-1, 2)
        R, T = tranform_matrix_fitting(points, points_after, dim=2)
        for i, p in enumerate(points_before):
            print("--")
            u, p = polar(R)
            print(u)  # the rotation part
            print(p)  # the sheer, scaling, and other deforming parts

    print("variable number", M.shape[1])
    print("matrix rank", matrix_rank(M))
