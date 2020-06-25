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
from numpy.linalg import matrix_rank

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
        R[i, q_ind * dim: (q_ind + 1) * dim] =  q_minus_p
        R[i, p_ind * dim: (p_ind + 1) * dim] = -q_minus_p
    
    return R


if __name__ == "__main__":
    points = np.array([
        [0, 0],
        [0, 1],
        [1, 0]
    ]) * 20 + 5
    edges = np.array([
        [0, 1],
        [1, 2],
        [2, 0]
    ])

    # points = np.hstack(
    #     (points, np.zeros((len(points), 1)))
    # )

    from solvers.rigidity_solver.xh_solver import show_graph
    from util.geometry_util import eigen
    norm = np.linalg.norm
    normalized = lambda v: v / norm(v)

    A = np.array([
        [1, -1, 0],
        [0, 1, -1],
        [-1, 0, 1],
    ])
    A = np.array([
        [1, 0, -1, 0, 0, 0],
        [0, 1, 0, -1, 0, 0],
        [0, 0, 1, 0, -1, 0],
        [0, 0, 0, 1, 0, -1],
        [-1, 0, 0, 0, 1, 0],
        [0, -1, 0, 0, 0, 1],
    ])
    print(A)

    n = np.array([
        normalized(points[0] - points[1]),
        normalized(points[1] - points[2]),
        normalized(points[2] - points[0]),
    ])
    print(n)
    Ne = np.diag(n.reshape((-1,)))
    Ne[0, 1] = n[0, 1]
    Ne[1, 0] = n[0, 0]
    Ne[2, 3] = n[1, 1]
    Ne[3, 2] = n[1, 0]
    Ne[4, 5] = n[2, 1]
    Ne[5, 4] = n[2, 0]
    print(Ne)
    N = np.diag(n.reshape((-1, ))) @ Ne
    print(N)

    K = np.identity(6)
    Q = A.T @  K @  A

    pairs = eigen(Q, symmetric=True)

    print("variable number", Q.shape[1])
    print("matrix rank", matrix_rank(Q))

    for i in range(Q.shape[1]):
        print(pairs[i][1].reshape((-1, 3)))
    
