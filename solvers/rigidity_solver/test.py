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

def spring_energy_matrix(points: np.ndarray, edges: np.ndarray, dim: int) -> np.ndarray:
    A = np.zeros(len(edges)* dim, len(points) * dim)
    K = np.zeros(len(edges)* dim, len(edges)* dim)
    for idx, e in enumerate(edges):
        for d in range(dim):
            A[3*idx+d][3*e[0]+d] = 1
            A[3*idx+d][3*e[1]+d] = -1

    for i in range(len(edges)):
        K[i][i] = 1 # set as the same material for now

    return A.T @ K @ A

if __name__ == "__main__":
    points = np.array([
        [0, 0],
        [0, 1],
        [1, 0]
    ]) * 20 + 5
    edges = np.array([
        [0, 1],
        [0, 2],
        [1, 2]
    ])

    points = np.hstack(
        (points, np.zeros((len(points), 1)))
    )

    from solvers.rigidity_solver.xh_solver import show_graph
    from util.geometry_util import eigen

    R = rigidity_matrix(points, edges, 3)
    M = R.T @ R
    pairs = eigen(M, symmetric=True)

    print("variable number", M.shape[1])
    print("matrix rank", matrix_rank(M))

    for i in range(M.shape[1]):
        print(pairs[i][1].reshape((-1, 3)))
    
    for i in range(6, 9):
        show_graph(points, [], pairs[i][1].reshape((-1, 3)))
