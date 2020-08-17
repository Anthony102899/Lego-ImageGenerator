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
import util.geometry_util as geo_util
from solvers.rigidity_solver.algo_core import (
    spring_energy_matrix,
    transform_matrix_fitting,
    solve_rigidity
)
from solvers.rigidity_solver.internal_structure import structure_sampling
import visualization.model_visualizer as vis
import copy

def trivial_basis(points: np.ndarray) -> np.ndarray:
    """
    Given n points in 3d space in form of a (n x 3) matrix, construct 6 'trivial' orthonormal vectors
    """
    P = points.reshape((-1, 3))
    n = len(P)

    # translation along x, y, and z
    translations = np.array([
       [1, 0, 0] * n, 
       [0, 1, 0] * n, 
       [0, 0, 1] * n,
    ])

    center = np.mean(P, axis=0)
    P_shifted = P - center # make the rotation vectors orthogonal
    x_axis, y_axis, z_axis = np.identity(3)
    rotations = np.array([
        np.cross(P_shifted, x_axis).reshape(-1),
        np.cross(P_shifted, y_axis).reshape(-1),
        np.cross(P_shifted, z_axis).reshape(-1),
    ])

    print("translation dim", translations.shape)
    print("rotation dim", rotations.shape)

    transformation = np.vstack((translations, rotations))
    # row-wise normalize the vectors into orthonormal basis
    basis = transformation / LA.norm(transformation, axis=1)[:, np.newaxis] 
    return basis



if __name__ == "__main__":
    debugger = MyDebugger("test")

    bricks = read_bricks_from_file("./data/full_models/hole_axle_test.ldr")
    structure_graph = ConnectivityGraph(bricks)
    points, edges, points_on_brick, abstract_edges = structure_sampling(structure_graph)

    is_rigid, eigen_pairs = solve_rigidity(points, edges + abstract_edges, dim=3)

    if is_rigid:
        vec, value = get_weakest_displacement(eigen_pairs, dim=dimension)
        vis.visualize_3D(points, edges, vec)
    else:
        motion_vecs = get_motions(eigen_pairs, dim=dimension)
        vis.visualize_3D(points, edges, motion_vecs[0])


