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
)
from solvers.rigidity_solver.internal_structure import structure_sampling
import solvers.rigidity_solver.visualization as vis
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

    
def simulate_step(structure_graph: ConnectivityGraph, n: int, bricks, step_size=1):
    structure_graph.bricks = bricks
    points, edges, points_on_brick, direction_for_abstract_edge = structure_sampling(structure_graph)

    M = spring_energy_matrix(points, edges, direction_for_abstract_edge)

    e_pairs = geo_util.eigen(M, symmetric=True)

    # collect all eigen vectors with zero eigen value
    zeroeigenspace = [e_vec for e_val, e_vec in e_pairs if abs(e_val) < 1e-6]

    print("Number of points", len(points))

    # Trivial basis -- orthonormalized translation along / rotation wrt 3 axes
    basis = geo_util.trivial_basis(points)

    # cast the eigenvectors corresponding to zero eigenvalues into nullspace of the trivial basis,
    # in other words, the new vectors doesn't have any components (projection) in the span of the trivial basis
    reduced_zeroeigenspace = [geo_util.subtract_orthobasis(vec, basis) for vec in zeroeigenspace]
    
    # count zero vectors in reduced eigenvectors
    num_zerovectors = sum([np.isclose(vec, np.zeros_like(vec)).all() for vec in reduced_zeroeigenspace])
    # In 3d cases, if the object only has 6 DOF, then exactly 6 eigenvectors for eigenvalue 0 are reduced to zerovector.
    assert num_zerovectors == 6

    e_vec = reduced_zeroeigenspace[n]
    e_vec = e_vec / LA.norm(e_vec)

    deformed_bricks = copy.deepcopy(bricks)
    delta_x = e_vec.reshape(-1, 3)

    for i in range(len(bricks)):
        indices_on_brick_i = np.array(points_on_brick[i])
        points_before = points[indices_on_brick_i]
        points_after = points_before + step_size * delta_x[indices_on_brick_i]
        R, T = transform_matrix_fitting(points_before, points_after)

        deformed_bricks[i].trans_matrix[:3, :3] = (
            R @ deformed_bricks[i].trans_matrix[:3, :3]
        )
        deformed_bricks[i].trans_matrix[:3, 3] = (
            R @ deformed_bricks[i].trans_matrix[:3, 3] + T
        )
        deformed_bricks[i].color = 4  # transparent color : 43

    return deformed_bricks

if __name__ == "__main__":
    debugger = MyDebugger("test")

    bricks = read_bricks_from_file("./data/full_models/hole_axle_test.ldr")
    structure_graph = ConnectivityGraph(bricks)



    for dim in range(6):
        d_bricks = copy.deepcopy(bricks)
        total_bricks = d_bricks
        for i in range(50):
            print("simulation step", i, "...")
            d_bricks = simulate_step(structure_graph, n=dim, bricks=d_bricks, step_size=1)
            total_bricks += d_bricks

        write_bricks_to_file(
            total_bricks, file_path=debugger.file_path(f"simulation_{dim}.ldr"), debug=False
        )
