from bricks_modeling.connectivity_graph import ConnectivityGraph
import numpy as np
from numpy import linalg as LA
import util.geometry_util as geo_util
from solvers.rigidity_solver.algo_core import (
    spring_energy_matrix,
    transform_matrix_fitting,
    solve_rigidity
)
from solvers.rigidity_solver.internal_structure import structure_sampling
import copy

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