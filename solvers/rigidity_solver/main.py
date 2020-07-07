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
    tranform_matrix_fitting,
)
from solvers.rigidity_solver.internal_structure import structure_sampling
import solvers.rigidity_solver.visualization as vis
import copy
from sympy import Matrix


def simulate_step(structure_graph: ConnectivityGraph, n: int, bricks, step_size=1):
    structure_graph.bricks = bricks
    points, edges, points_on_brick = structure_sampling(structure_graph)

    M = spring_energy_matrix(points, edges)

    C = geo_util.eigen(M, symmetric=True)

    # collect all eigen vectors with zero eigen value
    eigen_space = []
    for i in range(len(C)):
        e_val, e_vec = C[i]
        if abs(e_val) < 1e-6:
            eigen_space.append(e_vec)

    M = Matrix(np.array(eigen_space))
    M_rref = M.rref()[0] # reduced row echelon form

    e_vec = np.array(M_rref.row(n)).astype(np.float64)
    e_vec = e_vec / LA.norm(e_vec)

    deformed_bricks = copy.deepcopy(bricks)
    delta_x = e_vec.reshape(-1, 3)

    for i in range(len(bricks)):
        indices_on_brick_i = np.array(points_on_brick[i])
        points_before = points[indices_on_brick_i]
        points_after = points_before + step_size * delta_x[indices_on_brick_i]
        R, T = tranform_matrix_fitting(points_before, points_after)

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
    bricks = read_bricks_from_file("./data/full_models/single_pin.ldr")
    write_bricks_to_file(
        bricks, file_path=debugger.file_path("model_loaded.ldr"), debug=False
    )
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
