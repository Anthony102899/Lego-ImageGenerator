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
from solvers.rigidity_solver.algo_core import spring_energy_matrix, tranform_matrix_fitting
from solvers.rigidity_solver.internal_structure import structure_sampling
import solvers.rigidity_solver.visualization as vis
import copy

def simulate_step(structure_graph:ConnectivityGraph, n: int, bricks, step_size = 1):
    structure_graph.bricks = bricks
    points, edges, points_on_brick = structure_sampling(structure_graph)

    M = spring_energy_matrix(points, edges)

    C = geo_util.eigen(M, symmetric=True)

    e = C[n]
    e_val, e_vec = e

    deformed_bricks = copy.deepcopy(bricks)
    delta_x = e_vec.reshape(-1, 3)

    for i in range(len(bricks)):
        indices_on_brick_i = np.array(points_on_brick[i])
        points_before = points[indices_on_brick_i]
        points_after = points_before + step_size * delta_x[indices_on_brick_i]
        R, T = tranform_matrix_fitting(points_before, points_after)

        deformed_bricks[i].trans_matrix[:3, :3] = R @ deformed_bricks[i].trans_matrix[:3, :3]
        deformed_bricks[i].trans_matrix[:3, 3] =  R @ deformed_bricks[i].trans_matrix[:3, 3] + T
        deformed_bricks[i].color = 4 # transparent color : 43

    # vis.show_graph(list(points)+list(points+delta_x*step_size), edges + list(np.array(edges)+len(edges)), None)

    return deformed_bricks

if __name__ == "__main__":
    debugger = MyDebugger("test")
    bricks = read_bricks_from_file("./data/full_models/cube7.ldr")
    write_bricks_to_file(
        bricks, file_path=debugger.file_path("model_loaded.ldr"), debug=False
    )
    structure_graph = ConnectivityGraph(bricks)

    total_bricks = bricks

    d_bricks = bricks
    for i in range(100):
        print("simulation step", i,"...")
        d_bricks = simulate_step(structure_graph, n=0, bricks=d_bricks, step_size=0.5)
        total_bricks += d_bricks

    write_bricks_to_file(
        total_bricks, file_path=debugger.file_path(f"simulation.ldr"), debug=False
    )

