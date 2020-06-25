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
from solvers.rigidity_solver.algo_core import spring_energy_matrix
from solvers.rigidity_solver.internal_structure import structure_sampling
import solvers.rigidity_solver.visualization as vis


if __name__ == "__main__":
    debugger = MyDebugger("test")
    bricks = read_bricks_from_file("./data/full_models/test_single_brick.ldr")
    write_bricks_to_file(bricks, file_path=debugger.file_path("test_single_brick.ldr"), debug=False)
    structure_graph = ConnectivityGraph(bricks)
    points, edges = structure_sampling(structure_graph)

    M = spring_energy_matrix(points, edges)

    print("problem dimemsion:", M.shape[0])
    print("matrix rank:", matrix_rank(M))

    C = geo_util.eigen(M, symmetric=True)

    vis.show_graph(points, [], C[1][1].reshape((-1, 3)))
