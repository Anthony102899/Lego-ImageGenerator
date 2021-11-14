from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from solvers.rigidity_solver.internal_structure import structure_sampling
import os
from os.path import dirname as dir

# a rigid triangle
def lego_models(file_name):
    bricks = read_bricks_from_file(
        os.path.join(dir(dir(dir(dir(__file__)))), "data", "full_models", f"{file_name}.ldr")
    )
    structure_graph = ConnectivityGraph(bricks)
    points, edges, points_on_brick, abstract_edges = structure_sampling(structure_graph)

    return bricks, points, edges, abstract_edges, points_on_brick