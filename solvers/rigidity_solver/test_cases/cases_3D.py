from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from solvers.rigidity_solver.internal_structure import structure_sampling

# a rigid triangle
def case_1():
    bricks = read_bricks_from_file("./data/full_models/hole_axle_test.ldr")
    structure_graph = ConnectivityGraph(bricks)
    points, edges, points_on_brick, abstract_edges = structure_sampling(structure_graph)

    return bricks, points, edges, abstract_edges, points_on_brick