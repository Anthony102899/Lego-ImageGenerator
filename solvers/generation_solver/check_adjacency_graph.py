import os.path

import polygon_intersection as pi
import pickle
import numpy as np

from adjacency_graph import AdjacencyGraph
from precompute import PrecomputedModel
from solvers.generation_solver.sketch_util import calculate_bound_without_col


class Check_Adjacency_Graph:
    """
    Check the details of adjacency graph
    """
    def __init__(self, adjacency_graph_path, from_adjacency_graph=True):
        if from_adjacency_graph:
            # Directly load from Adjacency graph pickle
            self.adjacency_graph: AdjacencyGraph = pickle.load(open(adjacency_graph_path, "rb"))
        else:
            # Load from precompute model
            self.adjacency_graph: AdjacencyGraph = pickle.load(open(adjacency_graph_path, "rb")).get_structure_graph()
        self.alignment_types = [[1, 1], [1, -1], [-1, 1], [-1, -1]]

    def check_adjacency_graph(self, brick_types):
        for brick_type in brick_types:
            for alignment_type in self.alignment_types:
                same_align_bricks = []
                for brick in self.adjacency_graph.bricks:
                    if brick.template.id == brick_type and \
                            brick.trans_matrix[0][0] == alignment_type[0] and \
                            brick.trans_matrix[2][2] == alignment_type[1]:
                        same_align_bricks.append(brick)
                    if brick.template.id == brick_type and \
                            brick.trans_matrix[0][2] == alignment_type[0] and \
                            brick.trans_matrix[2][0] == alignment_type[1]:
                        same_align_bricks.append(brick)
                pi.group_display(same_align_bricks, 'r')


def compare_check_adjacency_graph(g1: Check_Adjacency_Graph, g2: Check_Adjacency_Graph):
    index_1 = 0
    index_2 = 0
    for i in range(max(len(g1.adjacency_graph.bricks), len(g2.adjacency_graph.bricks))):
        brick_1 = g1.adjacency_graph.bricks[index_1]
        brick_2 = g2.adjacency_graph.bricks[index_2]
        if not np.array_equiv(brick_1.trans_matrix, brick_2.trans_matrix):
            break
        index_1 += 1
        index_2 += 1



if __name__ == "__main__":
    check_adjacency_graph = Check_Adjacency_Graph(os.path.dirname(__file__) + "/precompute_models/LEGO_l/LEGO_l_white "
                                                                              "b=8 p=['3024', '3023', '24299', "
                                                                              "'24307', '43722', '43723']  g=5 "
                                                                              "t=32 C.pkl", False)
    bricks = check_adjacency_graph.adjacency_graph.bricks
    for brick in bricks:
        if brick.template.id == "24307":
            brick.template.vertices2D = \
                [[-0.8, 0.0, 0.8], [0.0, 0.0, -0.8], [0.8, 0.0, -0.8], [0.8, 0.0, 0.8], [-0.8, 0.0, 0.8]]
            polygon = calculate_bound_without_col(brick)
        elif brick.template.id == "24299":
            brick.template.vertices2D = \
                [[0.8, 0.0, 0.8], [0.0, 0.0, -0.8], [-0.8, 0.0, -0.8], [-0.8, 0.0, 0.8], [0.8, 0.0, 0.8]]
            polygon = calculate_bound_without_col(brick)
        else:
            continue

    """check_adjacency_graph2 = Check_Adjacency_Graph(os.path.dirname(__file__) + "/precompute_models/LEGO_l/LEGO_l_white "
                                                                              "b=8 p=['3024', '3023', '24299', "
                                                                              "'24307', '43722', '43723']  g=5 "
                                                                              "t=32 T.pkl", False)

    compare_check_adjacency_graph(check_adjacency_graph, check_adjacency_graph2)"""

    """check_adjacency_graph = Check_Adjacency_Graph(os.path.dirname(__file__) + "/connectivity/" +
                                                  "['3024', '3023', '24299', '24307', '43722', '43723'] base=24 "
                                                  "t=31205.38.pkl")"""
    # check_adjacency_graph.check_adjacency_graph(["43722", "43723"])
