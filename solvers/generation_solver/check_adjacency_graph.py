import os.path

import polygon_intersection as pi
import pickle
import bricks_modeling.file_IO.model_reader

from adjacency_graph import AdjacencyGraph


class Check_Adjacency_Graph:
    def __init__(self, adjacency_graph_path):
        self.adjacency_graph: AdjacencyGraph = pickle.load(open(adjacency_graph_path, "rb"))
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


if __name__ == "__main__":
    check_adjacency_graph = Check_Adjacency_Graph(os.path.dirname(__file__) + "/connectivity/" +
                                                  "['3024', '3023', '24299', '24307', '43722', '43723'] base=24 "
                                                  "t=31205.38.pkl")
    check_adjacency_graph.check_adjacency_graph(["24299", "24307"])
