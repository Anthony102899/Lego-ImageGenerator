import itertools
import json

import open3d as o3d

from bricks_modeling.connections.conn_type import compute_conn_type

"""
To use a graph to describe a LEGO structure
"""


class ConnectivityGraph:
    def __init__(self, bricks):
        self.bricks = bricks
        self.edges = []
        self.build_graph_from_bricks()

    def build_graph_from_bricks(self):
        for b_i, b_j in itertools.combinations(list(range(0, len(self.bricks))), 2):
            brick_i_conn_points = self.bricks[b_i].get_current_conn_points()
            brick_j_conn_points = self.bricks[b_j].get_current_conn_points()

            for m in range(0, len(brick_i_conn_points)):
                for n in range(0, len(brick_j_conn_points)):
                    cpoints_m = brick_i_conn_points[m]
                    cpoints_n = brick_j_conn_points[n]
                    type = compute_conn_type(cpoints_m, cpoints_n)
                    if type is not None:
                        self.edges.append(
                            {
                                "type": type.name,
                                "node_indices": [b_i, b_j],
                                "properties": {
                                    "contact_point_1": self.bricks[b_i]
                                    .template.c_points[m]
                                    .pos,
                                    "contact_orient_1": self.bricks[b_i]
                                    .template.c_points[m]
                                    .orient,
                                    "contact_point_2": self.bricks[b_j]
                                    .template.c_points[n]
                                    .pos,
                                    "contact_orient_2": self.bricks[b_j]
                                    .template.c_points[n]
                                    .orient,
                                },
                            }
                        )

    def to_json(self):
        nodes = []
        ##### Start json building
        for i in range(len(self.bricks)):
            brick = self.bricks[i]
            nodes.append(
                {
                    "translation": brick.get_translation(),
                    "orientation": [
                        brick.trans_matrix[0, 0],
                        brick.trans_matrix[0, 1],
                        brick.trans_matrix[0, 2],
                        brick.trans_matrix[1, 0],
                        brick.trans_matrix[1, 1],
                        brick.trans_matrix[1, 2],
                        brick.trans_matrix[2, 0],
                        brick.trans_matrix[2, 1],
                        brick.trans_matrix[2, 2],
                    ],
                }
            )

        return json.dumps({"nodes": nodes, "edges": self.edges})

    def show(self):
        # TODO: show nodes as balls, and show edges in different colors
        points = [b.get_translation() for b in self.bricks]
        lines = [e["node_indices"] for e in self.edges]
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([line_set])
