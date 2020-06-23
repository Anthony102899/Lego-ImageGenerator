import itertools
import json
import numpy as np

import open3d as o3d
import copy

from bricks_modeling.connections.conn_type import compute_conn_type
from util.json_encoder import NumpyArrayEncoder

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
                                "node_indices": (b_i, b_j),
                                "cpoint_indices": (m, n),
                                "properties": {
                                    "contact_point": self.bricks[b_i]
                                    .get_current_conn_points()[m]
                                    .pos,
                                    "contact_orient": self.bricks[b_i]
                                    .get_current_conn_points()[m]
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

        return json.dumps({"nodes": nodes, "edges": self.edges}, cls=NumpyArrayEncoder)

    def show(self):
        # TODO: show edges in different colors

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=2)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.9, 0.1, 0.1])

        points = [b.get_translation().tolist() for b in self.bricks]

        spheres = [
            copy.deepcopy(sphere).translate(b.get_translation().tolist())
            for b in self.bricks
        ]
        lines = [e["node_indices"] for e in self.edges]
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=20, origin=[0, 0, 0]
        )
        o3d.visualization.draw_geometries([mesh_frame, line_set] + spheres)
