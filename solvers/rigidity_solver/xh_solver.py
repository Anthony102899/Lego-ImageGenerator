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


def show_graph(points: List[np.array], edges:List[List]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=2)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.9, 0.1, 0.1])

    points = [p for p in points]

    spheres = [
        copy.deepcopy(sphere).translate(p) for p in points
    ]
    lines = [e for e in edges]
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


if __name__ == "__main__":
    debugger = MyDebugger("test")
    bricks = read_bricks_from_file("./data/full_models/hinged_L.ldr")
    write_bricks_to_file(bricks, file_path=debugger.file_path("test.ldr"), debug=False)
    structure_graph = ConnectivityGraph(bricks)

    points = []
    points_on_brick = {i: [] for i in range(len(bricks))}
    contraint_point_pairs = []

    #### sampling variables on contact points
    for edge in structure_graph.edges:
        if edge["type"] == ConnType.HOLE_PIN.name:
            bi = edge["node_indices"][0]
            bj = edge["node_indices"][1]
            contact_pt = edge["properties"]["contact_point"]
            contact_orient = edge["properties"]["contact_orient"]

            point_idx_base = len(points)

            p0 = contact_pt
            p1 = contact_pt + 5 * contact_orient
            p2 = contact_pt - 5 * contact_orient

            p_vec1, p_vec2 = geo_util.get_perpendicular_vecs(p1-p2)
            p3 = contact_pt + 5 * p_vec1
            p4 = contact_pt - 5 * p_vec1
            p5 = contact_pt + 5 * p_vec2
            p6 = contact_pt - 5 * p_vec2

            for i in range(7):
                exec(f"points.append(p{i})")
                points_on_brick[bi].append(point_idx_base + i)
            for i in range(7):
                exec(f"points.append(p{i})")
                points_on_brick[bj].append(point_idx_base + 7 + i)

            contraint_point_pairs.append((point_idx_base+0, point_idx_base+7+0))
            contraint_point_pairs.append((point_idx_base + 1, point_idx_base + 7 + 1))
            contraint_point_pairs.append((point_idx_base + 2, point_idx_base + 7 + 2))

    #### TODO: add additional sample points



    self_support_pairs = []
    for value in points_on_brick.values():
        self_support_pairs.extend(list(itertools.combinations(value, 2)))



    show_graph(points, contraint_point_pairs + self_support_pairs)
