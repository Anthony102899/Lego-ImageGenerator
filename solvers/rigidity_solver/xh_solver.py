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


def get_crystal_vertices(contact_pt: np.array, contact_orient: np.array):
    p0 = contact_pt
    p1 = contact_pt + 5 * contact_orient
    p2 = contact_pt - 5 * contact_orient
    p_vec1, p_vec2 = geo_util.get_perpendicular_vecs(p1 - p2)
    p3 = contact_pt + 5 * p_vec1
    p4 = contact_pt - 5 * p_vec1
    p5 = contact_pt + 5 * p_vec2
    p6 = contact_pt - 5 * p_vec2

    return [p0, p1, p2, p3, p4, p5, p6]


if __name__ == "__main__":
    debugger = MyDebugger("test")
    bricks = read_bricks_from_file("./data/full_models/hinged_L.ldr")
    write_bricks_to_file(bricks, file_path=debugger.file_path("test.ldr"), debug=False)
    structure_graph = ConnectivityGraph(bricks)

    points = []
    points_on_brick = {i: [] for i in range(len(bricks))}
    contraint_point_pairs = []
    feature_points_on_brick = {i: None for i in range(len(bricks))}

    for idx, b in enumerate(bricks):
        feature_points_on_brick[idx] = b.template.deg1_cpoint_indices()

    #### sampling variables on contact points
    for edge in structure_graph.edges:
        if edge["type"] == ConnType.HOLE_PIN.name:
            bi = edge["node_indices"][0]
            bj = edge["node_indices"][1]
            contact_pt = edge["properties"]["contact_point"]
            contact_orient = edge["properties"]["contact_orient"]

            feature_points_on_brick[bi].remove(edge["cpoint_indices"][0])
            feature_points_on_brick[bj].remove(edge["cpoint_indices"][1])

            point_idx_base = len(points)

            p = get_crystal_vertices(contact_pt, contact_orient)

            for i in range(7):
                exec(f"points.append(p[{i}])")
                points_on_brick[bi].append(point_idx_base + i)
            for i in range(7):
                exec(f"points.append(p[{i}])")
                points_on_brick[bj].append(point_idx_base + 7 + i)

            contraint_point_pairs.append((point_idx_base+0, point_idx_base+7+0))
            contraint_point_pairs.append((point_idx_base + 1, point_idx_base + 7 + 1))
            contraint_point_pairs.append((point_idx_base + 2, point_idx_base + 7 + 2))

    #### add additional sample points, by detecting if the connection points are already sampled
    for brick_id, c_id_set in feature_points_on_brick.items():
        brick = bricks[brick_id]
        for c_id in c_id_set:
            c_point = brick.get_current_conn_points()[c_id]
            contact_pt = c_point.pos
            contact_orient = c_point.orient
            p = get_crystal_vertices(contact_pt, contact_orient)
            point_idx_base = len(points)
            for i in range(7):
                exec(f"points.append(p[{i}])")
                points_on_brick[brick_id].append(point_idx_base + i)

    self_support_pairs = []
    for value in points_on_brick.values():
        self_support_pairs.extend(list(itertools.combinations(value, 2)))

    print(points_on_brick)

    all_edges = contraint_point_pairs + self_support_pairs

    K = np.empty([len(all_edges), len(all_edges)])
    A = np.empty([len(all_edges), len(points)])
    for idx, edge in enumerate(all_edges):
        e_1 = edge[0]
        e_2 = edge[1]
        distance = max(1, LA.norm(points[e_1] - points[e_2]))
        K[e_1][e_2] = 1/distance
        K[e_2][e_1] = 1 / distance
        A[idx][e_1] = 1
        A[idx][e_2] = -1

    # print(K)
    # print(A)
    M = A.transpose().dot(K).dot(A)
    print(M)
    show_graph(points, all_edges)
