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

def show_graph(points: List[np.array], edges: List[List], vectors: List[np.array]):
    assert len(points) == len(vectors)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=2)

    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.9, 0.1, 0.1])

    points = [p for p in points]

    spheres = [copy.deepcopy(sphere).translate(p) for p in points]
    arrows = []
    for idx, p in enumerate(points):
        vec = vectors[idx]
        rot_mat = geo_util.rot_matrix_from_vec_a_to_b([0,0,1],vec)
        vec_len = LA.norm(vec)
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.2, cone_radius=0.35, cylinder_height=5*vec_len, cone_height=4* vec_len,resolution=3)
        arrows.append(copy.deepcopy(arrow).translate(p).rotate(rot_mat, center = p))

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
    o3d.visualization.draw_geometries([mesh_frame, line_set] + spheres + arrows)


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
    bricks = read_bricks_from_file("./data/full_models/test_single_brick.ldr")
    write_bricks_to_file(bricks, file_path=debugger.file_path("test_single_brick.ldr"), debug=False)
    structure_graph = ConnectivityGraph(bricks)

    points = []
    points_on_brick = {i: [] for i in range(len(bricks))}
    feature_points_on_brick = {i: None for i in range(len(bricks))}

    for idx, b in enumerate(bricks):
        feature_points_on_brick[idx] = b.template.deg1_cpoint_indices()

    # Requirements on the sample points and their connection:
    # 1) respect symmetric property of the brick
    # 2) self-rigid connection inside each brick
    # 3) respect the joint property
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
                if i in {0,1,2}:
                    exec(f"points.append(p[{i}])")
                    points_on_brick[bi].append(len(points)-1)
                    points_on_brick[bj].append(len(points)-1)
                else:
                    exec(f"points.append(p[{i}])")
                    points_on_brick[bi].append(len(points) - 1)
                    exec(f"points.append(p[{i}])")
                    points_on_brick[bj].append(len(points) - 1)
    

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

    edges = []
    for value in points_on_brick.values():
        edges.extend(list(itertools.combinations(value, 2)))

    all_edges = edges

    # constructing the rigidity matrix R
    R = np.zeros((len(all_edges), 3 * len(points)))
    for i, (p_idx, q_idx) in enumerate(all_edges):
        q_minus_p = points[q_idx] - points[p_idx]
        assert LA.norm(q_minus_p) > 1e-6
        R[i, q_idx * 3: (q_idx + 1) * 3] =  q_minus_p
        R[i, p_idx * 3: (p_idx + 1) * 3] = -q_minus_p

    M: np.ndarray = R.T @ R

    print("problem dimemsion:", M.shape[0])
    print("matrix rank:", matrix_rank(M))

    C = geo_util.eigen(M, symmetric=True)

    print(C[0][1].shape)
    print(C[1][1].shape)

    show_graph(points, [], C[1][1].reshape((-1, 3)))
