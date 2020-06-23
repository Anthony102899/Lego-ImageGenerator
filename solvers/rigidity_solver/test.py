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

def show_graph(points: List[np.array], edges: List[List]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=2)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.9, 0.1, 0.1])

    points = [p for p in points]

    spheres = [copy.deepcopy(sphere).translate(p) for p in points]
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
    all_edges = [(0,1),(0,2),(1,2)]
    point_num = 3
    dim = 2

    K = np.zeros([dim * point_num, dim * point_num], dtype=np.float64)
    for edge in all_edges:
        p1, p2 = edge[0], edge[1]
        k = 1
        for d in range(dim):
            pd1 = p1 * dim + d
            pd2 = p2 * dim + d
            # the square terms
            K[pd1][pd1] += k
            K[pd2][pd2] += k
            # the x_i*x_j terms
            K[pd1][pd2] -= k
            K[pd2][pd1] -= k

    print("problem dimemsion:", K.shape[0])
    print("matrix rank:", matrix_rank(K))

    C = geo_util.eigen(K)
    for e in C:
        print(e[0])
