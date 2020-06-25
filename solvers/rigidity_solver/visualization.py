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
from solvers.rigidity_solver.algo_core import spring_energy_matrix



def show_graph(points: List[np.array], edges: List[List], vectors: List[np.array]):
    assert len(points) == len(vectors)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=2)

    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.9, 0.1, 0.1])

    points = [p for p in points]

    spheres = [copy.deepcopy(sphere).translate(p) for p in points]
    arrows = []
    for idx, p in enumerate(points):
        vec = vectors[idx]
        rot_mat = geo_util.rot_matrix_from_vec_a_to_b([0,0,1],vec)
        vec_len = LA.norm(vec)
        if vec_len > 0:
            arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.2, cone_radius=0.35, cylinder_height=10*vec_len, cone_height=8* vec_len,resolution=3)
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
