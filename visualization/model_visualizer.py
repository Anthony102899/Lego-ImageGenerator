import copy

import open3d as o3d

from bricks_modeling.connectivity_graph import ConnectivityGraph
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from solvers.rigidity_solver.algo_core import spring_energy_matrix
from solvers.rigidity_solver.internal_structure import structure_sampling
import util.geometry_util as geo_util
import numpy as np
from numpy import linalg as LA
from typing import List, Tuple

def get_mesh_for_points(points: List[np.ndarray]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=8)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.9, 0.1, 0.1])
    spheres = o3d.geometry.TriangleMesh()

    for b in points:
        spheres += copy.deepcopy(sphere).translate((b).tolist())

    return spheres

def get_lineset_for_edges(points, edges):
    colors = [[1, 0, 0] for i in range(len(edges))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(edges),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def get_mesh_for_arrows(points, vectors):
    arrows = o3d.geometry.TriangleMesh()
    for idx, p in enumerate(points):
        vec = vectors[idx]
        rot_mat = geo_util.rot_matrix_from_vec_a_to_b([0, 0, 1], vec)
        vec_len = LA.norm(vec)
        if vec_len > 0:
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.2,
                cone_radius=0.35,
                cylinder_height=400 * vec_len,
                cone_height=8 * vec_len,
                resolution=5,
            )
            norm_vec = vec / np.linalg.norm(vec)
            arrows += copy.deepcopy(arrow).translate(p).rotate(rot_mat, center=p)\
                .paint_uniform_color([(norm_vec[0]+1)/2,(norm_vec[1]+1)/2,(norm_vec[2]+1)/2])
    return arrows

def get_bricks_meshes(bricks):
    meshs = o3d.geometry.TriangleMesh()
    for brick in bricks:
        meshs += brick.get_mesh()
    return meshs

def visualize(points: np.array, lego_bricks = None, edges: List[Tuple] = None, arrows = None, show_axis = True):
    hybrid_mesh = o3d.geometry.TriangleMesh()
    point_meshes = get_mesh_for_points(points)
    hybrid_mesh += point_meshes

    if show_axis:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        hybrid_mesh += mesh_frame

    if lego_bricks is not None:
        model_meshes = get_bricks_meshes(bricks)
        hybrid_mesh += model_meshes

    if arrows is not None:
        arrow_meshes = get_mesh_for_arrows(points, vectors)
        hybrid_mesh += arrow_meshes

    if edges is not None:
        edge_line_set = get_lineset_for_edges(points, edges)
        o3d.visualization.draw_geometries([hybrid_mesh, edge_line_set])
    else:
        o3d.visualization.draw_geometries([hybrid_mesh])


if __name__ == "__main__":
    file_path = "../data/full_models/hinged_L.ldr"
    bricks = read_bricks_from_file(file_path)

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]])
    edges = [(0, 1), (1, 2), (2, 0)]
    vectors = np.array([[0, 0, 0.01], [0, 0, 0.01], [0, 0, 0.05]])

    visualize(points, lego_bricks=bricks, edges = edges, arrows=vectors, show_axis=True)
