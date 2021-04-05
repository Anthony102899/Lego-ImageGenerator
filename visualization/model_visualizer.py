import copy
import networkx as nx
import os
from util.debugger import MyDebugger
import random
import matplotlib.pyplot as plt
import open3d as o3d
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
import util.geometry_util as geo_util
import numpy as np
from numpy import linalg as LA
from typing import List, Tuple


def get_mesh_for_points(points: List[np.ndarray]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3, resolution=8)
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


def get_mesh_for_arrows(points, vectors, vec_len_coeff=200):
    arrows = o3d.geometry.TriangleMesh()
    for idx, p in enumerate(points):
        vec = vectors[idx]
        rot_mat = geo_util.rot_matrix_from_vec_a_to_b([0, 0, 1], vec)
        vec_len = LA.norm(vec)
        if vec_len > 0:
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.1,
                cone_radius=0.35,
                cylinder_height=vec_len_coeff * vec_len,
                cone_height=vec_len_coeff / 25 * vec_len,
                resolution=5,
            )
            norm_vec = vec / np.linalg.norm(vec)
            arrows += copy.deepcopy(arrow).translate(p + 1 * vec).rotate(rot_mat, center=p) \
                .paint_uniform_color([(norm_vec[0] + 1) / 2, (norm_vec[1] + 1) / 2, (norm_vec[2] + 1) / 2])
    return arrows


def get_bricks_meshes(bricks):
    meshs = o3d.geometry.TriangleMesh()
    for brick in bricks:
        meshs += brick.get_mesh()
    return meshs


def visualize_3D(points: np.array, lego_bricks=None, edges: List[Tuple] = None, arrows=None, show_axis=True, show_point=True):
    meshes = get_geometries_3D(points, lego_bricks, edges, arrows, show_axis, show_point)
    o3d.visualization.draw_geometries(meshes)

def get_geometries_3D(
        points: np.array,
        lego_bricks=None,
        edges: List[Tuple] = None,
        arrows=None,
        show_axis=True,
        show_point=True):

    hybrid_mesh = o3d.geometry.TriangleMesh()
    if show_point:
        point_meshes = get_mesh_for_points(points)
        hybrid_mesh += point_meshes

    if show_axis:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        hybrid_mesh += mesh_frame

    if lego_bricks is not None:
        model_meshes = get_bricks_meshes(lego_bricks)
        hybrid_mesh += model_meshes

    if arrows is not None:
        arrow_meshes = get_mesh_for_arrows(points, arrows)
        hybrid_mesh += arrow_meshes

    if edges is not None:
        edge_line_set = get_lineset_for_edges(points, edges)
        return [hybrid_mesh, edge_line_set]
    else:
        return [hybrid_mesh]


def visualize_hinges(points, edges, pivots, axes):
    hybrid_mesh = o3d.geometry.TriangleMesh()

    pivot_meshes = get_mesh_for_points(pivots)
    hybrid_mesh += pivot_meshes

    arrows = get_mesh_for_arrows(pivots, axes)
    hybrid_mesh += arrows

    edge_line_set = get_lineset_for_edges(points, edges)
    o3d.visualization.draw_geometries([hybrid_mesh, edge_line_set])


def visualize_2D(points: np.array, edges: List[Tuple] = None, arrows=None):
    # create Graph
    G_symmetric = nx.Graph()

    edge_color = ["gray" for i in range(len(edges))]

    # draw networks
    G_symmetric.add_nodes_from([i for i in range(len(points))])
    node_color = ["blue" for i in range(len(points))]
    node_pos = [[p[0], p[1]] for p in points]

    nx.draw_networkx(G_symmetric, pos=node_pos, node_size=10, node_color=node_color, width=0.7,
                     edgelist=edges, edge_color=edge_color,
                     with_labels=False, style="solid")

    if arrows is not None:
        ax = plt.axes()
        ax.autoscale(enable=True)
        for i in range(len(points)):
            if LA.norm(arrows[i]) > 1e-4:
                p_start = points[i]
                ax.arrow(p_start[0], p_start[1], arrows[i][0], arrows[i][1], head_width=0.05, head_length=0.1, fc='k',
                         ec='k')

    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin - 1, xmax + 1, ymin - 1, ymax + 1])
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def show_3D_example():
    file_path = "../data/full_models/hinged_L.ldr"
    bricks = read_bricks_from_file(file_path)

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]])
    edges = [(0, 1), (1, 2), (2, 0)]
    vectors = np.array([[0, 0, 0.01], [0, 0, 0.01], [0, 0, 0.05]])
    visualize_3D(points, lego_bricks=bricks, edges=edges, arrows=vectors, show_axis=True)


def show_2D_example():
    points = np.array([[0, 0], [0.5, 0], [0, 0.5]])
    edges = [(0, 1), (1, 2), (2, 0)]
    vectors = np.array([[0, 0.1], [0, 0.1], [0, 0.1]])
    visualize_2D(points, edges, vectors)


if __name__ == "__main__":
    show_3D_example()
    # show_2D_example()
