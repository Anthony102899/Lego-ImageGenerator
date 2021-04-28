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
from functools import reduce


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


colormap = {
    "rigid": [110 / 255, 179 / 255, 89 / 255],
    "motion": [1, 0, 0],
}


def get_mesh_for_arrows(points, vectors, length_coeff=200, radius_coeff=1, cutoff=0, return_single_mesh=True):
    """
    :param points: starting point (tail position) for each arrow
    :param vectors: pointing direction for each arrow
    :param length_coeff: coefficient for stretching the arrow
    :param radius_coeff: coefficient for thickening the arrow
    :param cutoff: arrow whose length < cutoff will not be generated
    :param return_single_mesh: if True, all arrow meshes will be merge into one mesh and return.
                               Otherwise, they return in a list
    :return: triangle mesh or list of meshes
    """
    arrows = []
    for idx, p in enumerate(points):
        vec = vectors[idx]
        rot_mat = geo_util.rot_matrix_from_vec_a_to_b([0, 0, 1], vec)
        vec_len = LA.norm(vec)
        if vec_len > cutoff:
            cylinder_height = 300 * vec_len * length_coeff
            cylinder_radius = 2 * radius_coeff
            cone_radius = 4 * radius_coeff
            cone_height = 100 * vec_len * length_coeff
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=cylinder_radius,
                cylinder_height=cylinder_height,
                cone_radius=cone_radius,
                cone_height=cone_height,
                resolution=5,
            )
            norm_vec = vec / np.linalg.norm(vec)
            color = (norm_vec[0] + 1) / 2, (norm_vec[1] + 1) / 2, (norm_vec[2] + 1) / 2
            transformed_mesh = copy.deepcopy(arrow).translate(p + 1 * vec).rotate(rot_mat, center=p).paint_uniform_color(color)
            arrows.append(transformed_mesh)

    if return_single_mesh:
        # merge all the meshes
        return reduce(lambda x, y: x + y, arrows, o3d.geometry.TriangleMesh())
    else:
        return arrows


def get_mesh_for_arrows_lego(points, vectors, Rigid):
    arrows = o3d.geometry.TriangleMesh()
    for idx, p in enumerate(points):
        vec = vectors[idx]
        rot_mat = geo_util.rot_matrix_from_vec_a_to_b([0, 0, 1], vec)
        vec_len = LA.norm(vec)
        if vec_len > 0:
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=2,
                cone_radius=4,
                cylinder_height=300 * vec_len,
                cone_height=100 * vec_len,
                resolution=5,
            )
            arrow.compute_vertex_normals()
            norm_vec = vec / np.linalg.norm(vec)
            # arrows.paint_uniform_color([(norm_vec[0]+1)/2,(norm_vec[1]+1)/2,(norm_vec[2]+1)/2])
            if Rigid:
                arrows.paint_uniform_color(colormap["rigid"])
                # arrows += copy.deepcopy(arrow).translate(p).rotate(rot_mat, center=p)\
                #   .paint_uniform_color([(norm_vec[0]+1)/2,(norm_vec[1]+1)/2,(norm_vec[2]+1)/2])
                arrows += copy.deepcopy(arrow).translate(p).rotate(rot_mat, center=p) \
                    .paint_uniform_color(colormap["rigid"])
            # arrows += copy.deepcopy(arrow).translate(p).rotate(rot_mat, center=p).paint_uniform_color([(norm_vec[0]+1)/2,(norm_vec[1]+1)/2,(norm_vec[2]+1)/2])
            else:
                arrows.paint_uniform_color(colormap["motion"])
                arrows += copy.deepcopy(arrow).translate(p).rotate(rot_mat, center=p) \
                    .paint_uniform_color(colormap["motion"])
    return arrows


def get_bricks_meshes(bricks):
    meshs = o3d.geometry.TriangleMesh()
    for brick in bricks:
        meshs += brick.get_mesh()
    return meshs


def visualize_3D(points: np.array,
                 lego_bricks=None,
                 edges: List[Tuple] = None,
                 arrows=None,
                 show_axis=True,
                 show_point=True,
                 show_edge=True,
                 ):
    meshes = get_geometries_3D(points, lego_bricks, edges, arrows, show_axis, show_point, show_edge)
    o3d.visualization.draw_geometries(meshes)


def get_geometries_3D(
        points: np.array,
        lego_bricks=None,
        edges: List[Tuple] = None,
        arrows=None,
        show_axis=True,
        show_point=True,
        show_edge=True,
):
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

    if edges is not None and show_edge:
        edge_line_set = get_lineset_for_edges(points, edges)
        return [hybrid_mesh, edge_line_set]
    else:
        return [hybrid_mesh]


def visualize_hinges(points, edges, pivots, axes):
    hybrid_mesh = o3d.geometry.TriangleMesh()

    pivot_meshes = get_mesh_for_points(pivots)
    hybrid_mesh += pivot_meshes

    max_p = np.max(points, axis=0)
    min_p = np.min(points, axis=0)
    dis = np.linalg.norm(max_p - min_p)
    arrows = get_mesh_for_arrows(pivots, axes, length_coeff=dis / 100, radius_coeff=dis / 200)
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
