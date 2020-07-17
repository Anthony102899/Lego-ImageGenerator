import copy

import open3d as o3d

from bricks_modeling.connectivity_graph import ConnectivityGraph
from bricks_modeling.file_IO.model_converter import color_phraser
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from solvers.rigidity_solver.algo_core import spring_energy_matrix
from solvers.rigidity_solver.internal_structure import structure_sampling
import util.geometry_util as geo_util
import numpy as np
from numpy import linalg as LA

def get_mesh_for_points(points):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=2)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.9, 0.1, 0.1])
    spheres = o3d.geometry.TriangleMesh()

    for b in points:
        spheres += copy.deepcopy(sphere).translate((b/2.5).tolist())

    return spheres

def get_mesh_for_edges(points, edges):
    colors = [[1, 0, 0] for i in range(len(edges))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points/2.5),
        lines=o3d.utility.Vector2iVector(edges),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def create_arrow_mesh(points, vectors):
    arrows = o3d.geometry.TriangleMesh()
    for idx, p in enumerate(points):
        vec = vectors[idx]
        rot_mat = geo_util.rot_matrix_from_vec_a_to_b([0, 0, 1], vec)
        vec_len = LA.norm(vec)
        if vec_len > 0:
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.2,
                cone_radius=0.35,
                cylinder_height=10 * vec_len,
                cone_height=8 * vec_len,
                resolution=3,
            )
            arrows += copy.deepcopy(arrow).translate(p/2.5).rotate(rot_mat, center=p/2.5)
    return arrows


def get_movement_direction(ldr_path, n:int):

    bricks = read_bricks_from_file(ldr_path)

    connect_graph = ConnectivityGraph(bricks)

    points, edges, points_on_brick = structure_sampling(connect_graph)

    M = spring_energy_matrix(points, edges)

    #TODO: Subetract meaningful eigenvecotrs

    '''e_pairs = geo_util.eigen(M, symmetric=True)

    # collect all eigen vectors with zero eigen value
    zeroeigenspace = [e_vec for e_val, e_vec in e_pairs if abs(e_val) < 1e-6]

    print("Number of points", len(points))

    # Trivial basis -- orthonormalized translation along / rotation wrt 3 axes
    basis = geo_util.trivial_basis(points)

    # cast the eigenvectors corresponding to zero eigenvalues into nullspace of the trivial basis,
    # in other words, the new vectors doesn't have any components (projection) in the span of the trivial basis
    reduced_zeroeigenspace = [geo_util.subtract_orthobasis(vec, basis) for vec in zeroeigenspace]

    # count zero vectors in reduced eigenvectors
    num_zerovectors = sum([np.isclose(vec, np.zeros_like(vec)).all() for vec in reduced_zeroeigenspace])
    # In 3d cases, exactly 6 eigenvectors for eigenvalue 0 are reduced to zerovector.
    #assert num_zerovectors == 6
    print(num_zerovectors)'''

    C = geo_util.eigen(M, symmetric=True)

    e = C[n]
    e_val, e_vec = e


    #e_vec = reduced_zeroeigenspace[n]
    e_vec = e_vec / LA.norm(e_vec)
    arrows = o3d.geometry.TriangleMesh()
    delta_x = e_vec.reshape(-1, 3)
    for i in range(len(connect_graph.bricks)):
        indices_on_brick_i = np.array(points_on_brick[i])
        point = points[indices_on_brick_i]
        arrows += create_arrow_mesh(point,delta_x[indices_on_brick_i])

    return arrows


def sampling_method_meshs(ldr_path, show_origin_model=True):
    color_dict = color_phraser()
    bricks = read_bricks_from_file(ldr_path)
    connect_graph = ConnectivityGraph(bricks)
    points, edges, points_on_brick = structure_sampling(connect_graph)

    meshs = o3d.geometry.TriangleMesh()
    if show_origin_model:
        for brick in bricks:
            meshs += brick.get_mesh(color_dict)
    line_set = get_mesh_for_edges(points, edges)
    meshs += get_mesh_for_points(points)
    return meshs, line_set



if __name__ == "__main__":
    path = "../data/full_models/test_case_5.ldr"
    meshs, line_set = sampling_method_meshs(path, show_origin_model=False)
    arrows = o3d.geometry.TriangleMesh()
    #for i in range(0,5):
    arrows += get_movement_direction(path,3)

    meshs += arrows
    o3d.visualization.draw_geometries([meshs, line_set])