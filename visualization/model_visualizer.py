import copy

import open3d as o3d

from bricks_modeling.connectivity_graph import ConnectivityGraph
from bricks_modeling.file_IO.model_converter import color_phraser
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from solvers.rigidity_solver.internal_structure import structure_sampling

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

if __name__ == "__main__":
    color_dict = color_phraser()
    ldr_path = "../data/full_models/test_case_5.ldr"
    bricks = read_bricks_from_file(ldr_path)
    connect_graph = ConnectivityGraph(bricks)
    points, edges, points_on_brick = structure_sampling(connect_graph)

    meshs = o3d.geometry.TriangleMesh()
    '''for brick in bricks:
        meshs += brick.get_mesh(color_dict)'''
    line_set = get_mesh_for_edges(points,edges)
    meshs+=get_mesh_for_points(points)
    o3d.visualization.draw_geometries([meshs, line_set])