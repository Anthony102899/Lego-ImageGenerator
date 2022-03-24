import open3d as o3d


def visualize_brick(brick_id):
    path = "../../bricks_modeling/database/obj/"
    mesh = o3d.io.read_triangle_mesh(path + brick_id + ".obj")
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
