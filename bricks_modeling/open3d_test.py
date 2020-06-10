import open3d as o3d
import numpy as np

if __name__ == "__main__":
    print("Testing IO for meshes ...")
    mesh = o3d.io.read_triangle_mesh(
        "./database/ldraw2stl/test.obj", print_progress=True
    )
    # print(mesh)
    # mesh = o3d.geometry.TriangleMesh.create_box(width=1.0,
    #                                             height=1.0,
    #                                             depth=1.0)
    print(mesh)
    print("Vertices:")
    print(np.asarray(mesh.vertices))
    print("Triangles:")
    print(np.asarray(mesh.triangles))
    o3d.visualization.draw_geometries(
        [mesh], mesh_show_wireframe=True, mesh_show_back_face=True
    )
    # o3d.io.write_triangle_mesh("copy_of_knot.ply", mesh)
