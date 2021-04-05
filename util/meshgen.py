"""
Given a .stl surface triangle soup file, generate a tetrahedral mesh
"""
import wildmeshing as wm
import open3d as o3d
import numpy as np


def tetrahedralize(surface_filename, out_filename):
    wm_filename = out_filename + '.wm.mesh'
    wm.tetrahedralize(surface_filename, wm_filename, edge_length_r=1/30)
    postprocess(wm_filename, out_filename)


def postprocess(mesh_filename, out_filename):
    with open(mesh_filename) as src:
        lines = src.readlines()
    with open(out_filename, "w") as dst:
        found = False
        for i, line in enumerate(lines):
            if line.strip() == "Triangles":
                found = True
            elif found:  # skipping the '0' followed by "Triangles"
                found = False
            else:
                print(line, end="", file=dst)




def cube_surface_mesh(filename, pivot, u, v, w):
    """
    u, v, w must be linearly independent!
    """
    assert np.linalg.matrix_rank(np.array([u, v, w])) == 3
    base = np.array([
        pivot,
        pivot + u,
        pivot + u + v,
        pivot + v,
        pivot + w,
        pivot + w + u,
        pivot + u + v + w,
        pivot + w + v
    ])
    cube_indices = np.array([
        [4, 3, 0],
        [3, 4, 7],
        [6, 3, 7],
        [3, 6, 2],
        [4, 6, 7],
        [6, 4, 5],
        [6, 1, 5],
        [1, 6, 2],
        [1, 3, 2],
        [3, 1, 0],
        [1, 4, 0],
        [4, 1, 5],
    ])
    o3dmesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(base),
        o3d.utility.Vector3iVector(cube_indices),
    )
    o3dmesh.compute_triangle_normals()
    o3d.io.write_triangle_mesh(filename, o3dmesh, write_ascii=True, write_vertex_normals=False)


def _main():
    pivot = np.array([1, 0, 0])
    a, b, c = np.eye(3)
    cube_surface_mesh("whatever.stl", pivot, a, b, c)


if __name__ == "__main__":
    _main()
