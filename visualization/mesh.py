import numpy as np
import open3d as o3d


def open3d_triangular_mesh(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_triangle_normals()
    return mesh


def _main():
    pass

if __name__ == '__main__':
    _main()