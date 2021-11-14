import numpy as np
import open3d as o3d
import copy

from visualization.model_visualizer import visualize_3D, visualize_hinges, get_mesh_for_arrows
import visualization.vismesh as vismesh
import util.geometry_util as geo_util


def open3d_triangular_mesh(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_triangle_normals()
    return mesh

def arrow_meshes(points, vectors, merged=False):
    arrows = []
    for idx, p in enumerate(points):
        vec = vectors[idx]
        rot_mat = geo_util.rot_matrix_from_vec_a_to_b([0, 0, 1], vec)
        vec_len = np.linalg.norm(vec)
        if vec_len > 0:
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.2,
                cone_radius=0.45,
                cylinder_height=400 * vec_len,
                cone_height=100 * vec_len,
                resolution=5,
            )
            norm_vec = vec / np.linalg.norm(vec)
            arrows.append(
                copy.deepcopy(arrow).translate(p).rotate(rot_mat, center=p) \
                    .paint_uniform_color([(norm_vec[0] + 1) / 2, (norm_vec[1] + 1) / 2, (norm_vec[2] + 1) / 2])
            )
    if merged:
        from functools import reduce
        return reduce(lambda x, y: x + y, arrows)
    else:
        return arrows


def _main():
    pass

if __name__ == '__main__':
    _main()