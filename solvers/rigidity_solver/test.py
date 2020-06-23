from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from bricks_modeling.connections.conn_type import ConnType
import numpy as np
import util.geometry_util as geo_util
import open3d as o3d
import copy
from typing import List
import itertools
from numpy import linalg as LA
from numpy.linalg import matrix_rank

def rigidity_matrix(points: np.ndarray, edges: np.ndarray, dim: int):
    """
    points: (n, d) array, n points in a d-dimensional space
    edges : (m, 2) array, m edges, store indices of the points they join
    dim   : int, dimension order
    """
    assert len(points.shape) == 2 and points.shape[1] == dim
    n, m = len(points), len(edges)

    # constructing the rigidity matrix R
    R = np.zeros((m, dim * n))
    for i, (p_ind, q_ind) in enumerate(edges):
        p_minus_q = points[p_ind, :] - points[q_ind, :]
        R[i, q_ind * dim: (q_ind + 1) * dim] =  p_minus_q
        R[i, p_ind * dim: (p_ind + 1) * dim] = -p_minus_q
    
    return R

def show_graph(points: List[np.array], edges: List[List]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=2)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.9, 0.1, 0.1])

    points = [p for p in points]

    spheres = [copy.deepcopy(sphere).translate(p) for p in points]
    lines = [e for e in edges]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=20, origin=[0, 0, 0]
    )
    o3d.visualization.draw_geometries([mesh_frame, line_set] + spheres)



if __name__ == "__main__":
    points = np.array([
        [0, 0],
        [0, 1],
        [1, 0]
    ]) 
    edges = np.array([
        [0, 1],
        [0, 2],
        [1, 2]
    ])

    R = rigidity_matrix(points, edges, 2)

    print(matrix_rank(R.T @ R))

    points_3d = np.hstack(
        (points, np.zeros((len(points), 1)))
    )
    R_3d = rigidity_matrix(points_3d, edges, 3)

    print(matrix_rank(R_3d.T @ R_3d))

    points_degenerated = np.array([
        [0, 0],
        [1, 1],
        [2, 2]
    ])
    R_degen = rigidity_matrix(points_degenerated, edges, 2)
    print(matrix_rank(R_degen.T @ R_degen))