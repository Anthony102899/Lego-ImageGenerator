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
from numpy.linalg import matrix_rank, matrix_power
import util.geometry_util as geo_util
from solvers.rigidity_solver.algo_core import spring_energy_matrix
from solvers.rigidity_solver.joint_construction import construct_joints
from scipy.spatial import transform


def get_crystal_vertices(contact_pt: np.array, contact_orient: np.array) -> np.ndarray:
    p0 = contact_pt
    p1 = contact_pt + 5 * contact_orient
    p2 = contact_pt - 5 * contact_orient
    p_vec1, p_vec2 = geo_util.get_perpendicular_vecs(p1 - p2)
    p3 = contact_pt + 5 * p_vec1
    p4 = contact_pt - 5 * p_vec1
    p5 = contact_pt + 5 * p_vec2
    p6 = contact_pt - 5 * p_vec2

    return np.array([p0, p1, p2, p3, p4, p5, p6])


def find_rotation_matrix(a, b):
    """
    find 3-by-3 matrix that rotates unit vector a to unit vector b
     i.e. find R such that R a = b
    """
    assert np.isclose(np.linalg.norm(a), 1)
    assert np.isclose(np.linalg.norm(b), 1)

    v = np.cross(a, b)
    if np.allclose(v, 0):
        return np.identity(3)
    s = np.linalg.norm(v)
    c = np.dot(a, b)

    v_skew = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])

    rot = np.identity(3) + v_skew + (1 - c) / (s * s) * matrix_power(v_skew, 2)

    return rot


def tetrahedral(p, q, thickness=1, ori=None):
    if ori is None:
        vertices = np.array([
            [0, 1, 0],
            [np.sqrt(3) / 2, -1 / 2, 0],
            [-np.sqrt(3) / 2, -1 / 2, 0],
        ])

        z = np.array([0, 0, 1])
        n = geo_util.normalize(p - q)
        rot = find_rotation_matrix(z, n)
        base = (rot @ vertices.T).T * thickness
    else:
        ori = geo_util.normalize(ori)
        offset = geo_util.normalize(np.cross(p - q, ori)) * np.sqrt(3) / 2
        base = np.vstack((
            ori,
            -ori * 0.5 + offset,
            -ori * 0.5 - offset
        )) * thickness

    start_center = (p + 0.01 * (q - p))
    end_center = (q + 0.01 * (p - q))
    start = base + start_center
    end = base + end_center

    num = 5
    triangles = np.linspace(start, end, num)
    centers = np.linspace(start_center, end_center, num) + (end_center - start_center) / (num - 1) / 2

    points = np.vstack(
         (np.vstack([np.vstack([tri, c]) for tri, c in zip(triangles[:-1], centers[:-1])]),
          triangles[-1])
    )
    edges = np.array(
        # triangle on the same layer
        [(4 * s + 0, 4 * s + 1) for s in range(num)] +
        [(4 * s + 1, 4 * s + 2) for s in range(num)] +
        [(4 * s + 2, 4 * s + 0) for s in range(num)] +

        # triangle vertices to center
        [(4 * s + 0, 4 * s + 3) for s in range(num - 1)] +
        [(4 * s + 1, 4 * s + 3) for s in range(num - 1)] +
        [(4 * s + 2, 4 * s + 3) for s in range(num - 1)] +

        # next triangle vertices to center
        [(4 * s + 3, 4 * s + 4) for s in range(num - 1)] +
        [(4 * s + 3, 4 * s + 5) for s in range(num - 1)] +
        [(4 * s + 3, 4 * s + 6) for s in range(num - 1)] +

        # vertical bars between neighbor triangles
        [(4 * s + 0, 4 * s + 4) for s in range(num - 1)] +
        [(4 * s + 1, 4 * s + 5) for s in range(num - 1)] +
        [(4 * s + 2, 4 * s + 6) for s in range(num - 1)] +

        # oblique bars between neighbor triangles
        [(4 * s + 0, 4 * s + 6) for s in range(num - 1)] +
        [(4 * s + 1, 4 * s + 4) for s in range(num - 1)] +
        [(4 * s + 2, 4 * s + 5) for s in range(num - 1)] +
        [(4 * s + 0, 4 * s + 5) for s in range(num - 1)] +
        [(4 * s + 1, 4 * s + 6) for s in range(num - 1)] +
        [(4 * s + 2, 4 * s + 4) for s in range(num - 1)]
    )

    edges = np.array(list(itertools.combinations(range(len(points)), 2)))

    return points, edges


# Requirements on the sample points and their connection:
# 1) respect symmetric property of the brick
# 2) self-rigid connection inside each brick
# 3) respect the joint property
def structure_sampling(structure_graph: ConnectivityGraph):
    bricks = structure_graph.bricks
    points = []
    edges = []
    abstract_edges = []
    # brick_id -> a list of point indices belongs to this brick
    points_on_brick = {i: [] for i in range(len(bricks))}

    # the representative connPoint on each brick
    representative_cpoints_on_brick = {i: None for i in range(len(bricks))}
    for idx, b in enumerate(bricks):
        representative_cpoints_on_brick[idx] = b.template.deg1_cpoint_indices()

    for edge in structure_graph.connect_edges:
        # get connecting bricks
        bi = edge["node_indices"][0]
        bj = edge["node_indices"][1]

        # get contact position and orientation
        contact_pt = edge["properties"]["contact_point"]
        contact_orient = edge["properties"]["contact_orient"]

        # generate points around the contact points
        p = get_crystal_vertices(contact_pt, contact_orient)

        construct_joints(abstract_edges, bi, bj, contact_orient, edge, edges, p, points, points_on_brick)

        # remove the generated points
        representative_cpoints_on_brick[bi].discard(edge["cpoint_indices"][0])
        representative_cpoints_on_brick[bj].discard(edge["cpoint_indices"][1])

    #### add additional sample points, by detecting if the connection points are already sampled
    for brick_id, c_id_set in representative_cpoints_on_brick.items():
        brick = bricks[brick_id]
        for c_id in c_id_set:
            c_point = brick.get_current_conn_points()[c_id]
            contact_pt = c_point.pos
            contact_orient = c_point.orient
            p = get_crystal_vertices(contact_pt, contact_orient)
            point_idx_base = len(points)
            for i in range(7):
                exec(f"points.append(p[{i}])")
                points_on_brick[brick_id].append(point_idx_base + i)

    for value in points_on_brick.values():
        edges.extend(list(itertools.combinations(value, 2)))

    return np.array(points), edges, points_on_brick, abstract_edges
