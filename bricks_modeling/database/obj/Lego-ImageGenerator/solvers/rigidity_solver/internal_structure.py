from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from bricks_modeling.connections.conn_type import ConnType
import numpy as np
import torch
import util.geometry_util as geo_util
import open3d as o3d
import copy
from typing import List
import itertools
from numpy import linalg as LA
from numpy.linalg import matrix_rank
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


def find_rotation_matrix(a, b, mode="numpy"):
    """
    find 3-by-3 matrix that rotates unit vector a to unit vector b
     i.e. find R such that R a = b
    """
    assert mode in ("numpy", "torch")
    if mode == "numpy":
        norm = np.linalg.norm
        isclose = np.isclose
        cross = np.cross
        allclose = np.allclose
        identity = np.identity
        dot = np.dot
        array = lambda x: np.asarray(x, dtype=np.double)
        matrix_power = np.linalg.matrix_power
    else:
        norm = torch.norm
        isclose = lambda x, constant: np.isclose(x.detach().numpy(), constant)
        cross = torch.cross
        allclose = lambda x, constant: np.allclose(x.detach().numpy(), constant)
        identity = torch.eye
        dot = torch.dot
        array = lambda x: torch.tensor(x, dtype=torch.double)
        matrix_power = torch.matrix_power


    assert isclose(norm(a), 1), f"{a}"
    assert isclose(norm(b), 1), f"{b}"

    v = cross(a, b)
    if allclose(v, 0):
        return identity(3)
    s = norm(v)
    c = dot(a, b)

    v_skew = array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])

    rot = identity(3) + v_skew + (1 - c) / (s * s) * matrix_power(v_skew, 2)

    return rot


def tetrahedron(p, q, density, thickness=1, ori=None, num=None, mode="numpy"):
    if mode == "numpy":
        norm = np.linalg.norm
        normalize = geo_util.normalize
        sqrt = np.sqrt
        vstack = np.vstack
        arange = np.arange
        cross = np.cross
        linspace = np.linspace
        array = lambda x: np.array(x, dtype=np.double)
        longarray = lambda x: np.asarray(x, dtype=int)
    else:
        norm = torch.norm
        normalize = lambda v: v / torch.norm(v)
        sqrt = torch.sqrt
        vstack = torch.vstack
        arange = torch.arange
        cross = torch.cross
        array = lambda x: torch.tensor(x, dtype=torch.double)
        longarray = lambda x: torch.tensor(x, dtype=torch.long)
        def linspace(start, end, num):
            slices = [
                torch.lerp(start, end, ratio)
                for ratio in torch.linspace(0, 1, num, dtype=torch.double)
            ]
            return torch.stack(slices)

    if ori is None:
        vertices = array([
            [0, 1, 0],
            [np.sqrt(3) / 2, -1 / 2, 0],
            [-np.sqrt(3) / 2, -1 / 2, 0],
        ])

        z = array([0, 0, 1])
        n = normalize(p - q)
        rot = find_rotation_matrix(z, n, mode)
        base = (rot @ vertices.T).T * thickness

    else:
        ori = normalize(ori)
        offset = normalize(cross(p - q, ori)) * sqrt(3) / 2
        base = vstack((
            ori,
            -ori * 0.5 + offset,
            -ori * 0.5 - offset
        )) * thickness

    start_center = (p + 0.01 * (q - p))
    end_center = (q + 0.01 * (p - q))
    start = base + start_center
    end = base + end_center

    if num is None:
        num = int(norm(p - q) / thickness * density)
        num = 2 if num < 2 else num

    triangles = linspace(start, end, num)
    centers = linspace(start_center, end_center, num) + ((end_center - start_center) / (num - 1) / 2)

    points = vstack(
         (vstack([vstack([tri, c]) for tri, c in zip(triangles[:-1], centers[:-1])]),
          triangles[-1])
    )
    edges = longarray(
        # triangle on the same layer
        [(4 * s + 0, 4 * s + 1) for s in arange(num)] +
        [(4 * s + 1, 4 * s + 2) for s in arange(num)] +
        [(4 * s + 2, 4 * s + 0) for s in arange(num)] +

        # triangle vertices to center
        [(4 * s + 0, 4 * s + 3) for s in arange(num - 1)] +
        [(4 * s + 1, 4 * s + 3) for s in arange(num - 1)] +
        [(4 * s + 2, 4 * s + 3) for s in arange(num - 1)] +

        # next triangle vertices to center
        [(4 * s + 3, 4 * s + 4) for s in arange(num - 1)] +
        [(4 * s + 3, 4 * s + 5) for s in arange(num - 1)] +
        [(4 * s + 3, 4 * s + 6) for s in arange(num - 1)] +

        # vertical bars between neighbor triangles
        [(4 * s + 0, 4 * s + 4) for s in arange(num - 1)] +
        [(4 * s + 1, 4 * s + 5) for s in arange(num - 1)] +
        [(4 * s + 2, 4 * s + 6) for s in arange(num - 1)] +

        # oblique bars between neighbor triangles
        [(4 * s + 0, 4 * s + 6) for s in arange(num - 1)] +
        [(4 * s + 1, 4 * s + 4) for s in arange(num - 1)] +
        [(4 * s + 2, 4 * s + 5) for s in arange(num - 1)] +
        [(4 * s + 0, 4 * s + 5) for s in arange(num - 1)] +
        [(4 * s + 1, 4 * s + 6) for s in arange(num - 1)] +
        [(4 * s + 2, 4 * s + 4) for s in arange(num - 1)]
    )

    # edges = np.array(list(itertools.combinations(range(len(points)), 2)))

    return points, edges


_rotation_90 = torch.tensor([
    [0, -1],
    [1, 0],
], dtype=torch.double)


def triangulation_with_torch(p, q, num, thickness=1.0):
    ori = torch.mv(_rotation_90, p - q)
    if num < 2:
        num = 2
    points = torch.vstack([
        torch.vstack([torch.lerp(p, q, w) for w in torch.linspace(1 / (num + 1), 1 - 1 / (num + 1), num, dtype=torch.double)]) + ori / ori.norm() / 2 * thickness,
        torch.vstack([torch.lerp(p, q, w) for w in torch.linspace(1 / (num + 1), 1 - 1 / (num + 1), num, dtype=torch.double)]) - ori / ori.norm() / 2 * thickness,
        p,
        q,
    ])
    p_index, q_index = 2 * num, 2 * num + 1
    edges = torch.vstack([
        torch.tensor([(s, s + num) for s in range(num)]),
        torch.tensor([(s, s + 1) for s in range(num - 1)]),
        torch.tensor([(s + 1, s + num) for s in range(num - 1)]),
        torch.tensor([(s, s + num + 1) for s in range(num - 1)]),
        torch.tensor([(s + num, s + num + 1) for s in range(num - 1)]),
        torch.tensor([[0, p_index], [num, p_index]]),
        torch.tensor([[num - 1, q_index], [2 * num - 1, q_index]]),
    ]).long()
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

    # add additional sample points, by detecting if the connection points are already sampled
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
