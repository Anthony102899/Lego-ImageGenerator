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
import util.geometry_util as geo_util
from solvers.rigidity_solver.algo_core import spring_energy_matrix
from solvers.rigidity_solver.joint_construction import construct_joints

def get_crystal_vertices(contact_pt: np.array, contact_orient: np.array):
    p0 = contact_pt
    p1 = contact_pt + 5 * contact_orient
    p2 = contact_pt - 5 * contact_orient
    p_vec1, p_vec2 = geo_util.get_perpendicular_vecs(p1 - p2)
    p3 = contact_pt + 5 * p_vec1
    p4 = contact_pt - 5 * p_vec1
    p5 = contact_pt + 5 * p_vec2
    p6 = contact_pt - 5 * p_vec2

    return [p0, p1, p2, p3, p4, p5, p6]


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
