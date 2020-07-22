from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.connections.connpointtype import ConnPointType, typeToBrick, isDoubleOriented
from util.geometry_util import rot_matrix_from_vec_a_to_b, rot_matrix_from_two_basis
import numpy as np
import itertools as iter
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA
from typing import List
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
import time
import os
import pickle5 as pickle

connect_type = [
    {ConnPointType.HOLE, ConnPointType.PIN},
    {ConnPointType.CROSS_HOLE, ConnPointType.AXLE},
    {ConnPointType.STUD, ConnPointType.TUBE},
]

class Tiling:
    def __init__(self, result_tiles, last_ring, last_ring_num, middle_debugger, tile_set, num_rings, start_ring, last_idx, total_time):
        self.result_tiles = result_tiles
        self.last_ring = last_ring  # the tiles in the last ring
        self.last_ring_num = last_ring_num
        self.middle_debugger = middle_debugger
        self.tile_set = tile_set
        self.num_rings = num_rings
        self.start_ring_idx = start_ring
        self.last_idx = last_idx
        self.total_time = total_time

# get eight or four matrices
def get_orient_matrices(cpoint_base, cpoint_align):
    transformations = []
    orient_matrices = get_orient_align_matrices(cpoint_base, cpoint_align)
    rotations = get_four_self_rotation(cpoint_base.orient)
    for orien_align_mat in orient_matrices:
        for orient_rotate_mat in rotations:
            transformation = np.identity(4)
            transform_mat = orient_rotate_mat @ orien_align_mat
            new_align_pos = transform_mat @ cpoint_align.pos
            transformation[:3, 3] = cpoint_base.pos - new_align_pos
            transformation[:3, :3] = transform_mat
            transformations.append(transformation)

    return transformations

def get_four_self_rotation(orient):
    assert abs(LA.norm(orient) - 1) < 1e-6
    rotations = []

    for i in {0, 1, 2, 3}:
        r = R.from_rotvec(np.pi / 2 * i * orient)
        rotations.append( r.as_matrix() ) # if you met problem in this line, please upgrade "scipy"

    return rotations

def get_orient_align_matrices(cpoint_base, cpoint_align):
    matrices = []
    is_double_side = isDoubleOriented[cpoint_align.type] or isDoubleOriented[cpoint_base.type]
    for direction in ({-1, 1} if is_double_side else {1}):
        rotation = rot_matrix_from_two_basis(cpoint_align.orient, cpoint_align.bi_orient, cpoint_base.orient * direction, cpoint_base.bi_orient)
        matrices.append(rotation)

    return matrices

""" Returns immediate possible aligns using "align_tile" for "base_brick" """
def generate_all_neighbor_tiles(
    base_brick: BrickInstance, align_tile: BrickInstance, color: int
):
    result_tiles = []
    base_cpoints = base_brick.get_current_conn_points()  # a list of cpoints in base
    align_cpoints = align_tile.get_current_conn_points()  # a list of cpoints in align

    for cpoint_base, cpoint_align in iter.product(base_cpoints, align_cpoints):
        if {cpoint_base.type, cpoint_align.type} in connect_type:  # can connect
            # get all possible rotation matrices
            matrices = get_orient_matrices(cpoint_base, cpoint_align)
            for trans_mat in matrices:  # 2 possible orientations consistent with the normal
                new_tile = BrickInstance(align_tile.template, trans_mat, color)
                # TODO: detect if new tile collide with the base tile (for concave shape)
                #if base_brick.collide(new_tile):
                result_tiles.append(new_tile)

    return result_tiles

def unique_brick_list(bricks: List[BrickInstance]):
    # remove self-repeat
    unique_list = []
    for x in bricks:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

""" Returns a list of "num_rings" neighbours of brick "base_brick" """
def find_brick_placements(num_rings: int, base_tile: BrickInstance, tile_set: list, initial_time):
    debugger = MyDebugger("middle")
    result_tiles = [base_tile]  # the resulting tiles
    last_ring = [base_tile]  # the tiles in the last ring
    snd_last_ring = []
    for i in range(0, num_rings):
        print(f"\ncomputing ring {i}")
        last_ring_num = len(last_ring)
        print("last ring num = ", last_ring_num)
        tmp = last_ring[:]
        this_ring = []
        # iterate over all bricks in the last ring
        for last_ring_idx in range(last_ring_num):
            print(f"last ring {last_ring_idx}")
            last_brick = last_ring.pop(0)  # brick instance in previous ring
            snd_last_ring.append(last_brick)

            # brick instance to be aligned
            for align_tile in tile_set:
                # a list of neighbour bricks of "base_brick"
                neighbour_tiles = generate_all_neighbor_tiles(
                    base_brick=last_brick, align_tile=align_tile, color=i + 1)

                neighbour_tiles = unique_brick_list(neighbour_tiles)

                for elem in neighbour_tiles:
                    if elem not in snd_last_ring and elem not in this_ring:
                        result_tiles.append(elem)
                        last_ring.append(elem)
                        this_ring.append(elem)
            if last_ring_idx % 30 == 0:
                tiling = Tiling(result_tiles, last_ring, last_ring_num,
                                debugger, tile_set, 
                                num_rings, i, last_ring_idx, 
                                time.time() - initial_time)
                pickle.dump(tiling, 
                            open(os.path.join(os.path.dirname(__file__), f'super_graph/r={i}#{num_rings}.pkl'), "wb"))
        write_bricks_to_file(result_tiles, 
            file_path=debugger.file_path(f"n={len(result_tiles)} r={i+1} t={round(time.time() - initial_time, 2)}.ldr"))
        snd_last_ring = tmp
    return result_tiles
    