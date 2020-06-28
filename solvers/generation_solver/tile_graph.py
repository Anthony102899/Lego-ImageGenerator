from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.connections.conn_type import compute_conn_type
from bricks_modeling.connections.connpointtype import ConnPointType, typeToBrick
from util.geometry_util import rot_matrix_from_vec_a_to_b

import numpy as np

connect_type = [
    {ConnPointType.HOLE, ConnPointType.PIN},
    {ConnPointType.CROSS_HOLE, ConnPointType.AXLE},
    {ConnPointType.HOLE, ConnPointType.AXLE},
    {ConnPointType.STUD, ConnPointType.TUBE}
    ]

def get_matrix(cpoint_base, cpoint_align, base_brick: BrickInstance):
    print("pos of base & algin cp: ", cpoint_base.pos, cpoint_align.pos)
    translation = np.identity(4)
    rotation = np.identity(4)

    scale_align = np.identity(3)
    for i in range(3):
        if typeToBrick[cpoint_align.type][4][i] != 0:
            scale_align[i][i] *= typeToBrick[cpoint_align.type][3]
    rot_align = rot_matrix_from_vec_a_to_b(typeToBrick[cpoint_align.type][1], cpoint_align.orient)
    """
    matrix_align = rot_align
    matrix_align = matrix_align @ scale_align
    offset_align = rot_align @ np.array(typeToBrick[cpoint_align.type][2])
    #print("matrix = ", matrix_align)
    """

    # TODO: get rotation
    cross = np.cross(cpoint_base.orient,cpoint_align.orient)
    if not np.linalg.norm(cross) < 1e-9:
        print("\ncross not zero, needs modification!\n")
    trans_vec = cpoint_base.pos - cpoint_align.pos
    translation[:,3] = np.append(trans_vec, 1) # + np.append(base_brick.trans_matrix[0:3][:,3], 0)
    print("translation of align:\n", translation)
    return translation @ rotation

""" returns a new brick instance """
def get_new_tile(align: BrickInstance, trans_mat):
    new = BrickInstance(align.template, trans_mat, 3)
    return new

""" Returns immediate possible aligns using "align_tile" for "base_brick" """
def get_all_tiles(base_brick: BrickInstance, align_tile: BrickInstance):
    result_tiles = []
    align_tags = []

    base_cpoints = base_brick.get_current_conn_points()  # a list of cpoints in base
    base_cpoint_num = len(base_cpoints)
    align_cpoints = align_tile.get_current_conn_points()  # a list of cpoints in align
    align_cpoint_num = len(align_cpoints)

    ls = [(x,y) for x in range(base_cpoint_num) for y in range(align_cpoint_num)]
    for base_cpoint_idx,align_cpoint_idx in ls:
        cpoint_base = base_cpoints[base_cpoint_idx]  # cpoint of base
        cpoint_align = align_cpoints[align_cpoint_idx]  # cpoint of align
        align_tag = (base_cpoint_idx, align_cpoint_idx, cpoint_base.type, cpoint_align.type)
        
        if {cpoint_base.type, cpoint_align.type} not in connect_type:  # cannot connect
            continue
    
        # TODO: add parameters if any
        trans_mat = get_matrix(cpoint_base, cpoint_align, base_brick)
        new_tile = get_new_tile(align_tile, trans_mat)  # brick instance, new tile based on "align_tile"

        new_cpoints = new_tile.get_current_conn_points()
        cpoint_new_tile = new_cpoints[align_cpoint_idx]  # cpoint of transformed align brick
        type_conn = compute_conn_type(cpoint_base, cpoint_new_tile)

        if type_conn is not None:
            result_tiles.append(new_tile)
            align_tags.append(align_tag)
    return result_tiles

""" Return a list of "num_rings" neighbours of brick "base_brick" """
def form_complete_graph(num_rings: int, base_tile: BrickInstance, tile_set: list):
    result_tiles = [base_tile]  # the resulting tiles
    last_ring = [base_tile]  # the tiles in the last ring
    for i in range(0, num_rings):
        print(f"\ncomputing ring_{i}")
        last_ring_num = len(last_ring)
        for last_ring_idx in range(last_ring_num):  # iterate over all bricks in the last ring
            print(f"last ring_{last_ring_idx}")
            last_layer_brick = last_ring.pop(0)  # brick instance in previous ring
            for align_tile in tile_set:  # brick instance to be aligned
                neighbour_tiles = get_all_tiles(base_brick=last_layer_brick, align_tile=align_tile)  # a list of neighbour bricks of "base_brick"

                for elem in neighbour_tiles:
                    if elem not in result_tiles:
                        result_tiles.append(elem)
                        last_ring.append(elem)
    return result_tiles
