from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.connections.connpointtype import ConnPointType, typeToBrick
from util.geometry_util import rot_matrix_from_vec_a_to_b

import numpy as np

connect_type = [
    {ConnPointType.HOLE, ConnPointType.PIN},
    {ConnPointType.CROSS_HOLE, ConnPointType.AXLE},
    {ConnPointType.STUD, ConnPointType.TUBE}
    ]
    
rot = np.array([
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
    [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
])

def get_rotation(cpoint_align, n, cpoint_base):
    align_orient = cpoint_align.orient
    base_orient = cpoint_base.orient
    base_direction = typeToBrick[cpoint_base.type][1]  # original base cpoint orient recorded
    orient_rot = np.identity(4)
    """
    TODO: test below
    align_direction = typeToBrick[cpoint_align.type][1]  # original cpoint orient recorded
    cross = np.cross(align_direction, align_orient)
    if np.linalg.norm(cross) == 0:
        orient_rot = np.identity(4)
        orient_rot_idx = -1
    else:
        orient_rot_idx = (np.where((align_direction + align_orient) == 0))[0][0]
        orient_rot = rot[orient_rot_idx]
    """
    cross = np.cross(base_orient, align_orient)
    if np.linalg.norm(cross) == 0:
        cal_rotation = np.identity(4)
    else:
        dif_index = (np.where(base_orient + align_orient == 0))[0][0]
        cal_rotation = rot[dif_index]
    self_rot = rot[(np.nonzero(base_direction))[0][0]]
    rotation = orient_rot @ cal_rotation
    if n == 0:
        #print("case 0\n")
        return rotation
    if n == 1:
        #print("case 1\n")
        return rotation @ self_rot
    #print("case 2\n")
    return rotation @ ((self_rot @ self_rot))

def get_matrix(cpoint_base, cpoint_align, base_brick: BrickInstance, i):
    translation = np.identity(4)
    rotation = get_rotation(cpoint_align, i, cpoint_base)
    #print("4. rotation = \n",rotation)
    rot_3d = rotation[0:3]
    rot_3d = rot_3d[:,0:3]
    new_orient = rot_3d @ (np.transpose(cpoint_align.orient))
    new_orient = np.transpose(new_orient)
    cross = np.cross(new_orient, cpoint_base.orient)
    align_pos_T = np.transpose(np.append(cpoint_align.pos, 1))
    new_align_pos = np.transpose(rotation @ align_pos_T)
    trans_vec = np.append(cpoint_base.pos, 1) - new_align_pos
    translation[:,3] = trans_vec
    return translation @ rotation, np.linalg.norm(cross) < 1e-9

""" returns a new brick instance """
def get_new_tile(align: BrickInstance, trans_mat, color: int):
    new = BrickInstance(align.template, trans_mat, color)
    return new

""" Returns immediate possible aligns using "align_tile" for "base_brick" """
def get_all_tiles(base_brick: BrickInstance, align_tile: BrickInstance, color: int):
    result_tiles = []
    align_tags = []

    base_cpoints = base_brick.get_current_conn_points()  # a list of cpoints in base
    base_cpoint_num = len(base_cpoints)
    align_cpoints = align_tile.get_current_conn_points()  # a list of cpoints in align
    align_cpoint_num = len(align_cpoints)

    ls = [(x,y) for x in range(base_cpoint_num) for y in range(align_cpoint_num)]
    for base_cpoint_idx, align_cpoint_idx in ls:
        cpoint_base = base_cpoints[base_cpoint_idx]  # one cpoint of base
        cpoint_align = align_cpoints[align_cpoint_idx]  # one cpoint of align
        
        if {cpoint_base.type, cpoint_align.type} not in connect_type:  # cannot connect
            continue
        align_tag = (base_cpoint_idx, align_cpoint_idx, cpoint_base.type, cpoint_align.type)
        
        for i in range(3):
            trans_mat, connect = get_matrix(cpoint_base, cpoint_align, base_brick, i)
            new_tile = get_new_tile(align_tile, trans_mat, color)  # brick instance, new tile based on "align_tile"
            if connect:
                result_tiles.append(new_tile)
                align_tags.append(align_tag)
    return result_tiles

""" Returns True if the brick is already in list """
def check_repeatability(elem: BrickInstance, result_tiles: list):
    tile_pos = list(map(lambda brick: list(brick.trans_matrix[:,3][:3]), result_tiles))  # a list of positions (#tiles * 3)
    elem_pos = list(elem.trans_matrix[:,3][:3])  # elem position (1 * 3)
    if list(elem_pos) not in list(tile_pos):
        return False
    result_tiles_idx = list(filter(lambda brick: (elem_pos == brick.trans_matrix[:,3][:3]).all(), result_tiles))  # filtered list of input bricks
    cpoints_ls = list(map(lambda brick: brick.get_current_conn_points(), result_tiles_idx))  # a list of cpoints (#filtered tiles * #cpoints each)
    cpoints_info = list(map(lambda cp_ls: list(map(lambda cp: [list(cp.pos), cp.type], cp_ls)), cpoints_ls)) # a list of (pos, type) (#filtered tiles * #cpoints each)
    elem_cpoints = elem.get_current_conn_points()
    elem_cpoints_info = list(map(lambda cp: [list(cp.pos), cp.type], elem_cpoints))  # (#cpoints in elem * 2)
    elem_cpoints_info.sort()
            
    for cp_info in cpoints_info:
        cp_info.sort()
        if np.array((np.array(cp_info) == np.array(elem_cpoints_info))).all():  # all cpoints in one input equal elem's
            #print("duplicate brick!")
            return True
            
    return False

""" Returns a list of "num_rings" neighbours of brick "base_brick" """
def form_complete_graph(num_rings: int, base_tile: BrickInstance, tile_set: list):
    result_tiles = [base_tile]  # the resulting tiles
    last_ring = [base_tile]  # the tiles in the last ring
    for i in range(0, num_rings):
        print(f"\ncomputing ring {i}")
        last_ring_num = len(last_ring)
        for last_ring_idx in range(last_ring_num):  # iterate over all bricks in the last ring
            print(f"last ring {last_ring_idx}")
            last_layer_brick = last_ring.pop(0)  # brick instance in previous ring
            for align_tile in tile_set:  # brick instance to be aligned
                neighbour_tiles = get_all_tiles(base_brick=last_layer_brick, align_tile=align_tile, color=i+1)  # a list of neighbour bricks of "base_brick"
                for elem in neighbour_tiles:
                    if not check_repeatability(elem, result_tiles):
                        result_tiles.append(elem)
                        last_ring.append(elem)
    return result_tiles
