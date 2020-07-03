from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.connections.connpointtype import ConnPointType, typeToBrick
from util.geometry_util import rot_matrix_from_vec_a_to_b
import numpy as np
import itertools as iter
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA

connect_type = [
    {ConnPointType.HOLE, ConnPointType.PIN},
    {ConnPointType.CROSS_HOLE, ConnPointType.AXLE},
    {ConnPointType.STUD, ConnPointType.TUBE},
]

rot = np.array(
    [
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    ]
)


def get_rotation(cpoint_align, n, cpoint_base):
    cal_rotation = np.c_[
        np.r_[
            rot_matrix_from_vec_a_to_b(cpoint_base.orient, cpoint_align.orient),
            np.array([[0, 0, 0]]),
        ],
        (np.array([[0, 0, 0, 1]])).T,
    ]
    rotation = cal_rotation
    self_rot = rot[n % 3]
    for i in range(int(n / 3)):
        rotation = rotation @ self_rot
    # print(f"\ncase {n}\n--------------------")
    return rotation

# get eight (2 x 4) matrics
def get_orient_matrices(cpoint_base, cpoint_align):
    transformations = []
    orient_matrices = get_two_orient_align_matrices(cpoint_base, cpoint_align)
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

def get_two_orient_align_matrices(cpoint_align, cpoint_base):
    matrices = []
    for direction in {-1, 1}:
        rotation = rot_matrix_from_vec_a_to_b(
            cpoint_align.orient, cpoint_base.orient * direction
        )
        rotation = (
            rot_matrix_from_vec_a_to_b(
                rotation @ cpoint_align.bi_orient, rotation @ cpoint_base.bi_orient
            )
            @ rotation
        )
        matrices.append(rotation)

    return matrices


""" returns a new brick instance """
def get_new_tile(align: BrickInstance, trans_mat, color: int):
    return BrickInstance(align.template, trans_mat, color)


# def get_matrix(cpoint_base, cpoint_align, base_brick: BrickInstance, i):
#     translation = np.identity(4)
#     rotation = get_rotation(cpoint_align, i, cpoint_base)
#     rot_3d = rotation[:3, :3]  # 3*3 matrix
#     new_align_pos = rotation @ np.append(cpoint_align.pos, 1)
#     translation[:, 3] = np.append(cpoint_base.pos, 1) - new_align_pos
#     new_orient = rot_3d @ cpoint_align.orient
#     same = np.linalg.norm(new_orient - cpoint_base.orient) < 1e-9
#     return (
#         translation @ rotation,
#         (np.linalg.norm(np.cross(new_orient, cpoint_base.orient)) < 1e-9),
#         same == 1,
#     )

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
                result_tiles.append(new_tile)

            # TODO: check collision with base_brick here

    return result_tiles


""" Returns True if the brick is already in list """


def check_repeatability(elem: BrickInstance, result_tiles: list):
    elem_pos = list(elem.trans_matrix[:, 3][:3])  # elem position (1 * 3)
    result_tiles_idx = list(
        filter(
            lambda brick: (elem_pos == brick.trans_matrix[:, 3][:3]).all(), result_tiles
        )
    )  # filtered list of input bricks (same position)
    if len(result_tiles_idx) == 0:
        return False
    cpoints_ls = list(
        map(lambda brick: brick.get_current_conn_points(), result_tiles_idx)
    )  # a list of cpoints (#filtered tiles * #cpoints each)
    cpoints_info = list(
        map(
            lambda cp_ls: list(map(lambda cp: [list(cp.pos), cp.type], cp_ls)),
            cpoints_ls,
        )
    )  # a list of (pos, type) (#filtered tiles * #cpoints each)
    elem_cpoints_info = list(
        map(lambda cp: [list(cp.pos), cp.type], elem.get_current_conn_points())
    )  # (#cpoints in elem * 2)
    elem_cpoints_info.sort()
    for cp_info in cpoints_info:
        cp_info.sort()
        if np.array(
            (np.array(cp_info) == np.array(elem_cpoints_info))
        ).all():  # all cpoints in one input equal elem's, duplicate brick
            return True
    return False


""" Returns a list of "num_rings" neighbours of brick "base_brick" """

def form_complete_graph(num_rings: int, base_tile: BrickInstance, tile_set: list):
    result_tiles = [base_tile]  # the resulting tiles
    last_ring = [base_tile]  # the tiles in the last ring
    for i in range(0, num_rings):
        print(f"\ncomputing ring {i}")
        last_ring_num = len(last_ring)

        # iterate over all bricks in the last ring
        for last_ring_idx in range(last_ring_num):
            print(f"last ring {last_ring_idx}")
            last_brick = last_ring.pop(0)  # brick instance in previous ring

            # brick instance to be aligned
            for align_tile in tile_set:
                # a list of neighbour bricks of "base_brick"
                neighbour_tiles = generate_all_neighbor_tiles(
                    base_brick=last_brick, align_tile=align_tile, color=i + 1
                )

                for elem in neighbour_tiles:
                    if not check_repeatability(elem, result_tiles):
                        result_tiles.append(elem)
                        last_ring.append(elem)

    return result_tiles
