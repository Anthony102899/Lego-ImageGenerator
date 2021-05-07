import numpy as np
import scipy
from util.geometry_util import normalize, get_perpendicular_vecs
from util import geometry_util as geo_util
from scipy.linalg import null_space
from itertools import combinations


def projection_matrix(source_points, target_point):
    """
    compute project matrix that maps deltas of source and target points in world coordinates (variables in optim setup)
    to the delta of target point in source point coordinate
    :param source_points: (3, 3) array, 3 points not on the same line
    :param target_point: (3, ) array
    :return: tuple[(3, 12) array, (3, 3) current source coordinates basis]
    """
    assert len(source_points) == 3
    dim = 3

    projection_matrix = np.zeros((dim, 4 * dim))

    x0, x1, x2 = source_points
    x0_x, x0_y, x0_z = x0
    x1_x, x1_y, x1_z = x1
    x2_x, x2_y, x2_z = x2
    t_x, t_y, t_z = target_point

    norm_1 = np.linalg.norm(x1 - x0)
    basis_1 = (x1 - x0) / norm_1

    norm_2 = np.linalg.norm(np.cross(x2 - x0, basis_1))
    basis_2 = np.cross(x2 - x0, basis_1) / norm_2

    basis_3 = np.cross(basis_1, basis_2)

    if np.isclose(norm_1, 0):
        raise Exception("Norm(x1 - x0) nears 0", "x0", x0, "x1", x1)
    if np.isclose(norm_2, 0):
        raise Exception("Norm(x2 - x0) nears 0", "x0", x0, "x1", x1, "x2", x2)

    # Sympy generated code
    # >>>>>>>>>>>>>>>>>>>>
    projection_matrix[0, 0] = (-t_x + 2*x0_x - x1_x)/norm_1
    projection_matrix[0, 1] = (-t_y + 2*x0_y - x1_y)/norm_1
    projection_matrix[0, 2] = (-t_z + 2*x0_z - x1_z)/norm_1
    projection_matrix[0, 3] = (t_x - x0_x)/norm_1
    projection_matrix[0, 4] = (t_y - x0_y)/norm_1
    projection_matrix[0, 5] = (t_z - x0_z)/norm_1
    projection_matrix[0, 6] = 0
    projection_matrix[0, 7] = 0
    projection_matrix[0, 8] = 0
    projection_matrix[0, 9] = (-x0_x + x1_x)/norm_1
    projection_matrix[0, 10] = (-x0_y + x1_y)/norm_1
    projection_matrix[0, 11] = (-x0_z + x1_z)/norm_1
    projection_matrix[1, 0] = (t_y*x1_z - t_y*x2_z - t_z*x1_y + t_z*x2_y + x1_y*x2_z - x1_z*x2_y)/(norm_1*norm_2)
    projection_matrix[1, 1] = (-t_x*x1_z + t_x*x2_z + t_z*x1_x - t_z*x2_x - x1_x*x2_z + x1_z*x2_x)/(norm_1*norm_2)
    projection_matrix[1, 2] = (t_x*x1_y - t_x*x2_y - t_y*x1_x + t_y*x2_x + x1_x*x2_y - x1_y*x2_x)/(norm_1*norm_2)
    projection_matrix[1, 3] = (-t_y*x0_z + t_y*x2_z + t_z*x0_y - t_z*x2_y - x0_y*x2_z + x0_z*x2_y)/(norm_1*norm_2)
    projection_matrix[1, 4] = (t_x*x0_z - t_x*x2_z - t_z*x0_x + t_z*x2_x + x0_x*x2_z - x0_z*x2_x)/(norm_1*norm_2)
    projection_matrix[1, 5] = (-t_x*x0_y + t_x*x2_y + t_y*x0_x - t_y*x2_x - x0_x*x2_y + x0_y*x2_x)/(norm_1*norm_2)
    projection_matrix[1, 6] = (t_y*x0_z - t_y*x1_z - t_z*x0_y + t_z*x1_y + x0_y*x1_z - x0_z*x1_y)/(norm_1*norm_2)
    projection_matrix[1, 7] = (-t_x*x0_z + t_x*x1_z + t_z*x0_x - t_z*x1_x - x0_x*x1_z + x0_z*x1_x)/(norm_1*norm_2)
    projection_matrix[1, 8] = (t_x*x0_y - t_x*x1_y - t_y*x0_x + t_y*x1_x + x0_x*x1_y - x0_y*x1_x)/(norm_1*norm_2)
    projection_matrix[1, 9] = (-x0_y*x1_z + x0_y*x2_z + x0_z*x1_y - x0_z*x2_y - x1_y*x2_z + x1_z*x2_y)/(norm_1*norm_2)
    projection_matrix[1, 10] = (x0_x*x1_z - x0_x*x2_z - x0_z*x1_x + x0_z*x2_x + x1_x*x2_z - x1_z*x2_x)/(norm_1*norm_2)
    projection_matrix[1, 11] = (-x0_x*x1_y + x0_x*x2_y + x0_y*x1_x - x0_y*x2_x - x1_x*x2_y + x1_y*x2_x)/(norm_1*norm_2)
    projection_matrix[2, 0] = (t_x*x0_y*x1_y - t_x*x0_y*x2_y + t_x*x0_z*x1_z - t_x*x0_z*x2_z - t_x*x1_y**2 + t_x*x1_y*x2_y - t_x*x1_z**2 + t_x*x1_z*x2_z - 2*t_y*x0_x*x1_y + 2*t_y*x0_x*x2_y + t_y*x0_y*x1_x - t_y*x0_y*x2_x + t_y*x1_x*x1_y - 2*t_y*x1_x*x2_y + t_y*x1_y*x2_x - 2*t_z*x0_x*x1_z + 2*t_z*x0_x*x2_z + t_z*x0_z*x1_x - t_z*x0_z*x2_x + t_z*x1_x*x1_z - 2*t_z*x1_x*x2_z + t_z*x1_z*x2_x + 2*x0_x*x1_y**2 - 2*x0_x*x1_y*x2_y + 2*x0_x*x1_z**2 - 2*x0_x*x1_z*x2_z - 2*x0_y*x1_x*x1_y + x0_y*x1_x*x2_y + x0_y*x1_y*x2_x - 2*x0_z*x1_x*x1_z + x0_z*x1_x*x2_z + x0_z*x1_z*x2_x + x1_x*x1_y*x2_y + x1_x*x1_z*x2_z - x1_y**2*x2_x - x1_z**2*x2_x)/(norm_1**2*norm_2)
    projection_matrix[2, 1] = (t_x*x0_x*x1_y - t_x*x0_x*x2_y - 2*t_x*x0_y*x1_x + 2*t_x*x0_y*x2_x + t_x*x1_x*x1_y + t_x*x1_x*x2_y - 2*t_x*x1_y*x2_x + t_y*x0_x*x1_x - t_y*x0_x*x2_x + t_y*x0_z*x1_z - t_y*x0_z*x2_z - t_y*x1_x**2 + t_y*x1_x*x2_x - t_y*x1_z**2 + t_y*x1_z*x2_z - 2*t_z*x0_y*x1_z + 2*t_z*x0_y*x2_z + t_z*x0_z*x1_y - t_z*x0_z*x2_y + t_z*x1_y*x1_z - 2*t_z*x1_y*x2_z + t_z*x1_z*x2_y - 2*x0_x*x1_x*x1_y + x0_x*x1_x*x2_y + x0_x*x1_y*x2_x + 2*x0_y*x1_x**2 - 2*x0_y*x1_x*x2_x + 2*x0_y*x1_z**2 - 2*x0_y*x1_z*x2_z - 2*x0_z*x1_y*x1_z + x0_z*x1_y*x2_z + x0_z*x1_z*x2_y - x1_x**2*x2_y + x1_x*x1_y*x2_x + x1_y*x1_z*x2_z - x1_z**2*x2_y)/(norm_1**2*norm_2)
    projection_matrix[2, 2] = (t_x*x0_x*x1_z - t_x*x0_x*x2_z - 2*t_x*x0_z*x1_x + 2*t_x*x0_z*x2_x + t_x*x1_x*x1_z + t_x*x1_x*x2_z - 2*t_x*x1_z*x2_x + t_y*x0_y*x1_z - t_y*x0_y*x2_z - 2*t_y*x0_z*x1_y + 2*t_y*x0_z*x2_y + t_y*x1_y*x1_z + t_y*x1_y*x2_z - 2*t_y*x1_z*x2_y + t_z*x0_x*x1_x - t_z*x0_x*x2_x + t_z*x0_y*x1_y - t_z*x0_y*x2_y - t_z*x1_x**2 + t_z*x1_x*x2_x - t_z*x1_y**2 + t_z*x1_y*x2_y - 2*x0_x*x1_x*x1_z + x0_x*x1_x*x2_z + x0_x*x1_z*x2_x - 2*x0_y*x1_y*x1_z + x0_y*x1_y*x2_z + x0_y*x1_z*x2_y + 2*x0_z*x1_x**2 - 2*x0_z*x1_x*x2_x + 2*x0_z*x1_y**2 - 2*x0_z*x1_y*x2_y - x1_x**2*x2_z + x1_x*x1_z*x2_x - x1_y**2*x2_z + x1_y*x1_z*x2_y)/(norm_1**2*norm_2)
    projection_matrix[2, 3] = (-t_x*x0_y**2 + t_x*x0_y*x1_y + t_x*x0_y*x2_y - t_x*x0_z**2 + t_x*x0_z*x1_z + t_x*x0_z*x2_z - t_x*x1_y*x2_y - t_x*x1_z*x2_z + t_y*x0_x*x0_y + t_y*x0_x*x1_y - 2*t_y*x0_x*x2_y - 2*t_y*x0_y*x1_x + t_y*x0_y*x2_x + 2*t_y*x1_x*x2_y - t_y*x1_y*x2_x + t_z*x0_x*x0_z + t_z*x0_x*x1_z - 2*t_z*x0_x*x2_z - 2*t_z*x0_z*x1_x + t_z*x0_z*x2_x + 2*t_z*x1_x*x2_z - t_z*x1_z*x2_x - 2*x0_x*x0_y*x1_y + x0_x*x0_y*x2_y - 2*x0_x*x0_z*x1_z + x0_x*x0_z*x2_z + x0_x*x1_y*x2_y + x0_x*x1_z*x2_z + 2*x0_y**2*x1_x - x0_y**2*x2_x - 2*x0_y*x1_x*x2_y + x0_y*x1_y*x2_x + 2*x0_z**2*x1_x - x0_z**2*x2_x - 2*x0_z*x1_x*x2_z + x0_z*x1_z*x2_x)/(norm_1**2*norm_2)
    projection_matrix[2, 4] = (t_x*x0_x*x0_y - 2*t_x*x0_x*x1_y + t_x*x0_x*x2_y + t_x*x0_y*x1_x - 2*t_x*x0_y*x2_x - t_x*x1_x*x2_y + 2*t_x*x1_y*x2_x - t_y*x0_x**2 + t_y*x0_x*x1_x + t_y*x0_x*x2_x - t_y*x0_z**2 + t_y*x0_z*x1_z + t_y*x0_z*x2_z - t_y*x1_x*x2_x - t_y*x1_z*x2_z + t_z*x0_y*x0_z + t_z*x0_y*x1_z - 2*t_z*x0_y*x2_z - 2*t_z*x0_z*x1_y + t_z*x0_z*x2_y + 2*t_z*x1_y*x2_z - t_z*x1_z*x2_y + 2*x0_x**2*x1_y - x0_x**2*x2_y - 2*x0_x*x0_y*x1_x + x0_x*x0_y*x2_x + x0_x*x1_x*x2_y - 2*x0_x*x1_y*x2_x - 2*x0_y*x0_z*x1_z + x0_y*x0_z*x2_z + x0_y*x1_x*x2_x + x0_y*x1_z*x2_z + 2*x0_z**2*x1_y - x0_z**2*x2_y - 2*x0_z*x1_y*x2_z + x0_z*x1_z*x2_y)/(norm_1**2*norm_2)
    projection_matrix[2, 5] = (t_x*x0_x*x0_z - 2*t_x*x0_x*x1_z + t_x*x0_x*x2_z + t_x*x0_z*x1_x - 2*t_x*x0_z*x2_x - t_x*x1_x*x2_z + 2*t_x*x1_z*x2_x + t_y*x0_y*x0_z - 2*t_y*x0_y*x1_z + t_y*x0_y*x2_z + t_y*x0_z*x1_y - 2*t_y*x0_z*x2_y - t_y*x1_y*x2_z + 2*t_y*x1_z*x2_y - t_z*x0_x**2 + t_z*x0_x*x1_x + t_z*x0_x*x2_x - t_z*x0_y**2 + t_z*x0_y*x1_y + t_z*x0_y*x2_y - t_z*x1_x*x2_x - t_z*x1_y*x2_y + 2*x0_x**2*x1_z - x0_x**2*x2_z - 2*x0_x*x0_z*x1_x + x0_x*x0_z*x2_x + x0_x*x1_x*x2_z - 2*x0_x*x1_z*x2_x + 2*x0_y**2*x1_z - x0_y**2*x2_z - 2*x0_y*x0_z*x1_y + x0_y*x0_z*x2_y + x0_y*x1_y*x2_z - 2*x0_y*x1_z*x2_y + x0_z*x1_x*x2_x + x0_z*x1_y*x2_y)/(norm_1**2*norm_2)
    projection_matrix[2, 6] = (t_x*x0_y**2 - 2*t_x*x0_y*x1_y + t_x*x0_z**2 - 2*t_x*x0_z*x1_z + t_x*x1_y**2 + t_x*x1_z**2 - t_y*x0_x*x0_y + t_y*x0_x*x1_y + t_y*x0_y*x1_x - t_y*x1_x*x1_y - t_z*x0_x*x0_z + t_z*x0_x*x1_z + t_z*x0_z*x1_x - t_z*x1_x*x1_z + x0_x*x0_y*x1_y + x0_x*x0_z*x1_z - x0_x*x1_y**2 - x0_x*x1_z**2 - x0_y**2*x1_x + x0_y*x1_x*x1_y - x0_z**2*x1_x + x0_z*x1_x*x1_z)/(norm_1**2*norm_2)
    projection_matrix[2, 7] = (-t_x*x0_x*x0_y + t_x*x0_x*x1_y + t_x*x0_y*x1_x - t_x*x1_x*x1_y + t_y*x0_x**2 - 2*t_y*x0_x*x1_x + t_y*x0_z**2 - 2*t_y*x0_z*x1_z + t_y*x1_x**2 + t_y*x1_z**2 - t_z*x0_y*x0_z + t_z*x0_y*x1_z + t_z*x0_z*x1_y - t_z*x1_y*x1_z - x0_x**2*x1_y + x0_x*x0_y*x1_x + x0_x*x1_x*x1_y + x0_y*x0_z*x1_z - x0_y*x1_x**2 - x0_y*x1_z**2 - x0_z**2*x1_y + x0_z*x1_y*x1_z)/(norm_1**2*norm_2)
    projection_matrix[2, 8] = (-t_x*x0_x*x0_z + t_x*x0_x*x1_z + t_x*x0_z*x1_x - t_x*x1_x*x1_z - t_y*x0_y*x0_z + t_y*x0_y*x1_z + t_y*x0_z*x1_y - t_y*x1_y*x1_z + t_z*x0_x**2 - 2*t_z*x0_x*x1_x + t_z*x0_y**2 - 2*t_z*x0_y*x1_y + t_z*x1_x**2 + t_z*x1_y**2 - x0_x**2*x1_z + x0_x*x0_z*x1_x + x0_x*x1_x*x1_z - x0_y**2*x1_z + x0_y*x0_z*x1_y + x0_y*x1_y*x1_z - x0_z*x1_x**2 - x0_z*x1_y**2)/(norm_1**2*norm_2)
    projection_matrix[2, 9] = (x0_x*x0_y*x1_y - x0_x*x0_y*x2_y + x0_x*x0_z*x1_z - x0_x*x0_z*x2_z - x0_x*x1_y**2 + x0_x*x1_y*x2_y - x0_x*x1_z**2 + x0_x*x1_z*x2_z - x0_y**2*x1_x + x0_y**2*x2_x + x0_y*x1_x*x1_y + x0_y*x1_x*x2_y - 2*x0_y*x1_y*x2_x - x0_z**2*x1_x + x0_z**2*x2_x + x0_z*x1_x*x1_z + x0_z*x1_x*x2_z - 2*x0_z*x1_z*x2_x - x1_x*x1_y*x2_y - x1_x*x1_z*x2_z + x1_y**2*x2_x + x1_z**2*x2_x)/(norm_1**2*norm_2)
    projection_matrix[2, 10] = (-x0_x**2*x1_y + x0_x**2*x2_y + x0_x*x0_y*x1_x - x0_x*x0_y*x2_x + x0_x*x1_x*x1_y - 2*x0_x*x1_x*x2_y + x0_x*x1_y*x2_x + x0_y*x0_z*x1_z - x0_y*x0_z*x2_z - x0_y*x1_x**2 + x0_y*x1_x*x2_x - x0_y*x1_z**2 + x0_y*x1_z*x2_z - x0_z**2*x1_y + x0_z**2*x2_y + x0_z*x1_y*x1_z + x0_z*x1_y*x2_z - 2*x0_z*x1_z*x2_y + x1_x**2*x2_y - x1_x*x1_y*x2_x - x1_y*x1_z*x2_z + x1_z**2*x2_y)/(norm_1**2*norm_2)
    projection_matrix[2, 11] = (-x0_x**2*x1_z + x0_x**2*x2_z + x0_x*x0_z*x1_x - x0_x*x0_z*x2_x + x0_x*x1_x*x1_z - 2*x0_x*x1_x*x2_z + x0_x*x1_z*x2_x - x0_y**2*x1_z + x0_y**2*x2_z + x0_y*x0_z*x1_y - x0_y*x0_z*x2_y + x0_y*x1_y*x1_z - 2*x0_y*x1_y*x2_z + x0_y*x1_z*x2_y - x0_z*x1_x**2 + x0_z*x1_x*x2_x - x0_z*x1_y**2 + x0_z*x1_y*x2_y + x1_x**2*x2_z - x1_x*x1_z*x2_x + x1_y**2*x2_z - x1_y*x1_z*x2_y)/(norm_1**2*norm_2)
    # <<<<<<<<<<<<<<<<<<<<

    return projection_matrix, np.vstack((basis_1, basis_2, basis_3))

def is_colinear(points):
    count = len(points)
    for p1_i, p2_i, p3_i in combinations(range(count), 3):
        p1, p2, p3 = points[p1_i], points[p2_i], points[p3_i]
        u = p2 - p1
        v = p3 - p1
        if np.abs(np.linalg.norm(np.cross(u, v))) > 1e-8:
            return False

    return True

def select_non_colinear_points(points, near=None):
    num = 3
    index_point_pairs = [(i, p) for i, p in enumerate(points)]
    if near is not None:
        index_point_pairs.sort(key=lambda p: np.linalg.norm(p[1] - near))

    for indices_points in combinations(index_point_pairs, num):
        indices, pts = map(np.array, zip(*indices_points))
        # if np.linalg.matrix_rank(pts) >= 3:
        #     return pts, indices
        u = pts[1] - pts[0]
        v = pts[2] - pts[0]
        if not np.isclose(np.linalg.norm(np.cross(u, v)), 0):
            return pts, indices

    raise Exception("Everything is on the same line")


def rigid_motion_for_coplanar_points(points):
    assert points.shape == (3, 3)
    u = points[1] - points[0]
    v = points[2] - points[0]
    normal = np.cross(u, v)
    motion = np.block([
        [u, u, u],  # translation along u
        [v, v, v],  # translation along v
        [np.cross(normal, u), np.cross(normal, v), np.zeros((3,))]  # rotation about point[0]
    ])
    assert motion.shape == (3, 9)
    return motion


def constraints_for_allowed_motions(
        source_points,
        target_points,
        rotation_axes=None,
        rotation_pivot=None,
        translation_vectors=None,
):
    assert source_points.shape == (3, 3)
    assert target_points.shape == (3, 3)
    assert rotation_pivot is None or rotation_pivot.shape == (3, )
    if rotation_axes is None:
        rotation_axes = np.zeros((1, 3))
    if translation_vectors is None:
        translation_vectors = np.zeros((1, 3))

    assert rotation_axes.ndim == 2
    assert translation_vectors.ndim == 2

    relative_rigid_motions = geo_util.trivial_basis(
        np.vstack((source_points, target_points)), dim=3
    )
    assert relative_rigid_motions.shape == (6, 18)

    relative_translation = np.hstack((
        (np.zeros((translation_vectors.shape[0], 9)),
         translation_vectors, translation_vectors, translation_vectors)
    ))
    assert relative_translation.shape == (translation_vectors.shape[0], 18)

    relative_targets = target_points - rotation_pivot
    relative_rotation = np.hstack((
        np.zeros((rotation_axes.shape[0], 9)),
        np.vstack([
            np.hstack([np.cross(ax, rel_tg) for rel_tg in relative_targets])
            for ax in rotation_axes
        ])
    ))
    assert relative_translation.shape == (rotation_axes.shape[0], 18)

    allowed_motion = np.vstack((
        relative_rigid_motions,                                                              # 6 rows
        np.hstack((null_space(geo_util.trivial_basis(source_points)).T, np.zeros((3, 9)))),  # 3 rows
        np.hstack((np.zeros((3, 9)), null_space(geo_util.trivial_basis(target_points)).T)),  # 3 rows

        relative_translation,  # len(t) rows
        relative_rotation,  # len(rot) rows
    ))  #

    constraints = null_space(allowed_motion).T

    return constraints
