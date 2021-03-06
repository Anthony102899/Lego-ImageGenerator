import numpy as np
import scipy
from util.geometry_util import normalize, get_perpendicular_vecs
from util import geometry_util as geo_util
from scipy.linalg import null_space
from itertools import combinations


def is_colinear(points):
    count = len(points)
    for p1_i, p2_i, p3_i in combinations(range(count), 3):
        p1, p2, p3 = points[p1_i], points[p2_i], points[p3_i]
        u = p2 - p1
        v = p3 - p1
        if np.abs(np.linalg.norm(np.cross(u, v))) > 1e-8:
            return False

    return True

def select_non_colinear_points(points, num, near=None):
    assert num in (2, 3)

    index_point_pairs = [(i, p) for i, p in enumerate(points)]
    if near is not None:
        index_point_pairs.sort(key=lambda p: np.linalg.norm(p[1] - near))

    if num == 3:
        for indices_points in combinations(index_point_pairs, num):
            indices, pts = map(np.array, zip(*indices_points))
            # if np.linalg.matrix_rank(pts) >= 3:
            #     return pts, indices
            u = pts[1] - pts[0]
            v = pts[2] - pts[0]
            if not np.isclose(np.linalg.norm(np.cross(u, v)), 0):
                return pts, indices

        raise Exception("Everything is on the same line")
    else:
        for indices_points in combinations(index_point_pairs, num):
            indices, pts = map(np.array, zip(*indices_points))
            # if np.linalg.matrix_rank(pts) >= 3:
            #     return pts, indices
            if not np.allclose(pts[0], pts[1]):
                return pts, indices

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

    # `motion`: no orthogonality guarantee
    return motion


def direction_for_relative_disallowed_motions(
        source_points,
        target_points,
        rotation_axes=None,
        rotation_pivot=None,
        translation_vectors=None,
        returns_allowed_motion=False,
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

    # allowed motion 1: the source and target points move as a whole
    relative_rigid_motions = geo_util.trivial_basis(
        np.vstack((source_points, target_points)), dim=3
    )
    assert relative_rigid_motions.shape == (6, 18)

    # allowed motion 2: based on the given translation vectors
    relative_translation = np.hstack((
        (np.zeros((translation_vectors.shape[0], 9)),
         translation_vectors, translation_vectors, translation_vectors)
    ))
    assert relative_translation.shape == (translation_vectors.shape[0], 18) or np.allclose(translation_vectors, 0)

    # allowed motion 3: possible instantaneous displacements for rotation per the given rotation axes
    relative_targets = target_points - rotation_pivot
    relative_rotation = np.hstack((
        np.zeros((rotation_axes.shape[0], 9)),
        np.vstack([
            np.hstack([np.cross(ax, rel_tg) for rel_tg in relative_targets])
            for ax in rotation_axes
        ])
    ))
    assert relative_rotation.shape == (rotation_axes.shape[0], 18) or np.allclose(rotation_axes, 0)

    # aggregate all allowed motions
    allowed_motion = np.vstack((
        relative_rigid_motions,                                                              # 6 rows, motion #1

        relative_translation,                                                                # len(t) rows
        relative_rotation,                                                                   # len(rot) rows

        np.hstack((null_space(geo_util.trivial_basis(source_points)).T, np.zeros((3, 9)))),  # 3 rows, deform in sources
        np.hstack((np.zeros((3, 9)), null_space(geo_util.trivial_basis(target_points)).T)),  # 3 rows, deform in targets
    ))  #

    if returns_allowed_motion:
        # dimension: m x 18, m is the number of disallowed motion
        return allowed_motion

    else:
        constraints = null_space(allowed_motion).T  # take null space, get orthogonal `prohibitive direction'
        # dimension: m x 18, m is the number of disallowed motion
        return constraints
