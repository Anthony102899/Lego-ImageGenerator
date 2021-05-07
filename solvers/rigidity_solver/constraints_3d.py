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
