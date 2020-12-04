import numpy as np
from util.geometry_util import normalize, get_perpendicular_vecs
from solvers.rigidity_solver.algo_core import project_matrix_3D
from scipy.linalg import null_space
from itertools import combinations


def select_points_on_plane(points):
    count = len(points)
    for p1_i, p2_i, p3_i in combinations(range(count), 3):
        p1, p2, p3 = points[p1_i], points[p2_i], points[p3_i]
        u = p2 - p1
        v = p3 - p1
        if np.abs(np.linalg.norm(np.cross(u, v))) > 1e-6:
            return np.vstack((p1, p2, p3)), np.array([p1_i, p2_i, p3_i])

    raise Exception("Everything is on the same line")


def null_space_of_allowed_motions(
        translations: np.ndarray,
        rotations: np.ndarray,
        target_points: np.ndarray,
):
    dim = 3
    target_point_count = len(target_points)
    allowed_translations = np.zeros((len(translations) * target_point_count, target_point_count * dim))
    allowed_rotations = np.zeros((len(rotations) * target_point_count, target_point_count * dim))

    for index, allowed_motion in enumerate(allowed_translations):
        for j in range(target_point_count):
            row_num = index * target_point_count + j
            allowed_translations[row_num, j * dim: j * dim + dim] = allowed_motion

    for index, allowed_motion in enumerate(allowed_rotations):
        for j in range(0, target_point_count * 2, 2):
            row_num = index * target_point_count + j
            r1, r2 = null_space(np.array([allowed_motion, allowed_motion])).T
            allowed_rotations[row_num, j * dim: j * dim + dim] = r1
            allowed_rotations[row_num + 1, j * dim: j * dim + dim] = r2

    allowed_motions = np.vstack([allowed_translations, allowed_rotations])
    return null_space(allowed_motions).T


def constraints_for_allowed_motions(
        model,
        source_points,
        source_point_indices,
        target_points,
        target_point_indices,
        translations=None,
        rotations=None
) -> np.ndarray:
    dim = 3
    forbidden_motion_bases = null_space_of_allowed_motions(
        translations if translations is not None else [],
        rotations if rotations is not None else [],
        target_points,
    )
    projection, basis1, basis2, basis3 = project_matrix_3D(
        source_points,
        source_point_indices,
        target_points,
        target_point_indices,
        points=[0] * model.point_count
    )

    bases = (basis1, basis2, basis3)

    target_point_count = len(target_points)
    constraints = np.zeros((len(forbidden_motion_bases), target_point_count * 3))

    # motion basis : np motion allowed in the direction of basis
    for index, motion_basis in enumerate(forbidden_motion_bases):
        for j in range(target_point_count):
            constraints[index, j * dim] = np.dot(motion_basis, bases[0])
            constraints[index, j * dim + 1] = np.dot(motion_basis, bases[1])
            constraints[index, j * dim + 2] = np.dot(motion_basis, bases[2])

    return constraints @ projection
