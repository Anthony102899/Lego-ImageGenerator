import numpy as np
import scipy
from util.geometry_util import normalize, get_perpendicular_vecs
from scipy.linalg import null_space
from itertools import combinations


def projection_matrix(source_points, target_point):
    """
    compute project matrix that maps deltas of source and target points in world coordinates
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


def prohibitive_space_of_allowed_relative_rotation(
        target_point: np.ndarray,
        pivot_point: np.ndarray,
        rotation_axis: np.ndarray,
):
    """
    Compute the null space of a rotation wrt to axis
    :param target_point: (3, )
    :param pivot_point: (3, )
    :param rotation_axis: (3, )
    :return: (m, 3)
    """
    allowed_direction = np.cross(target_point - pivot_point, rotation_axis)
    if np.abs(np.linalg.norm(allowed_direction)) > 1e-8:
        allowed_direction = normalize(allowed_direction)
        null_basis = np.vstack(get_perpendicular_vecs(allowed_direction))
        return null_basis
    else:
        sqrt_half = np.sqrt(0.5)
        null_basis = np.array([
            [1, 0, 0],
            [0, sqrt_half, sqrt_half],
            [0, sqrt_half, -sqrt_half],
        ])
        return null_basis



def constraints_for_allowed_motions(
        source_points,
        target_points,
        rotation_axis=None,
        rotation_pivot=None,
):
    constraints = []
    for target_point in target_points:
        relative_projection, source_transform = projection_matrix(
            source_points,
            target_point,
        )

        if rotation_axis is not None and rotation_pivot is not None:
            assert len(rotation_axis) == len(rotation_pivot)

            relative_target_point = source_transform @ target_point
            relative_rotation_pivot = source_transform @ rotation_pivot
            relative_rotation_axis = source_transform @ rotation_axis

            allowed_motion = np.cross(relative_target_point - relative_rotation_pivot, relative_rotation_axis)
            allowed_motion /= np.linalg.norm(allowed_motion)

            prohibitive_space = prohibitive_space_of_allowed_relative_rotation(
                relative_target_point,
                relative_rotation_pivot,
                relative_rotation_axis,
            )

            assert np.linalg.matrix_rank(source_transform.T @ prohibitive_space.T) >= 2

            prohibitive_space_on_deltas = prohibitive_space @ relative_projection

            constraints.append(prohibitive_space_on_deltas)

    return constraints
