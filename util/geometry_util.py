import numpy as np
import math
from numpy import linalg as LA
from typing import List


def vec_local2world(rot_mat: np.ndarray, local_vec: np.ndarray) -> np.ndarray:
    return np.dot(rot_mat, local_vec)


def point_local2world(
    rot_mat: np.ndarray, translation: np.ndarray, local_point: np.ndarray
) -> np.ndarray:
    return np.dot(rot_mat, local_point) + translation


# return a axis-aligned unit vector that perpendicular to the input_images normal
def gen_lateral_vec(vec: np.array):
    norm = vec / LA.norm(vec)
    result_vec = np.array([0, 0, 0])
    for i in range(3):  # every coordinate in 3 dimension
        result_vec[i] = 1
        if (
            abs(LA.norm(np.cross(result_vec, norm)) - 1) < 1e-6
        ):  # two vectors perpendicular
            return result_vec
        result_vec[i] = 0
    input("error normal input_images!", norm)


def rot_matrix_from_vec_a_to_b(a, b):
    cross = np.cross(a, b)
    if np.linalg.norm(cross) == 0:  # parallel
        if (a == b).all():
            return np.identity(3, dtype=float)
        return -np.identity(3, dtype=float)
    else:
        dot = np.dot(a, b)
        angle = math.acos(dot)
        rotation_axes = cross / np.linalg.norm(cross)
        M = np.array(
            [
                [0, -rotation_axes[2], rotation_axes[1]],
                [rotation_axes[2], 0, -rotation_axes[0]],
                [-rotation_axes[1], rotation_axes[0], 0],
            ]
        )

        return (
            np.identity(3, dtype=float)
            + math.sin(angle) * M
            + (1 - math.cos(angle)) * np.dot(M, M)
        )


def get_perpendicular_vec(vec: np.array) -> np.array:
    assert LA.norm(vec) > 0
    perp_vec = None
    if abs(vec[0]) > 1e-10:
        perp_vec = np.array([(-vec[1] - vec[2]) / vec[0], 1, 1])
    elif abs(vec[1]) > 1e-10:
        perp_vec = np.array([1, (-vec[0] - vec[2]) / vec[1], 1])
    else:
        perp_vec = np.array([1, 1, (-vec[0] - vec[1]) / vec[2]])

    return perp_vec / LA.norm(perp_vec)


def get_perpendicular_vecs(vec: np.array) -> np.array:
    vec1 = get_perpendicular_vec(vec)
    vec2 = np.cross(vec1, vec / LA.norm(vec))
    return vec1, vec2


def points_span_dim(points: np.ndarray) -> bool:
    """
    points: shape(n, 3)
    If the points spans a
        i) 0D space (the points are identical), return 0
        i) 1D space (on the same line), return 1
        ii) 2D space (on the same plane), return 2
        iii) higher-than-2D space, return 3
    """
    assert len(np.shape(points)) == 2 and np.shape(points)[1] == 3
    rank = np.linalg.matrix_rank(points)

    if rank == 1:
        column_comp = np.all(
            points == points[0, :], axis=0
        )  # compare the entries columnwise
        if np.all(column_comp):  # all rows are identical to the first row
            return 0
        else:
            return 1

    return min(rank, 3)

def project(v: np.ndarray, base: np.ndarray):
    """
    Project vector v on base. Return the projection
    """
    return np.dot(v, base) / np.linalg.norm(base)

def eigen(matrix: np.ndarray, symmetric: bool) -> List:
    """
    Compute eigenvalues/vectors, return a list of eigenvalue/vectors, sorted by the eigenvalue ascendingly
        symmetric: a boolean that indicates the input_images matrix is symmetric

    Note: if the matrix is symmetric (Hermitian), the eigenvectors shall be real
    """
    if symmetric:
        assert np.allclose(matrix, matrix.T)

    eig_func = np.linalg.eigh if symmetric else np.linalg.eig
    w, V = eig_func(matrix)
    V_normalized = map(lambda v: v / np.linalg.norm(v), V.T)

    eigen_pairs = sorted(list(zip(w, V_normalized)), key=lambda pair: pair[0])

    return eigen_pairs
