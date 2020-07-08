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

def rot_matrix_from_two_basis(a1, a2, b1, b2):
    assert abs(a1 @ a2) < 1e-6 and abs(b1 @ b2) < 1e-6
    assert abs(LA.norm(a1) - 1) < 1e-4 and abs(LA.norm(a2) - 1) < 1e-4 and abs(LA.norm(b1) - 1) < 1e-4 and abs(LA.norm(b2) - 1) < 1e-4
    a3 = np.cross(a1, a2)
    b3 = np.cross(b1, b2)
    X_before = np.empty([3,3])
    X_before[:,0] = a1
    X_before[:, 1] = a2
    X_before[:, 2] = a3

    X_after = np.empty([3, 3])
    X_after[:,0] = b1
    X_after[:, 1] = b2
    X_after[:, 2] = b3

    return X_after @ np.linalg.inv(X_before)

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


def get_perpendicular_vecs(vec: np.ndarray) -> np.ndarray:
    vec1 = get_perpendicular_vec(vec) 
    vec2 = np.cross(vec1, vec / LA.norm(vec))
    return vec1, vec2

def project(v: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    Project vector v on base. Return the projection
    """
    length = np.dot(v, base) / np.dot(base, base)
    return length * base

def rowwise_normalize(mat: np.ndarray) -> np.ndarray:
    """
    Row wise normalize a matrix, using L2 norm
    """
    assert len(mat.shape) == 2
    return mat / LA.norm(mat, axis=1)[:, np.newaxis] 


def orthonormalize(basis: np.ndarray) -> np.ndarray:
    """
    Take a set of linearly independent vectors, return an orthogonal basis via Modified Gram-Schmidt Process
    """
    U = np.zeros_like(basis)
    for k, v in enumerate(basis):
        u = np.copy(v)
        for i in range(k):
            u -= project(u, U[i])

        U[k] = u[:]


    U_norm = U / LA.norm(U, axis=1)[:, np.newaxis] 
    return U_norm

def trivial_basis(points: np.ndarray) -> np.ndarray:
    """
    Given n points in 3d space in form of a (n x 3) matrix, construct 6 'trivial' orthonormal vectors
    """
    P = points.reshape((-1, 3))
    n = len(P)

    # translation along x, y, and z
    translations = np.array([
       [1, 0, 0] * n, 
       [0, 1, 0] * n, 
       [0, 0, 1] * n,
    ])

    center = np.mean(P, axis=0)
    P_shifted = P - center # make the rotation vectors orthogonal
    x_axis, y_axis, z_axis = np.identity(3)
    rotations = np.array([
        np.cross(P_shifted, x_axis).reshape(-1),
        np.cross(P_shifted, y_axis).reshape(-1),
        np.cross(P_shifted, z_axis).reshape(-1),
    ])

    transformation = np.vstack((translations, rotations))
    # row-wise normalize the vectors into orthonormal basis
    basis = transformation / LA.norm(transformation, axis=1)[:, np.newaxis] 
    orthonormal_basis = orthonormalize(basis)
    return basis

def subtract_orthobasis(vector: np.ndarray, orthobasis: np.ndarray) -> np.ndarray:
    """
    Given a vector and a orthonormal matrix, project the vector into the null space of the matrix
    """

    projections = np.apply_along_axis(lambda base: project(vector, base), axis=1, arr=orthobasis)
    subtraction = vector - np.sum(projections, axis=0)
    return subtraction

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


def eigen(matrix: np.ndarray, symmetric: bool) -> List:
    """
    Compute eigenvalues/vectors, return a list of <eigenvalue, vector> pairs, sorted by the eigenvalue ascendingly
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
