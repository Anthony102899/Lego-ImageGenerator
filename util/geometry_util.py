import numpy as np
import math

from typing import List

# TODO: to add function annotation
def vec_local2world(rot_mat: np.ndarray, local_vec: np.ndarray) -> np.ndarray:
    return np.dot(rot_mat, local_vec)


# TODO: to add function annotation
def point_local2world(
    rot_mat: np.ndarray, translation: np.ndarray, local_point: np.ndarray
) -> np.ndarray:
    return np.dot(rot_mat, local_point) + translation

def rot_matrix_from_vec_a_to_b(a, b):
    cross = np.cross(a, b)
    if np.linalg.norm(cross) == 0: # parallel
        return np.identity(3, dtype=float)
    else:
        dot = np.dot(a, b)
        angle = math.acos(dot)
        rotation_axes = cross / np.linalg.norm(cross)
        M = np.array([[0, -rotation_axes[2], rotation_axes[1]],
                      [rotation_axes[2], 0, -rotation_axes[0]],
                      [-rotation_axes[1], rotation_axes[0], 0]])

        return np.identity(3, dtype=float) + math.sin(angle) * M + (1 - math.cos(angle)) * np.dot(M, M)

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
        column_comp = np.all(points == points[0,:], axis=0) # compare the entries columnwise
        if np.all(column_comp): # all rows are identical to the first row
            return 0
        else:
            return 1

    return min(rank, 3)

def eigen(matrix: np.ndarray) -> List:
    """
    Compute eigenvalues/vectors, return a list of eigenvalue/vectors, sorted by the eigenvalue ascendingly
    """
    w, v = np.linalg.eig(matrix)
    
    eigen_pairs = sorted(
        list(zip(w, v)),
        key=lambda pair: pair[0]
    )

    return eigen_pairs
