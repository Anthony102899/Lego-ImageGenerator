import numpy as np
import math

# TODO: to add function annotation
def vec_local2world(rot_mat: np.ndarray, local_vec: np.ndarray):
    return np.dot(rot_mat, local_vec)


# TODO: to add function annotation
def point_local2world(
    rot_mat: np.ndarray, translation: np.ndarray, local_point: np.ndarray
):
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