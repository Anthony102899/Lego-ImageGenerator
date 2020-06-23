import numpy as np
import math
from numpy import linalg as LA

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
    if np.linalg.norm(cross) == 0:  # parallel
        return np.identity(3, dtype=float)
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
        perp_vec =  np.array([(-vec[1] - vec[2]) / vec[0], 1, 1])
    elif abs(vec[1]) > 1e-10:
        perp_vec =  np.array([1, (-vec[0] - vec[2]) / vec[1], 1])
    else:
        perp_vec =  np.array([1, 1, (-vec[0] - vec[1]) / vec[2]])

    return perp_vec/LA.norm(perp_vec)

if __name__ == "__main__":
    for i in range(10):
        a = np.random.rand(3)
        b = get_perpendicular_vec(a)
        print(a, b)
        print(a.dot(b))

    a = np.array([1, 0, 0])
    b = get_perpendicular_vec(a)
    print(a, b)
    print(a.dot(b))

    a = np.array([0, 5, 0])
    b = get_perpendicular_vec(a)
    print(a, b)
    print(a.dot(b))

    a = np.array([0, 0, 36])
    b = get_perpendicular_vec(a)
    print(a, b)
    print(a.dot(b))
