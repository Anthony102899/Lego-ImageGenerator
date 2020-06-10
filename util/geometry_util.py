import numpy as np

# TODO: to add function annotation
def vec_local2world(rot_mat: np.ndarray, local_vec: np.ndarray):
    return np.dot(rot_mat, local_vec)

# TODO: to add function annotation
def point_local2world(rot_mat: np.ndarray, translation: np.ndarray, local_point: np.ndarray):
    return np.dot(rot_mat, local_point) + translation