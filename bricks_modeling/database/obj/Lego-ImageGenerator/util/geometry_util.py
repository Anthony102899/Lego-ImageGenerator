import numpy as np
import math
from numpy import linalg as LA
from typing import List
from sympy import Matrix
from scipy.spatial.transform import Rotation as R


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
    X_before = np.empty([3, 3])
    X_before[:, 0] = a1
    X_before[:, 1] = a2
    X_before[:, 2] = a3

    X_after = np.empty([3, 3])
    X_after[:, 0] = b1
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
        dot = np.dot(a, b) / (LA.norm(a) * LA.norm(b))
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
    Project vector v on base. Return the projected vector
    """
    length = np.dot(v, base) / np.dot(base, base)
    return length * base

def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / LA.norm(vec)

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

def decompose_on_orthobasis(vector, orthobasis):
    projection_norms = np.apply_along_axis(lambda base: LA.norm(project(vector, base)), axis=1, arr=orthobasis)
    return projection_norms

def trivial_basis(points: np.ndarray, dim=None, orthonormal=True) -> np.ndarray:
    """
    Given n points in 3d space in form of a (n x 3) matrix, construct 6 'trivial' orthonormal vectors
    """
    if dim is None:
        dim = points.shape[1]
    assert dim in [2, 3]
    P = points.reshape((-1, dim))
    n = len(P)

    # translation along x, y (and z if in 3d)
    translations = np.hstack([np.identity(dim)] * n)

    # note that here we cast 2d points into 3d,
    # this is to get the perpendicular vectors via cross product which has to be done in 3d
    P_as3d = np.hstack((P, np.zeros((n, 1)))) if dim == 2 else P
    center = np.mean(P_as3d, axis=0) 
    P_shifted = P_as3d - center # centralize the object, to make the rotation vectors orthogonal

    x_axis, y_axis, z_axis = np.identity(3)

    if dim == 3:
        rotations = np.array([
            np.cross(P_shifted, x_axis).reshape(-1),
            np.cross(P_shifted, y_axis).reshape(-1),
            np.cross(P_shifted, z_axis).reshape(-1),
        ])
    else: # dim == 2
        # rotate wrt z_axis, discard the third dimension
        rotated = np.cross(P_shifted, z_axis)[:, :dim]
        rotations = rotated.reshape((1, -1))

    transformation = np.vstack((translations, rotations))
    # row-wise normalize the vectors so that each row is unitary
    basis = rowwise_normalize(transformation)
    if orthonormal:
        return orthonormalize(basis)
    else:
        return basis

def subtract_orthobasis(vector: np.ndarray, orthobasis: np.ndarray) -> np.ndarray:
    """
    Given a vector and a orthonormal matrix, project the vector into the null space of the matrix
    """

    projections = np.apply_along_axis(lambda base: project(vector, base), axis=1, arr=orthobasis)
    subtraction = vector - np.sum(projections, axis=0)
    return subtraction


def is_subspace(vecs, basis) -> bool:
    """
    Return if the space spanned by "vecs" is a subspace of the space spanning by "basis"
    """
    for vec in vecs:
        projected_vec = np.zeros_like(vec)
        for zero_base in basis:
            projected_vec += project(vec, zero_base)
        if LA.norm(projected_vec-vec) > 1e-5:
            return False

    return True


def is_trivial_motions(eigen_vecs, points, dim) -> bool:
    basis = trivial_basis(points, dim)
    return is_subspace(eigen_vecs, basis)


def points_span_dim(points: np.ndarray) -> bool:
    """
    points: shape(n, 3)
    If the points spans a
        i) 0D space (the points are identical), return 0
        ii) 1D space (on the same line), return 1
        iii) 2D space (on the same plane), return 2
        iv) higher-than-2D space, return 3
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


def eigen(matrix: np.ndarray, symmetric: bool = True) -> List:
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


def rref(matrix: np.ndarray) -> np.ndarray:
    """
    Return the reduced echelon form of a matrix
    """
    M = Matrix(matrix)
    M_rref, pivot_indices = M.rref(iszerofunc = lambda x: True if abs(x)<1e-9 else False) # reduced row echelon form
    return np.array(M_rref).astype(np.float64)

def gen_random_rotation() -> np.ndarray:
    rand_rot_axis = np.random.rand(3)
    rot = R.from_rotvec(np.random.randint(0, 100) * rand_rot_axis)
    return rot.as_matrix()

def get_random_transformation() -> np.ndarray:
    trans_matrix = np.identity(4, dtype=float)
    trans_matrix[:3, :3] = gen_random_rotation()
    trans_matrix[:3, 3] = 5 * np.random.rand(3)
    return trans_matrix


def cart2sphere(cartesian: np.ndarray) -> np.ndarray:
    """
    Convert cartesian coordinates to spherical coordinates
    """
    def _cart2sphere(row):
        x, y, z = row
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        theta = np.arctan2(hxy, z)
        phi = np.arctan2(y, x)
        return np.array([r, theta, phi], dtype=np.double)

    spherical = np.apply_along_axis(_cart2sphere, axis=1, arr=cartesian)
    return spherical


def sphere2cart(spherical: np.ndarray) -> np.ndarray:
    "Order of the coordinate: r theta phi"
    def _sphere2cart(row):
        r, theta, phi = row
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z], dtype=np.double)

    cart = np.apply_along_axis(_sphere2cart, axis=1, arr=spherical)
    return cart

def unitsphere2cart(thetaphi: np.ndarray) -> np.ndarray:
    spherical = np.hstack((np.ones((len(thetaphi), 1)), thetaphi))
    cart = sphere2cart(spherical)
    return cart


def clear_redundance_vecs(vecs):

    res = np.zeros_like(vecs)
    for k,v in enumerate(vecs):
        u = v.copy()
        for i in range(k):
            u -= u.dot(res[i])*res[i]
        if LA.norm(u) < 1e-5:
            continue
        u = u/LA.norm(u)
        res[k] = u[:]
    a = res

    return  res


def clear_trivial_motion(zero_vecs:np.ndarray ,trivial_bases:np.ndarray,choose_largest_vec = True,dim = 3):
    res = np.array(zero_vecs)
    for i in range(len(zero_vecs)):
        for trivial_basis in trivial_bases:
            assert abs(np.linalg.norm(trivial_basis) - 1) < 1e-6
            res[i] = res[i]  - res[i].dot(trivial_basis)*trivial_basis
        if np.linalg.norm(res[i]) < 1e-6:
            res[i] = 0
        else:
            res[i] = res[i]/np.linalg.norm(res[i])
    return res