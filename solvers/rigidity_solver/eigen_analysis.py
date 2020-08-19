import util.geometry_util as geo_util
from numpy import linalg as LA


# to get basis forming the motion space
def get_motions(eigen_pairs, points, dim):
    # collect all eigen vectors with zero eigen value
    zeroeigenspace = [e_vec for e_val, e_vec in eigen_pairs]

    # Trivial basis -- orthonormalized translation along / rotation wrt 3 axes
    basis = geo_util.trivial_basis(points, dim)

    # cast the eigenvectors corresponding to zero eigenvalues into nullspace of the trivial basis,
    # in other words, the new vectors don't have any components (projection) in the span of the trivial basis

    if geo_util.is_subspace(zeroeigenspace, basis):
        reduced_zeroeigenspace = [geo_util.subtract_orthobasis(vec, basis) for vec in zeroeigenspace]
        motion_basis = geo_util.rref(reduced_zeroeigenspace)
        trivial_motion_dim = 3 if dim == 2 else 6
    else:
        motion_basis = zeroeigenspace
        trivial_motion_dim = 0

    result_motions = []
    for i in range(len(motion_basis) - trivial_motion_dim):
        e_vec = motion_basis[i]
        e_vec = e_vec / LA.norm(e_vec)
        delta_x = e_vec.reshape(-1, dim)
        result_motions.append(delta_x)

    return result_motions

def get_weakest_displacement(eigen_pairs, dim):
    e_val, e_vec = eigen_pairs[0]
    return e_vec.reshape(-1, dim), e_val