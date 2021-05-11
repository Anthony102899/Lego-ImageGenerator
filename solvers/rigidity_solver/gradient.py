import numpy as np
import torch

import util.geometry_util as geo_util

"""
Gradient analysis: 
    - points position
    - joint position
    - joint orientation
"""

rotation_matrix = torch.tensor([
    [0, 1],
    [-1, 0],
], dtype=torch.double)


def spring_energy_matrix(
        points,
        edges,
        dim,
):
    n, m = len(points), len(edges)

    K = torch.zeros((m, m), dtype=torch.double)
    P = torch.zeros((m, m * dim), dtype=torch.double)
    A = torch.zeros((m * dim, n * dim), dtype=torch.double)

    for idx, e in enumerate(edges):
        edge_vec: torch.Tensor = points[e[0]] - points[e[1]]
        P[idx, idx * dim: idx * dim + dim] = edge_vec / edge_vec.norm()
        K[idx, idx] = 1 / edge_vec.norm()

        for d in range(dim):
            A[dim * idx + d, dim * e[0] + d] = 1
            A[dim * idx + d, dim * e[1] + d] = -1

    return torch.chain_matmul(A.T, P.T, K, P, A)


def constraint_matrix(
        points,
        pivots,
        translation_vectors,
        rotation_centers,
        joint_point_indices,
):
    """
    PyTorch implemented constraint matrix construction; ONLY SUPPORT 2D!
    :param points: (n, 2)
    :param pivots: iterable of vectors
    :param translation_vectors: iterable of [translation 2-vectors or None]
    :param rotation_centers: iterable of [rotation center 1-vector or None]
    :param joint_point_indices: iterable of [4-vector], indices into rows of points matrix
    :return: constraint matrix
    """
    constraint_matrices = []

    assert len(pivots) == len(translation_vectors) == len(rotation_centers) == len(joint_point_indices)

    joint_info = zip(pivots, translation_vectors, rotation_centers, joint_point_indices)

    for pivot, translation, rotation, point_indices in joint_info:
        sp = points[point_indices]

        basis = []
        basis.append(rigid_motion(sp))

        nonrigid_motion = torch.vstack((
            torch.cat([sp[0] - sp[1], sp[1] - sp[0], torch.zeros(4)]),
            torch.cat([torch.zeros(4), sp[2] - sp[3], sp[3] - sp[2]]),
        ))
        basis.append(nonrigid_motion)

        if translation is not None:
            assert not torch.allclose(translation, torch.tensor(0, dtype=torch.double))
            translation_motion = torch.cat((
                translation, translation, torch.zeros_like(translation), torch.zeros_like(translation)
            ))
            basis.append(translation_motion)

        if rotation is not None:
            rotation_motion = torch.cat((
                rotation_matrix @ (sp[0] - rotation), rotation_matrix @ (sp[1] - rotation), torch.zeros(4)
            ))
            basis.append(rotation_motion)

        allowed_motion_space = torch.vstack(basis)
        prohibitive_motion_space = torch_null_space(allowed_motion_space).t()

        i, j, k, l = point_indices
        zero_constraint = torch.zeros((prohibitive_motion_space.size()[0], points.size()[0] * 2), dtype=torch.double)
        zero_constraint[:, i * 2: (i + 1) * 2] = prohibitive_motion_space[:, 0: 2]
        zero_constraint[:, j * 2: (j + 1) * 2] = prohibitive_motion_space[:, 2: 4]
        zero_constraint[:, k * 2: (k + 1) * 2] = prohibitive_motion_space[:, 4: 6]
        zero_constraint[:, l * 2: (l + 1) * 2] = prohibitive_motion_space[:, 6: 8]

        constraint_matrices.append(zero_constraint)

    return torch.vstack(constraint_matrices)


def rigid_motion(points: torch.Tensor) -> torch.Tensor:
    n = points.size()[0] * 2
    even_ind = torch.arange(0, n, 2)
    odd_ind = even_ind + 1

    motion = torch.zeros((3, n))
    motion[0, even_ind] = 1.0
    motion[1, odd_ind] = 1.0
    motion[2] = (rotation_matrix @ points.t()).t().reshape(-1)

    return motion


def torch_null_space(A: torch.Tensor, disturb_s=False) -> torch.Tensor:
    dist = 1e-9
    with torch.no_grad():
        g, d, h = torch.svd(A, some=False)
        max_dist = dist * len(d)
        eps = torch.arange(0, len(d), dtype=torch.double) * dist
        eps_diag = torch.zeros_like(A)
        eps_diag[:len(d), :len(d)] = torch.diag_embed(eps)
        C = torch.chain_matmul(g, eps_diag, h.t())

    if disturb_s:
        B = A + C
    else:
        B = A

    u, s, v = torch.svd(B, some=False)
    vt = v.t()
    M, N = u.size()[0], vt.size()[1]
    rcond = torch.finfo(s.dtype).eps * max(M, N)
    tol = torch.max(s) * rcond + max_dist
    num = torch.sum(s > tol, dtype=torch.int)

    Q = vt[num:, :].t().conj()

    return Q

