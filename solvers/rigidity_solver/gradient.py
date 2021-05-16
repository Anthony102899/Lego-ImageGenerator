import numpy as np
from scipy.linalg import null_space
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

    edge_endpoints = points[edges]
    edge_vectors = edge_endpoints[:, 0, :] - edge_endpoints[:, 1, :]
    edge_length = edge_vectors.norm(dim=1)
    K = torch.diag_embed(1 / edge_length)

    indices = torch.arange(len(edges))
    for offset in torch.arange(dim):
        P[indices, indices * dim + offset] = edge_vectors[:, offset]

        A[indices * dim + offset, edges[:, 0] * dim + offset] = 1
        A[indices * dim + offset, edges[:, 1] * dim + offset] = -1


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
    dim = points.size()[1]
    constraint_matrices = []

    assert len(pivots) == len(translation_vectors) == len(rotation_centers) == len(joint_point_indices)

    joint_info = zip(pivots, translation_vectors, rotation_centers, joint_point_indices)

    for pivot, translation, rotation, point_indices in joint_info:
        sp = points[point_indices]

        basis = []
        basis.append(rigid_motion(sp))

        if dim == 2:
            nonrigid_motion = torch.vstack((
                torch.cat([sp[0] - sp[1], sp[1] - sp[0], torch.zeros(4)]),
                torch.cat([torch.zeros(4), sp[2] - sp[3], sp[3] - sp[2]]),
            ))
        else:
            nonrigid_motion = torch.vstack([
                torch.cat([sp[0] - sp[1], sp[1] - sp[0], torch.zeros(3), torch.zeros(9)]),
                torch.cat([torch.zeros(3), sp[1] - sp[2], sp[2] - sp[1], torch.zeros(9)]),
                torch.cat([sp[2] - sp[0], torch.zeros(3), sp[0] - sp[2], torch.zeros(9)]),
                torch.cat([torch.zeros(9), sp[3] - sp[4], sp[4] - sp[3], torch.zeros(3)]),
                torch.cat([torch.zeros(9), torch.zeros(3), sp[4] - sp[5], sp[5] - sp[4]]),
                torch.cat([torch.zeros(9), sp[5] - sp[3], torch.zeros(3), sp[3] - sp[5]]),
            ])
        basis.append(nonrigid_motion)

        if translation is not None:
            assert not torch.allclose(translation, torch.tensor(0, dtype=torch.double))
            if dim == 2:
                translation_motion = torch.cat((
                    translation, translation, torch.zeros_like(translation), torch.zeros_like(translation)
                ))
            else:
                translation_motion = torch.cat((
                    translation, translation, translation,
                    torch.zeros_like(translation), torch.zeros_like(translation), torch.zeros_like(translation),
                ))
            basis.append(translation_motion)

        if rotation is not None:
            if dim == 2:
                rotation_motion = torch.cat((
                    rotation_matrix @ (sp[0] - rotation), rotation_matrix @ (sp[1] - rotation), torch.zeros(4)
                ))
            else:
                rotation_list = []
                for rotation_axis in rotation:
                    rotation_list.append(
                        torch.cat((torch.cross(rotation_axis, sp[0] - pivot),
                                   torch.cross(rotation_axis, sp[1] - pivot),
                                   torch.cross(rotation_axis, sp[2] - pivot),
                                   torch.zeros(9)))
                    )

                rotation_motion = torch.vstack(rotation_list)

            basis.append(rotation_motion)

        allowed_motion_space = torch.vstack(basis)
        prohibitive_motion_space = torch_null_space(allowed_motion_space, disturb_s=True).t()

        zero_constraint = torch.zeros((prohibitive_motion_space.size()[0], points.size()[0] * dim), dtype=torch.double)
        if dim == 2:
            i, j, k, l = point_indices
            zero_constraint[:, i * 2: (i + 1) * 2] = prohibitive_motion_space[:, 0: 2]
            zero_constraint[:, j * 2: (j + 1) * 2] = prohibitive_motion_space[:, 2: 4]
            zero_constraint[:, k * 2: (k + 1) * 2] = prohibitive_motion_space[:, 4: 6]
            zero_constraint[:, l * 2: (l + 1) * 2] = prohibitive_motion_space[:, 6: 8]
        if dim == 3:
            for i, index in enumerate(point_indices):
                zero_constraint[:, index * 3: (index + 1) * 3] = prohibitive_motion_space[:, i * 3: (i + 1) * 3]

        constraint_matrices.append(zero_constraint)

    return torch.vstack(constraint_matrices)


def rigid_motion(points: torch.Tensor) -> torch.Tensor:
    dim = points.size()[1]
    n = points.size()[0] * dim
    assert dim in (2, 3)
    if dim == 2:
        even_ind = torch.arange(0, n, 2)
        odd_ind = even_ind + 1

        motion = torch.zeros((3, n))
        motion[0, even_ind] = 1.0
        motion[1, odd_ind] = 1.0
        motion[2] = (rotation_matrix @ points.t()).t().reshape(-1)
    else:
        motion = torch.zeros((6, n), dtype=torch.double)
        motion[0, torch.arange(0, n, 3)] = 1.0
        motion[1, torch.arange(0, n, 3) + 1] = 1.0
        motion[2, torch.arange(0, n, 3) + 2] = 1.0

        x = torch.tensor([1, 0, 0], dtype=torch.double)
        y = torch.tensor([0, 1, 0], dtype=torch.double)
        z = torch.tensor([0, 0, 1], dtype=torch.double)
        for i, point in enumerate(points):
            motion[3, i * 3: (i + 1) * 3] = torch.cross(point, x)
            motion[4, i * 3: (i + 1) * 3] = torch.cross(point, y)
            motion[5, i * 3: (i + 1) * 3] = torch.cross(point, z)
        assert (i + 1) * 3 == n

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

