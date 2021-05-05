import numpy as np
import torch

import util.geometry_util as geo_util

"""
Gradient analysis: 
    - points position
    - joint position
    - joint orientation
"""



def smallest_eigenpair(
        points: torch.Tensor,
        edges: torch.Tensor,
        hinge_axes,
        hinge_pivots,
        hinge_point_indices,
        extra_constraints=None,
):
    eigenvalues, eigenvectors = differentiable_eigen(
        points, edges,
        hinge_axes, hinge_pivots, hinge_point_indices,
        extra_constraints
    )
    return eigenvalues[0], eigenvectors[0]

def gradient_analysis(
        points: np.ndarray,
        edges: np.ndarray,
        hinge_axes,
        hinge_pivots,
        hinge_point_indices,
        extra_constraints=None,
        iters=1000
) -> torch.Tensor:
    assert iters > 0

    hinge_rad = torch.tensor(geo_util.cart2sphere(hinge_axes)[:, 1:], requires_grad=True)
    theta, phi = hinge_rad[:, 0], hinge_rad[:, 1]
    points = torch.tensor(points, dtype=torch.double, requires_grad=True)
    edges = torch.tensor(edges)
    # hinge_axes = torch.tensor(hinge_axes, dtype=torch.double, requires_grad=True)
    hinge_pivots = torch.tensor(hinge_pivots, dtype=torch.double, requires_grad=True)
    hinge_point_indices = torch.tensor(hinge_point_indices, dtype=torch.long)

    optimizer = torch.optim.Adam([hinge_rad, hinge_pivots, points], lr=1e-3)

    for i in range(iters):
        optimizer.zero_grad()

        hinge_axes = torch.hstack([
            torch.unsqueeze(torch.sin(theta) * torch.cos(phi), 1),
            torch.unsqueeze(torch.sin(theta) * torch.sin(phi), 1),
            torch.unsqueeze(torch.cos(theta), 1)
        ])

        eigenvalue, _ = smallest_eigenpair(
            points, edges, hinge_axes, hinge_pivots, hinge_point_indices, extra_constraints
        )

        # Negate it as torch optimizer minimizes objective by default
        obj = -eigenvalue

        obj.backward()
        optimizer.step()

    return obj



def rigidity_matrix(
        points,
        edges,
        dim
):
    n, m = len(points), len(edges)

    R = torch.zeros((m, dim * n), dtype=torch.double)
    for i, (p_ind, q_ind) in enumerate(edges):
        q_minus_p = points[q_ind, :] - points[p_ind, :]
        R[i, q_ind * dim: (q_ind + 1) * dim] = q_minus_p
        R[i, p_ind * dim: (p_ind + 1) * dim] = -q_minus_p

    return R


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
        hinge_axes,
        hinge_pivots,
        hinge_point_indices,
):
    dim = 3
    constraint_matrices = []

    for axis, pivot, point_indices in zip(hinge_axes, hinge_pivots, hinge_point_indices):
        for source_point_indices, target_point_indices in [
            (point_indices[0], point_indices[1]),
            (point_indices[1], point_indices[0]),
        ]:
            source_points = points[source_point_indices, :]
            target_points = points[target_point_indices, :]

            constraints = hinge_constraints(
                source_points,
                target_points,
                axis,
                pivot,
            )

            i, j, k = source_point_indices
            for constraint, target_index in zip(constraints, target_point_indices):
                l = target_index
                zero_constraint = torch.zeros((constraint.shape[0], len(points) * dim), dtype=torch.double)
                zero_constraint[:, i * 3: (i + 1) * 3] = constraint[:, 0: 3]
                zero_constraint[:, j * 3: (j + 1) * 3] = constraint[:, 3: 6]
                zero_constraint[:, k * 3: (k + 1) * 3] = constraint[:, 6: 9]
                zero_constraint[:, l * 3: (l + 1) * 3] = constraint[:, 9: 12]

                constraint_matrices.append(zero_constraint)

    return torch.vstack(constraint_matrices)


def perpendicular_vectors(v: torch.Tensor):
    index = torch.nonzero(v)[0]
    non_parallel = torch.ones_like(v)
    non_parallel[index] = 0
    u = torch.cross(non_parallel, v)
    w = torch.cross(u, v)
    return u, w

def torch_null_space(A: torch.Tensor, disturb_s=False):
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


def hinge_constraints(
        source_points,
        target_points,
        rotation_axis,
        rotation_pivot,
):
    constraints = []
    for target_point in target_points:
        relative_projection, source_transform = projection_matrix(
            source_points,
            target_point,
        )

        assert len(rotation_axis) == len(rotation_pivot)

        relative_target_point = torch.mv(source_transform, target_point)
        relative_rotation_pivot = torch.mv(source_transform, rotation_pivot)
        relative_rotation_axis = torch.mv(source_transform, rotation_axis)

        prohibitive_space = prohibitive_space_of_allowed_relative_rotation(
            relative_target_point,
            relative_rotation_pivot,
            relative_rotation_axis,
        )

        prohibitive_space_on_deltas = torch.mm(prohibitive_space, relative_projection)

        constraints.append(prohibitive_space_on_deltas)

    return constraints


def prohibitive_space_of_allowed_relative_rotation(
        target_point: torch.Tensor,
        pivot: torch.Tensor,
        axis: torch.Tensor,
) -> torch.Tensor:
    allowed_direction = torch.cross(target_point - pivot, axis)
    if allowed_direction.norm() > 1e-8:
        # unit_direction = allowed_direction / allowed_direction.norm()
        null_basis = torch.vstack(perpendicular_vectors(allowed_direction))
        return null_basis
    else:
        sqrt_half = np.sqrt(0.5)
        null_basis = torch.tensor([
            [1, 0, 0],
            [0, sqrt_half, sqrt_half],
            [0, sqrt_half, -sqrt_half],
        ], dtype=torch.double)
        return null_basis


def projection_matrix(
        source_points: torch.Tensor,
        target_point: torch.Tensor
):
    dim = 3
    projection_matrix = torch.zeros((dim, 4 * dim), dtype=torch.double)

    x0, x1, x2 = source_points
    x0_x, x0_y, x0_z = x0
    x1_x, x1_y, x1_z = x1
    x2_x, x2_y, x2_z = x2
    t_x, t_y, t_z = target_point

    norm_1 = (x1 - x0).norm()
    basis_1 = (x1 - x0) / norm_1

    norm_2 = torch.cross(x2 - x0, basis_1).norm()
    basis_2 = torch.cross(x2 - x0, basis_1) / norm_2

    basis_3 = torch.cross(basis_1, basis_2)

    if torch.isclose(norm_1, torch.tensor(0.0, dtype=torch.double)):
        raise Exception("Norm(x1 - x0) nears 0", "x0", x0, "x1", x1)
    if torch.isclose(norm_2, torch.tensor(0.0, dtype=torch.double)):
        raise Exception("Norm(x2 - x0) nears 0", "x0", x0, "x1", x1, "x2", x2)

    # Sympy generated code
    # >>>>>>>>>>>>>>>>>>>>
    projection_matrix[0, 0] = (-t_x + 2*x0_x - x1_x)/norm_1
    projection_matrix[0, 1] = (-t_y + 2*x0_y - x1_y)/norm_1
    projection_matrix[0, 2] = (-t_z + 2*x0_z - x1_z)/norm_1
    projection_matrix[0, 3] = (t_x - x0_x)/norm_1
    projection_matrix[0, 4] = (t_y - x0_y)/norm_1
    projection_matrix[0, 5] = (t_z - x0_z)/norm_1
    projection_matrix[0, 6] = 0
    projection_matrix[0, 7] = 0
    projection_matrix[0, 8] = 0
    projection_matrix[0, 9] = (-x0_x + x1_x)/norm_1
    projection_matrix[0, 10] = (-x0_y + x1_y)/norm_1
    projection_matrix[0, 11] = (-x0_z + x1_z)/norm_1
    projection_matrix[1, 0] = (t_y*x1_z - t_y*x2_z - t_z*x1_y + t_z*x2_y + x1_y*x2_z - x1_z*x2_y)/(norm_1*norm_2)
    projection_matrix[1, 1] = (-t_x*x1_z + t_x*x2_z + t_z*x1_x - t_z*x2_x - x1_x*x2_z + x1_z*x2_x)/(norm_1*norm_2)
    projection_matrix[1, 2] = (t_x*x1_y - t_x*x2_y - t_y*x1_x + t_y*x2_x + x1_x*x2_y - x1_y*x2_x)/(norm_1*norm_2)
    projection_matrix[1, 3] = (-t_y*x0_z + t_y*x2_z + t_z*x0_y - t_z*x2_y - x0_y*x2_z + x0_z*x2_y)/(norm_1*norm_2)
    projection_matrix[1, 4] = (t_x*x0_z - t_x*x2_z - t_z*x0_x + t_z*x2_x + x0_x*x2_z - x0_z*x2_x)/(norm_1*norm_2)
    projection_matrix[1, 5] = (-t_x*x0_y + t_x*x2_y + t_y*x0_x - t_y*x2_x - x0_x*x2_y + x0_y*x2_x)/(norm_1*norm_2)
    projection_matrix[1, 6] = (t_y*x0_z - t_y*x1_z - t_z*x0_y + t_z*x1_y + x0_y*x1_z - x0_z*x1_y)/(norm_1*norm_2)
    projection_matrix[1, 7] = (-t_x*x0_z + t_x*x1_z + t_z*x0_x - t_z*x1_x - x0_x*x1_z + x0_z*x1_x)/(norm_1*norm_2)
    projection_matrix[1, 8] = (t_x*x0_y - t_x*x1_y - t_y*x0_x + t_y*x1_x + x0_x*x1_y - x0_y*x1_x)/(norm_1*norm_2)
    projection_matrix[1, 9] = (-x0_y*x1_z + x0_y*x2_z + x0_z*x1_y - x0_z*x2_y - x1_y*x2_z + x1_z*x2_y)/(norm_1*norm_2)
    projection_matrix[1, 10] = (x0_x*x1_z - x0_x*x2_z - x0_z*x1_x + x0_z*x2_x + x1_x*x2_z - x1_z*x2_x)/(norm_1*norm_2)
    projection_matrix[1, 11] = (-x0_x*x1_y + x0_x*x2_y + x0_y*x1_x - x0_y*x2_x - x1_x*x2_y + x1_y*x2_x)/(norm_1*norm_2)
    projection_matrix[2, 0] = (t_x*x0_y*x1_y - t_x*x0_y*x2_y + t_x*x0_z*x1_z - t_x*x0_z*x2_z - t_x*x1_y**2 + t_x*x1_y*x2_y - t_x*x1_z**2 + t_x*x1_z*x2_z - 2*t_y*x0_x*x1_y + 2*t_y*x0_x*x2_y + t_y*x0_y*x1_x - t_y*x0_y*x2_x + t_y*x1_x*x1_y - 2*t_y*x1_x*x2_y + t_y*x1_y*x2_x - 2*t_z*x0_x*x1_z + 2*t_z*x0_x*x2_z + t_z*x0_z*x1_x - t_z*x0_z*x2_x + t_z*x1_x*x1_z - 2*t_z*x1_x*x2_z + t_z*x1_z*x2_x + 2*x0_x*x1_y**2 - 2*x0_x*x1_y*x2_y + 2*x0_x*x1_z**2 - 2*x0_x*x1_z*x2_z - 2*x0_y*x1_x*x1_y + x0_y*x1_x*x2_y + x0_y*x1_y*x2_x - 2*x0_z*x1_x*x1_z + x0_z*x1_x*x2_z + x0_z*x1_z*x2_x + x1_x*x1_y*x2_y + x1_x*x1_z*x2_z - x1_y**2*x2_x - x1_z**2*x2_x)/(norm_1**2*norm_2)
    projection_matrix[2, 1] = (t_x*x0_x*x1_y - t_x*x0_x*x2_y - 2*t_x*x0_y*x1_x + 2*t_x*x0_y*x2_x + t_x*x1_x*x1_y + t_x*x1_x*x2_y - 2*t_x*x1_y*x2_x + t_y*x0_x*x1_x - t_y*x0_x*x2_x + t_y*x0_z*x1_z - t_y*x0_z*x2_z - t_y*x1_x**2 + t_y*x1_x*x2_x - t_y*x1_z**2 + t_y*x1_z*x2_z - 2*t_z*x0_y*x1_z + 2*t_z*x0_y*x2_z + t_z*x0_z*x1_y - t_z*x0_z*x2_y + t_z*x1_y*x1_z - 2*t_z*x1_y*x2_z + t_z*x1_z*x2_y - 2*x0_x*x1_x*x1_y + x0_x*x1_x*x2_y + x0_x*x1_y*x2_x + 2*x0_y*x1_x**2 - 2*x0_y*x1_x*x2_x + 2*x0_y*x1_z**2 - 2*x0_y*x1_z*x2_z - 2*x0_z*x1_y*x1_z + x0_z*x1_y*x2_z + x0_z*x1_z*x2_y - x1_x**2*x2_y + x1_x*x1_y*x2_x + x1_y*x1_z*x2_z - x1_z**2*x2_y)/(norm_1**2*norm_2)
    projection_matrix[2, 2] = (t_x*x0_x*x1_z - t_x*x0_x*x2_z - 2*t_x*x0_z*x1_x + 2*t_x*x0_z*x2_x + t_x*x1_x*x1_z + t_x*x1_x*x2_z - 2*t_x*x1_z*x2_x + t_y*x0_y*x1_z - t_y*x0_y*x2_z - 2*t_y*x0_z*x1_y + 2*t_y*x0_z*x2_y + t_y*x1_y*x1_z + t_y*x1_y*x2_z - 2*t_y*x1_z*x2_y + t_z*x0_x*x1_x - t_z*x0_x*x2_x + t_z*x0_y*x1_y - t_z*x0_y*x2_y - t_z*x1_x**2 + t_z*x1_x*x2_x - t_z*x1_y**2 + t_z*x1_y*x2_y - 2*x0_x*x1_x*x1_z + x0_x*x1_x*x2_z + x0_x*x1_z*x2_x - 2*x0_y*x1_y*x1_z + x0_y*x1_y*x2_z + x0_y*x1_z*x2_y + 2*x0_z*x1_x**2 - 2*x0_z*x1_x*x2_x + 2*x0_z*x1_y**2 - 2*x0_z*x1_y*x2_y - x1_x**2*x2_z + x1_x*x1_z*x2_x - x1_y**2*x2_z + x1_y*x1_z*x2_y)/(norm_1**2*norm_2)
    projection_matrix[2, 3] = (-t_x*x0_y**2 + t_x*x0_y*x1_y + t_x*x0_y*x2_y - t_x*x0_z**2 + t_x*x0_z*x1_z + t_x*x0_z*x2_z - t_x*x1_y*x2_y - t_x*x1_z*x2_z + t_y*x0_x*x0_y + t_y*x0_x*x1_y - 2*t_y*x0_x*x2_y - 2*t_y*x0_y*x1_x + t_y*x0_y*x2_x + 2*t_y*x1_x*x2_y - t_y*x1_y*x2_x + t_z*x0_x*x0_z + t_z*x0_x*x1_z - 2*t_z*x0_x*x2_z - 2*t_z*x0_z*x1_x + t_z*x0_z*x2_x + 2*t_z*x1_x*x2_z - t_z*x1_z*x2_x - 2*x0_x*x0_y*x1_y + x0_x*x0_y*x2_y - 2*x0_x*x0_z*x1_z + x0_x*x0_z*x2_z + x0_x*x1_y*x2_y + x0_x*x1_z*x2_z + 2*x0_y**2*x1_x - x0_y**2*x2_x - 2*x0_y*x1_x*x2_y + x0_y*x1_y*x2_x + 2*x0_z**2*x1_x - x0_z**2*x2_x - 2*x0_z*x1_x*x2_z + x0_z*x1_z*x2_x)/(norm_1**2*norm_2)
    projection_matrix[2, 4] = (t_x*x0_x*x0_y - 2*t_x*x0_x*x1_y + t_x*x0_x*x2_y + t_x*x0_y*x1_x - 2*t_x*x0_y*x2_x - t_x*x1_x*x2_y + 2*t_x*x1_y*x2_x - t_y*x0_x**2 + t_y*x0_x*x1_x + t_y*x0_x*x2_x - t_y*x0_z**2 + t_y*x0_z*x1_z + t_y*x0_z*x2_z - t_y*x1_x*x2_x - t_y*x1_z*x2_z + t_z*x0_y*x0_z + t_z*x0_y*x1_z - 2*t_z*x0_y*x2_z - 2*t_z*x0_z*x1_y + t_z*x0_z*x2_y + 2*t_z*x1_y*x2_z - t_z*x1_z*x2_y + 2*x0_x**2*x1_y - x0_x**2*x2_y - 2*x0_x*x0_y*x1_x + x0_x*x0_y*x2_x + x0_x*x1_x*x2_y - 2*x0_x*x1_y*x2_x - 2*x0_y*x0_z*x1_z + x0_y*x0_z*x2_z + x0_y*x1_x*x2_x + x0_y*x1_z*x2_z + 2*x0_z**2*x1_y - x0_z**2*x2_y - 2*x0_z*x1_y*x2_z + x0_z*x1_z*x2_y)/(norm_1**2*norm_2)
    projection_matrix[2, 5] = (t_x*x0_x*x0_z - 2*t_x*x0_x*x1_z + t_x*x0_x*x2_z + t_x*x0_z*x1_x - 2*t_x*x0_z*x2_x - t_x*x1_x*x2_z + 2*t_x*x1_z*x2_x + t_y*x0_y*x0_z - 2*t_y*x0_y*x1_z + t_y*x0_y*x2_z + t_y*x0_z*x1_y - 2*t_y*x0_z*x2_y - t_y*x1_y*x2_z + 2*t_y*x1_z*x2_y - t_z*x0_x**2 + t_z*x0_x*x1_x + t_z*x0_x*x2_x - t_z*x0_y**2 + t_z*x0_y*x1_y + t_z*x0_y*x2_y - t_z*x1_x*x2_x - t_z*x1_y*x2_y + 2*x0_x**2*x1_z - x0_x**2*x2_z - 2*x0_x*x0_z*x1_x + x0_x*x0_z*x2_x + x0_x*x1_x*x2_z - 2*x0_x*x1_z*x2_x + 2*x0_y**2*x1_z - x0_y**2*x2_z - 2*x0_y*x0_z*x1_y + x0_y*x0_z*x2_y + x0_y*x1_y*x2_z - 2*x0_y*x1_z*x2_y + x0_z*x1_x*x2_x + x0_z*x1_y*x2_y)/(norm_1**2*norm_2)
    projection_matrix[2, 6] = (t_x*x0_y**2 - 2*t_x*x0_y*x1_y + t_x*x0_z**2 - 2*t_x*x0_z*x1_z + t_x*x1_y**2 + t_x*x1_z**2 - t_y*x0_x*x0_y + t_y*x0_x*x1_y + t_y*x0_y*x1_x - t_y*x1_x*x1_y - t_z*x0_x*x0_z + t_z*x0_x*x1_z + t_z*x0_z*x1_x - t_z*x1_x*x1_z + x0_x*x0_y*x1_y + x0_x*x0_z*x1_z - x0_x*x1_y**2 - x0_x*x1_z**2 - x0_y**2*x1_x + x0_y*x1_x*x1_y - x0_z**2*x1_x + x0_z*x1_x*x1_z)/(norm_1**2*norm_2)
    projection_matrix[2, 7] = (-t_x*x0_x*x0_y + t_x*x0_x*x1_y + t_x*x0_y*x1_x - t_x*x1_x*x1_y + t_y*x0_x**2 - 2*t_y*x0_x*x1_x + t_y*x0_z**2 - 2*t_y*x0_z*x1_z + t_y*x1_x**2 + t_y*x1_z**2 - t_z*x0_y*x0_z + t_z*x0_y*x1_z + t_z*x0_z*x1_y - t_z*x1_y*x1_z - x0_x**2*x1_y + x0_x*x0_y*x1_x + x0_x*x1_x*x1_y + x0_y*x0_z*x1_z - x0_y*x1_x**2 - x0_y*x1_z**2 - x0_z**2*x1_y + x0_z*x1_y*x1_z)/(norm_1**2*norm_2)
    projection_matrix[2, 8] = (-t_x*x0_x*x0_z + t_x*x0_x*x1_z + t_x*x0_z*x1_x - t_x*x1_x*x1_z - t_y*x0_y*x0_z + t_y*x0_y*x1_z + t_y*x0_z*x1_y - t_y*x1_y*x1_z + t_z*x0_x**2 - 2*t_z*x0_x*x1_x + t_z*x0_y**2 - 2*t_z*x0_y*x1_y + t_z*x1_x**2 + t_z*x1_y**2 - x0_x**2*x1_z + x0_x*x0_z*x1_x + x0_x*x1_x*x1_z - x0_y**2*x1_z + x0_y*x0_z*x1_y + x0_y*x1_y*x1_z - x0_z*x1_x**2 - x0_z*x1_y**2)/(norm_1**2*norm_2)
    projection_matrix[2, 9] = (x0_x*x0_y*x1_y - x0_x*x0_y*x2_y + x0_x*x0_z*x1_z - x0_x*x0_z*x2_z - x0_x*x1_y**2 + x0_x*x1_y*x2_y - x0_x*x1_z**2 + x0_x*x1_z*x2_z - x0_y**2*x1_x + x0_y**2*x2_x + x0_y*x1_x*x1_y + x0_y*x1_x*x2_y - 2*x0_y*x1_y*x2_x - x0_z**2*x1_x + x0_z**2*x2_x + x0_z*x1_x*x1_z + x0_z*x1_x*x2_z - 2*x0_z*x1_z*x2_x - x1_x*x1_y*x2_y - x1_x*x1_z*x2_z + x1_y**2*x2_x + x1_z**2*x2_x)/(norm_1**2*norm_2)
    projection_matrix[2, 10] = (-x0_x**2*x1_y + x0_x**2*x2_y + x0_x*x0_y*x1_x - x0_x*x0_y*x2_x + x0_x*x1_x*x1_y - 2*x0_x*x1_x*x2_y + x0_x*x1_y*x2_x + x0_y*x0_z*x1_z - x0_y*x0_z*x2_z - x0_y*x1_x**2 + x0_y*x1_x*x2_x - x0_y*x1_z**2 + x0_y*x1_z*x2_z - x0_z**2*x1_y + x0_z**2*x2_y + x0_z*x1_y*x1_z + x0_z*x1_y*x2_z - 2*x0_z*x1_z*x2_y + x1_x**2*x2_y - x1_x*x1_y*x2_x - x1_y*x1_z*x2_z + x1_z**2*x2_y)/(norm_1**2*norm_2)
    projection_matrix[2, 11] = (-x0_x**2*x1_z + x0_x**2*x2_z + x0_x*x0_z*x1_x - x0_x*x0_z*x2_x + x0_x*x1_x*x1_z - 2*x0_x*x1_x*x2_z + x0_x*x1_z*x2_x - x0_y**2*x1_z + x0_y**2*x2_z + x0_y*x0_z*x1_y - x0_y*x0_z*x2_y + x0_y*x1_y*x1_z - 2*x0_y*x1_y*x2_z + x0_y*x1_z*x2_y - x0_z*x1_x**2 + x0_z*x1_x*x2_x - x0_z*x1_y**2 + x0_z*x1_y*x2_y + x1_x**2*x2_z - x1_x*x1_z*x2_x + x1_y**2*x2_z - x1_y*x1_z*x2_y)/(norm_1**2*norm_2)
    # <<<<<<<<<<<<<<<<<<<<

    return projection_matrix, torch.vstack((basis_1, basis_2, basis_3))
