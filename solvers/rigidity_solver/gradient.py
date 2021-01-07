import numpy as np
import torch

"""
Gradient analysis: 
    - points position
    - joint position
    - joint orientation
"""


def gradient_analysis(
        points: np.ndarray,
        edges: np.ndarray,
        joints,
        dim: int,
) -> np.ndarray:
    points = torch.tensor(points, requires_grad=True)
    edges = torch.tensor(edges)
    joint_positions = torch.tensor(joints, requires_grad=True)
    joint_orients = torch.tensor(joints, requires_grad=True)


def rigidity_matrix(
        points,
        edges,
        dim
):
    n, m = len(points), len(edges)

    R = torch.zeros((m, dim * n))
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

    K = torch.zeros((m, m))
    P = torch.zeros((m, m * dim))
    A = torch.zeros((m * dim, n * dim))

    for idx, e in enumerate(edges):
        edge_vec: torch.Tensor = points[e[0]] - points[e[1]]
        P[idx, idx * dim: idx * dim + dim] = edge_vec / edge_vec.norm()
        K[idx, idx] = 1 / edge_vec.norm()

        for d in range(dim):
            A[dim * idx + d, dim * e[0] + d] = 1
            A[dim * idx + d, dim * e[1] + d] = -1

    return torch.chain_matmul([A.T, P.T, K, P, A])


def constraint_matrix(
        points,
        hinges_pos,
        hinges_orients,
):
    pass


def hinge_constraint(
        source
):
    pass
