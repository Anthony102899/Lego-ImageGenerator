# %%
import sys

sys.path.append("../../..")

from scipy.linalg import null_space
import copy
import numpy as np
from numpy.linalg import matrix_rank, matrix_power, cholesky, inv
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import util.geometry_util as geo_util
import solvers.rigidity_solver.gradient as gradient
from solvers.rigidity_solver.internal_structure import tetrahedron, triangulation_with_torch
from solvers.rigidity_solver.constraints_3d import select_non_colinear_points
from solvers.rigidity_solver import gradient, algo_core as core, extra_constraint
from solvers.rigidity_solver.eigen_analysis import eigen_analysis

from visualization.model_visualizer import visualize_3D, visualize_2D
from matplotlib import pyplot as plt

data = np.array([
    [0, 0],
    [0, 10],
    [10, 10],
    [10, 0],
])

# mutable
parameter_nodes = {
    "left-down": torch.tensor(data[0], dtype=torch.double),
    "left-up": torch.tensor(data[1], dtype=torch.double),
    "right-up": torch.tensor(data[2], dtype=torch.double),
    "right-down": torch.tensor(data[3], dtype=torch.double),
}
parameter_scalars = {}
immutable = {}

# for param in parameter_nodes.values():
#     param.requires_grad_(True)

part_node_connectivity = {
    "left": ("left-up", "left-down"),
    "right": ("right-up", "right-down"),
    "up": ("left-up", "right-up"),
}


def describe_nodes():
    node_map = {**parameter_nodes, **immutable}
    return node_map


part_map = {}

from collections import namedtuple

Part = namedtuple("Part", "points, edges, index_offset")
Joint = namedtuple("Joint", "pivot, part1_ind, part2_ind, translation, rotation_center")


def empty(_):
    return None


joints = [
    Joint(lambda nm: nm["left-up"], "left", "up", empty, lambda nm: nm["left-up"]),
    Joint(lambda nm: nm["right-up"], "right", "up", empty, lambda nm: nm["right-up"]),
]


def describe_model(part_nodes, only_points=False):
    offset = 0
    for key, (i, j) in part_node_connectivity.items():
        _points, _edges = triangulation_with_torch(part_nodes[i], part_nodes[j], 10, thickness=0.3)
        part_map[key] = Part(_points, _edges, offset)
        assert not torch.any(torch.isnan(_points)), f"exists nan, {part_nodes[i], part_nodes[j]}"

        offset += len(_points)

    point_matrix = torch.vstack([part_map[key].points for key in part_node_connectivity.keys()])
    assert not torch.any(torch.isnan(point_matrix))

    if only_points:
        return point_matrix

    edge_matrix = torch.vstack([
        part_map[key].edges + part_map[key].index_offset for key in part_node_connectivity.keys()])
    constraint_point_indices = torch.tensor(np.vstack([
        np.concatenate(
            [select_non_colinear_points(
                part_map[j.part1_ind].points.detach().numpy(),
                2,
                near=j.pivot(part_nodes).detach().numpy()
             )[1] + part_map[j.part1_ind].index_offset,
             select_non_colinear_points(
                 part_map[j.part2_ind].points.detach().numpy(),
                 2,
                 near=j.pivot(part_nodes).detach().numpy()
             )[1] + part_map[j.part2_ind].index_offset]
        ) for j in joints
    ]), dtype=torch.long)

    return point_matrix, edge_matrix, constraint_point_indices


def total_length(nodes, connectivity):
    len = torch.tensor(0, dtype=torch.double)
    for i, j in connectivity.values():
        len += torch.norm(nodes[i] - nodes[j])
    return len


nodes = describe_nodes()
points, edges, constraint_point_indices = describe_model(nodes)
init_len = total_length(nodes, part_node_connectivity)
    # visualize_2D(points, edges)

joint_constraints = gradient.constraint_matrix(
    points,
    pivots=[j.pivot(nodes) for j in joints],
    translation_vectors=[j.translation(nodes) for j in joints],
    rotation_centers=[j.rotation_center(nodes) for j in joints],
    joint_point_indices=constraint_point_indices,
)

fix_point_constraints = torch.zeros((4, points.size()[0] * 2), dtype=torch.double)
_, (ind_a, _) = select_non_colinear_points(points.detach().numpy(), num=2, near=nodes["left-down"].numpy())
_, (ind_b, _) = select_non_colinear_points(points.detach().numpy(), num=2, near=nodes["right-down"].numpy())
print(points[ind_a])
print(points[ind_b])
fix_point_constraints[0, ind_a * 2] = 1.0
fix_point_constraints[1, ind_a * 2 + 1] = 1.0
fix_point_constraints[2, ind_b * 2] = 1.0
fix_point_constraints[3, ind_b * 2 + 1] = 1.0


extra_constraints = torch.vstack([
    # gradient.rigid_motion(points),
    fix_point_constraints,
])

constraints = torch.vstack([
    joint_constraints,
    extra_constraints
])

B = gradient.torch_null_space(constraints)

K = gradient.spring_energy_matrix(points, edges, dim=2)

Q = torch.chain_matmul(B.t(), K, B)

# the eigenvalues are already in ascending order!
eigenvalues, eigenvectors = torch.symeig(Q, eigenvectors=True)

eigind = 0
smallest_eigenvalue = eigenvalues[eigind]
print(smallest_eigenvalue)
corresponding_eigenvector = torch.mv(B, eigenvectors[:, eigind])

arrows = geo_util.normalize(corresponding_eigenvector.reshape(-1, 2).detach().numpy()) * len(points)
visualize_2D(points, edges, arrows)

