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
from collections import namedtuple

from visualization.model_visualizer import visualize_3D, visualize_2D
from matplotlib import pyplot as plt

Part = namedtuple("Part", "points, edges, index_offset")
Joint = namedtuple("Joint", "pivot, part1_ind, part2_ind, translation, rotation_center")
def empty(_):
    return None

# mutable
parameter_nodes = {
    "A": torch.tensor([0, 0, 0], dtype=torch.double),
    "B": torch.tensor([1, 0, 0], dtype=torch.double),
    "C": torch.tensor([1, 1, 0], dtype=torch.double),
    "D": torch.tensor([0, 1, 0], dtype=torch.double),
}

for value in parameter_nodes.values():
    value *= 10

parameter_scalars = {}
immutable = {}

# for param in parameter_nodes.values():
#     param.requires_grad_(True)

node_connectivity = {
    "AB": ("A", "B"),
    "BC": ("B", "C"),
    "CD": ("C", "D"),
    "DA": ("D", "A"),
}

part_map = {}

az = torch.tensor([0, 0, 1], dtype=torch.double)
axes = torch.tensor([
    [1, 0, 1],
    [-1, 0, 1],
    [-1/2, -1, 1],
    [1/2, -1, 1],
], dtype=torch.double)
joints = [
    Joint(lambda nm: nm["A"], "AB", "DA",
          empty, lambda nm: axes[0]),
    Joint(lambda nm: nm["B"], "BC", "AB",
          empty, lambda nm: axes[1]),
    Joint(lambda nm: nm["C"], "CD", "BC",
          empty, lambda nm: axes[2]),
    Joint(lambda nm: nm["D"], "DA", "CD",
          empty, lambda nm: axes[3]),
]


def describe_nodes():
    node_map = {**parameter_nodes, **immutable}
    return node_map



def describe_model(part_nodes, only_points=False):
    offset = 0
    part_map = {}
    for key, (i, j) in node_connectivity.items():
        _points, _edges = tetrahedron(part_nodes[i], part_nodes[j], density=0.3, num=5, thickness=2, mode="torch")
        part_map[key] = Part(_points, _edges, offset)
        assert not torch.any(torch.isnan(_points)), f"exists nan, {part_nodes[i], part_nodes[j]}"

        offset += len(_points)

    point_matrix = torch.vstack([part_map[key].points for key in node_connectivity.keys()])
    assert not torch.any(torch.isnan(point_matrix))

    if only_points:
        return point_matrix

    edge_matrix = torch.vstack([
        part_map[key].edges + part_map[key].index_offset for key in node_connectivity.keys()])
    constraint_point_indices = torch.tensor(np.vstack([
        np.concatenate(
            [select_non_colinear_points(
                part_map[j.part1_ind].points.detach().numpy(),
                3,
                near=j.pivot(part_nodes).detach().numpy()
             )[1] + part_map[j.part1_ind].index_offset,
             select_non_colinear_points(
                 part_map[j.part2_ind].points.detach().numpy(),
                 3,
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
init_len = total_length(nodes, node_connectivity)

#### >>>>>>>>>>>>>>>>>>>>>> BUILDING MODEL FOR OUTPUT JSON
from solvers.rigidity_solver.models import Model, Beam, Joint
model = Model()
beam_map = {
    k: Beam.tetra(nodes[na].numpy(), nodes[nb].numpy()) for k, (na, nb) in node_connectivity.items()
}
joints = [
    Joint(beam_map["AB"], beam_map["DA"], nodes["A"], rotation_axes=axes[0]),
    Joint(beam_map["BC"], beam_map["AB"], nodes["B"], rotation_axes=axes[1]),
    Joint(beam_map["CD"], beam_map["BC"], nodes["C"], rotation_axes=axes[2]),
    Joint(beam_map["DA"], beam_map["CD"], nodes["D"], rotation_axes=axes[3]),
]
model.add_beams(beam_map.values())
model.add_joints(joints)
model.save_json("frame_axes.json")
#### <<<<<<<<<<<<<<<<<<<<<< FINISH BUILDING MODEL FOR OUTPUT JSON

arrows = torch.zeros_like(points, dtype=torch.double)
for i, value in enumerate(nodes.values()):
    _, (ind, *_) = select_non_colinear_points(points.numpy(), 3, near=value.numpy())
    arrows[ind] = axes[i]
visualize_3D(points.numpy(), edges=edges, arrows=arrows.numpy() / 10)

joint_constraints = gradient.constraint_matrix(
    points,
    pivots=[j.pivot(nodes) for j in joints],
    translation_vectors=[j.translation(nodes) for j in joints],
    rotation_centers=[j.rotation_center(nodes) for j in joints],
    joint_point_indices=constraint_point_indices,
)

fix_point_constraints = torch.zeros((15, joint_constraints.size()[1]))
for i, w in enumerate(torch.linspace(0, 1, 5, dtype=torch.double)):
    _, indices = select_non_colinear_points(points.numpy(), 3, torch.lerp(nodes["A"], nodes["B"], w).numpy())
    fix_point_constraints[i, 3 * indices[0]] = 1
    fix_point_constraints[i + 1, 3 * indices[0] + 1] = 1
    fix_point_constraints[i + 2, 3 * indices[0] + 2] = 1

extra_constraints = torch.vstack([
    # gradient.rigid_motion(points),
    fix_point_constraints,
])

constraints = torch.vstack([
    joint_constraints,
    extra_constraints
])


B = gradient.torch_null_space(constraints)

K = gradient.spring_energy_matrix(points, edges, dim=3)

from solvers.rigidity_solver.algo_core import generalized_courant_fischer
Q, _ = generalized_courant_fischer(K.numpy(), constraints.numpy())
print(geo_util.eigen(Q, True)[:5])

Q = torch.chain_matmul(B.t(), K, B)

# the eigenvalues are already in ascending order!
eigenvalues, eigenvectors = torch.symeig(Q, eigenvectors=True)

eigind = 0

smallest_eigenvalue = eigenvalues[eigind]
print(smallest_eigenvalue)
print(eigenvalues[:10])
corresponding_eigenvector = torch.mv(B, eigenvectors[:, eigind])

arrows = geo_util.normalize(corresponding_eigenvector.reshape(-1, 3).detach().numpy()) * len(points) / 100
visualize_3D(points, edges=edges, arrows=arrows)

# plt.clf()
# plt.scatter(points[:, 0], points[:, 1])
# num = 5
# ratios = np.linspace(1 / (num + 1), 1 - 1 / (num + 1), num)
# for i, j in node_connectivity.values():
#     for r in ratios:
#         x, y = torch.lerp(nodes[i], nodes[j], r)
#         _, (ind_p, ind_q) = select_non_colinear_points(points.numpy(), num=2, near=(x, y))
#         dx, dy = (arrows[ind_p] + arrows[ind_q]) * 0.5
#         plt.arrow(*points[ind_p], *arrows[ind_p], color="blue")
#         plt.arrow(*points[ind_q], *arrows[ind_q], color="green")
#         plt.arrow(x, y, dx, dy, color="red")
# plt.show()

