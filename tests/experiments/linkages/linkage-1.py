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
from util.timer import SimpleTimer
import solvers.rigidity_solver.gradient as gradient
from solvers.rigidity_solver.internal_structure import tetrahedron, triangulation_with_torch
from solvers.rigidity_solver.constraints_3d import select_non_colinear_points
from solvers.rigidity_solver import gradient, algo_core as core, extra_constraint
from solvers.rigidity_solver.eigen_analysis import eigen_analysis
from collections import namedtuple

from visualization.model_visualizer import visualize_3D, visualize_2D
from visualization import model_visualizer as vis
from matplotlib import pyplot as plt

Part = namedtuple("Part", "points, edges, index_offset")
Joint = namedtuple("Joint", "pivot, part1_ind, part2_ind, translation, rotation_center")
def empty(_):
    return None

# mutable
parameter_nodes = {
    "A": torch.tensor([-118, 25], dtype=torch.double),
    # "A-slider-1": torch.tensor([-118 + 46, 25 + 135], dtype=torch.double),
    # "A-slider-2": torch.tensor([-118 - 46, 25 - 135], dtype=torch.double),
    "B": torch.tensor([0, 0], dtype=torch.double),
    "C": torch.tensor([63, -202], dtype=torch.double),
    "C-slider-1": torch.tensor([63 - 81 / 2, -202 + 39 / 2], dtype=torch.double),
    "C-slider-2": torch.tensor([63 + 81 / 2, -202 - 39 / 2], dtype=torch.double),
    "D": torch.tensor([266, -60], dtype=torch.double),
    "E": torch.tensor([336, 198], dtype=torch.double),
    "E-slider-1": torch.tensor([336 - 63 / 2, 198 + 63 / 2], dtype=torch.double),
    "E-slider-2": torch.tensor([336 + 63 / 2, 198 - 63 / 2], dtype=torch.double),
    "F": torch.tensor([210, 73], dtype=torch.double),
}

for value in parameter_nodes.values():
    value *= 0.05

parameter_scalars = {}
immutable = {}

# for param in parameter_nodes.values():
#     param.requires_grad_(True)

node_connectivity = {
    # "A-slider": ("A-slider-1", "A-slider-2"),
    "C-slider": ("C-slider-1", "C-slider-2"),
    "E-slider": ("E-slider-1", "E-slider-2"),
    "AB": ("A", "B"),
    "BC": ("B", "C"),
    "CD": ("C", "D"),
    "DE": ("D", "E"),
    "EF": ("E", "F"),
    "BF": ("B", "F"),
}

part_map = {}

joints = [
    # Joint(lambda nm: nm["A"], "A-slider", "AB",
    #       lambda nm: nm["A-slider-1"] - nm["A-slider-2"],
          # lambda nm: nm["A"]),

    Joint(lambda nm: nm["B"], "AB", "BF",
          empty, lambda nm: nm["B"]),
    Joint(lambda nm: nm["B"], "AB", "BC",
          empty, lambda nm: nm["B"]),
    Joint(lambda nm: nm["B"], "BC", "BF",
          empty, lambda nm: nm["B"]),

    Joint(lambda nm: nm["C"], "BC", "C-slider",
          lambda nm: nm["C-slider-1"] - nm["C-slider-2"],
          lambda nm: nm["C"]),
    Joint(lambda nm: nm["C"], "CD", "C-slider",
          lambda nm: nm["C-slider-1"] - nm["C-slider-2"],
          lambda nm: nm["C"]),
    Joint(lambda nm: nm["C"], "BC", "CD",
          empty,
          lambda nm: nm["C"]),

    Joint(lambda nm: nm["D"], "DE", "CD",
          empty, lambda nm: nm["D"]),

    Joint(lambda nm: nm["E"], "DE", "E-slider",
          lambda nm: nm["E-slider-1"] - nm["E-slider-2"],
          lambda nm: nm["E"]),
    Joint(lambda nm: nm["E"], "EF", "E-slider",
          lambda nm: nm["E-slider-1"] - nm["E-slider-2"],
          lambda nm: nm["E"]),
    Joint(lambda nm: nm["E"], "DE", "EF",
          empty, lambda nm: nm["E"]),

    Joint(lambda nm: nm["F"], "BF", "EF",
          empty, lambda nm: nm["F"]),
]


def describe_nodes():
    node_map = {**parameter_nodes, **immutable}
    return node_map



def describe_model(part_nodes, only_points=False):
    offset = 0
    part_map = {}
    for key, (i, j) in node_connectivity.items():
        _points, _edges = triangulation_with_torch(part_nodes[i], part_nodes[j], 5, thickness=0.3)
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
init_len = total_length(nodes, node_connectivity)
    # visualize_2D(points, edges)
timer = SimpleTimer()

K = gradient.spring_energy_matrix(points, edges, dim=2)
timer.checkpoint("K")


joint_constraints = gradient.constraint_matrix(
    points,
    pivots=[j.pivot(nodes) for j in joints],
    translation_vectors=[j.translation(nodes) for j in joints],
    rotation_centers=[j.rotation_center(nodes) for j in joints],
    joint_point_indices=constraint_point_indices,
)

fix_point_constraints = torch.zeros((12, points.size()[0] * 2), dtype=torch.double)
fixed_points = [
    nodes["A"],
    nodes["C-slider-1"], nodes["C-slider-2"],
    nodes["E-slider-1"], nodes["E-slider-2"],
]
for i, pt in enumerate(fixed_points):
    _, (ind_a, _) = select_non_colinear_points(points.detach().numpy(), num=2, near=pt.detach().numpy())
    fix_point_constraints[i * 2, ind_a * 2] = 1.0
    fix_point_constraints[i * 2 + 1, ind_a * 2 + 1] = 1.0


extra_constraints = torch.vstack([
    # gradient.rigid_motion(points),
    fix_point_constraints,
])

constraints = torch.vstack([
    joint_constraints,
    extra_constraints
])


B = gradient.torch_null_space(constraints)

Q = torch.chain_matmul(B.t(), K, B)
timer.checkpoint("Q")
print("#parts", len(node_connectivity))
print("#joints", len(joints))
print("#points", len(points))
timer.report()

# the eigenvalues are already in ascending order!
eigenvalues, eigenvectors = torch.symeig(Q, eigenvectors=True)

eigind = 0

smallest_eigenvalue = eigenvalues[eigind]
print(smallest_eigenvalue)
print(eigenvalues[:10])
corresponding_eigenvector = torch.mv(B, eigenvectors[:, eigind])

arrows = geo_util.normalize(corresponding_eigenvector.reshape(-1, 2).detach().numpy()) * len(points)
# visualize_2D(points, edges, arrows)

plt.clf()
plt.axis("equal")
plt.axis("off")
# plt.scatter(points[:, 0], points[:, 1])
num = 5
ratios = [0, *np.linspace(1 / (num + 1), 1 - 1 / (num + 1), num), 1]
for i, j in node_connectivity.values():
    for r in ratios:
        x, y = torch.lerp(nodes[i], nodes[j], r)
        _, (ind_p, ind_q) = select_non_colinear_points(points.numpy(), num=2, near=(x, y))

        width = 0.1
        length = 0.1

        if 0 < r < 1:
            dx, dy = (arrows[ind_p] + arrows[ind_q]) * 0.5 * length
        else:
            dx, dy = arrows[ind_p] * length
        # plt.arrow(*points[ind_p], *arrows[ind_p], color="blue")
        # plt.arrow(*points[ind_q], *arrows[ind_q], color="green")
        if np.linalg.norm((dx, dy)) > 0.1:
            plt.arrow(x, y, dx, dy, color=vis.colormap["orange"], width=width)

plt.savefig("linkage-1-arrow.svg", transparent=True)
plt.show()

