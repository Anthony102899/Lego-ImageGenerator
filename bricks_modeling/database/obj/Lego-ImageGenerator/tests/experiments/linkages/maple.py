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
from util.timer import SimpleTimer

from visualization.model_visualizer import visualize_3D, visualize_2D
from visualization import model_visualizer as vis
from matplotlib import pyplot as plt
from collections import namedtuple
import itertools

Part = namedtuple("Part", "points, edges, index_offset")
Joint = namedtuple("Joint", "pivot, part1_ind, part2_ind, translation, rotation_center")

def empty(_):
    return None

# mutable
t = lambda x, y: torch.tensor([x, y], dtype=torch.double)
parameter_nodes = {
    "A": t(625, -1296),
    "B": t(744, -1376),
    "C": t(724, -1456),
    "D": t(909, -1485),
    "E": t(913, -1612),
    "F": t(677, -1633),
    "G": t(812, -1735),
    "H": t(649, -1735),
    "I": t(644, -1901),
    "J": t(606, -1901),
    "K": t(600, -1735),
    "L": t(438, -1735),
    "M": t(572, -1633),
    "N": t(336, -1612),
    "O": t(339, -1485),
    "P": t(526, -1456),
    "Q": t(505, -1376),
    "R": t(625, -1548),
}

for value in parameter_nodes.values():
    value *= 0.05

parameter_scalars = {}
immutable = {}

# for param in parameter_nodes.values():
#     param.requires_grad_(True)

node_connectivity = {
    key: (key[0], key[1])
    for key in (
        "AC", "AP", "BJ", "BP", "CQ", "DN", "DH", "EO", "EH", "FG",
        "GL", "IQ", "IJ", "LM", "KN", "KO"
    )
}

part_map = {}

def gen_joint(pvname, conn):
    return [Joint(lambda nm: nm[pvname], *conn_pair, empty, lambda nm: nm[pvname])
            for conn_pair in itertools.combinations(conn, 2)]

joints = [
    *gen_joint("A", ["AC", "AP"]),
    *gen_joint("B", ["BP", "BJ"]),
    *gen_joint("C", ["BJ", "CQ", "AC"]),
    *gen_joint("D", ["DN", "DH"]),
    *gen_joint("E", ["EO", "EH"]),
    *gen_joint("F", ["BJ", "FG"]),
    *gen_joint("G", ["FG", "GL"]),
    *gen_joint("H", ["BJ", "GL", "DH", "EH"]),
    *gen_joint("I", ["IQ", "IJ"]),
    *gen_joint("J", ["IJ", "BJ"]),
    *gen_joint("K", ["GL", "IQ", "KN", "KO"]),
    *gen_joint("L", ["GL", "LM"]),
    *gen_joint("M", ["LM", "IQ"]),
    *gen_joint("N", ["KN", "DN"]),
    *gen_joint("O", ["KO", "EO"]),
    *gen_joint("P", ["BP", "AP", "IQ"]),
    *gen_joint("Q", ["IQ", "CQ"]),
    # *gen_joint("R", ["EO", "DN"]),
]


def describe_nodes():
    node_map = {**parameter_nodes, **immutable}
    return node_map


def total_length(nodes, connectivity):
    len = torch.tensor(0, dtype=torch.double)
    for i, j in connectivity.values():
        len += torch.norm(nodes[i] - nodes[j])
    return len

hoped_num_per_beam = 5
density = total_length(describe_nodes(), node_connectivity) / (hoped_num_per_beam * len(node_connectivity))

def describe_model(part_nodes, only_points=False):
    offset = 0
    part_map = {}
    for key, (i, j) in node_connectivity.items():
        node_dist = torch.norm(part_nodes[i] - part_nodes[j])
        num = int(torch.round(node_dist / density))
        _points, _edges = triangulation_with_torch(part_nodes[i], part_nodes[j], num=num, thickness=0.3)
        part_map[key] = Part(_points, _edges, offset)
        assert not torch.any(torch.isnan(_points)), f"exists nan, {part_nodes[i], part_nodes[j]}"

        offset += len(_points)

    point_matrix = torch.vstack([part_map[key].points for key in node_connectivity.keys()])
    assert not torch.any(torch.isnan(point_matrix))

    if only_points:
        return point_matrix

    edge_matrix = torch.vstack([
        part_map[key].edges + part_map[key].index_offset for key in node_connectivity.keys()])
    import time
    now = time.time()
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

# fix_point_constraints = torch.zeros((12, points.size()[0] * 2), dtype=torch.double)
# fixed_points = [
#     nodes["A"],
#     nodes["C-slider-1"], nodes["C-slider-2"],
#     nodes["E-slider-1"], nodes["E-slider-2"],
# ]
# for i, pt in enumerate(fixed_points):
#     _, (ind_a, _) = select_non_colinear_points(points.detach().numpy(), num=2, near=pt.detach().numpy())
#     fix_point_constraints[i * 2, ind_a * 2] = 1.0
#     fix_point_constraints[i * 2 + 1, ind_a * 2 + 1] = 1.0


extra_constraints = torch.vstack([
    gradient.rigid_motion(points),
    # fix_point_constraints,
])

constraints = torch.vstack([
    joint_constraints,
    extra_constraints
])


B = gradient.torch_null_space(constraints)

from solvers.rigidity_solver.algo_core import generalized_courant_fischer
Q, _ = generalized_courant_fischer(K.numpy(), constraints.numpy())

Q = torch.chain_matmul(B.t(), K, B)
timer.checkpoint("Q")

# the eigenvalues are already in ascending order!
eigenvalues, eigenvectors = torch.symeig(Q, eigenvectors=True)
timer.checkpoint("symeig")
timer.report()
print("#parts", len(node_connectivity))
print("#joints", len(joints))
print("#points", len(points))

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
for i, j in node_connectivity.values():
    node_dist = torch.norm(nodes[i] - nodes[j])
    num = int(torch.round(node_dist / density))
    ratios = [0, *np.linspace(1 / (num + 1), 1 - 1 / (num + 1), num), 1]
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

plt.savefig(f"{__file__.replace('.py', '')}-arrow.svg", transparent=True)
plt.show()

