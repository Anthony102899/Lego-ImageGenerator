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
from itertools import product, combinations

from visualization.model_visualizer import visualize_3D, visualize_2D
from visualization import model_visualizer as vis
from matplotlib import pyplot as plt

Part = namedtuple("Part", "points, edges, index_offset")
Joint = namedtuple("Joint", "pivot, part1_ind, part2_ind, translation, rotation_center")

def empty(_):
    return None

# mutable
nodes = {
    "A": torch.tensor([0, 0], dtype=torch.double),
    "B": torch.tensor([127, -125], dtype=torch.double),
    "C": torch.tensor([127, -209], dtype=torch.double),
    "D": torch.tensor([481, -209], dtype=torch.double),
    "E": torch.tensor([0, -375], dtype=torch.double),
    "F": torch.tensor([608, -375], dtype=torch.double),
    "G": torch.tensor([608, 0], dtype=torch.double),
    "H": torch.tensor([0, -209], dtype=torch.double),
    "I": torch.tensor([608, -209], dtype=torch.double),

    "upleft": torch.tensor([127 - 20, -80], dtype=torch.double),
    "downleft": torch.tensor([127 - 20, -295], dtype=torch.double),
    "downright": torch.tensor([481 + 20, -295], dtype=torch.double),
    "upright": torch.tensor([481 + 20, -80], dtype=torch.double),
    # "upleft": torch.tensor([100, -80], dtype=torch.double),
    # "downleft": torch.tensor([100, -295], dtype=torch.double),
    # "downright": torch.tensor([515, -295], dtype=torch.double),
    # "upright": torch.tensor([515, -80], dtype=torch.double),
}

for value in nodes.values():
    value *= 0.05

parameter_scalars = {}
immutable = {}

# for param in parameter_nodes.values():
#     param.requires_grad_(True)

node_connectivity = {
    "AB": ("A", "B"),

    "AE": ("A", "E"),
    "EF": ("E", "F"),
    "FG": ("F", "G"),
    "HI": ("H", "I"),

    "up": ("upright", "upleft"),
    "down": ("downright", "downleft"),
    "left": ("upleft", "downleft"),
    "right": ("upright", "downright"),
}

def square_part(upleft, downleft, downright, upright):
    vertical_num, horizontal_num = 13, 13
    points = torch.vstack([
        torch.lerp(
            torch.lerp(upleft, downleft, vr),
            torch.lerp(upright, downright, vr),
            hr
        )
        for vr, hr in product(
            torch.linspace(0, 1, vertical_num, dtype=torch.double),
            torch.linspace(0, 1, horizontal_num, dtype=torch.double))
    ])
    edges = torch.tensor([(i, j) for i, j in combinations(torch.arange(len(points)), 2)], dtype=torch.long)
    return points, edges


def describe_parts(nodes):
    part_map = {}
    offset = 0
    for key, (i, j) in node_connectivity.items():
        part_map[key] = Part(*triangulation_with_torch(nodes[i], nodes[j], 5, thickness=1.0), offset)
        offset += len(part_map[key].points)

    # part_map["middle"] = Part(*square_part(nodes["upleft"], nodes["downleft"], nodes["downright"], nodes["upright"]), offset)
    # offset += len(part_map["middle"].points)

    return part_map

joints = [
    Joint(lambda nm: nm["H"], "AE", "HI",
          empty, empty),
    Joint(lambda nm: nm["I"], "FG", "HI",
          empty, empty),
    Joint(lambda nm: nm["E"], "AE", "EF",
          empty, empty),
    Joint(lambda nm: nm["F"], "FG", "EF",
          empty, empty),

    Joint(lambda nm: nm["A"], "AE", "AB",
          empty, lambda nm: nm["A"]),

    Joint(lambda nm: nm["upleft"], "up", "left", empty, empty),
    Joint(lambda nm: nm["upright"], "up", "right", empty, empty),
    Joint(lambda nm: nm["downleft"], "down", "left", empty, empty),
    Joint(lambda nm: nm["downright"], "down", "right", empty, empty),
    Joint(lambda nm: nm["B"], "AB", "left",
          empty, lambda nm: nm["B"]),
    Joint(lambda nm: nm["C"], "HI", "left",
          lambda nm: nm["H"] - nm["I"],
          lambda nm: nm["C"]),
    Joint(lambda nm: nm["D"], "HI", "right",
          lambda nm: nm["H"] - nm["I"],
          lambda nm: nm["D"]),

    # Joint(lambda nm: nm["B"], "AB", "middle",
    #       empty, lambda nm: nm["B"]),
    # Joint(lambda nm: nm["C"], "HI", "middle",
    #       lambda nm: nm["H"] - nm["I"],
    #       lambda nm: nm["C"]),
    # Joint(lambda nm: nm["D"], "HI", "middle",
    #       lambda nm: nm["H"] - nm["I"],
    #       lambda nm: nm["D"]),

]


def describe_nodes():
    node_map = nodes
    return node_map



def describe_model(nodes, part_map, only_points=False):
    point_matrix = torch.vstack([part_map[key].points for key in part_map.keys()])
    assert not torch.any(torch.isnan(point_matrix))

    if only_points:
        return point_matrix

    edge_matrix = torch.vstack([
        part_map[key].edges + part_map[key].index_offset for key in part_map.keys()])
    constraint_point_indices = torch.tensor(np.vstack([
        np.concatenate(
            [select_non_colinear_points(
                part_map[j.part1_ind].points.detach().numpy(),
                2,
                near=j.pivot(nodes).detach().numpy()
             )[1] + part_map[j.part1_ind].index_offset,
             select_non_colinear_points(
                 part_map[j.part2_ind].points.detach().numpy(),
                 2,
                 near=j.pivot(nodes).detach().numpy()
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
part_map = describe_parts(nodes)
points, edges, constraint_point_indices = describe_model(nodes, part_map)
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


# fixed_points = [
#     nodes["A"], nodes["G"],
#     nodes["E"], nodes["F"],
#     *[torch.lerp(nodes["A"], nodes["E"], w) for w in torch.linspace(0, 1, 5, dtype=torch.double)],
#     *[torch.lerp(nodes["G"], nodes["F"], w) for w in torch.linspace(0, 1, 5, dtype=torch.double)],
#     *[torch.lerp(nodes["E"], nodes["F"], w) for w in torch.linspace(0, 1, 5, dtype=torch.double)],
# ]
# fix_point_constraints = torch.zeros((len(fixed_points) * 2, points.size()[0] * 2), dtype=torch.double)
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

# the eigenvalues are already in ascending order!
eigenvalues, eigenvectors = torch.symeig(Q, eigenvectors=True)
timer.checkpoint("eig")
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
node_points = np.vstack([v for k, v in nodes.items()])
# for constr_ind in constraint_point_indices:
    # plt.scatter(points[constr_ind[:2]][:, 0], points[constr_ind[:2]][:, 1])
    # plt.scatter(points[constr_ind[2:]][:, 0], points[constr_ind[2:]][:, 1])
#
# plt.scatter(points[:, 0], points[:, 1], s=3)

num = 5
ratios = [0, *np.linspace(1 / (num + 1), 1 - 1 / (num + 1), num), 1]
for i, j in node_connectivity.values():
    for r in ratios:
        x, y = torch.lerp(nodes[i], nodes[j], r)
        _, (ind_p, ind_q) = select_non_colinear_points(points.numpy(), num=2, near=(x, y))

        width = 0.12
        length = 0.10

        if 0 < r < 1:
            dx, dy = (arrows[ind_p] + arrows[ind_q]) * 0.5 * length
        else:
            dx, dy = arrows[ind_p] * length
        # plt.arrow(*points[ind_p], *arrows[ind_p] * 0.5 * length, color="blue", width=width)
        # plt.arrow(*points[ind_q], *arrows[ind_q] * 0.5 * length, color="green", width=width)
        if np.linalg.norm((dx, dy)) > 0.025:
            plt.arrow(x, y, dx, dy, color=vis.colormap["orange"], width=width)

plt.savefig(f"{__file__.replace('.py', '')}-arrow.svg", transparent=True)
plt.show()
