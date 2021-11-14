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

from visualization.model_visualizer import visualize_3D, visualize_2D, colormap
from matplotlib import pyplot as plt

from model_square import define, Part, Joint, empty

definition = define()
parameter_nodes = definition["parameter_nodes"]
immutable = definition["immutable"]
node_connectivity = definition["node_connectivity"]
joints = definition["joints"]

stage = 3
print(f"stage: {stage}")

if stage == 1:
    parameter_nodes.update({
        "rightup-up": torch.tensor([8, 10], dtype=torch.double),
        "rightup-right": torch.tensor([10, 8], dtype=torch.double),
    })
    node_connectivity.update({
        "rightup": ("rightup-up", "rightup-right"),
    })
    joints.extend([
        Joint(lambda nm: nm["rightup-up"], "up", "rightup", rotation_center=lambda nm: nm["rightup-up"], translation=empty),
        Joint(lambda nm: nm["rightup-right"], "right", "rightup", rotation_center=lambda nm: nm["rightup-right"], translation=empty)
    ])
elif stage == 2:
    parameter_nodes.update({
        "rightup-up": torch.tensor([8, 10], dtype=torch.double),
        "rightup-right": torch.tensor([10, 8], dtype=torch.double),
        "leftup-up": torch.tensor([2, 10], dtype=torch.double),
        "leftup-left": torch.tensor([0, 8], dtype=torch.double),
    })
    node_connectivity.update({
        "rightup": ("rightup-up", "rightup-right"),
        "leftup": ("leftup-up", "leftup-left"),
    })
    joints.extend([
        Joint(lambda nm: nm["rightup-up"], "up", "rightup", rotation_center=lambda nm: nm["rightup-up"], translation=empty),
        Joint(lambda nm: nm["rightup-right"], "right", "rightup", rotation_center=lambda nm: nm["rightup-right"], translation=empty),
        Joint(lambda nm: nm["leftup-up"], "up", "leftup", rotation_center=lambda nm: nm["leftup-up"], translation=empty),
        Joint(lambda nm: nm["leftup-left"], "left", "leftup", rotation_center=lambda nm: nm["leftup-left"], translation=empty),
    ])
elif stage == 3:
    parameter_nodes.update({
        "rightup-up": torch.tensor([5, 10], dtype=torch.double),
        "rightup-right": torch.tensor([10, 5], dtype=torch.double),
        "leftup-up": torch.tensor([5, 10], dtype=torch.double),
        "leftup-left": torch.tensor([0, 5], dtype=torch.double),
    })
    node_connectivity.update({
        "rightup": ("rightup-up", "rightup-right"),
        "leftup": ("leftup-up", "leftup-left"),
    })
    joints.extend([
        Joint(lambda nm: nm["rightup-up"], "up", "rightup", rotation_center=lambda nm: nm["rightup-up"], translation=empty),
        Joint(lambda nm: nm["rightup-right"], "right", "rightup", rotation_center=lambda nm: nm["rightup-right"], translation=empty),
        Joint(lambda nm: nm["leftup-up"], "up", "leftup", rotation_center=lambda nm: nm["leftup-up"], translation=empty),
        Joint(lambda nm: nm["leftup-left"], "left", "leftup", rotation_center=lambda nm: nm["leftup-left"], translation=empty),
    ])


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
print(smallest_eigenvalue.numpy())
corresponding_eigenvector = torch.mv(B, eigenvectors[:, eigind])

arrows = geo_util.normalize(corresponding_eigenvector.reshape(-1, 2).detach().numpy()) * len(points)
# visualize_2D(points, edges, arrows)

plt.clf()
# plt.scatter(points[:, 0], points[:, 1])
plt.axis('equal')
plt.axis('off')
num = 5
for key, (i, j) in node_connectivity.items():
    print(key)
    if key in ("rightup", "leftup"):
        ratios = np.array([0, *np.linspace(1 / (num + 1), 1 - 1 / (num + 1), num), 1])
        ratios = ratios[np.array((0, 2, 4, 6))]
    else:
        ratios = [0, *np.linspace(1 / (num + 1), 1 - 1 / (num + 1), num), 1]
    for r in ratios:
        x, y = torch.lerp(nodes[i], nodes[j], r)
        _, (ind_p, ind_q) = select_non_colinear_points(points.numpy(), num=2, near=(x, y))
        if 0 < r < 1:
            dxy = (arrows[ind_p] + arrows[ind_q]) * 0.5
        else:
            dxy = arrows[ind_p]
        # plt.arrow(*points[ind_p], *arrows[ind_p], color="blue")
        # plt.arrow(*points[ind_q], *arrows[ind_q], color="green")
        color = colormap["motion"] if stage == 0 else (255 / 255, 165 / 255, 0)
        if np.linalg.norm(dxy) > 0.1:
            plt.arrow(x, y, *dxy / 5, color=color, width=0.1, length_includes_head=False)

plt.savefig(f"square-{stage}.svg", transparent=True)
