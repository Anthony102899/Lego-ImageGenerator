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
   [503, 353],
   [1067, 27],
   [866, 128],
   [1067, 167],
   [1067, 367],
   [261, 432],
]) * 0.01

# mutable
parameter_nodes = {
    "up-right-conn": torch.tensor(data[1], dtype=torch.double),
    "right-down-node": torch.tensor(data[4], dtype=torch.double),
}
parameter_scalars = {
    "sliding-ratio": torch.tensor(0.75, dtype=torch.double),
    "main-right-ratio": torch.tensor(0.583, dtype=torch.double),
}
immutable = {
    "base-main-conn": torch.tensor(data[5], dtype=torch.double),
    "main-up-conn": torch.tensor(data[0], dtype=torch.double),
}

for param in parameter_nodes.values():
    param.requires_grad_(True)

part_node_connectivity = {
    "main": ("base-main-conn", "main-right-conn"),
    "up-left": ("main-up-conn", "up-sliding-conn"),
    "up-right": ("up-sliding-conn", "up-right-conn"),
    "right": ("up-right-conn", "right-down-node"),
}


def describe_nodes():
    nm = {**parameter_nodes, **immutable}
    computed_nodes = {
        "up-sliding-conn": torch.lerp(nm["main-up-conn"], nm["up-right-conn"], parameter_scalars["sliding-ratio"]),
        "main-right-conn": torch.lerp(nm["up-right-conn"], nm["right-down-node"], parameter_scalars["main-right-ratio"]),
    }
    node_map = {**nm, **computed_nodes}
    return node_map


part_map = {}

from collections import namedtuple

Part = namedtuple("Part", "points, edges, index_offset")
Joint = namedtuple("Joint", "pivot, part1_ind, part2_ind, translation, rotation_center")


def empty(_):
    return None


joints = [
    Joint(lambda nm: nm["main-up-conn"], "main", "up-left", empty, lambda nm: nm["main-up-conn"]),
    Joint(lambda nm: nm["up-sliding-conn"], "up-left", "up-right", lambda nm: nm["up-right-conn"] - nm["main-up-conn"],
          empty),
    Joint(lambda nm: nm["up-right-conn"], "up-right", "right", empty, lambda nm: nm["up-right-conn"]),
    Joint(lambda nm: nm["main-right-conn"], "main", "right", empty, lambda nm: nm["main-right-conn"]),
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

# %%
# initialization for edges and constraint_point_indices
with torch.no_grad():
    nodes = describe_nodes()
    points, edges, constraint_point_indices = describe_model(nodes)
    init_len = total_length(nodes, part_node_connectivity)
    # visualize_2D(points, edges)

# %%
n_iters = 500
optimizer = Adam([
    {"params": [*parameter_nodes.values()], "lr": 0.01},
    {"params": [*parameter_scalars.values()], "lr": 0.002},
])

traces = []

for it in tqdm(range(n_iters)):
    optimizer.zero_grad()

    nodes = describe_nodes()

    points = describe_model(nodes, only_points=True)
    assert not torch.any(torch.isnan(torch.vstack(tuple(nodes.values())))), f"exists nan in nodes, {nodes}"

    with torch.no_grad():
        joint_constraints = gradient.constraint_matrix(
            points,
            pivots=[j.pivot(nodes) for j in joints],
            translation_vectors=[j.translation(nodes) for j in joints],
            rotation_centers=[j.rotation_center(nodes) for j in joints],
            joint_point_indices=constraint_point_indices,
        )

        extra_constraints = torch.vstack([
            gradient.rigid_motion(points)
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

    eigind = 1
    smallest_eigenvalue = eigenvalues[eigind]
    corresponding_eigenvector = torch.mv(B, eigenvectors[:, eigind])

    assert not torch.allclose(eigenvalues[eigind],
                              torch.tensor(0.0, dtype=torch.double),
                              atol=1e-12), f"more than expected num dof: {eigenvalues}"

    # peripheral = (vertex_a - vertex_b).norm() + (vertex_b - vertex_c).norm() + (vertex_c - vertex_a).norm()
    # peripheral_penalty = 5 * torch.pow(peripheral - init_peripheral, 2), thickness=0.3
    length_penalty = 0.001 * torch.pow(total_length(nodes, part_node_connectivity) - init_len, 2)
    # Negate eigenvalue in the objective as we're trying to increase it
    objective = -smallest_eigenvalue + length_penalty
    objective.backward()

    optimizer.step()

    with torch.no_grad():
        for value in parameter_scalars.values():
            value.clamp_(0.0, 1.0)

    trace = {
        "eigenvalue": smallest_eigenvalue.detach().cpu().numpy(),
        "eigenvector": corresponding_eigenvector.detach().cpu().numpy(),
        "nodes": copy.deepcopy({k: v.detach().numpy() for k, v in nodes.items()}),
        "points": points.detach().cpu().numpy(),
    }
    traces.append(trace)

# %%
# visualize the optimization process
from matplotlib import pyplot as plt

# objective against time
objectives = [t["eigenvalue"] for t in traces]
plt.plot(np.arange(n_iters), objectives)
plt.show()


# shape of the triangle against time
def plot_shape(ax, vertices, edges):
    for a, b in edges:
        p, q = vertices[a], vertices[b]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=[0, 1, 0])


plt.clf()
fig, ax = plt.subplots()
ax.set(xlim=(0, 1.5), ylim=(0, 2))
ax.axis('equal')

# points = np.array([t["nodes"][2] for t in traces])
# points_x = points[:, 0]
# points_y = points[:, 1]
# ax.scatter(points_x, points_y, s=1)

for it in np.round(np.linspace(0, n_iters - 1, 8)).astype(np.int):
    trace = traces[it]
    vertices = trace["nodes"]
    plot_shape(ax, vertices, part_node_connectivity.values())

plt.show()

for it in np.round(np.linspace(0, n_iters - 1, 8)).astype(np.int):
    trace = traces[it]
    visualize_2D(trace["points"], edges, trace["eigenvector"].reshape(-1, 2)[:, :2])
