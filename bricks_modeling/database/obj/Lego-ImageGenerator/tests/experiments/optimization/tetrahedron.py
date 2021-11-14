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
    [0, 0, 0],
    [1, 0, 0],
    [0, 2.5, 0],
    [2, 2, -2.5],
], dtype=np.double) * 3

parameter_nodes = {
    "bottom-ori": torch.from_numpy(data[0]),
    "bottom-x": torch.from_numpy(data[1]),
    "bottom-y": torch.from_numpy(data[2]),
    "top": torch.from_numpy(data[3]),
}
for value in parameter_nodes.values():
    value.requires_grad_(True)

parameter_scalars = {}

immutable_nodes = {
}

computed_nodes = {}

node_connectivity = {
    "bottom-ori-x": ("bottom-ori", "bottom-x"),
    "bottom-x-y": ("bottom-x", "bottom-y"),
    "bottom-ori-y": ("bottom-ori", "bottom-y"),
    "bt-ori": ("bottom-ori", "top"),
    "bt-x": ("bottom-x", "top"),
    "bt-y": ("bottom-y", "top"),
}


def describe_nodes():
    nm = {**parameter_nodes, **immutable_nodes}
    return nm


part_map = {}

from collections import namedtuple

Part = namedtuple("Part", "points, edges, index_offset")
Joint = namedtuple("Joint", "pivot, part1_ind, part2_ind, translation, rotation_center")


empty = lambda _: None
ball = lambda _: torch.eye(3, dtype=torch.double)

joints = [
    Joint(lambda nm: nm["bottom-ori"], "bottom-ori-x", "bottom-ori-y", empty, ball),
    Joint(lambda nm: nm["bottom-ori"], "bottom-ori-x", "bt-ori", empty, ball),
    Joint(lambda nm: nm["bottom-ori"], "bottom-ori-y", "bt-ori", empty, ball),
    Joint(lambda nm: nm["bottom-x"], "bottom-ori-x", "bottom-x-y", empty, ball),
    Joint(lambda nm: nm["bottom-x"], "bottom-ori-x", "bt-x", empty, ball),
    Joint(lambda nm: nm["bottom-x"], "bottom-x-y", "bt-x", empty, ball),
    Joint(lambda nm: nm["bottom-y"], "bottom-ori-y", "bottom-x-y", empty, ball),
    Joint(lambda nm: nm["bottom-y"], "bottom-ori-y", "bt-y", empty, ball),
    Joint(lambda nm: nm["bottom-y"], "bottom-x-y", "bt-y", empty, ball),
    Joint(lambda nm: nm["top"], "bt-ori", "bt-x", empty, ball),
    Joint(lambda nm: nm["top"], "bt-x", "bt-y", empty, ball),
    Joint(lambda nm: nm["top"], "bt-y", "bt-ori", empty, ball),
]


def describe_model(part_nodes, only_points=False):
    offset = 0
    for key, (i, j) in node_connectivity.items():
        _points, _edges = tetrahedron(part_nodes[i], part_nodes[j], density=0.3, num=4, thickness=0.3, mode="torch")
        part_map[key] = Part(_points, _edges, offset)
        assert not torch.any(torch.isnan(_points)), f"exists nan, {part_nodes[i], part_nodes[j]}"

        offset += len(_points)

    point_matrix = torch.vstack([part_map[key].points for key in node_connectivity.keys()])
    assert not torch.any(torch.isnan(point_matrix))

    if only_points:
        return point_matrix

    edge_matrix = torch.vstack([
        part_map[key].edges + part_map[key].index_offset for key in node_connectivity.keys()
    ])

    if len(joints) > 0:
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
    else:
        constraint_point_indices = []

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
    init_len = total_length(nodes, node_connectivity)
    # visualize_2D(points, edges)

# %%
n_iters = 500
optimizer = torch.optim.SGD([
    {"params": [*parameter_nodes.values()], "lr": 10},
    # {"params": [*parameter_scalars.values()], "lr": 0.002},
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

    K = gradient.spring_energy_matrix(points, edges, dim=3)

    Q = torch.chain_matmul(B.t(), K, B)

    # the eigenvalues are already in ascending order!
    eigenvalues, eigenvectors = torch.symeig(Q, eigenvectors=True)

    eigind = 3
    smallest_eigenvalue = eigenvalues[eigind]
    corresponding_eigenvector = torch.mv(B, eigenvectors[:, eigind])

    assert torch.allclose(eigenvalues[:eigind],
                              torch.tensor(0.0, dtype=torch.double),
                              atol=1e-12)

    # length_penalty = 0.2 * torch.pow(total_length(nodes, node_connectivity) - init_len, 2)
    # Negate eigenvalue in the objective as we're trying to increase it
    objective = -smallest_eigenvalue
    objective.backward()

    optimizer.step()

    # print(smallest_eigenvalue, eigenvalues[:eigind])
    # for node in parameter_nodes.values():
    #     print(node.grad.data)
    # exit()

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

ax.axis('off')
for key in nodes:
    plt.cla()
    ax.axis('equal')
    ax.axis('off')
    points = np.array([t["nodes"][key] for t in traces])
    points_x = points[:, 0]
    points_y = points[:, 1]
    ax.plot(points_x, points_y, color="black")
    plt.savefig(f"excavator-{key}-points.svg", transparent=True)

print(traces[0]["nodes"])
print(traces[-1]["nodes"])
for key, (i, j) in node_connectivity.items():
    print(key, traces[-1]["nodes"][i], traces[-1]["nodes"][j], np.linalg.norm(traces[-1]["nodes"][i] - traces[-1]["nodes"][j]))
for key, (i, j) in node_connectivity.items():
    print(key, traces[0]["nodes"][i], traces[0]["nodes"][j], np.linalg.norm(traces[0]["nodes"][i] - traces[0]["nodes"][j]))

for it in np.round(np.linspace(0, n_iters - 1, 8)).astype(np.int):
    trace = traces[it]
    vertices = trace["nodes"]
    plot_shape(ax, vertices, node_connectivity.values())

plt.show()

for it in np.round(np.linspace(0, n_iters - 1, 8)).astype(np.int):
    trace = traces[it]
    # visualize_2D(trace["points"], edges, trace["eigenvector"].reshape(-1, 2)[:, :2])
