#%%
import sys
sys.path.append("../../..")

from scipy.linalg import null_space
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

#%%
part_nodes = np.array([
    [0, 0],
    [1, 0],
    [1, 2],
]) * 5

optimizable_ind = np.array([2, ])
optimizable_nodes = torch.tensor(
    part_nodes[optimizable_ind],
    dtype=torch.double, requires_grad=True)
fixed_nodes = torch.tensor(
    np.delete(part_nodes, optimizable_ind, axis=0),
    dtype=torch.double)

part_node_connectivity = np.array([
    (0, 1),
    (1, 2),
    (2, 0),
])

part_map = {}

from collections import namedtuple
Part = namedtuple("Part", "points, edges, index_offset")
Joint = namedtuple("Joint", "pivot, part1_ind, part2_ind, translation, rotation_center")

def empty(_):
    return None

joints = [
    Joint(lambda nodes: part_nodes[0], (0, 1), (2, 0), empty, lambda nodes: nodes[0]),
    Joint(lambda nodes: part_nodes[1], (0, 1), (1, 2), empty, lambda nodes: nodes[1]),
    Joint(lambda nodes: part_nodes[2], (1, 2), (2, 0), empty, lambda nodes: nodes[2]),
]

def describe_model(part_nodes, only_points=False):
    offset = 0
    for i, j in part_node_connectivity:
        _points, _edges = triangulation_with_torch(part_nodes[i], part_nodes[j], 10, thickness=0.3)
        part_map[(i, j)] = Part(_points, _edges, offset)
        assert not torch.any(torch.isnan(_points)), f"exists nan, {part_nodes[i], part_nodes[j]}"

        offset += len(_points)

    point_matrix = torch.vstack([part_map[(i, j)].points for i, j in part_node_connectivity])
    assert not torch.any(torch.isnan(point_matrix))

    if only_points:
        return point_matrix

    edge_matrix = torch.vstack([
        part_map[(i, j)].edges + part_map[(i, j)].index_offset for i, j in part_node_connectivity])
    constraint_point_indices = torch.tensor(np.vstack([
        np.concatenate((
            select_non_colinear_points(part_map[j.part1_ind].points.detach().numpy(), 2, near=j.pivot(part_nodes))[1] + part_map[j.part1_ind].index_offset,
            select_non_colinear_points(part_map[j.part2_ind].points.detach().numpy(), 2, near=j.pivot(part_nodes))[1] + part_map[j.part2_ind].index_offset))
        for j in joints
    ]), dtype=torch.long)

    return point_matrix, edge_matrix, constraint_point_indices

# %%
# initialization for edges and constraint_point_indices
with torch.no_grad():
    nodes = torch.vstack([fixed_nodes, optimizable_nodes])
    _, edges, constraint_point_indices = describe_model(nodes)

# %%
n_iters = 1000
optimizer = Adam(params=[optimizable_nodes], lr=0.01)

traces = []

for it in tqdm(range(n_iters)):
    optimizer.zero_grad()

    nodes = torch.vstack([fixed_nodes, optimizable_nodes])

    points = describe_model(nodes, only_points=True)
    assert not torch.any(torch.isnan(nodes)), f"exists nan in nodes, {nodes}"

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

    eigind = 0
    smallest_eigenvalue = eigenvalues[eigind]
    corresponding_eigenvector = torch.mv(B, eigenvectors[:, eigind])

    assert not torch.allclose(eigenvalues[eigind], torch.tensor(0.0, dtype=torch.double)), f"more than expected num dof: {eigenvalues}"

    # peripheral = (vertex_a - vertex_b).norm() + (vertex_b - vertex_c).norm() + (vertex_c - vertex_a).norm()
    # peripheral_penalty = 5 * torch.pow(peripheral - init_peripheral, 2), thickness=0.3
    peripheral_penalty = 0
    # Negate eigenvalue in the objective as we're trying to increase it
    objective = -smallest_eigenvalue + peripheral_penalty
    objective.backward()

    optimizer.step()

    trace = {
        "eigenvalue": smallest_eigenvalue.detach().cpu().numpy(),
        "eigenvector": corresponding_eigenvector.detach().cpu().numpy(),
        "vertices": nodes.detach().cpu().numpy(),
        "points": points.detach().cpu().numpy(),
    }
    traces.append(trace)

#%%
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

points = np.array([t["vertices"][2] for t in traces])
points_x = points[:, 0]
points_y = points[:, 1]
ax.scatter(points_x, points_y, s=1)

for it in np.round(np.linspace(0, n_iters - 1, 4)).astype(np.int):
    trace = traces[it]
    vertices = trace["vertices"]
    plot_shape(ax, vertices, part_node_connectivity)

plt.show()

for it in np.round(np.linspace(0, n_iters - 1, 4)).astype(np.int):
    trace = traces[it]
    visualize_2D(trace["points"], edges, trace["eigenvector"].reshape(-1, 2)[:, :2])

