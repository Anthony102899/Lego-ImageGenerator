#%%
import sys
sys.path.append("../../..")

import scipy
import numpy as np
from numpy.linalg import matrix_rank, matrix_power, cholesky, inv
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import util.geometry_util as geo_util
from solvers.rigidity_solver.gradient import gradient_analysis
from solvers.rigidity_solver.internal_structure import tetrahedron, triangulation_with_torch
from solvers.rigidity_solver.algo_core import solve_rigidity, spring_energy_matrix
from solvers.rigidity_solver.models import Beam, Model, Joint
from solvers.rigidity_solver import gradient, algo_core as core
from solvers.rigidity_solver.eigen_analysis import eigen_analysis

from visualization.model_visualizer import visualize_3D, visualize_2D

from matplotlib import pyplot as plt

#%%
objectives = []

numpy_vertices = np.array([
    [0, 0],
    [1, 0],
    [1 / 2 , np.sqrt(3) / 2],
])
vertex_a, vertex_b, vertex_c = torch.tensor(numpy_vertices, dtype=torch.double)
vertex_c.requires_grad_(True)

part_nodes = np.array([
    [0, 0],
    [1, 0],
    [1, 2],
])

part_node_connectivity = np.array([
    [0, 1],
    [1, 2],
    [2, 0],
])

topology = np.array([
    []
])

model = Model()
with torch.no_grad():
    for i, j in part_node_connectivity:
        points, edges = triangulation_with_torch(part_nodes[i], part_nodes[j], 5)
        model.add_beam(Beam(points, edges))

    model.add_joint(Joint(model.beams[0], model.beams[1], pivot=part_nodes[1]))
    model.add_joint(Joint(model.beams[1], model.beams[2], pivot=part_nodes[2]))
    model.add_joint(Joint(model.beams[2], model.beams[0], pivot=part_nodes[0]))


def select_points(points, near):
    dist = (points - near).norm(dim=1)
    sorted_indices = torch.argsort(dist)[:2]
    return sorted_indices


def _decide_geometry(assembly_nodes, topo):
    point_map = {}
    for i, j in topo:
        points, _ = part_discretization(assembly_nodes[i], assembly_nodes[j])
        points, _ = triangulation_with_torch()
        point_map[(i, j)] = points

    return point_map

def _decide_edges(assembly_nodes, topo, point_map):
    offset = 0
    edge_map = {}
    for index, (i, j) in enumerate(topo):
        points = point_map[(i, j)]
        _, edges = part_discretization(assembly_nodes[i], assembly_nodes[j])
        edge_map[(i, j)] = edges + offset
        offset += len(points)

    for index, (i, j) in enumerate(topo):
        points


def decide_edges():
    vertex_map = {
        "a": vertex_a,
        "b": vertex_b,
        "c": vertex_c,
    }
    point_map = {}
    edge_map = {}

    point_map["ab"], part_ab_edges = part_discretization(vertex_map['a'], vertex_map['b'], thickness=0.2)
    edge_map["ab"] = part_ab_edges * 0 * len(point_map["ab"])
    point_map["bc"], part_bc_edges = part_discretization(vertex_map['b'], vertex_map['c'], thickness=0.2)
    edge_map["bc"] = part_ab_edges * 1 * len(point_map["ab"])
    point_map["ca"], part_ca_edges = part_discretization(vertex_map['c'], vertex_map['a'], thickness=0.2)
    edge_map["ca"] = part_ab_edges * 2 * len(point_map["ab"])

    point_per_part = len(point_map["ab"])

    from itertools import product
    indices_ab_a = select_points(point_map["ab"], near=vertex_map["a"])
    indices_ab_b = select_points(point_map["ab"], near=vertex_map["b"]) + point_per_part

    indices_bc_b = select_points(point_map["bc"], near=vertex_map["b"])
    indices_bc_c = select_points(point_map["bc"], near=vertex_map["c"]) + 2 * point_per_part

    indices_ca_c = select_points(point_map["ca"], near=vertex_map["c"])
    indices_ca_a = select_points(point_map["ca"], near=vertex_map["a"])

    edges = torch.vstack([
        part_ab_edges,
        part_bc_edges + point_per_part,
        part_ca_edges + point_per_part * 2,
        torch.vstack([
            torch.tensor(list(product(indices_ab_a, indices_ab_b))),
            torch.tensor(list(product(indices_bc_b, indices_bc_c))),
            torch.tensor(list(product(indices_ca_c, indices_ca_a))),
        ])
    ])
    return edges


def decide_geometry(vertex_map):
    point_map = {}

    point_map["ab"], _ = part_discretization(vertex_map['a'], vertex_map['b'], thickness=0.2)
    point_map["bc"], _ = part_discretization(vertex_map['b'], vertex_map['c'], thickness=0.2)
    point_map["ca"], _ = part_discretization(vertex_map['c'], vertex_map['a'], thickness=0.2)

    points = torch.vstack([point_map[ind] for ind in ("ab", "bc", "ca")])
    return points


with torch.no_grad():
    vertex_map = {
        "a": vertex_a,
        "b": vertex_b,
        "c": vertex_c,
    }
    edges = decide_edges()
    init_peripheral = sum([(vertex_map[p] - vertex_map[q]).norm() for p, q in ("ab", "bc", "ca")])

n_iters = 50000
optimizer = Adam(params=[vertex_map["c"]], lr=0.05)

traces = []

for it in tqdm(range(n_iters)):
    optimizer.zero_grad()

    vertex_map = {
        "a": vertex_a,
        "b": vertex_b,
        "c": vertex_c,
    }
    points = decide_geometry(vertex_map)

    K = gradient.spring_energy_matrix(points, edges, dim=2)
    eigenvalues, eigenvectors = torch.symeig(K, eigenvectors=True)
    indices = torch.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[indices]
    sorted_eigenvectors = eigenvectors[indices]

    smallest_nonzero_eigenval = sorted_eigenvalues[3]
    assert not torch.allclose(smallest_nonzero_eigenval, torch.tensor(0.0, dtype=torch.double)), f"more than 3 dof {sorted_eigenvalues}"

    visualize_2D(points.detach().numpy(), edges, eigenvectors[3].detach().numpy().reshape(-1, 2))
    print(eigenvectors[3].detach().numpy().reshape(-1, 2).shape)

    vertices = torch.vstack([vertex_a, vertex_b, vertex_c])

    # peripheral = (vertex_a - vertex_b).norm() + (vertex_b - vertex_c).norm() + (vertex_c - vertex_a).norm()
    # peripheral_penalty = 5 * torch.pow(peripheral - init_peripheral, 2)
    peripheral_penalty = 0
    # Negate eigenvalue in the objective as we're trying to increase it
    objective = -smallest_nonzero_eigenval + peripheral_penalty
    objective.backward()

    optimizer.step()

    trace = {
        "eigenvalue": smallest_nonzero_eigenval.detach().cpu().numpy(),
        "vertices": vertices.detach().cpu().numpy(),
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

for it in range(3):
    trace = traces[it * 300]
    vertices = trace["vertices"]
    plot_shape(ax, vertices, np.array([[0, 1], [1, 2], [2, 0]]))

plt.show()


