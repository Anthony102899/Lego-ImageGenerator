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
from solvers.rigidity_solver.gradient import gradient_analysis
from solvers.rigidity_solver.internal_structure import tetrahedron, triangulation_with_torch
from solvers.rigidity_solver.algo_core import solve_rigidity, spring_energy_matrix
from solvers.rigidity_solver.models import Beam, Model, Joint
from solvers.rigidity_solver import gradient, algo_core as core, extra_constraint
from solvers.rigidity_solver.eigen_analysis import eigen_analysis

from visualization.model_visualizer import visualize_3D, visualize_2D

from matplotlib import pyplot as plt

#%%
objectives = []

part_nodes = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 2, 0],
]) * 5

optimizable_ind = np.array([2, ])
optimizable_nodes = torch.tensor(
    part_nodes[optimizable_ind],
    dtype=torch.double, requires_grad=True)
fixed_nodes = torch.tensor(
    np.delete(part_nodes, optimizable_ind, axis=0),
    dtype=torch.double)

part_node_connectivity = np.array([
    [0, 1],
    [1, 2],
    [2, 0],
])

def model_info(part_nodes):
    model = Model()
    with torch.no_grad():
        for i, j in part_node_connectivity:
            points, edges = triangulation_with_torch(part_nodes[i], part_nodes[j], 10, thickness=0.3)
            model.add_beam(Beam(points.detach().numpy(), edges.detach().numpy()))

        model.add_joint(Joint(model.beams[0], model.beams[1], pivot=part_nodes[1]))
        model.add_joint(Joint(model.beams[1], model.beams[2], pivot=part_nodes[2]))
        model.add_joint(Joint(model.beams[2], model.beams[0], pivot=part_nodes[0]))

    points = torch.vstack(
        [triangulation_with_torch(part_nodes[i], part_nodes[j], 10, thickness=0.3)[0] for i, j in part_node_connectivity]
    )

    return points, model


n_iters = 1000
optimizer = Adam(params=[optimizable_nodes], lr=0.01)

traces = []

_, model = model_info(torch.vstack([fixed_nodes, optimizable_nodes]))
edges = torch.tensor(model.edge_matrix(), dtype=torch.long)

for it in tqdm(range(n_iters)):
    optimizer.zero_grad()

    nodes = torch.vstack([fixed_nodes, optimizable_nodes])

    points, model = model_info(nodes)

    extra_constraints = np.vstack([
        extra_constraint.z_static(len(points)),
        extra_constraint.trivial_basis(points.detach().numpy(), dim=3),
    ])

    constraints = np.vstack([
        model.constraint_matrix(),
        extra_constraints,
    ])
    np_B = null_space(constraints)
    B = torch.tensor(np_B, dtype=torch.double)

    K = gradient.spring_energy_matrix(points, edges, dim=3)

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
    visualize_2D(trace["points"], edges, trace["eigenvector"].reshape(-1, 3)[:, :2])
