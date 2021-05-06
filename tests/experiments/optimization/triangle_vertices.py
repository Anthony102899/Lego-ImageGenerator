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
from solvers.rigidity_solver import gradient, algo_core as core
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
fixed_ind = np.arange(len(part_nodes))
optimizable_nodes = torch.tensor(part_nodes[optimizable_ind], dtype=torch.double, requires_grad=True)
fixed_nodes = torch.tensor(part_nodes[fixed_ind], dtype=torch.double)

part_node_connectivity = np.array([
    [0, 1],
    [1, 2],
    [2, 0],
])

def model_info(part_nodes, edges=None):
    model = Model()
    with torch.no_grad():
        for i, j in part_node_connectivity:
            points, edges = triangulation_with_torch(part_nodes[i], part_nodes[j], 5)
            model.add_beam(Beam(points.detach().numpy(), edges.detach().numpy()))

        model.add_joint(Joint(model.beams[0], model.beams[1], pivot=part_nodes[1]))
        model.add_joint(Joint(model.beams[1], model.beams[2], pivot=part_nodes[2]))
        model.add_joint(Joint(model.beams[2], model.beams[0], pivot=part_nodes[0]))

    points = torch.vstack(
        [triangulation_with_torch(part_nodes[i], part_nodes[j], 5)[0] for i, j in part_node_connectivity]
    )

    return points, model

def z_static(point_count):
    constr = np.zeros((point_count, point_count * 3))
    constr[np.arange(point_count), np.arange(point_count) * 3 + 2] = 1
    return constr

n_iters = 3
optimizer = Adam(params=[optimizable_nodes], lr=0.01)

traces = []

_, model = model_info(torch.vstack([fixed_nodes, optimizable_nodes]))
edges = torch.tensor(model.edge_matrix(), dtype=torch.long)

for it in tqdm(range(n_iters)):
    optimizer.zero_grad()

    nodes = torch.vstack([fixed_nodes, optimizable_nodes])

    points, model = model_info(nodes)

    extra_cosntraints = np.vstack([
        z_static(len(points)),
        geo_util.trivial_basis(points.detach().numpy(), dim=3),
    ])

    constraints = np.vstack([
        torch.tensor(model.constraint_matrix(), dtype=torch.double),
        torch.tensor(extra_cosntraints, dtype=torch.double),
    ])
    np_B = null_space(constraints)
    B = torch.tensor(np_B, dtype=torch.double)

    np_K = core.spring_energy_matrix_accelerate_3D(points.detach().numpy(), edges.numpy())
    K = gradient.spring_energy_matrix(points, edges, dim=3)

    Q = torch.chain_matmul(B.t(), K, B)

    # the eigenvalues are already in ascending order!
    eigenvalues, eigenvectors = torch.symeig(Q, eigenvectors=True)

    eigind = 0
    smallest_eigenvalue = eigenvalues[eigind]
    corresponding_eigenvector = torch.mv(B, eigenvectors[:, eigind])

    assert not torch.allclose(eigenvalues[eigind], torch.tensor(0.0, dtype=torch.double)), f"more than expected num dof: {eigenvalues}"

    # peripheral = (vertex_a - vertex_b).norm() + (vertex_b - vertex_c).norm() + (vertex_c - vertex_a).norm()
    # peripheral_penalty = 5 * torch.pow(peripheral - init_peripheral, 2)
    peripheral_penalty = 0
    # Negate eigenvalue in the objective as we're trying to increase it
    objective = -smallest_eigenvalue + peripheral_penalty
    objective.backward()

    print(objective)

    optimizer.step()

    trace = {
        "eigenvalue": smallest_eigenvalue.detach().cpu().numpy(),
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

for it in range(3):
    trace = traces[it * 300]
    vertices = trace["vertices"]
    plot_shape(ax, vertices, np.array([[0, 1], [1, 2], [2, 0]]))

plt.show()


