import scipy
import numpy as np
from numpy.linalg import matrix_rank, matrix_power, cholesky, inv
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import util.geometry_util as geo_util
from solvers.rigidity_solver.gradient import gradient_analysis
from solvers.rigidity_solver.internal_structure import tetrahedron
from solvers.rigidity_solver.algo_core import solve_rigidity, spring_energy_matrix
from solvers.rigidity_solver.models import Beam, Model, Joint
from solvers.rigidity_solver import gradient, algo_core as core
from solvers.rigidity_solver.eigen_analysis import eigen_analysis

from visualization.model_visualizer import visualize_3D

from tests.testsamples import tetra
from matplotlib import pyplot as plt

objectives = []
x_range = np.linspace(-2., 2., num=50)
y_range = np.linspace(-0.5, 1.5, num=50)
from itertools import product
xy_range = product(x_range, y_range)

rt3o2 = np.sqrt(3) / 2

fixed_vertices, movable_vertices = [
    torch.tensor([
        [0, 0],
        [1, 0],
    ], dtype=torch.double),
    torch.tensor([
        [1, 1],
        [2, 1],
    ], dtype=torch.double, requires_grad=True)
]

edges = torch.tensor([
    [0, 1],
    [1, 2],
    [2, 0],
    [1, 3],
    [2, 3]
], dtype=torch.long)

n_iters = 3000
optimizer = Adam(params=[movable_vertices])

traces = []

for it in tqdm(range(n_iters)):
    optimizer.zero_grad()

    vertices = torch.vstack([
        fixed_vertices, movable_vertices
    ])

    K = gradient.spring_energy_matrix(vertices, edges, dim=2)
    eigenvalues, eigenvectors = torch.symeig(K, eigenvectors=True)
    indices = torch.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[indices]
    sorted_eigenvectors = eigenvectors[indices]

    smallest_nonzero_index = 3
    smallest_nonzero_eigenval = sorted_eigenvalues[smallest_nonzero_index]
    assert torch.allclose(sorted_eigenvalues[:smallest_nonzero_index], torch.tensor(0.0, dtype=torch.double)), f"more than 3 dof {sorted_eigenvalues}"
    assert not torch.allclose(sorted_eigenvalues[smallest_nonzero_index], torch.tensor(0.0, dtype=torch.double)), f"more than 3 dof {sorted_eigenvalues}"

    # Negate the objective as we're trying to increase it
    (-smallest_nonzero_eigenval).backward()

    grad = movable_vertices.grad.data

    optimizer.step()

    trace = {
        "eigenvalue": smallest_nonzero_eigenval.detach().cpu().numpy(),
        "vertices": vertices.detach().cpu().numpy(),
        "grad": np.copy(grad.detach().cpu().numpy()),
    }
    traces.append(trace)

# visualize the optimization process
from matplotlib import pyplot as plt

# objective against time
# objectives = [t["eigenvalue"] for t in traces]
# plt.plot(np.arange(n_iters), objectives)
# plt.show()


# shape of the triangle against time
def plot_shape(ax, vertices, edges):
    for a, b in edges:
        p, q = vertices[a], vertices[b]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=[0, 1, 0])


for it in range(9):
    ax = plt.subplot(3, 3, it + 1)
    trace = traces[it * 350]

    vertices = trace["vertices"]
    grad = trace["grad"] / 5
    plot_shape(plt, vertices, edges)
    for v, g in zip(vertices[2:], grad):
        plt.arrow(
            v[0], v[1],
            g[0], g[1],
            length_includes_head=True,
            width=0.01,
        )
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1.5)
    ax.set_title(f"it={it * 350}, eigenval={trace['eigenvalue']}")

plt.show()

for it in range(9):
    ax = plt.subplot(3, 3, it + 1)
    trace = traces[it * 350]
    vertices = trace["vertices"]
    grad = trace["grad"] / 5

    for v, g in zip(vertices[2:], grad):
        ax.arrow(
            v[0], v[1],
            g[0], g[1],
            length_includes_head=True,
            width=0.01,
        )
    ax.set(xlim=(0, 2.5), ylim=(0, 1.5))

plt.show()


