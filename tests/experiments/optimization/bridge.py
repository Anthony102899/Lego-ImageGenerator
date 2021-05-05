# %%
import sys
sys.path.append("../../..")

from tqdm import tqdm
import torch
from torch.optim import Adam
import numpy as np

from solvers.rigidity_solver import gradient
from solvers.rigidity_solver import algo_core as core
from util import geometry_util

init_vertices = np.array([
    [0, 0], [1, 0], [2, 0], [4, 0], [5, 0], [6, 0],
    [1, 1], [2, 2], [3, 3], [4, 2], [5, 1]
], dtype=np.double)

init_vertices -= np.mean(init_vertices, axis=0)

optimizable_vertices = torch.tensor(
    init_vertices[np.array([6, 7, 8, 9, 10])], dtype=torch.double, requires_grad=True)
fixed_vertices = torch.tensor(
    init_vertices[np.array([0, 1, 2, 3, 4, 5])], dtype=torch.double
)

fixed_points_index = torch.tensor([0, 5], dtype=torch.long)
edges = torch.tensor(
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 5), (1, 6), (2, 7),
     (3, 9), (4, 10), (2, 6), (2, 8), (3, 8), (3, 10)],
    dtype=torch.long)

n_iters = 3000
optimizer = Adam(params=[optimizable_vertices])

traces = []

for it in tqdm(range(n_iters)):
    optimizer.zero_grad()

    vertices = torch.vstack([
        fixed_vertices,
        optimizable_vertices,
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

    grad = optimizable_vertices.grad.data

    optimizer.step()

    trace = {
        "eigenvalue": smallest_nonzero_eigenval.detach().cpu().numpy(),
        "vertices": vertices.detach().cpu().numpy(),
        "grad": np.copy(grad.detach().cpu().numpy()),
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
def plot_shape(ax, vertices, edges, **style):
    for a, b in edges:
        p, q = vertices[a], vertices[b]
        ax.plot([p[0], q[0]], [p[1], q[1]], **style)

plt.clf()
fig, ax = plt.subplots()
ax.set(xlim=(0, 1.5), ylim=(0, 2))
ax.axis('equal')

points = np.vstack([t["vertices"] for t in traces])
points_x = points[:, 0]
points_y = points[:, 1]
ax.scatter(points_x, points_y, s=1, c='red')

# for it in range(n_iters):
#     trace = traces[it]
#     vertices = trace["vertices"]
#     points_x = vertices[:, 0]
#     points_y = vertices[:, 1]
#     ax.scatter(points_x, points_y, s=1, c=[it / n_iters, 0, 0])


for it in np.round(np.linspace(0, n_iters - 1, 2)).astype(np.int):
    trace = traces[it]
    vertices = trace["vertices"]
    plot_shape(ax, vertices, edges, color=np.array([it / n_iters, 0, 0]))

plt.show()
