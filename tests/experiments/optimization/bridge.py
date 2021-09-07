# %%
import sys
sys.path.append("../../..")

from tqdm import tqdm
import torch
from torch.optim import Adam
import numpy as np
from scipy.linalg import null_space

from solvers.rigidity_solver import (
    algo_core as core,
    gradient,
    extra_constraint
)
from solvers.rigidity_solver.models import Model, Joint, Beam
from solvers.rigidity_solver.internal_structure import triangulation_with_torch
from visualization.model_visualizer import visualize_2D
from util import geometry_util
from util.timer import SimpleTimer

root3 = np.sqrt(3)
part_nodes = np.array([
    [0, 0], [1, 0], [2, 0], [3, 0],
    [1 / 2, root3 / 2], [3 / 2, root3 / 2], [5 / 2, root3 / 2],
], dtype=np.double) * 5
part_nodes = np.hstack((
    part_nodes, np.zeros((len(part_nodes), 1))
))

part_nodes -= np.mean(part_nodes, axis=0)

optimzable_ind = np.array([4, 5, 6])

optimizable_nodes = torch.tensor(
    part_nodes[optimzable_ind], dtype=torch.double, requires_grad=True)
fixed_nodes = torch.tensor(
    np.delete(part_nodes, optimzable_ind, axis=0), dtype=torch.double
)

part_node_connectivity = torch.tensor(
    [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6),
     (0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6)],
    dtype=torch.long)

def model_info(part_nodes):
    model = Model()
    seg_num = 3
    with torch.no_grad():
        for i, j in part_node_connectivity:
            points2d, edges = triangulation_with_torch(part_nodes[i][:2], part_nodes[j][:2], seg_num, thickness=0.3)
            points = torch.hstack((points2d, torch.zeros(points2d.size()[0], 1)))
            model.add_beam(Beam(points.detach().numpy(), edges.detach().numpy()))

        # pivot index, part_1 index, part_2 index
        joints_info = [
            (0, 5, 0),
            (0, 6, 1),
            (6, 7, 1),
            (7, 1, 1),
            (1, 8, 2),
            (8, 9, 2),
            (9, 2, 2),
            (2, 10, 3),
            (5, 6, 4),
            (6, 3, 4),
            (3, 7, 5),
            (7, 8, 5),
            (8, 4, 5),
            (4, 9, 6),
            (9, 10, 6),
        ]
        for i, j, n in joints_info:
            model.add_joint(Joint(model.beams[i], model.beams[j], pivot=part_nodes[n]))

    points2d = torch.vstack(
        [triangulation_with_torch(part_nodes[i][:2], part_nodes[j][:2], seg_num, thickness=0.3)[0] for i, j in part_node_connectivity]
    )
    points = torch.hstack((points2d, torch.zeros(points2d.size()[0], 1)))

    return points, model

_, model = model_info(torch.vstack([fixed_nodes, optimizable_nodes]))
points = model.point_matrix()
edges = torch.tensor(model.edge_matrix()).long()

pairs = model.eigen_solve(num_pairs=1, extra_constr=np.vstack((
    extra_constraint.z_static(len(points)),
    extra_constraint.trivial_basis(points, dim=3),
)), verbose=True)
eigenvector = pairs[0][1]
import matplotlib.pyplot as plt
plt.axis("equal")
plt.axis("off")
for pt, v in zip(points, eigenvector.reshape(-1, 3)):
    v *= 1.8
    plt.arrow(pt[0], pt[1], v[0], v[1], color=(255 / 255, 165 / 255, 0), width=0.05)
plt.savefig("bridge-load.svg", transparent=True)

n_iters = 500
optimizer = Adam(params=[optimizable_nodes], lr=0.01)

traces = []

for it in tqdm(range(n_iters)):
    optimizer.zero_grad()

    nodes = torch.vstack([
        fixed_nodes,
        optimizable_nodes,
    ])

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

    assert torch.allclose(eigenvalues[:eigind], torch.tensor(0.0, dtype=torch.double)), f"more than expected dof {eigenvalues}"
    assert not torch.allclose(eigenvalues[eigind], torch.tensor(0.0, dtype=torch.double)), f"more than expected num dof: {eigenvalues}"

    # Negate the objective as we're trying to increase it
    objective = -smallest_eigenvalue
    objective.backward()

    grad = optimizable_nodes.grad.data

    optimizer.step()

    trace = {
        "eigenvalue": smallest_eigenvalue.detach().cpu().numpy(),
        "eigenvector": corresponding_eigenvector.detach().cpu().numpy(),
        "vertices": nodes.detach().cpu().numpy(),
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
    plot_shape(ax, vertices, part_node_connectivity, color=np.array([it / n_iters, 0, 0]))

plt.show()
