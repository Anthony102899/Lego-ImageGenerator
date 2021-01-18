#%%
import sys
sys.path.append("..")

import scipy
import numpy as np
from numpy.linalg import matrix_rank, matrix_power, cholesky, inv
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import util.geometry_util as geo_util

from solvers.rigidity_solver.gradient import gradient_analysis
from solvers.rigidity_solver.internal_structure import tetrahedralize
from solvers.rigidity_solver.algo_core import solve_rigidity, spring_energy_matrix
from solvers.rigidity_solver.joints import Beam, Model, Hinge
from solvers.rigidity_solver import gradient as gd

from visualization.model_visualizer import visualize_3D

import testcases
#%%

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(npoints, ndim)
    vec = geo_util.rowwise_normalize(vec)
    return vec


axes = np.array([
        [-1, 1, 2],
        [-1, -1, 2],
        [1, -1, 2],
        [1, -1, 2],
    ])


objectives = []

from testcases import tetra
# if True:
axes_string = "c".join([str(n) for n in axes.reshape(-1)])
model = tetra.square(axes)

points = model.point_matrix()
edges = model.edge_matrix()
A = model.constraint_matrix()

hinges = model.joints
hinge_axes = np.array([h.axis for h in hinges])
hinge_pivots = np.array([h.pivot_point for h in hinges])
hinge_point_indices = model.joint_point_indices()

extra_constraints = np.zeros((len(model.beams[0].points) * 3, points.shape[0] * 3))
for r, c in enumerate(range(len(model.beams[0].points) * 3)):
    extra_constraints[r, c] = 1
trivial_motions = geo_util.trivial_basis(points, dim=3)
extra_constraints = trivial_motions
A = np.vstack([A, extra_constraints])

M = spring_energy_matrix(points, edges, dim=3)

# mathematical computation
B = scipy.linalg.null_space(A)
T = np.transpose(B) @ B
S = B.T @ M @ B

L = cholesky(T)
L_inv = inv(L)

Q = np.linalg.multi_dot([L_inv.T, S, L_inv])

pairs = geo_util.eigen(Q, symmetric=True)
eigenvalues = np.array([v for v, e in pairs])
np_obj, eigenvector = pairs[0]
arrows = B @ eigenvector

print("DoF:", np.sum(np.isclose(eigenvalues, 0)))
print("numpy obj:", np_obj)

hinge_rad = torch.tensor(geo_util.cart2sphere(hinge_axes)[:, 1:], requires_grad=True)
theta, phi = hinge_rad[:, 0], hinge_rad[:, 1]
points = torch.tensor(points, dtype=torch.double, requires_grad=True)
edges = torch.tensor(edges)
# hinge_axes = torch.tensor(hinge_axes, dtype=torch.double, requires_grad=True)
hinge_pivots = torch.tensor(hinge_pivots, dtype=torch.double, requires_grad=True)
hinge_point_indices = torch.tensor(hinge_point_indices, dtype=torch.long)

step_size = 5e-4
iters = 2000

optimizer = torch.optim.Adam([hinge_rad], lr=step_size)
writer = SummaryWriter(comment=axes_string)
writer.add_text("Adam", f"Step size: {step_size}")

for it in tqdm(range(iters)):
    optimizer.zero_grad()

    hinge_axes = torch.hstack([
        torch.unsqueeze(torch.sin(theta) * torch.cos(phi), 1),
        torch.unsqueeze(torch.sin(theta) * torch.sin(phi), 1),
        torch.unsqueeze(torch.cos(theta), 1)
    ])

    # Negate it as torch optimizer minimizes objective
    eigenvalues, eigenvectors = gd.differentiable_eigen(
        points, edges,
        hinge_axes, hinge_pivots, hinge_point_indices,
        # extra_constraints
    )

    obj = -eigenvalues[7]

    axes_cart = hinge_axes.detach().numpy()
    writer.add_scalar("Objective/Least Eigenvalue", -obj.detach().numpy(), it)
    for i, cartesian in enumerate(axes_cart):
        writer.add_scalars(f"Params/Hinge {i}", {
            "x": cartesian[0],
            "y": cartesian[1],
            "z": cartesian[2],
        }, it)

    obj.backward()
    optimizer.step()

    if it == 0:
        print(obj)

    if it % 500 == 500 - 1:
        print("obj", obj, "init:", axes, "opt:", axes_cart)


# %%
