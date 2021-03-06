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
from solvers.rigidity_solver.internal_structure import tetrahedron
from solvers.rigidity_solver.algo_core import solve_rigidity, spring_energy_matrix
from solvers.rigidity_solver.models import Beam, Model, Joint
from solvers.rigidity_solver import gradient as gd
from solvers.rigidity_solver.eigen_analysis import eigen_analysis

from visualization.model_visualizer import visualize_3D

import testcases
#%%

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(npoints, ndim)
    vec = geo_util.rowwise_normalize(vec)
    return vec

axes = np.array([
        [1, 1, 0],
        [-1, -1, 1],
        [1, -1, 2],
        [1, 1, 3],
    ])


objectives = []

from testcases import tetra
# if True:
axes_string = "c".join([str(n) for n in axes.reshape(-1)])
# model = tetra.square(axes)

model = tetra.equilateral_triangle(np.array([
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
]))

points = model.point_matrix()
edges = model.edge_matrix()

hinges = model.joints
hinge_axes = np.array([h.axis for h in hinges])
hinge_pivots = np.array([h.pivot_point for h in hinges])
hinge_point_indices = model.joint_point_indices()

hinge_rad = torch.tensor(geo_util.cart2sphere(hinge_axes)[:, 1:], requires_grad=True)
theta, phi = hinge_rad[:, 0], hinge_rad[:, 1]
points = torch.tensor(points, dtype=torch.double)
edges = torch.tensor(edges)
# hinge_axes = torch.tensor(hinge_axes, dtype=torch.double, requires_grad=True)
hinge_pivots = torch.tensor(hinge_pivots, dtype=torch.double)
hinge_point_indices = torch.tensor(hinge_point_indices, dtype=torch.long)

# extra_constraints = torch.hstack((
#     torch.eye(len(model.beams[0].points) * 3),
#     torch.zeros((len(model.beams[0].points) * 3, len(points) * 3 - len(model.beams[0].points) * 3))
# ))
# extra_constraints = torch.from_numpy(geo_util.trivial_basis(points.detach().numpy(), dim=3)).double()
extra_constraints = None

choice = 6 if extra_constraints is None else 0

with torch.no_grad():
    hinge_axes = torch.hstack([
        torch.unsqueeze(torch.sin(theta) * torch.cos(phi), 1),
        torch.unsqueeze(torch.sin(theta) * torch.sin(phi), 1),
        torch.unsqueeze(torch.cos(theta), 1)
    ])
    pairs = eigen_analysis(
        points.detach().numpy(),
        edges.detach().numpy(),
        np.vstack((
            gd.constraint_matrix(points, hinge_axes, hinge_pivots, hinge_point_indices).detach().numpy(),
            # extra_constraints.detach().numpy(),
        ))
    )
eigenvalues = np.array([e for e, _ in pairs])
np_obj, eigenvector = pairs[choice]
arrows = eigenvector

print("DoF:", np.sum(np.abs(eigenvalues) < 1e-8))
print("numpy obj:", np_obj)

visualize_3D(points, edges=edges, arrows=arrows.reshape(-1, 3))

step_size = 5e-4
iters = 5000

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
        extra_constraints
    )

    obj = -eigenvalues[choice]

    axes_cart = hinge_axes.detach().numpy()
    # np_obj = eigen_analysis(
    #     points.detach().numpy(),
    #     edges.detach().numpy(),
    #     np.vstack((
    #         gd.constraint_matrix(points, hinge_axes, hinge_pivots, hinge_point_indices).detach().numpy(),
    #         extra_constraints.detach().numpy(),
    #     ))
    # )[choice][0]
    #
    writer.add_scalar("Objective/Least Eigenvalue (torch)", obj.detach().numpy(), it)
    # writer.add_scalar("Objective/Least Eigenvalue (numpy)", np_obj, it)

    for i, cartesian in enumerate(axes_cart):
        writer.add_scalars(f"Params/Hinge {i}", {
            "x": cartesian[0],
            "y": cartesian[1],
            "z": cartesian[2],
        }, it)

    obj.backward()
    optimizer.step()

    if np.any(np.isnan(hinge_rad.detach().numpy())):
        msg = f"NaN at {it} iterations"
        writer.add_text("ERROR", msg)
        print(msg)
        exit(1)

    if it <= 5:
        print(obj)
        print(hinge_rad.grad)

    if it % 500 == 500 - 1:
        print("obj", obj, "init:", axes, "opt:", axes_cart)


# %%
