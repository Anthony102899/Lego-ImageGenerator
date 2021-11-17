#%%
import sys
sys.path.append("..")

import scipy
from scipy.spatial import transform
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
from visualization.model_visualizer import visualize_3D, visualize_hinges

from visualization.model_visualizer import visualize_3D

import testcases
#%%

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(npoints, ndim)
    vec = geo_util.rowwise_normalize(vec)
    return vec

pyramid_axes = np.array([
    [-1, 1, 2],
    [-1, -1, 2],
    [1, -1, 2],
    [1, 1, 2],
])

axes_list = [
    pyramid_axes, pyramid_axes,
    pyramid_axes, pyramid_axes,
    pyramid_axes,
]


objectives = []

rotations = [
    transform.Rotation.from_rotvec(np.array([0, 0, 1]) * 0 * np.pi),
    transform.Rotation.from_rotvec(np.array([0, 0, 1]) * 0.5 * np.pi),
    transform.Rotation.from_rotvec(np.array([0, 0, 1]) * 1 * np.pi),
    transform.Rotation.from_rotvec(np.array([0, 0, 1]) * 1.5 * np.pi),
    transform.Rotation.from_rotvec(np.array([0, 0, 1]) * 2 * np.pi),
]

params_trace = []
gradients_trace = []

from testcases import tetra
for index, axes in enumerate(axes_list):
    print(f"""
    =====================================================
    =========================={index}==========================
    =====================================================
    """)
    model = tetra.square(axes)

    points = model.point_matrix()
    edges = model.edge_matrix()

    hinges = model.joints

    hinge_axes = np.array([h.axis for h in hinges])
    hinge_pivots = np.array([h.pivot_point for h in hinges])
    hinge_point_indices = model.joint_point_indices()

    rot = rotations[index]
    points = rot.apply(points)
    hinge_axes = rot.apply(hinge_axes)
    hinge_pivots = rot.apply(hinge_pivots)

    points = torch.from_numpy(points).double()
    edges = torch.from_numpy(edges).long()
    hinge_axes = torch.from_numpy(hinge_axes).double()
    hinge_pivots = torch.from_numpy(hinge_pivots).double()
    hinge_point_indices = torch.from_numpy(hinge_point_indices).long()

    # visualize_hinges(points, edges, hinge_pivots, hinge_axes)

    points.requires_grad_()

    eigenvalues, eigenvectors = gd.differentiable_eigen(points, edges, hinge_axes, hinge_pivots, hinge_point_indices)
    params_trace.append((points, edges, hinge_axes, hinge_pivots, hinge_point_indices))


    objective = -eigenvalues[7]
    objective.backward()

    gradients = points.grad.numpy().reshape(-1, 3)

    gradients_trace.append(gradients)

    visualize_3D(points.detach().numpy(), edges=edges.detach().numpy(), arrows=gradients * 1e6)

original_param = params_trace[0]
final_param = params_trace[-1]
for p, q in zip(original_param, final_param):
    print(torch.allclose(p, q))

print(np.allclose(gradients_trace[0], gradients_trace[-1]))
# %%
