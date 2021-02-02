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
from solvers.rigidity_solver.eigen_analysis import eigen_analysis

from visualization.model_visualizer import visualize_3D

from tests.testcases import tetra
from matplotlib import pyplot as plt


axes = np.array([
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
])

model = tetra.tetrahedron(
    np.array([
        [0, 0, 0],
        [1 / 2, np.sqrt(3) / 2, 0],
        [1, 0, 0],
        [1 / 2, np.sqrt(3) / 6, np.sqrt(6) / 3],
    ]),
    axes
)
pairs = eigen_analysis(model.point_matrix(), model.edge_matrix(), model.constraint_matrix())

zero_eigenvalues = [e for e, v in pairs if e < 1e-8]
print("DoF:", len(zero_eigenvalues))
print("Least Eigenvalue:", pairs[6][0])

objectives = []
y_range = np.linspace(-1.5, 1.5, num=50)
z_range = np.linspace(-1.5, 1.5, num=50)
from itertools import product

yz_range = product(y_range, z_range)

for it, (y, z) in enumerate(tqdm(yz_range)):
    vertices = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1 - y, y, z],
    ])
    # vertices = np.array([
    #     [0, 0, 0],
    #     [1 / 2, np.sqrt(3) / 2, 0],
    #     [1, 0, 0],
    #     [1 / 2, y, z],
    # ])

    model = tetra.tetrahedron(vertices=vertices, axes=axes)

    points = model.point_matrix()
    extra_constraints = geo_util.trivial_basis(points, 3)

    pairs = eigen_analysis(
        points,
        model.edge_matrix(),
        np.vstack((
            model.constraint_matrix(),
            extra_constraints,
        ))
    )

    seventh_eig, eigv = pairs[0]

    objectives.append(seventh_eig)

Z = np.array(objectives).reshape(len(y_range), len(z_range)).transpose()
ax = plt.gca()
ax.set_aspect('equal')
cs = plt.contour(y_range, z_range, Z, levels=50)
fig = plt.gcf()
fig.colorbar(cs)

plt.title("On plane x + y = 1")
plt.xlabel("y axis")
plt.ylabel("z axis")
z_ind, y_ind = np.unravel_index(np.argmax(Z), Z.shape)
plt.scatter([y_range[y_ind]], [z_range[z_ind]])
plt.show()
