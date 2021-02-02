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
])
rt3o2 = np.sqrt(3) / 2
vertices = np.array([
    [-rt3o2, -0.5, 0],
    [rt3o2, -0.5, 0],
    [0, 1, 0],
])

scale_range = np.linspace(200, 800, num=100)
objectives = []
nonfixed_objectives = []
for it, scale in enumerate(tqdm(scale_range)):
    model = tetra.triangle(vertices=vertices, axes=axes, scale=scale)

    points = model.point_matrix()

    try:
        fixed_pairs = eigen_analysis(
            model.point_matrix(),
            model.edge_matrix(),
            model.constraint_matrix(),
            fix_stiffness=True,
        )
        nonfixed_pairs = eigen_analysis(
            model.point_matrix(),
            model.edge_matrix(),
            model.constraint_matrix(),
            fix_stiffness=False,
        )
    except AssertionError:
        model.visualize()

    objectives.append(fixed_pairs[6][0])
    nonfixed_objectives.append(nonfixed_pairs[6][0])


plt.xlabel("scale")
plt.ylabel("7-th smallest eigenvalue")
plt.plot(scale_range, objectives, label="Fix stiffness")
plt.plot(scale_range, nonfixed_objectives, label="No fix stiffness")
plt.legend()
plt.show()
