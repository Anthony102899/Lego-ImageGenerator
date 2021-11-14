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

from tests.testcases import tetra
from matplotlib import pyplot as plt


model = tetra.tetrahedron(
    np.array([
        [0, 0, 0],
        [1 / 2, np.sqrt(3) / 2, 0],
        [1, 0, 0],
        [1 / 2, np.sqrt(3) / 6, np.sqrt(6) / 3],
    ]),
    np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, np.sqrt(3) / np.sqrt(6), 1]
    ])
)

pairs = eigen_analysis(
    model.point_matrix(),
    model.edge_matrix(),
    model.constraint_matrix()
)

e, v = pairs[6]
print(e)
model.visualize(arrows=v)

