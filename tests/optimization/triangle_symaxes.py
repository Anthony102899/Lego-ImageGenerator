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

polar_range = np.linspace(0, 1, num=100, endpoint=False) * np.pi

objectives = []
for it, polar in enumerate(polar_range):
    axes = geo_util.unitsphere2cart(np.array([
        [polar, 5/6 * np.pi],
        [polar, 3/2 * np.pi],
        [polar, 1/6 * np.pi],
    ]))
    model = tetra.equilateral_triangle(axes)

    points = model.point_matrix()
    extra_constraints = np.hstack((
        np.identity(len(model.beams[0].points) * 3),
        np.zeros((len(model.beams[0].points) * 3, len(points) * 3 - len(model.beams[0].points) * 3))
    ))

    pairs = eigen_analysis(
        model.point_matrix(),
        model.edge_matrix(),
        np.vstack((
            model.constraint_matrix(),
            extra_constraints
        ))
    )

    seventh_eig, eigv = pairs[0]

    objectives.append(seventh_eig)

plt.xlabel("angle (deg)")
plt.ylabel("7-th smallest eigenvalue")
plt.plot(90 - polar_range / np.pi * 180, objectives)
plt.show()
