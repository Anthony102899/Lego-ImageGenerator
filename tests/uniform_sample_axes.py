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
from solvers.rigidity_solver.eigen_analysis import eigen_analysis
from solvers.rigidity_solver.internal_structure import tetrahedralize
from solvers.rigidity_solver.algo_core import solve_rigidity, spring_energy_matrix
from solvers.rigidity_solver.joints import Beam, Model, Hinge
from solvers.rigidity_solver import gradient as gd

from visualization.model_visualizer import visualize_3D

from testcases import tetra, simple
from itertools import product

radians = np.linspace(0, 2, num=4, endpoint=False) * np.pi
axes_radians = product(radians, repeat=8)

trace = []

for rad in tqdm(axes_radians):
    axes_rad = np.fromiter(rad, np.double).reshape(-1, 2)
    axes = geo_util.unitsphere2cart(axes_rad)
    model = tetra.square(axes)
    points, edges = model.point_matrix(), model.edge_matrix()
    constraints = model.constraint_matrix()
    eigen_pairs = eigen_analysis(points, edges, constraints)
    objective, eigenvector = eigen_pairs[7]
    trace.append((objective, list(eigenvector)))

import pickle
with open("uniform.pickle", "w") as fp:
    pickle.dump(trace, fp)
