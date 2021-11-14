import sys
sys.path.append("..")
import numpy as np
from tqdm import tqdm

from solvers.rigidity_solver.eigen_analysis import eigen_analysis

from tests.testcases import tetra
from matplotlib import pyplot as plt

scale_range = np.linspace(200, 800, num=100)
objectives = []
nonfixed_objectives = []
for it, scale in enumerate(tqdm(scale_range)):
    model = tetra.single_beam(scale=scale)

    points = model.point_matrix()

    try:
        pseudo_constraints = np.zeros((3, model.point_matrix().size))
        pseudo_constraints[:3, :3] = np.identity(3)
        fixed_pairs = eigen_analysis(
            model.point_matrix(),
            model.edge_matrix(),
            pseudo_constraints,
            fix_stiffness=True,
        )
        nonfixed_pairs = eigen_analysis(
            model.point_matrix(),
            model.edge_matrix(),
            pseudo_constraints,
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
