import numpy as np
from solvers.rigidity_solver.joints import Model, Hinge, Beam

def sqaure():
    model = Model()

    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ]) * 150

    beams = [
        Beam.tetra(points[i], points[(i + 1) % 4], thickness=20) for i in range(4)
    ]

    model.add_beams(beams)
