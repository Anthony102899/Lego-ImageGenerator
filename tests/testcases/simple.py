import numpy as np
from solvers.rigidity_solver.joints import Model, Hinge, Beam

def case_hinge(ax):
    beams = [
        Beam(np.array([
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]) + np.array([1, 0, 0])),
        Beam(np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ]) + np.array([0, 1, 0])),
    ]
    pivot = np.array([0, 0, 0])
    hinges = [
        Hinge(beams[0], beams[1], axis=ax, pivot_point=pivot)
    ]
    model = Model()
    model.add_beams(beams)
    model.add_joints(hinges)

    return model

def hinge_with_perpendicular_axis():
    return case_hinge(np.array([0, 0, 1]))

def hinge_with_mid_axis():
    return case_hinge(np.array([1, 1, 0]))

def hinge_with_yaw_axis():
    return case_hinge(np.array([1, 1, 1]))
