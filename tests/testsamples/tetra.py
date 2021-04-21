import numpy as np
from solvers.rigidity_solver.joints import Model, Hinge, Beam


def square(axes):
    model = Model()

    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ]) * 100

    beams = [
        Beam.tetra(points[i], points[(i + 1) % 4], thickness=20) for i in range(4)
    ]

    hinges = [
        Hinge(beams[i], beams[(i + 1) % 4], axis=ax, pivot_point=points[(i + 1) % 4])
        for i, ax in zip(range(4), axes)
    ]

    model.add_beams(beams)
    model.add_joints(hinges)

    return model


def square_perpendicular_axes():
    axes = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ])
    return square(axes)


def square_diverting_axes():
    axes = np.array([
        [1, 1, 0],
        [-1, 1, 0],
        [-1, 1, 0],
        [1, 1, 0],
    ])
    return square(axes)

def square_centering_axes():
    axes = np.array([
        [-1, 1, 0],
        [1, 1, 0],
        [-1, 1, 0],
        [1, 1, 0],
    ])
    return square(axes)


def square_pyramid_axes():
    axes = np.array([
        [-1, 1, 1],
        [1, 1, 1],
        [-1, 1, 1],
        [1, 1, 1],
    ])
    return square(axes)
