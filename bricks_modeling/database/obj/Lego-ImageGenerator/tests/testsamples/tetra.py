import numpy as np
from solvers.rigidity_solver.models import Model, Joint, Beam
from itertools import combinations


def beam():
    model = Model()
    beam = Beam.tetra(np.zeros(3), np.ones(3) * 250, thickness=20)
    model.add_beam(beam)
    return model

def square(axes):
    model = Model()

    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ]) - np.array([0.5, 0.5, 0])
    points *= 240

    beams = [
        Beam.tetra(points[i], points[(i + 1) % 4], thickness=20, ori=np.array([0, 0, 1])) for i in range(4)
    ]

    hinges = [
        Joint(beams[i], beams[(i + 1) % 4], rotation_axes=[ax], pivot=points[(i + 1) % 4])
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
        [-1, 1, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
    ])
    return square(axes)

def square_centering_axes():
    axes = np.array([
        [-1, 1, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
    ])
    return square(axes)


def square_pyramid_axes():
    axes = np.array([
        [-1, 1, 1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
    ])
    return square(axes)

def square_closed_axes():
    axes = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
    ])
    return square(axes)

def triangle(vertices, axes, scale=10):
    model = Model()
    vertices = vertices.copy() * scale
    beams = [
        Beam.tetra(vertices[i], vertices[(i + 1) % 3], thickness=1.5, ori=np.array([0, 0, 1]))
        for i in range(3)
    ]
    hinges = [Joint(beams[i], beams[(i + 1) % 3], pivot=vertices[(i + 1) % 3], rotation_axes=axes[i]) for i in range(3)]

    model.add_beams(beams)
    model.add_joints(hinges)

    return model


def equilateral_triangle(axes):
    rt3o2 = np.sqrt(3) / 2
    vertices = np.array([
        [-rt3o2, -0.5, 0],
        [rt3o2, -0.5, 0],
        [0, 1, 0],
    ])
    return triangle(vertices, axes)


def _hinge3(beams, pivot, axis):
    assert len(beams) == 3
    return (Joint(b1, b2, pivot_point=pivot, axis=axis)
            for b1, b2 in combinations(beams, 2))


def tetrahedron(vertices, axes, scale=240):
    model = Model()
    vertices = vertices.copy() * scale

    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]

    beams = [
        Beam.tetra(vertices[p], vertices[q], thickness=20, ori=np.array([0, 0, 1]))
        for p, q in edges
    ]

    hinges = [
        *_hinge3([beams[2], beams[0], beams[3]], pivot=vertices[0], axis=axes[0]),
        *_hinge3([beams[0], beams[1], beams[4]], pivot=vertices[1], axis=axes[1]),
        *_hinge3([beams[1], beams[2], beams[5]], pivot=vertices[2], axis=axes[2]),
        *_hinge3([beams[3], beams[4], beams[5]], pivot=vertices[3], axis=axes[3]),
    ]

    model.add_beams(beams)
    model.add_joints(hinges)

    return model


def single_beam(scale=240):
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
    ]) * scale
    model = Model()
    model.add_beam(Beam.tetra(vertices[0], vertices[1], thickness=20))

    return model


def table_slider():
    pts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [0, -1, 0],
        [0, -2, 0],
        [0, -0., 0],
        [0.5, -0.5, 0],
    ]) * 100
    beams = [
        Beam.tetra(pts[0], pts[4]),
        Beam.tetra(pts[0], pts[2]),
        Beam.tetra(pts[1], pts[3]),
        Beam.tetra(pts[5], pts[6]),
    ]
    joints = [
        Joint(beams[0], beams[1], pts[0], rotation_axes=np.array([0, 0, 1])),
        Joint(beams[1], beams[2], pts[1], rotation_axes=np.array([0, 0, 1])),
        Joint(beams[0], beams[2], pts[3], rotation_axes=np.array([0, 0, 1]), translation_vectors=np.array([0, 1, 0])),
        Joint(beams[0], beams[3], pts[5], rotation_axes=np.array([0, 0, 1])),
        Joint(beams[2], beams[3], pts[6]),
    ]
    model = Model()
    model.add_beams(beams)
    model.add_joints(joints)

    return model
