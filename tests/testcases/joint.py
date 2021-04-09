import numpy as np
from solvers.rigidity_solver.models import Model, Joint, Beam

_beams = [
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
_beams = [
    Beam.tetra(
        np.array([0, 0, 1]),
        np.array([0, 0, 2]),
    ),
    Beam.tetra(
        np.array([0, 1, 0]),
        np.array([0, 2, 0]),
    ),
]
_pivot = np.array([0, 0, 0])


def hinge():
    model = Model()
    model.add_beams(_beams)
    model.add_joint(Joint(
        _beams[0], _beams[1],
        _pivot,
        rotation_axes=np.array([0, 0, 1])
    ))
    return model


def prismatic():
    model = Model()
    model.add_beams(_beams)
    model.add_joint(Joint(
        _beams[0], _beams[1], _pivot,
        translation_vectors=np.array([0, 0, 1])
    ))
    return model


def cylindrical():
    model = Model()
    model.add_beams(_beams)
    model.add_joint(Joint(
        _beams[0], _beams[1], _pivot,
        translation_vectors=np.array([0, 0, 1]),
        rotation_axes=np.array([0, 0, 1])
    ))
    return model


def ball():
    model = Model()
    model.add_beams(_beams)
    model.add_joint(Joint(
        _beams[0], _beams[1], _pivot,
        rotation_axes=np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
    ))
    return model


def ball_planar():
    model = Model()
    model.add_beams(_beams)
    model.add_joint(Joint(
        _beams[0], _beams[1], _pivot,
        rotation_axes=np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]),
        translation_vectors=np.array([
            [1, 0, 0],
            [0, 0, 1],
        ])
    ))
    return model

def universal():
    model = Model()
    model.add_beams(_beams)
    model.add_joint(Joint(
        _beams[0], _beams[1], _pivot,
        rotation_axes=np.array([
            [1, 1, 0],
            [0, 1, 0],
        ]),
    ))
    return model
