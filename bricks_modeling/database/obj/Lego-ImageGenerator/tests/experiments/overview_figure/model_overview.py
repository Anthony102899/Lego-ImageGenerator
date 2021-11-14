from solvers.rigidity_solver.models import *
import json
import numpy as np

p = lambda x, y: np.array([x, y, 0], dtype=np.double) * 20
v = lambda x, y, z: np.array([x, y, z], dtype=np.double)
lerp = lambda p, q, w: p * (1 - w) + q * w

def define_from_file():
    scale = 0.1 * np.array([1, -1, 1])

    def beam_from_file(filename):
        with open(filename) as fp:
            obj = json.load(fp)
        return Beam(
            np.hstack((
                np.array(obj["points"]),
                np.zeros((len(obj["points"]), 1))
            )) * scale,
            np.array(obj["edges"]),
        )

    p = lambda x, y: np.array([x, y, 0], dtype=np.double) * scale
    model = Model()
    names = ("base", "base-support", "lower", "main", "support-top-left", "support-top-right")
    _bmap = {
        name: beam_from_file(f"definition/points-edges-{name}.json")
        for name in names
    }
    joints = [
        Joint(_bmap["base"], _bmap["main"], pivot=p(261, 432), rotation_axes=v(0, 0, 1)),
        Joint(_bmap["base"], _bmap["base-support"], pivot=p(303, 581), rotation_axes=v(0, 0, 1)),
        Joint(_bmap["base-support"], _bmap["main"], pivot=p(413, 273), rotation_axes=v(0, 0, 1)),
        Joint(_bmap["main"], _bmap["support-top-left"], pivot=p(500, 184), rotation_axes=v(0, 0, 1)),
        Joint(_bmap["main"], _bmap["lower"], pivot=p(1060, 167), rotation_axes=v(0, 0, 1)),
        Joint(_bmap["support-top-left"], _bmap["support-top-right"], pivot=p(866, 83), translation_vectors=geo_util.normalize(p(567, -157))),
        Joint(_bmap["support-top-right"], _bmap["lower"], pivot=p(1067, 27), rotation_axes=v(0, 0, 1)),
    ]

    beams = list(_bmap.values())
    model.add_beams(beams)
    model.add_joints(joints)

    return locals()


def define():

    def init_beam(p, q, density=0.2, thickness=2):
        return Beam.tetra(p, q, thickness=thickness, density=density, ori=v(0, 0, 1))

    model = Model()
    _bmap = {
        "base": init_beam(p(-1.2, -0.2), p(-0.8, -1.8)),
        "upper-main-left": init_beam(p(-1, -0.8), p(0, 0)),
        "upper-main-right": init_beam(p(0, 0), p(3.2, 0.5), thickness=3),
        "upper-support-left": init_beam(p(0.5, 0.5), p(2.3, 1.2)),
        "upper-support-right": init_beam(p(2.3, 1.2), p(3.2, 1.55), density=0.3),
        "base-support": init_beam(p(-0.8, -1.8), p(0, 0)),
        "lower": init_beam(p(3.2, 1.45), p(3., -2), thickness=3),
        # "lower-support-up": init_beam(p(5, 3), p(6, 0)),
        # "lower-support-mid": init_beam(p(5, 1), p(6, 0)),
        # "flop": init_beam(p(5, 0), p(4, -1)),
    }

    ax_z = v(0, 0, 1)
    joints = [
        Joint(_bmap["upper-main-right"], _bmap["upper-support-left"], pivot=p(0.5, 0.5), rotation_axes=ax_z),
        Joint(_bmap["upper-support-left"], _bmap["upper-support-right"], pivot=p(2.3, 1.2), translation_vectors=v(1.8, 0.7, 0)),
        Joint(_bmap["upper-support-right"], _bmap["lower"], pivot=p(3.2, 1.55), rotation_axes=ax_z),
        Joint(_bmap["upper-main-right"], _bmap["lower"], pivot=p(3.0, 0.4), rotation_axes=ax_z),
        Joint(_bmap["base"], _bmap["upper-main-left"], pivot=p(-1, -0.8), rotation_axes=ax_z),
        Joint(_bmap["base"], _bmap["base-support"], pivot=p(-0.8, -1.8), rotation_axes=ax_z),
        Joint(_bmap["upper-main-left"], _bmap["base-support"], pivot=p(0, 0), rotation_axes=ax_z),
        Joint(_bmap["upper-main-right"], _bmap["upper-main-right"], pivot=p(0, 0))
        # Joint(_bmap["lower"], _bmap["flop"], pivot=p(5, 0), rotation_axes=None),
    ]
    beams = list(_bmap.values())
    model.add_beams(beams)
    model.add_joints(joints)

    return locals()
