from solvers.rigidity_solver.models import *
import numpy as np

p = lambda x, y, z: np.array([x, y, z], dtype=np.double)
v = lambda x, y, z: np.array([x, y, z], dtype=np.double)
lerp = lambda p, q, w: p * (1 - w) + q * w

def define(stage: int):
    assert 0 <= stage

    model = Model()
    density = 0.4

    _p = {
        "ul-b": p(0, 0, 0),  # up left back
        "dl-b": p(0, 0, -30),  # down left back
        "ur-b": p(30, 0, 0),  # up right back
        "dr-b": p(30, 0, -30),  # down right back
        "ul-f": p(0, 30, 0),  # up left front
        "dl-f": p(0, 30, -30),  # down left front
        "ur-f": p(30, 30, 0),  # up right front
        "dr-f": p(30, 30, -30),  # down right front
    }
    _p.update({
        "cross-b": p(15, 0, -30 / 4),
        "cross-f": p(15, 30, -30 / 4),

        "ul-ur-1/3-b": p(10, 0, 0),
        "ul-ur-2/3-b": p(20, 0, 0),

        "ul-ur-1/3-f": p(10, 30, 0),
        "ul-ur-2/3-f": p(20, 30, 0),
    })

    _bmap = {
        "top": Beam.tetra(p(0, 0, 0), p(30, 0, 0), density=density),
        "bottom": Beam.tetra(p(0, 0, -30), p(30, 0, -30), density=density),
        "oblique-1": Beam.tetra(p(10, 0, 0), p(30, 0, -30), density=density),
        "oblique-2": Beam.tetra(p(20, 0, 0), p(0, 0, -30), density=density),

        "top-f": Beam.tetra(p(0, 30, 0), p(30, 30, 0), density=density),
        "bottom-f": Beam.tetra(p(0, 30, -30), p(30, 30, -30), density=density),
        "oblique-1-f": Beam.tetra(p(10, 30, 0), p(30, 30, -30), density=density),
        "oblique-2-f": Beam.tetra(p(20, 30, 0), p(0, 30, -30), density=density),

        # "top-mid": Beam.tetra(p(0, 15, 0), p(30, 15, 0), density=density),
        "horizontal-top-1": Beam.tetra(p(0, 0, 0), p(0, 30, 0), density=density),
        # "horizontal-top-mid": Beam.tetra(p(15, 0, 0), p(15, 30, 0), density=density),
        "horizontal-top-2": Beam.tetra(p(30, 0, 0), p(30, 30, 0), density=density),
    }

    _ay = p(0, 1, 0)
    _ax = p(1, 0, 0)

    joints = [
        Joint(_bmap["top"], _bmap["oblique-1"], pivot=_p["ul-ur-1/3-b"], rotation_axes=_ay),
        Joint(_bmap["top"], _bmap["oblique-2"], pivot=_p["ul-ur-2/3-b"], rotation_axes=_ay),
        Joint(_bmap["bottom"], _bmap["oblique-1"], pivot=_p["dr-b"], rotation_axes=_ay),
        Joint(_bmap["bottom"], _bmap["oblique-2"], pivot=_p["dl-b"], rotation_axes=_ay),
        # Joint(_bmap["oblique-1"], _bmap["oblique-2"], pivot=_p["cross-b"], rotation_axes=_ay),

        Joint(_bmap["top-f"], _bmap["oblique-1-f"], pivot=_p["ul-ur-1/3-f"], rotation_axes=_ay),
        Joint(_bmap["top-f"], _bmap["oblique-2-f"], pivot=_p["ul-ur-2/3-f"], rotation_axes=_ay),
        Joint(_bmap["bottom-f"], _bmap["oblique-1-f"], pivot=_p["dr-f"], rotation_axes=_ay),
        Joint(_bmap["bottom-f"], _bmap["oblique-2-f"], pivot=_p["dl-f"], rotation_axes=_ay),
        # Joint(_bmap["oblique-1-f"], _bmap["oblique-2-f"], pivot=_p["cross-f"], rotation_axes=_ay),
        #
        Joint(_bmap["horizontal-top-1"], _bmap["top"], pivot=p(0, 0, 0)),
        Joint(_bmap["horizontal-top-1"], _bmap["top-f"], pivot=p(0, 30, 0)),
        Joint(_bmap["horizontal-top-2"], _bmap["top"], pivot=p(30, 0, 0)),
        Joint(_bmap["horizontal-top-2"], _bmap["top-f"], pivot=p(30, 30, 0)),

        # Joint(_bmap["top-mid"], _bmap["horizontal-top-mid"], pivot=p(15, 15, 0)),
        # Joint(_bmap["horizontal-top-mid"], _bmap["top"], pivot=p(15, 0, 0)),
        # Joint(_bmap["horizontal-top-mid"], _bmap["top-f"], pivot=p(15, 30, 0)),
        # Joint(_bmap["top-mid"], _bmap["horizontal-top-1"], pivot=p(0, 15, 0)),
        # Joint(_bmap["top-mid"], _bmap["horizontal-top-2"], pivot=p(30, 15, 0)),
    ]

    if stage >= 2:
        _p.update({
            # "dl-dr-1/3-b": lerp(_p["dl-b"], _p["dr-b"], 1 / 3),
            # "dl-dr-2/3-b": lerp(_p["dl-b"], _p["dr-b"], 2 / 3),
            # "dl-dr-1/3-f": lerp(_p["dl-f"], _p["dr-f"], 1 / 3),
            # "dl-dr-2/3-f": lerp(_p["dl-f"], _p["dr-f"], 2 / 3),
            #
            "dl-dr-1/2-f": lerp(_p["dl-f"], _p["dr-f"], 1 / 2),
            "dl-dr-1/2-b": lerp(_p["dl-b"], _p["dr-b"], 1 / 2),
        })
        _bmap.update({
            # "horizontal-bottom-1": Beam.tetra(_p["dl-dr-1/3-b"], _p["dl-dr-1/3-f"], density=density),
            # "horizontal-bottom-2": Beam.tetra(_p["dl-dr-2/3-b"], _p["dl-dr-2/3-f"], density=density),
            "horizontal-bottom-m": Beam.tetra(_p["dl-dr-1/2-b"], _p["dl-dr-1/2-f"], density=density),
        })
        ax_z = v(0, 0, 1)
        joints.extend([
            # Joint(_bmap["horizontal-bottom-1"], _bmap["bottom"], pivot=_p["dl-dr-1/3-b"], rotation_axes=ax_z),
            # Joint(_bmap["horizontal-bottom-1"], _bmap["bottom-f"], pivot=_p["dl-dr-1/3-f"], rotation_axes=ax_z),
            # Joint(_bmap["horizontal-bottom-2"], _bmap["bottom"], pivot=_p["dl-dr-2/3-b"], rotation_axes=ax_z),
            # Joint(_bmap["horizontal-bottom-2"], _bmap["bottom-f"], pivot=_p["dl-dr-2/3-f"], rotation_axes=ax_z),
            Joint(_bmap["horizontal-bottom-m"], _bmap["bottom"], pivot=_p["dl-dr-1/2-b"], rotation_axes=ax_z),
            Joint(_bmap["horizontal-bottom-m"], _bmap["bottom-f"], pivot=_p["dl-dr-1/2-f"], rotation_axes=ax_z),
        ])

    if stage >= 3:
        s3_frac = '0.1'
        _p.update({
            f"oblique-1-{s3_frac}": lerp(_p["ul-ur-1/3-b"], _p["dr-b"], float(s3_frac)),
            f"oblique-2-{s3_frac}": lerp(_p["ul-ur-2/3-b"], _p["dl-b"], float(s3_frac)),
            f"oblique-1-{s3_frac}-f": lerp(_p["ul-ur-1/3-f"], _p["dr-f"], float(s3_frac)),
            f"oblique-2-{s3_frac}-f": lerp(_p["ul-ur-2/3-f"], _p["dl-f"], float(s3_frac)),
        })
        _bmap.update({
            "stage3-hori-1": Beam.tetra(_p[f"oblique-1-{s3_frac}"], _p[f"oblique-1-{s3_frac}-f"], density=density),
            "stage3-hori-2": Beam.tetra(_p[f"oblique-2-{s3_frac}"], _p[f"oblique-2-{s3_frac}-f"], density=density),
        })
        joints.extend([
            Joint(_bmap["stage3-hori-1"], _bmap["oblique-1"], pivot=_p[f"oblique-1-{s3_frac}"]),
            Joint(_bmap["stage3-hori-1"], _bmap["oblique-1-f"], pivot=_p[f"oblique-1-{s3_frac}-f"]),
            Joint(_bmap["stage3-hori-2"], _bmap["oblique-2"], pivot=_p[f"oblique-2-{s3_frac}"]),
            Joint(_bmap["stage3-hori-2"], _bmap["oblique-2-f"], pivot=_p[f"oblique-2-{s3_frac}-f"]),
        ])

    if stage >= 4:
        s4_frac = '0.8'
        _p.update({
            f"oblique-1-{s4_frac}": lerp(_p["ul-ur-1/3-b"], _p["dr-b"], float(s4_frac)),
            f"oblique-2-{s4_frac}": lerp(_p["ul-ur-2/3-b"], _p["dl-b"], float(s4_frac)),
            f"oblique-1-{s4_frac}-f": lerp(_p["ul-ur-1/3-f"], _p["dr-f"], float(s4_frac)),
            f"oblique-2-{s4_frac}-f": lerp(_p["ul-ur-2/3-f"], _p["dl-f"], float(s4_frac)),
        })
        _bmap.update({
            "stage4-hori-1": Beam.tetra(_p[f"oblique-1-{s4_frac}"], _p[f"oblique-1-{s4_frac}-f"], density=density),
            "stage4-hori-2": Beam.tetra(_p[f"oblique-2-{s4_frac}"], _p[f"oblique-2-{s4_frac}-f"], density=density),
        })
        joints.extend([
            Joint(_bmap["stage4-hori-1"], _bmap["oblique-1"], pivot=_p[f"oblique-1-{s4_frac}"]),
            Joint(_bmap["stage4-hori-1"], _bmap["oblique-1-f"], pivot=_p[f"oblique-1-{s4_frac}-f"]),
            Joint(_bmap["stage4-hori-2"], _bmap["oblique-2"], pivot=_p[f"oblique-2-{s4_frac}"]),
            Joint(_bmap["stage4-hori-2"], _bmap["oblique-2-f"], pivot=_p[f"oblique-2-{s4_frac}-f"]),
        ])

    beams = list(_bmap.values())

    model.add_beams(beams)
    model.add_joints(joints)

    return locals()
