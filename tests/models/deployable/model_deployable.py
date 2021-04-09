from solvers.rigidity_solver.models import *
import numpy as np

_scale = lambda arr: arr * 15
v = lambda x, y, z: np.array([x, y, z], dtype=np.double)
p = lambda x, y, z: (_scale(np.array([x, y, z], dtype=np.double)))

def lerp(p, q, weight):
    return p + (q - p) * weight

def define(stage):
    _p = {
        "a": p(0, 0, 0),
        "b": p(1, 0, 0),
        "c": p(1 / 2, np.sqrt(3) / 2, 0),

        "A-u": p(3 / 2, np.sqrt(3) / 2, 1),
        "A-d": p(3 / 2, np.sqrt(3) / 2, -1),

        "B-u": p(-1 / 2, np.sqrt(3) / 2, 1),
        "B-d": p(-1 / 2, np.sqrt(3) / 2, -1),

        "C-u": p(1 / 2, -np.sqrt(3) / 2, 1),
        "C-d": p(1 / 2, -np.sqrt(3) / 2, -1),
    }

    _p.update({
        "ab-mid": lerp(_p["A-u"], _p["B-u"], 0.5),
        "bc-mid": lerp(_p["B-u"], _p["C-u"], 0.5),
        "ca-mid": lerp(_p["C-u"], _p["A-u"], 0.5),
    })

    den = 0.5
    stage_2_frac = 0.3
    stage_3_frac = 0.7

    _stage_2_points = {
        f"{a}-u-{b}-d-{stage_2_frac}": lerp(_p[f"{a}-u"], _p[f"{b}-d"], stage_2_frac)
        for a in "ABC" for b in "ABC" if a != b
    }
    _p.update(_stage_2_points)
    _stage_3_points = {
        f"{a}-u-{b}-d-{stage_3_frac}": lerp(_p[f"{a}-u"], _p[f"{b}-d"], stage_3_frac)
        for a in "ABC" for b in "ABC" if a != b
    }
    _p.update(_stage_3_points)

    normalize = lambda x: x / np.linalg.norm(x)
    _da = normalize(_p["c"] - _p["b"])
    _db = normalize(_p["a"] - _p["c"])
    _dc = normalize(_p["b"] - _p["a"])

    model = Model()
    _bmap = {
        "top-A": Beam.tetra(_p["B-u"], _p["C-u"], density=den),
        "top-B": Beam.tetra(_p["C-u"], _p["A-u"], density=den),
        "top-C": Beam.tetra(_p["A-u"], _p["B-u"], density=den),

        "top-ab-bc": Beam.tetra(_p["ab-mid"], _p["bc-mid"], density=den),
        "top-bc-ca": Beam.tetra(_p["bc-mid"], _p["ca-mid"], density=den),
        "top-ca-ab": Beam.tetra(_p["ca-mid"], _p["ab-mid"], density=den),

        "core-ab": Beam.tetra(_p['a'], _p["b"], density=den),
        "core-bc": Beam.tetra(_p["b"], _p["c"], density=den),
        "core-ca": Beam.tetra(_p["c"], _p["a"], density=den),

        "A-c": Beam.tetra(_p["A-u"], _p["C-d"], density=den),
        "A-b": Beam.tetra(_p["A-u"], _p["B-d"], density=den),
        "B-a": Beam.tetra(_p["B-u"], _p["A-d"], density=den),
        "B-c": Beam.tetra(_p["B-u"], _p["C-d"], density=den),
        "C-b": Beam.tetra(_p["C-u"], _p["B-d"], density=den),
        "C-a": Beam.tetra(_p["C-u"], _p["A-d"], density=den),
    }
    joints = [
        Joint(_bmap["core-ab"], _bmap["core-bc"], pivot=_p["b"], rotation_axes=v(0, 0, 1)),
        Joint(_bmap["core-bc"], _bmap["core-ca"], pivot=_p["c"], rotation_axes=v(0, 0, 1)),
        Joint(_bmap["core-ca"], _bmap["core-ab"], pivot=_p["a"], rotation_axes=v(0, 0, 1)),

        Joint(_bmap["top-ab-bc"], _bmap["top-bc-ca"], pivot=_p["bc-mid"], rotation_axes=v(0, 0, 1)),
        Joint(_bmap["top-bc-ca"], _bmap["top-ca-ab"], pivot=_p["ca-mid"], rotation_axes=v(0, 0, 1)),
        Joint(_bmap["top-ca-ab"], _bmap["top-ab-bc"], pivot=_p["ab-mid"], rotation_axes=v(0, 0, 1)),
        Joint(_bmap["top-ab-bc"], _bmap["top-B"], pivot=_p["bc-mid"], rotation_axes=v(0, 0, 1)),
        Joint(_bmap["top-bc-ca"], _bmap["top-C"], pivot=_p["ca-mid"], rotation_axes=v(0, 0, 1)),
        Joint(_bmap["top-ca-ab"], _bmap["top-A"], pivot=_p["ab-mid"], rotation_axes=v(0, 0, 1)),

        Joint(_bmap["A-c"], _bmap["B-c"], pivot=_p["C-d"], rotation_axes=_dc),
        Joint(_bmap["B-a"], _bmap["B-c"], pivot=_p["B-u"], rotation_axes=_db),
        Joint(_bmap["B-a"], _bmap["C-a"], pivot=_p["A-d"], rotation_axes=_da),
        Joint(_bmap["C-b"], _bmap["C-a"], pivot=_p["C-u"], rotation_axes=_dc),
        Joint(_bmap["C-b"], _bmap["A-b"], pivot=_p["B-d"], rotation_axes=_db),
        Joint(_bmap["A-c"], _bmap["A-b"], pivot=_p["A-u"], rotation_axes=_da),
        #
        Joint(_bmap["A-c"], _bmap["core-bc"],
              pivot=_p["c"], rotation_axes=_da),
        Joint(_bmap["A-b"], _bmap["core-bc"],
              pivot=_p["b"], rotation_axes=_da),
        Joint(_bmap["B-a"], _bmap["core-ca"],
              pivot=_p["a"], rotation_axes=_db),
        Joint(_bmap["B-c"], _bmap["core-ca"],
              pivot=_p["c"], rotation_axes=_db),
        Joint(_bmap["C-a"], _bmap["core-ab"],
              pivot=_p["a"], rotation_axes=_dc),
        Joint(_bmap["C-b"], _bmap["core-ab"],
              pivot=_p["b"], rotation_axes=_dc),

        Joint(_bmap["top-C"], _bmap["top-A"], pivot=_p["B-u"], rotation_axes=v(0, 0, 1)),
        Joint(_bmap["top-A"], _bmap["top-B"], pivot=_p["C-u"], rotation_axes=v(0, 0, 1)),
        Joint(_bmap["top-B"], _bmap["top-C"], pivot=_p["A-u"], rotation_axes=v(0, 0, 1)),

        Joint(_bmap["top-B"], _bmap["A-b"], pivot=_p["A-u"], rotation_axes=v(0, 0, 1)),
        Joint(_bmap["top-C"], _bmap["A-c"], pivot=_p["A-u"], rotation_axes=v(0, 0, 1)),
        Joint(_bmap["top-C"], _bmap["B-c"], pivot=_p["B-u"], rotation_axes=v(0, 0, 1)),
        Joint(_bmap["top-A"], _bmap["B-a"], pivot=_p["B-u"], rotation_axes=v(0, 0, 1)),
        Joint(_bmap["top-A"], _bmap["C-a"], pivot=_p["C-u"], rotation_axes=v(0, 0, 1)),
        Joint(_bmap["top-B"], _bmap["C-b"], pivot=_p["C-u"], rotation_axes=v(0, 0, 1)),
    ]

    ax_z = v(0, 0, 1)
    if stage >= 2:
        _stage_2_beam = {
            f"s2-{a}{b}": Beam.tetra(_p[f"{a}-u-{b}-d-{stage_2_frac}"], _p[f"{b}-u-{a}-d-{stage_2_frac}"], density=den)
            for a, b in ("AB", "BC", "CA")
        }
        _bmap.update(_stage_2_beam)
        _stage_2_joint = [
            Joint(_bmap[f"s2-{a}{b}"], _bmap[f"{a}-{b.lower()}"], pivot=_p[f"{a}-u-{b}-d-{stage_2_frac}"], rotation_axes=ax_z)
            for a, b in ("AB", "BC", "CA")
        ] + [
            Joint(_bmap[f"s2-{a}{b}"], _bmap[f"{b}-{a.lower()}"], pivot=_p[f"{b}-u-{a}-d-{stage_2_frac}"], rotation_axes=ax_z)
            for a, b in ("AB", "BC", "CA")
        ]
        joints.extend(_stage_2_joint)

    if stage >= 3:
        _stage_3_beam = {
            f"s3-{a}{b}": Beam.tetra(_p[f"{a}-u-{b}-d-{stage_3_frac}"], _p[f"{b}-u-{a}-d-{stage_3_frac}"], density=den * 2)
            for a, b in ("AB", "BC", "CA")
        }
        _bmap.update(_stage_3_beam)
        _stage_3_joint = [
                             Joint(_bmap[f"s3-{a}{b}"], _bmap[f"{a}-{b.lower()}"], pivot=_p[f"{a}-u-{b}-d-{stage_3_frac}"], rotation_axes=ax_z)
                             for a, b in ("AB", "BC", "CA")
                         ] + [
                             Joint(_bmap[f"s3-{a}{b}"], _bmap[f"{b}-{a.lower()}"], pivot=_p[f"{b}-u-{a}-d-{stage_3_frac}"], rotation_axes=ax_z)
                             for a, b in ("AB", "BC", "CA")
                         ]
        joints.extend(_stage_3_joint)

    if stage >= 4:
        _indices = ["AB", "BC", "CA", "BA", "CB", "AC"]
        _stage_4_beam = {
            f"s4-{_indices[i % 3]}": Beam.tetra(_p[f"{a}-u-{b}-d-{stage_2_frac}"], _p[f"{a}-u-{b}-d-{stage_3_frac}"], density=den * 2)
            for i, (a, b) in enumerate(_indices)
        }
        _bmap.update(_stage_4_beam)
        _stage_4_joint = [
            Joint(_bmap[f"s4-{_indices[i % 3]}"], _bmap[f"s2-{_indices[i % 3]}"], pivot=_p[f"{a}-u-{b}-d-{stage_2_frac}"],)
            for i, (a, b) in enumerate(_indices)
        ] + [
            Joint(_bmap[f"s4-{_indices[i % 3]}"], _bmap[f"s3-{_indices[i % 3]}"], pivot=_p[f"{a}-u-{b}-d-{stage_3_frac}"], )
            for i, (a, b) in enumerate(_indices)
        ]
        joints.extend(_stage_4_joint)

    beams = list(_bmap.values())
    model.add_beams(beams)
    model.add_joints(joints)

    return locals()
