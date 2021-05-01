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
        "ab-0.1": lerp(_p["A-u"], _p["B-u"], 0.1),
        "bc-0.1": lerp(_p["B-u"], _p["C-u"], 0.1),
        "ca-0.1": lerp(_p["C-u"], _p["A-u"], 0.1),
        "ba-0.1": lerp(_p["B-u"], _p["A-u"], 0.1),
        "cb-0.1": lerp(_p["C-u"], _p["B-u"], 0.1),
        "ac-0.1": lerp(_p["A-u"], _p["C-u"], 0.1),
        "ab-0.9": lerp(_p["A-u"], _p["B-u"], 0.9),
        "bc-0.9": lerp(_p["B-u"], _p["C-u"], 0.9),
        "ca-0.9": lerp(_p["C-u"], _p["A-u"], 0.9),
        "ba-0.9": lerp(_p["B-u"], _p["A-u"], 0.9),
        "cb-0.9": lerp(_p["C-u"], _p["B-u"], 0.9),
        "ac-0.9": lerp(_p["A-u"], _p["C-u"], 0.9),
    })

    def beam_init(p, q, density=0.5):
        return Beam.tetra(p, q, density=density, thickness=1)

    stage_2_frac = 0.25
    stage_3_frac = 0.7


    normalize = lambda x: x / np.linalg.norm(x)
    _da = normalize(_p["c"] - _p["b"])
    _db = normalize(_p["a"] - _p["c"])
    _dc = normalize(_p["b"] - _p["a"])
    _dz = v(0, 0, 1)

    model = Model()
    _bmap = {
        "top-A": beam_init(_p["B-u"], _p["C-u"]),
        "top-B": beam_init(_p["C-u"], _p["A-u"]),
        "top-C": beam_init(_p["A-u"], _p["B-u"]),

        # "top-ab-bc": beam_init(_p["ab-mid"], _p["bc-mid"]),
        # "top-bc-ca": beam_init(_p["bc-mid"], _p["ca-mid"]),
        # "top-ca-ab": beam_init(_p["ca-mid"], _p["ab-mid"]),
        #
        # "core-ab": beam_init(_p['a'], _p["b"]),
        # "core-bc": beam_init(_p["b"], _p["c"]),
        # "core-ca": beam_init(_p["c"], _p["a"]),
        #
        "A-c": beam_init(_p["ca-0.9"], _p["C-d"]),
        "A-b": beam_init(_p["ab-0.1"], _p["B-d"]),
        "B-a": beam_init(_p["ab-0.9"], _p["A-d"]),
        "B-c": beam_init(_p["bc-0.1"], _p["C-d"]),
        "C-b": beam_init(_p["bc-0.9"], _p["B-d"]),
        "C-a": beam_init(_p["ca-0.1"], _p["A-d"]),
    }
    joints = [
        Joint(_bmap["B-a"], _bmap["C-a"], pivot=_p["A-d"], rotation_axes=_da),
        Joint(_bmap["C-b"], _bmap["A-b"], pivot=_p["B-d"], rotation_axes=_db),
        Joint(_bmap["A-c"], _bmap["B-c"], pivot=_p["C-d"], rotation_axes=_dc),

        Joint(_bmap["top-C"], _bmap["top-A"], pivot=_p["B-u"], rotation_axes=-v(0, 0, 1)),
        Joint(_bmap["top-A"], _bmap["top-B"], pivot=_p["C-u"], rotation_axes=-v(0, 0, 1)),
        Joint(_bmap["top-B"], _bmap["top-C"], pivot=_p["A-u"], rotation_axes=-v(0, 0, 1)),

        Joint(_bmap["top-B"], _bmap["A-b"], pivot=_p["ab-0.1"], rotation_axes=_da),
        Joint(_bmap["top-C"], _bmap["A-c"], pivot=_p["ca-0.9"], rotation_axes=_da),
        Joint(_bmap["top-C"], _bmap["B-c"], pivot=_p["bc-0.1"], rotation_axes=_db),
        Joint(_bmap["top-A"], _bmap["B-a"], pivot=_p["ab-0.9"], rotation_axes=_db),
        Joint(_bmap["top-A"], _bmap["C-a"], pivot=_p["ca-0.1"], rotation_axes=_dc),
        Joint(_bmap["top-B"], _bmap["C-b"], pivot=_p["bc-0.9"], rotation_axes=_dc),

        Joint(_bmap["A-b"], _bmap["B-a"],
              pivot=(_p["ab-0.1"] + _p["ab-0.9"] + _p["A-d"] + _p["B-d"]) / 4,
              rotation_axes=np.cross(_dc, _dz)),
        Joint(_bmap["B-c"], _bmap["C-b"],
              pivot=(_p["bc-0.1"] + _p["bc-0.9"] + _p["B-d"] + _p["C-d"]) / 4,
              rotation_axes=np.cross(_da, _dz)),
        Joint(_bmap["C-a"], _bmap["A-c"],
              pivot=(_p["ca-0.1"] + _p["ca-0.9"] + _p["C-d"] + _p["A-d"]) / 4,
              rotation_axes=np.cross(_db, _dz)),
    ]

    ax_z = v(0, 0, 1)
    if stage >= 2:
        _stage_2_points = {
            f"{a}-u-{b}-d-{stage_2_frac}": lerp(_p[f"{a.lower()}{b.lower()}-0.1"], _p[f"{b}-d"], stage_2_frac)
            for a in "ABC" for b in "ABC" if a != b
        }
        _p.update(_stage_2_points)
        _stage_2_beam = {
            f"s2-{a}{b}": beam_init(_p[f"{a}-u-{b}-d-{stage_2_frac}"], _p[f"{b}-u-{a}-d-{stage_2_frac}"])
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
        _stage_3_points = {
            f"{a}-u-{b}-d-{stage_3_frac}": lerp(_p[f"{a}-u"], _p[f"{b}-d"], stage_3_frac)
            for a in "ABC" for b in "ABC" if a != b
        }
        _p.update(_stage_3_points)
        _stage_3_beam = {
            f"s3-{a}{b}": beam_init(_p[f"{a}-u-{b}-d-{stage_3_frac}"], _p[f"{b}-u-{a}-d-{stage_3_frac}"])
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
        _indices = ["AB", "BC", "CA"]
        _stage_4_points = {
            f"s4-{_indices[i % 3]}": lerp(_p[f"{a}-u-{b}-d-{stage_2_frac}"], _p[f"{b}-u-{a}-d-{stage_2_frac}"], 0.5)
            for i, (a, b) in enumerate(_indices)
        }
        _p.update(_stage_4_points)
        _stage_4_beam = {
            f"s4-{_indices[i % 3]}": beam_init(_p[f"s4-{_indices[i]}"], _p[f"{a.lower()}{b.lower()}-mid"])
            for i, (a, b) in enumerate(_indices)
        }
        _bmap.update(_stage_4_beam)
        _stage_4_joint = [
            Joint(_bmap[f"s4-{_indices[i % 3]}"], _bmap[f"s2-{_indices[i % 3]}"],
                  pivot=_p[f"s4-{_indices[i]}"],
                  rotation_axes=np.cross((_dc, _da, _db)[i], v(0, 0, 1))
                  )
            for i, (a, b) in enumerate(_indices)
        ] + [
            Joint(_bmap[f"s4-{_indices[i % 3]}"], _bmap[f"top-{'CAB'[i]}"],
                  pivot=_p[f"{a.lower()}{b.lower()}-mid"],
                  rotation_axes=np.cross((_dc, _da, _db)[i], v(0, 0, 1))
                  )
        for i, (a, b) in enumerate(_indices)
        ]
        joints.extend(_stage_4_joint)

    beams = list(_bmap.values())
    model.add_beams(beams)
    model.add_joints(joints)

    return locals()


if __name__ == "__main__":
    model = define(1)["model"]
    model.visualize(show_hinge=True)

    points = model.point_matrix()
    edges = model.edge_matrix()
    stiffness = spring_energy_matrix_accelerate_3D(points, edges, abstract_edges=[]),
    constraints = model.constraint_matrix()

    new_stiffness, B = generalized_courant_fischer(
        stiffness,
        constraints
    )

    pairs = model.eigen_solve(num_pairs=20)
    print([e for e, v in pairs])
    for stage in range(1, 4 + 1):
        model = define(stage)["model"]
        model.save_json(f"output/table-stage{stage}.json")
