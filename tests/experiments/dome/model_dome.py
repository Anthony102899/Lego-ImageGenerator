from solvers.rigidity_solver.models import *
import numpy as np

scale = 10
p = lambda x, y, z: np.array([x, y, z], dtype=np.double) * scale
v = lambda x, y, z: np.array([x, y, z], dtype=np.double)
lerp = lambda p, q, w: p * (1 - w) + q * w

def polar_to_cart(r, phi, theta):
    return np.asarray([
        r * np.cos(theta) * np.sin(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(phi),
    ], dtype=np.double) * scale

def define(stage):
    v_portions = 2
    h_portions = 4
    radius = 10
    v_max = np.pi / 2
    h_max = np.pi

    def beam_init(p, q, density=0.3):
        return Beam.tetra(p, q, thickness=2, density=density)

    ###### temp:
    theta = 0
    i = 0

    model = Model()

    _p = {
        # axial points
        **{
            f"ax-far-{i}-{j}": polar_to_cart(radius, phi, theta)
            for j, phi in enumerate(np.linspace(0, h_max, h_portions + 1))
            # for i, theta in enumerate(np.linspace(0, v_max, v_portions, endpoint=False))
        },
        **{
            f"ax-near-{i}-{j}": polar_to_cart(radius * 0.85, phi, theta)
            for j, phi in enumerate(np.linspace(0, h_max, h_portions + 1))
            # for i, theta in enumerate(np.linspace(0, v_max, v_portions, endpoint=False))
        },
    # "top": polar_to_cart(radius, 0, 0.5 * np.pi),
    }
    _p.update({
        **{
            f"cross-{i}-{j}": np.sum([_p[f"ax-{n}-{i}-{k}"] for n in ("near", "far") for k in (j, j + 1)], axis=0) / 4
            for j in range(h_portions)
            # for i, theta in enumerate(np.linspace(0, v_max, v_portions, endpoint=False))
        }
    })
    _bmap = {
        **{
            # f"node-{i}-{j}": beam_init(_p[f"ax-far-{i}-{j}"] * 1.05, _p[f"ax-far-{i}-{j}"] * 0.95, density=1)
            # for j in range(h_portions + 1)
            # for i in range(v_portions)
        },
        **{
            f"ax-{i}-{j}": beam_init(_p[f"ax-far-{i}-{j}"], _p[f"ax-near-{i}-{j}"])
            for j in range(h_portions + 1)
            # for i in range(v_portions)
        },
        **{
            f"left-{i}-{j}": beam_init(_p[f"ax-far-{i}-{j}"], _p[f"ax-near-{i}-{j + 1}"])
            for j in range(h_portions)
            # for i in range(v_portions)
        },
        **{
            f"right-{i}-{j}": beam_init(_p[f"ax-near-{i}-{j}"], _p[f"ax-far-{i}-{j + 1}"])
            for j in range(h_portions)
            # for i in range(v_portions)
        },
    }
    joints = [
        *[
            Joint(_bmap[f"left-{i}-{j}"], _bmap[f"right-{i}-{j}"],
                  pivot=_p[f"cross-{i}-{j}"],
                  rotation_axes=geo_util.normalize(np.cross(_p[f"cross-{i}-{j}"], _p[f"ax-far-{i}-{j}"])),
            )
            for j in range(h_portions)
            # for i in range(v_portions)
        ],
        *[
            Joint(_bmap[f"ax-{i}-{j}"], _bmap[f"left-{i}-{j}"],
                  pivot=_p[f"ax-far-{i}-{j}"],
                  rotation_axes=geo_util.normalize(np.cross(_p[f"cross-{i}-{j}"], _p[f"ax-far-{i}-{j}"])))
            for j in range(h_portions)
            # for i in range(v_portions)
        ],
        *[
            Joint(_bmap[f"ax-{i}-{j}"], _bmap[f"right-{i}-{j}"],
                  pivot=_p[f"ax-near-{i}-{j}"],
                  rotation_axes=geo_util.normalize(np.cross(_p[f"cross-{i}-{j}"], _p[f"ax-far-{i}-{j}"])))
            for j in range(h_portions)
            # for i in range(v_portions)
        ],
        *[
            Joint(_bmap[f"ax-{i}-{j + 1}"], _bmap[f"left-{i}-{j}"],
                  pivot=_p[f"ax-near-{i}-{j + 1}"],
                  rotation_axes=geo_util.normalize(np.cross(_p[f"cross-{i}-{j}"], _p[f"ax-far-{i}-{j}"])))
            for j in range(h_portions)
            # for i in range(v_portions)
        ],
        *[
            Joint(_bmap[f"ax-{i}-{j + 1}"], _bmap[f"right-{i}-{j}"],
                  pivot=_p[f"ax-far-{i}-{j + 1}"],
                  rotation_axes=geo_util.normalize(np.cross(_p[f"cross-{i}-{j}"], _p[f"ax-far-{i}-{j}"])))
            for j in range(h_portions)
            # for i in range(v_portions)
        ],
    ]

    beams = list(_bmap.values())
    model.add_beams(beams)
    model.add_joints(joints)

    return locals()
