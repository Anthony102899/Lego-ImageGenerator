from solvers.rigidity_solver.models import *
import numpy as np

from functools import reduce

scale = 10
p = lambda x, y, z: np.array([x, y, z], dtype=np.double) * scale
v = lambda x, y, z: np.array([x, y, z], dtype=np.double)
lerp = lambda p, q, w: p * (1 - w) + q * w

def define(stage):
    width = 5
    length = 6
    height = 5
    hinged_ratio = 0.6
    foot_ratio = 0.6
    p_len = p(1, 0, 0)
    p_width = p(0, 1, 0)
    p_leg = p(0, 0, -1)
    cross_on_leg_ratio = 0.3
    cross_on_arm_ratio = 0.5

    def coord(a, b, c):
        return a * width * p_width + b * length * p_len + c * height * p_leg
    def init_beam(p, q, density=0.3):
        return Beam.tetra(p, q, density=density)

    model = Model()

    pmap = reduce(
        lambda x, y: {**x, **y},
        ({
            f"{letter}-hinge": coord(x_sign * 0.5 * hinged_ratio, y_sign * 0.5, 0),
            f"{letter}-tip": coord(x_sign * 0.5, y_sign * 0.5, 0),
            f"{letter}-foot": coord(x_sign * 0.5 * foot_ratio, y_sign * 0.5, 1),
            f"{letter}-knee": coord(x_sign * 0.5 * foot_ratio, y_sign * 0.5, 0),
            f"{letter}-arm-crosspoint": coord(x_sign * 0.5 * (hinged_ratio + (1 - hinged_ratio) * cross_on_arm_ratio), y_sign * 0.5, 0),
            f"{letter}-leg-crosspoint": coord(x_sign * 0.5 * foot_ratio, y_sign * 0.5, cross_on_leg_ratio),
        } for letter, (x_sign, y_sign) in zip("abcd", ((-1, 1), (1, 1), (1, -1), (-1, -1))))
    )
    bmap = {
        **reduce(
            lambda x, y: {**x, **y},
            [{
                f"{letter}-arm": init_beam(pmap[f"{letter}-hinge"], pmap[f"{letter}-tip"]),
                f"{letter}-leg": init_beam(pmap[f"{letter}-knee"], pmap[f"{letter}-foot"]),
                f"{letter}-cross": init_beam(pmap[f"{letter}-arm-crosspoint"], pmap[f"{letter}-leg-crosspoint"]),
            } for letter in "abcd"] + [{
                f"{m}{n}-top": init_beam(pmap[f"{m}-hinge"], pmap[f"{n}-hinge"])
            } for m, n in zip("abcd", "bcda")]
        ),
        f"ad-tip": init_beam(pmap[f"a-tip"], pmap[f"d-tip"]),
        f"bc-tip": init_beam(pmap[f"b-tip"], pmap[f"c-tip"]),
    }

    joints = [
        *[
            Joint(bmap[f"{letter}-arm"], bmap[f"{letter}-cross"],
                  pivot=pmap[f"{letter}-arm-crosspoint"], rotation_axes=p_len)
            for letter in "abcd"
        ],
        *[
            Joint(bmap[f"{letter}-leg"], bmap[f"{letter}-cross"],
                  pivot=pmap[f"{letter}-leg-crosspoint"])
            for letter in "abcd"
        ],
        *[
            Joint(bmap[f"{m}{n}-top"], bmap[f"{n}{o}-top"], pivot=pmap[f"{n}-hinge"], rotation_axes=p_leg)
            for m, n, o in zip("abcd", "bcda", "cdab")],
        Joint(bmap[f"ad-tip"], bmap[f"a-arm"], pivot=pmap[f"a-tip"], rotation_axes=p_leg),
        Joint(bmap[f"ad-tip"], bmap[f"d-arm"], pivot=pmap[f"d-tip"], rotation_axes=p_leg),
        Joint(bmap[f"bc-tip"], bmap[f"b-arm"], pivot=pmap[f"b-tip"], rotation_axes=p_leg),
        Joint(bmap[f"bc-tip"], bmap[f"c-arm"], pivot=pmap[f"c-tip"], rotation_axes=p_leg),

        Joint(bmap[f"ab-top"], bmap[f"a-arm"], pivot=pmap[f"a-hinge"], rotation_axes=p_width),
        Joint(bmap[f"cd-top"], bmap[f"d-arm"], pivot=pmap[f"d-hinge"], rotation_axes=p_width),
        Joint(bmap[f"ab-top"], bmap[f"b-arm"], pivot=pmap[f"b-hinge"], rotation_axes=p_width),
        Joint(bmap[f"cd-top"], bmap[f"c-arm"], pivot=pmap[f"c-hinge"], rotation_axes=p_width),
    ]

    beams = list(bmap.values())
    model.add_beams(beams)
    model.add_joints(joints)

    return locals()


if __name__ == "__main__":
    model = define(1)["model"]
    pairs = model.eigen_solve(num_pairs=30)
    print(*[(i, e) for i, (e, _) in enumerate(pairs)], sep="\n")
    e, v = pairs[14]
    # model.visualize(arrows=v.reshape(-1, 3))
