from solvers.rigidity_solver.models import *
import numpy as np
from functools import reduce

scale = 10
p = lambda x, y, z: np.array([x, y, z], dtype=np.double) * scale
v = lambda x, y, z: np.array([x, y, z], dtype=np.double)
lerp = lambda p, q, w: p * (1 - w) + q * w


def define(stage):
    width = 6
    length = 3
    height = 5
    hinged_ratio = 0.5
    foot_ratio = 0.4
    p_len = p(1, 0, 0)
    p_width = p(0, 1, 0)
    p_leg = p(0, 0, -1)
    cross_on_leg_ratio = 0.3
    cross_on_arm_ratio = 0.5

    def coord(a, b, c):
        return a * width * p_width + b * length * p_len + c * height * p_leg
    def init_beam(p, q, density=0.3):
        return Beam.tetra(p, q, density=density, thickness=1.5)

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
                f"{m}{n}-top": init_beam(pmap[f"{m}-hinge"], pmap[f"{n}-hinge"])
            } for m, n in zip("abcd", "bcda")] +
            [{
                f"{letter}-arm": init_beam(pmap[f"{letter}-hinge"], pmap[f"{letter}-tip"]),
                f"{letter}-leg": init_beam(pmap[f"{letter}-knee"], pmap[f"{letter}-foot"]),
                f"{letter}-cross": init_beam(pmap[f"{letter}-arm-crosspoint"], pmap[f"{letter}-leg-crosspoint"]),
            }
                for letter in "abcd"
                # for letter in "a"
            ]
        ),
        f"ad-tip": init_beam(pmap[f"a-tip"], pmap[f"d-tip"]),
        f"bc-tip": init_beam(pmap[f"b-tip"], pmap[f"c-tip"]),
    }

    joints = [
        *[
            Joint(bmap[f"{letter}-arm"], bmap[f"{letter}-cross"],
                  pivot=pmap[f"{letter}-arm-crosspoint"],
                  rotation_axes=p_len
                  )
            for letter in "abcd"
            # for letter in "a"
        ],
        *[
            Joint(bmap[f"{letter}-leg"], bmap[f"{letter}-cross"],
                  pivot=pmap[f"{letter}-leg-crosspoint"],
                  rotation_axes=p_len,
                  translation_vectors=p_leg
                  )
            for letter in "abcd"
            # for letter in "a"
        ],
        *[
            Joint(bmap[f"{m}{n}-top"], bmap[f"{n}{o}-top"], pivot=pmap[f"{n}-hinge"], rotation_axes=None)
            for m, n, o in zip("abcd", "bcda", "cdab")
        ],
        Joint(bmap[f"ad-tip"], bmap[f"a-arm"], pivot=pmap[f"a-tip"], rotation_axes=None),
        Joint(bmap[f"ad-tip"], bmap[f"d-arm"], pivot=pmap[f"d-tip"], rotation_axes=None),
        Joint(bmap[f"bc-tip"], bmap[f"b-arm"], pivot=pmap[f"b-tip"], rotation_axes=None),
        Joint(bmap[f"bc-tip"], bmap[f"c-arm"], pivot=pmap[f"c-tip"], rotation_axes=None),

        Joint(bmap[f"ab-top"], bmap[f"a-arm"], pivot=pmap[f"a-hinge"], rotation_axes=p_len),
        Joint(bmap[f"cd-top"], bmap[f"d-arm"], pivot=pmap[f"d-hinge"], rotation_axes=p_len),
        Joint(bmap[f"ab-top"], bmap[f"b-arm"], pivot=pmap[f"b-hinge"], rotation_axes=p_len),
        Joint(bmap[f"cd-top"], bmap[f"c-arm"], pivot=pmap[f"c-hinge"], rotation_axes=p_len),

        Joint(bmap[f"ab-top"], bmap[f"a-leg"], pivot=pmap[f"a-knee"], rotation_axes=None),
        Joint(bmap[f"cd-top"], bmap[f"d-leg"], pivot=pmap[f"d-knee"], rotation_axes=None),
        Joint(bmap[f"ab-top"], bmap[f"b-leg"], pivot=pmap[f"b-knee"], rotation_axes=None),
        Joint(bmap[f"cd-top"], bmap[f"c-leg"], pivot=pmap[f"c-knee"], rotation_axes=None),
    ]

    if stage >= 2:
        s2_ratio = 0.4
        pmap.update(**{
            f"s2-{letter}-leg-point": lerp(pmap[f"{letter}-knee"], pmap[f"{letter}-leg-crosspoint"], s2_ratio - 0.1)
            for letter in 'abcd'
        }, **{
            f"s2-{letter}-cross-point": lerp(pmap[f"{letter}-arm-crosspoint"], pmap[f"{letter}-leg-crosspoint"], s2_ratio + 0.1)
            for letter in 'abcd'
        })
        bmap.update(**{
            f"s2-{letter}": init_beam(pmap[f"s2-{letter}-leg-point"], pmap[f"s2-{letter}-cross-point"])
            for letter in 'abcd'
        })
        joints.extend([
            *[
                Joint(bmap[f"s2-{letter}"], bmap[f"{letter}-leg"], pivot=pmap[f"s2-{letter}-leg-point"],
                      rotation_axes=p_len)
                for letter in 'abcd'
            ], *[
                Joint(bmap[f"s2-{letter}"], bmap[f"{letter}-cross"], pivot=pmap[f"s2-{letter}-cross-point"],
                      rotation_axes=p_len)
                for letter in 'abcd'
            ]
        ])

    if stage >= 3:
        s3_ratio = 0.7
        pmap.update({
            f"{letter}-leg-{s3_ratio}": lerp(pmap[f"{letter}-knee"], pmap[f"{letter}-foot"], s3_ratio)
            for letter in "abcd"
        })
        bmap.update({
            "s3-ac": init_beam(pmap[f"a-leg-{s3_ratio}"], pmap[f"c-leg-{s3_ratio}"]),
            "s3-bd": init_beam(pmap[f"b-leg-{s3_ratio}"], pmap[f"d-leg-{s3_ratio}"]),
        })
        joints.extend([
            Joint(
                bmap["s3-ac"], bmap["s3-bd"],
                pivot=np.average([pmap[f"{letter}-leg-{s3_ratio}"] for letter in "abcd"], axis=0),
                rotation_axes=v(0, 0, 1),
            ),
            Joint(bmap["s3-ac"], bmap["a-leg"], pivot=pmap[f"a-leg-{s3_ratio}"], ),
            Joint(bmap["s3-ac"], bmap["c-leg"], pivot=pmap[f"c-leg-{s3_ratio}"], ),
            Joint(bmap["s3-bd"], bmap["b-leg"], pivot=pmap[f"b-leg-{s3_ratio}"], ),
            Joint(bmap["s3-bd"], bmap["d-leg"], pivot=pmap[f"d-leg-{s3_ratio}"], ),
        ])

    if stage >= 4:
        s4_ratio = 0.85
        pmap.update({
            f"{letter}-leg-{s4_ratio}": lerp(pmap[f"{letter}-knee"], pmap[f"{letter}-foot"], s4_ratio)
            for letter in "abcd"
        })
        bmap.update({
            "s4-ab": init_beam(pmap[f"a-leg-{s4_ratio}"], pmap[f"b-leg-{s4_ratio}"]),
            "s4-cd": init_beam(pmap[f"c-leg-{s4_ratio}"], pmap[f"d-leg-{s4_ratio}"]),
        })
        joints.extend([
            Joint(bmap["s4-ab"], bmap["a-leg"], pivot=pmap[f"a-leg-{s4_ratio}"], ),
            Joint(bmap["s4-ab"], bmap["b-leg"], pivot=pmap[f"b-leg-{s4_ratio}"], ),
            Joint(bmap["s4-cd"], bmap["c-leg"], pivot=pmap[f"c-leg-{s4_ratio}"], ),
            Joint(bmap["s4-cd"], bmap["d-leg"], pivot=pmap[f"d-leg-{s4_ratio}"], ),
        ])

    if stage >= 5:
        s5_ratios = [1 / 3, 2 / 3]
        s5_ratios = [0.5]
        for ratio in s5_ratios:
            pmap.update({
                f"s5-low-{p}{q}-{ratio}": lerp(pmap[f"{p}-leg-{s4_ratio}"], pmap[f"{q}-leg-{s4_ratio}"], ratio)
                for p, q in ("ab", "cd")
            }, **{
                f"s5-top-{p}{q}-{ratio}": lerp(pmap[f"{p}-knee"], pmap[f"{q}-knee"], ratio)
                for p, q in ("ab", "cd")
            })
            bmap.update({
                f"s5-ab-{ratio}": init_beam(pmap[f"s5-low-ab-{ratio}"], pmap[f"s5-top-ab-{ratio}"]),
                f"s5-cd-{ratio}": init_beam(pmap[f"s5-low-cd-{ratio}"], pmap[f"s5-top-cd-{ratio}"]),
            })
            joints.extend([
                Joint(bmap[f"s5-ab-{ratio}"], bmap["ab-top"], pivot=pmap[f"s5-top-ab-{ratio}"], ),
                Joint(bmap[f"s5-ab-{ratio}"], bmap["s4-ab"], pivot=pmap[f"s5-low-ab-{ratio}"], ),
                Joint(bmap[f"s5-cd-{ratio}"], bmap["cd-top"], pivot=pmap[f"s5-top-cd-{ratio}"], ),
                Joint(bmap[f"s5-cd-{ratio}"], bmap["s4-cd"], pivot=pmap[f"s5-low-cd-{ratio}"], ),
            ])

    beams = list(bmap.values())
    model.add_beams(beams)
    model.add_joints(joints)

    return locals()


def save_json_for_all_stages():
    for stage in range(1, 4 + 1):
        model = define(stage)["model"]
        model.save_json(f"output/foldable-stage{stage}.json")


if __name__ == "__main__":
    save_json_for_all_stages()
    stage = 4
    model = define(stage)["model"]
    pairs = model.soft_solve(8)
    pairs = model.eigen_solve(
        8,
        extra_constr=geo_util.trivial_basis(model.point_matrix())
    )
    # model.visualize()
    print(*[(i, e) for i, (e, _) in enumerate(pairs)], sep="\n")
    for i in range(4):
        e, v = pairs[i]
        model.visualize(v.reshape(-1, 3), show_hinge=False)
