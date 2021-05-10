from solvers.rigidity_solver.models import *
from itertools import product
import numpy as np

scale = 30
p = lambda x, y, z: np.array([x, y, z], dtype=np.double) * scale
v = lambda x, y, z: np.array([x, y, z], dtype=np.double)
lerp = lambda p, q, w: p * (1 - w) + q * w
avg = lambda pts: np.average(np.asarray(pts), axis=0)


def beam_init(p, q, density=0.3):
    return Beam.tetra(p, q, thickness=2, density=density)


def polar_to_cart(r, azimuth, polar):
    return np.asarray([
        r * np.cos(azimuth) * np.sin(polar),
        r * np.sin(azimuth) * np.sin(polar),
        r * np.cos(polar),
    ], dtype=np.double) * scale


def scissor_2d(num_portions, inner_ratio, start_angles, end_angles, axial_joint_type):
    # inferred variable
    closed = np.allclose(polar_to_cart(1, *start_angles), polar_to_cart(1, *end_angles))
    num_axes = num_portions + (0 if closed else 1)
    num_shared_axes = num_axes + (0 if closed else -2)

    angles_range = np.linspace(start_angles, end_angles, num_axes, endpoint=(not closed))
    print(angles_range / np.pi)

    pmap = {
        **{
            f"axial-{i}-outer": polar_to_cart(1, az, pl)
            for i, (az, pl) in enumerate(angles_range)
        },
        **{
            f"axial-{i}-inner": polar_to_cart(1 * inner_ratio, az, pl)
            for i, (az, pl) in enumerate(angles_range)
        },
    }
    pmap = {
        **pmap,
        **{
            f"cross-{i}": avg([pmap[f"axial-{(i + offset) % num_axes}-{inout}"] for offset in (0, 1) for inout in ("inner", "outer")])
            for i in range(num_portions)
        }
    }
    bmap = {
        # **{
        #       f"confluence-{i}-inner": beam_init(
        #           lerp(pmap[f"axial-{i}-outer"], pmap[f"axial-{i}-inner"], -0.1),
        #           lerp(pmap[f"axial-{i}-outer"], pmap[f"axial-{i}-inner"], 0.1),
        #       )
        #       for i in range(num_axes)
        # },
        # **{
        #     f"confluence-{i}-outer": beam_init(
        #         lerp(pmap[f"axial-{i}-outer"], pmap[f"axial-{i}-inner"], 1.1),
        #         lerp(pmap[f"axial-{i}-outer"], pmap[f"axial-{i}-inner"], 0.9),
        #     )
        #     for i in range(num_axes)
        # },
        **{
            f"left-{i}": beam_init(pmap[f"axial-{i}-outer"],
                                   pmap[f"axial-{(i + 1) % num_axes}-inner"])
            for i in range(num_portions)
        },
        **{
            f"right-{i}": beam_init(pmap[f"axial-{i}-inner"],
                                    pmap[f"axial-{(i + 1) % num_axes}-outer"])
            for i in range(num_portions)
        },
    }
    ortho = geo_util.normalize(np.cross(pmap["cross-0"], pmap["axial-0-outer"]))
    print(pmap["cross-0"], pmap["axial-0-outer"])
    jmap = {}
    for i in range(num_portions):
        jmap[f"cross-{i}"] = Joint(bmap[f"left-{i}"], bmap[f"right-{i}"], pivot=pmap[f"cross-{i}"],
                              rotation_axes=ortho)
        # model.add_joint(Joint(bmap[f"left-{i}"], bmap[f"axis-{i}"], pivot=pmap[f"axial-{i}-outer"],
        #                       rotation_axes=ortho)),
        # model.add_joint(Joint(bmap[f"right-{i}"], bmap[f"axis-{i}"], pivot=pmap[f"axial-{i}-inner"],
        #                       rotation_axes=ortho)),
        # model.add_joint(Joint(bmap[f"left-{i}"], bmap[f"axis-{i + 1}"], pivot=pmap[f"axial-{i + 1}-outer"],
        #                       rotation_axes=ortho)),
        # model.add_joint(Joint(bmap[f"right-{i}"], bmap[f"axis-{i + 1}"], pivot=pmap[f"axial-{i + 1}-inner"],
        #                       rotation_axes=ortho)),
    for i in range(num_shared_axes):
        if axial_joint_type == "ball" and i in list(range(0, num_axes, 2)):
            axes = np.eye(3)
        else:
            axes = ortho
        jmap[f"axial-{(i + 1) % num_axes}-inner"] = Joint(bmap[f"left-{i}"], bmap[f"right-{(i + 1) % num_axes}"],
                              pivot=pmap[f"axial-{(i + 1) % num_axes}-inner"],
                              rotation_axes=axes)
        jmap[f"axial-{(i + 1) % num_axes}-outer"] = Joint(bmap[f"right-{i}"], bmap[f"left-{(i + 1) % num_axes}"],
                              pivot=pmap[f"axial-{(i + 1) % num_axes}-outer"],
                              rotation_axes=axes)
        #     model.add_joint(Joint(bmap[f"confluence-{i}-outer"], bmap[f"axis-{i}"], pivot=pmap[f"axial-{i}-outer"],
        #                           rotation_axes=ortho)),
        #     model.add_joint(Joint(bmap[f"confluence-{i}-inner"], bmap[f"axis-{i}"], pivot=pmap[f"axial-{i}-inner"],
        #                           rotation_axes=ortho)),

    return pmap, bmap, jmap

def define(stage):
    v_portions = 2
    h_portions = 4
    radius = 10
    v_max = np.pi / 2
    h_max = np.pi

    model = Model()

    inner_ratio = 0.75
    s1_pmap, s1_bmap, s1_jmap = scissor_2d(
        8, inner_ratio,
        np.array([0, np.pi / 2]), np.array([np.pi * 2, np.pi / 2]),
        axial_joint_type="hinge",
    )
    s1_beams = list(s1_bmap.values())
    model.add_beams(s1_beams)
    model.add_joints(s1_jmap.values())

    s2_pmap, s2_bmap, s2_jmap = scissor_2d(
        4, inner_ratio,
        np.array([0, -np.pi / 2]), np.array([0, np.pi / 2]),
        axial_joint_type="hinge"
    )
    s2_beams = list(s2_bmap.values())
    model.add_beams(s2_beams)
    model.add_joints(s2_jmap.values())

    s3_pmap, s3_bmap, s3_jmap = scissor_2d(
        4, inner_ratio,
        np.array([np.pi / 2, -np.pi / 2]), np.array([np.pi / 2, np.pi / 2]),
        axial_joint_type="ball"
    )
    s3_beams = list(s3_bmap.values())
    model.add_beams(s3_beams)
    model.add_joints(s3_jmap.values())

    print(list(s2_jmap.keys()))
    joint_map_connection = [
        [s2_pmap["axial-2-inner"], s2_jmap, s3_jmap],
        [s2_pmap["axial-2-outer"], s2_jmap, s3_jmap],
    ]
    beam_map_connection = [
        [s2_pmap["axial-0-inner"], s2_bmap, s1_bmap],
        [s2_pmap["axial-0-outer"], s2_bmap, s1_bmap],
        [s2_pmap["axial-4-inner"], s2_bmap, s1_bmap],
        [s2_pmap["axial-4-outer"], s2_bmap, s1_bmap],
        [s3_pmap["axial-0-inner"], s3_bmap, s1_bmap],
        [s3_pmap["axial-0-outer"], s3_bmap, s1_bmap],
        [s3_pmap["axial-4-inner"], s3_bmap, s1_bmap],
        [s3_pmap["axial-4-outer"], s3_bmap, s1_bmap],
    ]

    extra_joints = []
    for pivot, first_map, other_map in joint_map_connection:
        first_key, first_joint = min([j for j in first_map.items()], key=lambda j: (np.linalg.norm(j[1].pivot - pivot)))
        other_key, other_joint = min([j for j in other_map.items()], key=lambda j: (np.linalg.norm(j[1].pivot - pivot)))
        part_pairs = product(
            (first_joint.part1, first_joint.part2),
            (other_joint.part1, other_joint.part2),
        )
        extra_joints.extend([
            Joint(par_a, par_b, pivot, rotation_axes=np.eye(3)) for par_a, par_b in part_pairs
        ])

    for pivot, first_map, s1_map in beam_map_connection:
        def min_dist(beam):
            return min(np.linalg.norm(beam.points - pivot, axis=1))

        first_beams = sorted([beam for beam in first_map.values()], key=min_dist)
        s1_beams = sorted([beam for beam in s1_map.values()], key=min_dist)

        first_beam = first_beams[0]
        other_beams = s1_beams[:2]

        part_pairs = product(
            (first_beam, ), other_beams
        )

        extra_joints.extend([
            Joint(par_a, par_b, pivot, rotation_axes=np.eye(3)) for par_a, par_b in part_pairs
        ])

    model.add_joints(extra_joints)

        # print([(beam.principle_points, min_dist(beam)) for beam in s1_map.values()])
        # print(min_dist(other_beams[1]))

    return locals()


if __name__ == '__main__':
    model = define(1)["model"]
    indices = model.point_indices()
    pairs = model.eigen_solve(
        num_pairs=20,
        # extra_constr=model.constraints_fixing_first_part(),
        extra_constr=geo_util.trivial_basis(model.point_matrix()),
    )

    print(*[e for e, _ in pairs])
    model.visualize(pairs[0][1].reshape(-1, 3), show_hinge=True,)
