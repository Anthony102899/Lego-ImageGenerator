from solvers.rigidity_solver.models import *
import numpy as np

scale = 30
p = lambda x, y, z: np.array([x, y, z], dtype=np.double) * scale
v = lambda x, y, z: np.array([x, y, z], dtype=np.double)
lerp = lambda p, q, w: p * (1 - w) + q * w
avg = lambda pts: np.average(np.asarray(pts), axis=0)

def polar_to_cart(r, azimuth, polar):
    return np.asarray([
        r * np.cos(azimuth) * np.sin(polar),
        r * np.sin(azimuth) * np.sin(polar),
        r * np.cos(polar),
    ], dtype=np.double) * scale


def define(stage):
    v_portions = 2
    h_portions = 4
    radius = 10
    v_max = np.pi / 2
    h_max = np.pi

    def beam_init(p, q, density=0.3):
        return Beam.tetra(p, q, thickness=2, density=density)


    model = Model()

    num_portion = 5
    inner_ratio = 0.5
    max_azimuth = np.pi
    pmap = {
        **{
            f"axial-{i}-outer": polar_to_cart(1, 0, p)
            for i, p in enumerate(np.linspace(0, max_azimuth, num_portion + 1))
        },
        **{
            f"axial-{i}-inner": polar_to_cart(1 * inner_ratio, 0, p)
            for i, p in enumerate(np.linspace(0, max_azimuth, num_portion + 1))
        },
    }
    pmap = {
        **pmap,
        **{
            f"cross-{i}": avg([pmap[f"axial-{i + offset}-{inout}"] for offset in (0, 1) for inout in ("inner", "outer")])
            for i in range(num_portion)
        }
    }
    bmap = {
        # **{
        #       f"confluence-{i}-inner": beam_init(
        #           lerp(pmap[f"axial-{i}-outer"], pmap[f"axial-{i}-inner"], -0.1),
        #           lerp(pmap[f"axial-{i}-outer"], pmap[f"axial-{i}-inner"], 0.1),
        #       )
        #       for i in range(num_portion + 1)
        # },
        # **{
        #     f"confluence-{i}-outer": beam_init(
        #         lerp(pmap[f"axial-{i}-outer"], pmap[f"axial-{i}-inner"], 1.1),
        #         lerp(pmap[f"axial-{i}-outer"], pmap[f"axial-{i}-inner"], 0.9),
        #     )
        #     for i in range(num_portion + 1)
        # },
        **{
            f"left-{i}": beam_init(pmap[f"axial-{i}-outer"], pmap[f"axial-{i + 1}-inner"])
            for i in range(num_portion)
        },
        **{
            f"right-{i}": beam_init(pmap[f"axial-{i}-inner"], pmap[f"axial-{i + 1}-outer"])
            for i in range(num_portion)
        },
    }
    ortho = geo_util.normalize(np.cross(pmap["cross-0"], pmap["axial-0-outer"]))
    [(
        model.add_joint(Joint(bmap[f"left-{i}"], bmap[f"right-{i}"], pivot=pmap[f"cross-{i}"],
                              rotation_axes=ortho)),
        # model.add_joint(Joint(bmap[f"left-{i}"], bmap[f"axis-{i}"], pivot=pmap[f"axial-{i}-outer"],
        #                       rotation_axes=ortho)),
        # model.add_joint(Joint(bmap[f"right-{i}"], bmap[f"axis-{i}"], pivot=pmap[f"axial-{i}-inner"],
        #                       rotation_axes=ortho)),
        # model.add_joint(Joint(bmap[f"left-{i}"], bmap[f"axis-{i + 1}"], pivot=pmap[f"axial-{i + 1}-outer"],
        #                       rotation_axes=ortho)),
        # model.add_joint(Joint(bmap[f"right-{i}"], bmap[f"axis-{i + 1}"], pivot=pmap[f"axial-{i + 1}-inner"],
        #                       rotation_axes=ortho)),
        ) for i in range(num_portion)]
    [(
        model.add_joint(Joint(bmap[f"left-{i}"], bmap[f"right-{i + 1}"], pivot=pmap[f"axial-{i + 1}-inner"],
                              rotation_axes=ortho)),
        model.add_joint(Joint(bmap[f"right-{i}"], bmap[f"left-{i + 1}"], pivot=pmap[f"axial-{i + 1}-outer"],
                              rotation_axes=ortho)),
    #     model.add_joint(Joint(bmap[f"confluence-{i}-outer"], bmap[f"axis-{i}"], pivot=pmap[f"axial-{i}-outer"],
    #                           rotation_axes=ortho)),
    #     model.add_joint(Joint(bmap[f"confluence-{i}-inner"], bmap[f"axis-{i}"], pivot=pmap[f"axial-{i}-inner"],
    #                           rotation_axes=ortho)),
    ) for i in range(num_portion - 1)]

    beams = list(bmap.values())
    model.add_beams(beams)
    joints = model.joints

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
