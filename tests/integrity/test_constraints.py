import unittest

import numpy as np
from scipy.linalg import null_space
import util.geometry_util as geo_util
from solvers.rigidity_solver.constraints_3d import constraints_for_allowed_motions


def rigid_motion_for_coplanar_points(points):
    assert points.shape == (3, 3)
    u = points[1] - points[0]
    v = points[2] - points[0]
    normal = np.cross(u, v)
    motion = np.block([
        [u, u, u],  # translation along u
        [v, v, v],  # translation along v
        [np.cross(normal, u), np.cross(normal, v), np.zeros((3,))] # rotation about point[0]
    ])
    assert motion.shape == (3, 9)
    return motion


class TestConstraints(unittest.TestCase):
    def test_basics(self):
        source_points = np.eye(3)
        target_points = np.eye(3) + np.array([1, 0, 1])
        pivot = np.zeros((3, ))
        translation_vectors = np.array([[1, 1, 1]])
        rotation_axes = np.array([[0, 0, 1]])
        # relative rigid translation + rotation (6 DoF)
        relative_rigid_motions = geo_util.trivial_basis(
            np.vstack((source_points, target_points)), dim=3
        )
        assert relative_rigid_motions.shape == (6, 18), f"actual shape {relative_rigid_motions.shape}"

        # 3 points, in 2D plane, 3 rigid motion (2 translation + 1 rotation),
        source_rigid_motion = np.hstack((
            np.zeros((3, 9)), rigid_motion_for_coplanar_points(source_points),
        ))
        # 3 points, in 2D plane, 3 rigid motion (2 translation + 1 rotation),
        target_rigid_motion = np.hstack((
            rigid_motion_for_coplanar_points(target_points), np.zeros((3, 9)),
        ))

        relative_translation = np.hstack((
            (np.zeros((translation_vectors.shape[0], 9)),
             translation_vectors, translation_vectors, translation_vectors)
        ))
        assert relative_translation.shape == (translation_vectors.shape[0], 18)

        relative_targets = target_points - pivot
        relative_rotation = np.hstack((
            np.zeros((rotation_axes.shape[0], 9)),
            np.vstack([
                np.hstack([np.cross(ax, rel_tg) for rel_tg in relative_targets])
                for ax in rotation_axes
            ])
        ))
        assert relative_translation.shape == (rotation_axes.shape[0], 18)

        allowed_motion = np.vstack((
            relative_rigid_motions,
            source_rigid_motion,
            target_rigid_motion,
            relative_translation,
            relative_rotation,
        ))

        rank = np.linalg.matrix_rank
        nullity = lambda arr: np.linalg.matrix_rank(null_space(arr))

        print("6 points rigid", rank(relative_rigid_motions))
        print("src rigid", rank(source_rigid_motion))
        print("target rigid", rank(target_rigid_motion))
        print("relative translation", rank(relative_translation))
        print("relative rotation", rank(relative_rotation))

        print("A non-rigid", null_space(geo_util.trivial_basis(source_points)).T.shape)
        print("experimenting", rank(np.vstack((
            relative_rigid_motions,  # 6
            relative_translation,
            relative_rotation,
            # source_rigid_motion,
            # np.hstack((geo_util.trivial_basis(source_points), np.zeros((6, 9)))),
            np.hstack((null_space(geo_util.trivial_basis(source_points)).T, np.zeros((3, 9)))),
            np.hstack((np.zeros((3, 9)), null_space(geo_util.trivial_basis(target_points)).T)),
            # np.hstack(
            #     (geo_util.trivial_basis(source_points), np.zeros((6, 9)))
            # ),
        ))))

        prohibitive_motion = null_space(allowed_motion)

        constraints_rank = np.linalg.matrix_rank(prohibitive_motion)
        print(prohibitive_motion.shape, constraints_rank)

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
