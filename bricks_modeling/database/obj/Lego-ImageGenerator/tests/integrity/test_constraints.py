import unittest

import numpy as np
from scipy.linalg import null_space
import util.geometry_util as geo_util
from solvers.rigidity_solver.constraints_3d import direction_for_relative_disallowed_motions

class TestConstraints(unittest.TestCase):
    def test_basics(self):
        source_points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        target_points = np.eye(3) + np.array([1, 0, 1])
        pivot = np.array([0, 1, 0])
        translation_vectors = np.array([[1, 1, 1]])
        rotation_axes = np.array([[0, 0, 1]])

        constraints = direction_for_relative_disallowed_motions(source_points, target_points,
                                                                rotation_axes, pivot, translation_vectors)


        expected_accepted_deformation = np.block([
            [np.ones(9, ), np.zeros(9, )],
            [np.ones(9, ), np.ones(9, )],
            [np.zeros(9, ), np.zeros(9, )],
            [np.zeros(9, ), np.cross(target_points - pivot, rotation_axes[0]).reshape(-1)],
            [np.ones(9, ), np.ones(9,) + np.cross(target_points - pivot, rotation_axes[0]).reshape(-1)],
            [np.ones(9, ), np.cross(target_points - pivot, rotation_axes[0]).reshape(-1) + np.ones(9)],
        ])

        expected_rejected_deformation = np.block([
            [np.arange(9), np.zeros(9, )],
            [np.zeros(9, ), np.cross(target_points, rotation_axes[0]).reshape(-1)],
            [np.arange(9), np.cross(target_points - pivot, rotation_axes[0]).reshape(-1)],
            [np.ones(9, ), np.cross(target_points - pivot, rotation_axes[0] + 1).reshape(-1)],
            [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]), np.zeros(9)],
        ])

        for deformation in expected_accepted_deformation:
            self.assertTrue(np.allclose(constraints @ deformation, 0))

        for deformation in expected_rejected_deformation:
            self.assertFalse(np.allclose(constraints @ deformation, 0))


    def test_direction_for_disallowed_motion(self):
        source_points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ])
        target_points = np.array([
            [0, 1, 0],
            [0, 2, 0],
            [1, 2, 0],
        ])

        pivot = np.array([0, 0, 0])

        relative_translation_1 = direction_for_relative_disallowed_motions(source_points, target_points,
                                                                           rotation_pivot=pivot,
                                                                           translation_vectors=np.eye(3)[(0, 1), :],
                                                                           rotation_axes=np.eye(3))
        relative_translation_2 = direction_for_relative_disallowed_motions(source_points, target_points,
                                                                           rotation_pivot=pivot,
                                                                           translation_vectors=np.eye(3)[(1, 2), :],
                                                                           rotation_axes=np.eye(3))
        relative_translation_3 = direction_for_relative_disallowed_motions(source_points, target_points,
                                                                           rotation_pivot=pivot,
                                                                           translation_vectors=np.eye(3)[(2, 0), :],
                                                                           rotation_axes=np.eye(3))

        relative_rotation_1 = direction_for_relative_disallowed_motions(source_points, target_points,
                                                                        rotation_pivot=pivot,
                                                                        translation_vectors=np.eye(3),
                                                                        rotation_axes=np.eye(3)[(0, 1), :])
        relative_rotation_2 = direction_for_relative_disallowed_motions(source_points, target_points,
                                                                        rotation_pivot=pivot,
                                                                        translation_vectors=np.eye(3),
                                                                        rotation_axes=np.eye(3)[(1, 2), :])
        relative_rotation_3 = direction_for_relative_disallowed_motions(source_points, target_points,
                                                                        rotation_pivot=pivot,
                                                                        translation_vectors=np.eye(3),
                                                                        rotation_axes=np.eye(3)[(2, 0), :])

        relatives = (
            relative_translation_1, relative_translation_2, relative_translation_3,
            relative_rotation_1, relative_rotation_2, relative_rotation_3,
        )
        relative_motion_basis = np.vstack(relatives)

        for motion in relatives:
            self.assertEqual(motion.shape, (1, 18))

        self.assertEqual(np.linalg.matrix_rank(relative_motion_basis), 6)


if __name__ == '__main__':
    unittest.main()
