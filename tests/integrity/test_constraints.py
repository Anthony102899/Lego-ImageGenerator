import unittest

import numpy as np
from scipy.linalg import null_space
import util.geometry_util as geo_util
from solvers.rigidity_solver.constraints_3d import constraints_for_allowed_motions

class TestConstraints(unittest.TestCase):
    def test_basics(self):
        source_points = np.eye(3)
        target_points = np.eye(3) + np.array([1, 0, 1])
        pivot = np.array([0, 1, 0])
        translation_vectors = np.array([[1, 1, 1]])
        rotation_axes = np.array([[0, 0, 1]])

        constraints = constraints_for_allowed_motions(source_points, target_points,
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


if __name__ == '__main__':
    unittest.main()
