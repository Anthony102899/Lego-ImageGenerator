import unittest

import numpy as np
from scipy.linalg import null_space
import util.geometry_util as geo_util
from solvers.rigidity_solver import algo_core
from solvers.rigidity_solver.constraints_3d import direction_for_relative_disallowed_motions
from solvers.rigidity_solver.models import Model, Joint, Beam

class TestJointsAsStiffness(unittest.TestCase):
    def test_3d_sliding_joint_no_special_soft_constraints(self):
        '''
        We have two tets connected through a sliding joint, all the DoFs apart from the allowed ones are of
        uniform stiffness.
        '''
        source_points = np.eye(3)
        target_points = np.eye(3) + np.array([1, 0, 1])
        pivot = np.array([0, 1, 0])
        rotation_axes = None
        translation_vectors = np.array([[1, 1, 1]])

        disallowed_motions = direction_for_relative_disallowed_motions(source_points, target_points,
                                                                       rotation_axes, pivot, translation_vectors)

        part_stiffness = algo_core.spring_energy_matrix(
            np.vstack((source_points, target_points, np.ones((1, 3)) * 3, np.ones((1, 3)) * 4)),
            np.array([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
                      (0, 6), (1, 6), (2, 6),
                      (3, 7), (4, 7), (5, 7)])
        )

        joint_stiffness = np.zeros_like(part_stiffness)
        joint_stiffness[:18, :18] = disallowed_motions.T @ disallowed_motions

        global_stiffness = part_stiffness + joint_stiffness

        joint_stiffness_rank = np.linalg.matrix_rank(joint_stiffness)
        part_stiffness_rank = np.linalg.matrix_rank(part_stiffness)
        global_stiffness_rank = np.linalg.matrix_rank(global_stiffness)

        self.assertEqual(joint_stiffness_rank, 5)  # get rid of 3 rotation and 2 translation, hence 5
        self.assertEqual(part_stiffness_rank, 6 + 6)  # each part: 4 * 3 - 6 = 6; two parts: 6 + 6
        self.assertEqual(global_stiffness_rank, 17)  # 5 + 12


    def test_3d_sliding_joint_no_special_soft_constraints(self):
        '''
        We have two tets connected through a sliding joint, all the DoFs apart from the allowed ones are of
        uniform stiffness.
        '''
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]) * 10
        beams = [
            Beam.tetra(points[0], points[1]),
            Beam.tetra(points[0], points[2]),
        ]
        pivot = points[0]
        rotation_axes = np.array([0, 0, 1])
        for index, coeff in enumerate((0, 0.0000002)):
            hinge = Joint(beams[0], beams[1], pivot, rotation_axes=rotation_axes,
                          soft_translation=np.array([0, 0, 1]), soft_translation_coeff=np.array([coeff])
                          )
            model = Model()
            model.add_beams(beams)
            model.add_joints([hinge])
            pairs = model.soft_solve()
            if index == 0:
                self.assertTrue(np.isclose(pairs[6][0], 0))
                self.assertTrue(np.isclose(pairs[7][0], 0))
            elif index == 1:
                self.assertTrue(np.isclose(pairs[6][0], 0))
                self.assertFalse(np.isclose(pairs[7][0], 0))


if __name__ == '__main__':
    unittest.main()
