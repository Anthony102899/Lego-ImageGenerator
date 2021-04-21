import unittest

import solvers.rigidity_solver.algo_core as core
import numpy as np
import tests.testsamples.tetra as tetra


class TestSpringEnergyMatrixAccelerate3D(unittest.TestCase):
    def test_consistency(self):
        model = tetra.square_perpendicular_axes()
        old = core.spring_energy_matrix(points=model.point_matrix(), edges=model.edge_matrix())
        print(model.point_matrix())
        print(model.edge_matrix())
        new = core.spring_energy_matrix_accelerate_3D(
            points=model.point_matrix(),
            edges=model.edge_matrix(),
            virtual_edges=False,
            abstract_edges = []
        )
        print(np.linalg.norm(new-old))
        self.assertTrue(np.allclose(old, new), msg="inconsistent result")


if __name__ == '__main__':
    unittest.main()
