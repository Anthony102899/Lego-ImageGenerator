import unittest

import torch
from solvers.rigidity_solver import gradient
from solvers.rigidity_solver import algo_core as core


class TestGradient(unittest.TestCase):
    def test_simple_forward_2d(self):
        vertices = torch.tensor([
            [0, 0],
            [1, 0],
            [1, 1],
        ], dtype=torch.double)

        edges = torch.tensor([
            [0, 1],
            [1, 2],
            [2, 0],
        ], dtype=torch.long)

        K = gradient.spring_energy_matrix(vertices, edges, dim=2)
        eigenvalues, eigenvectors = torch.symeig(K, eigenvectors=True)
        indices = torch.argsort(eigenvalues)
        sorted_eigenvalues = eigenvalues[indices]

        zero = torch.tensor(0, dtype=torch.double)
        self.assertEqual(sorted_eigenvalues.size(), (6, ))
        self.assertTrue(torch.allclose(sorted_eigenvalues[:3], zero))
        self.assertFalse(torch.allclose(sorted_eigenvalues[3:], zero))


if __name__ == '__main__':
    unittest.main()
