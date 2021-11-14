import numpy as np
import torch
from collections import namedtuple

from solvers.rigidity_solver.internal_structure import tetrahedron, triangulation_with_torch
from solvers.rigidity_solver.constraints_3d import select_non_colinear_points
from solvers.rigidity_solver import gradient

Part = namedtuple("Part", "points, edges, index_offset")
Joint = namedtuple("Joint", "pivot, part1_ind, part2_ind, translation, rotation_center")

class Model:
    def __init__(self, node_map, connectivity, joints):
        self.dim = len(next(iter(node_map.values())))
        assert self.dim in (2, 3)

        self.node_map = node_map
        self.connectivity = connectivity
        self.joints = joints

    def describe_model(self, only_points=False):
        offset = 0
        part_map = {}
        for key, (i, j) in self.connectivity.items():
            if self.dim == 3:
                _points, _edges = tetrahedron(self.node_map[i], self.node_map[j], density=0.3, num=5, thickness=2, mode="torch")
            else:
                _points, _edges = triangulation_with_torch(self.node_map[i], self.node_map[j], num=5, thickness=2)
            part_map[key] = Part(_points, _edges, offset)
            assert not torch.any(torch.isnan(_points)), f"exists nan, {self.node_map[i], self.node_map[j]}"

            offset += len(_points)

        point_matrix = torch.vstack([part_map[key].points for key in self.connectivity.keys()])
        assert not torch.any(torch.isnan(point_matrix))

        if only_points:
            return point_matrix

        edge_matrix = torch.vstack([
            part_map[key].edges + part_map[key].index_offset for key in self.connectivity.keys()])
        constraint_point_indices = torch.tensor(np.vstack([
            np.concatenate(
                [select_non_colinear_points(
                    part_map[j.part1_ind].points.detach().numpy(),
                    self.dim,
                    near=j.pivot(self.node_map).detach().numpy()
                )[1] + part_map[j.part1_ind].index_offset,
                 select_non_colinear_points(
                     part_map[j.part2_ind].points.detach().numpy(),
                     self.dim,
                     near=j.pivot(self.node_map).detach().numpy()
                 )[1] + part_map[j.part2_ind].index_offset]
            ) for j in self.joints
        ]), dtype=torch.long)

        return point_matrix, edge_matrix, constraint_point_indices


    def eigen_solve(self, extra_constraints=None):
        points, edges, constraint_point_indices = self.describe_model()
        with torch.no_grad():
            joint_constraints = gradient.constraint_matrix(
                points,
                pivots=[j.pivot(self.node_map) for j in self.joints],
                translation_vectors=[j.translation(self.node_map) for j in self.joints],
                rotation_centers=[j.rotation_center(self.node_map) for j in self.joints],
                joint_point_indices=constraint_point_indices,
            )

            if extra_constraints is None:
                extra_constraints = torch.vstack([
                    gradient.rigid_motion(points)
                ])

        constraints = torch.vstack([
            joint_constraints,
            extra_constraints
        ])

        B = gradient.torch_null_space(constraints)
        K = gradient.spring_energy_matrix(points, edges, dim=self.dim)
        Q = torch.chain_matmul(B.t(), K, B)

        return torch.symeig(Q)

