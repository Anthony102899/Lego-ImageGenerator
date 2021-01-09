import numpy as np
import itertools
import util.geometry_util as geo_util
from solvers.rigidity_solver.internal_structure import get_crystal_vertices
from .constraint_3d import select_non_colinear_points, constraints_for_allowed_motions


class Model:
    def __init__(self):
        self.beams = []
        self.joints = []

    def point_matrix(self) -> np.ndarray:
        beam_points = np.vstack([b.points for b in self.beams]).reshape(-1, 3)
        # joint_points = np.array([j.virtual_points for j in self.joints]).reshape(-1, 3)
        return np.vstack((
            beam_points,
            # joint_points
        ))

    def edge_matrix(self) -> np.ndarray:
        edge_indices = []
        index_offset = 0
        for beam in self.beams:
            edge_indices.append(beam.edges() + index_offset)
            index_offset += beam.point_count
        # for joint in self.joints:
        #     edge_indices.append(joint.edges() + index_offset)
        #     index_offset += joint.virtual_point_count
        matrix = np.vstack([edges for edges in edge_indices if edges.size > 0])
        return matrix

    def constraint_matrix(self) -> np.ndarray:
        matrix = []
        # collect constraints for each joint and stack them
        for joint in self.joints:
            constraints = joint.linear_constraints(self)
            matrix.append(constraints)

        numpy_matrix = np.vstack(matrix) if len(matrix) > 0 else np.empty(0)
        return numpy_matrix

    @property
    def point_count(self):
        return sum(beam.point_count for beam in self.beams) + sum(joint.virtual_point_count for joint in self.joints)

    def add_beam(self, beam):
        self.beams.append(beam)

    def add_beams(self, beams):
        for beam in beams:
            self.add_beam(beam)

    def add_joint(self, joint):
        self.joints.append(joint)

    def add_joints(self, joints):
        for joint in joints:
            self.add_joint(joint)

    def beam_point_index(self, beam):
        beam_index = self.beams.index(beam)
        return sum(b.point_count for b in self.beams[:beam_index])

    def joint_point_indices(self):
        indices = []
        for joint in self.joints:
            offset_part_1 = self.beam_point_index(joint.part1)
            offset_part_2 = self.beam_point_index(joint.part2)

            indice_on_part_1 = select_non_colinear_points(joint.part1.points, near=joint.pivot_point)[1] + offset_part_1
            indice_on_part_2 = select_non_colinear_points(joint.part2.points, near=joint.pivot_point)[1] + offset_part_2

            indices.append((indice_on_part_1, indice_on_part_2))

        return indices


class Beam:
    def __init__(self, p1, p2, crystal_counts):
        orient = (p2 - p1) / np.linalg.norm(p2 - p1)
        self.crystals = [get_crystal_vertices(c, orient) for c in np.linspace(p1, p2, num=crystal_counts)]
        self.points = np.vstack(self.crystals)

    @classmethod
    def points(cls, points):
        beam = Beam(points[0], points[1], 2)
        beam.points = points
        return beam

    @classmethod
    def vertices(cls, points, orient):
        beam = Beam(points[0], points[1], 2)
        orient = orient / np.linalg.norm(orient) * 10
        beam.points = np.vstack((points, points + orient))
        return beam

    def edges(self) -> np.ndarray:
        index_range = range(len(self.points))
        pair_indices = np.array(list(itertools.combinations(index_range, 2)))
        return pair_indices

    @property
    def point_count(self):
        return len(self.points)


class Hinge:
    def __init__(self, part1, part2, axis, pivot_point):
        self.part1 = part1
        self.part2 = part2
        self.axis = axis
        self.pivot_point = pivot_point

    def linear_constraints(self, model: Model) -> np.ndarray:
        dim = 3

        constraint_matrix = []
        for source, target in [
            (self.part1, self.part2),
            (self.part2, self.part1)
        ]:
            source_points, source_point_indices = select_non_colinear_points(source.points, near=self.pivot_point)
            target_points, target_point_indices = select_non_colinear_points(target.points, near=self.pivot_point)

            source_point_indices += model.beam_point_index(source)
            target_point_indices += model.beam_point_index(target)

            constraints = constraints_for_allowed_motions(
                source_points,
                target_points,
                rotation_axis=self.axis,
                rotation_pivot=self.pivot_point,
            )

            i, j, k = source_point_indices
            for constraint, target_index in zip(constraints, target_point_indices):
                l = target_index
                zero_constraint = np.zeros((constraint.shape[0], model.point_count * dim))
                zero_constraint[:, i * 3: (i + 1) * 3] = constraint[:, 0: 3]
                zero_constraint[:, j * 3: (j + 1) * 3] = constraint[:, 3: 6]
                zero_constraint[:, k * 3: (k + 1) * 3] = constraint[:, 6: 9]
                zero_constraint[:, l * 3: (l + 1) * 3] = constraint[:, 9: 12]
                constraint_matrix.append(zero_constraint)

        matrix = np.vstack(constraint_matrix)
        return matrix
